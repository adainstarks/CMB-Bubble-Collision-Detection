from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi


DEFAULT_PATCH_PIX = 256
DEFAULT_RESO_ARCMIN = 13.0


def patch_center_pixel(npix: int) -> float:
    return (int(npix) - 1) / 2.0


def pixel_to_patch_offsets_deg(
    x_pix: float,
    y_pix: float,
    npix: int = DEFAULT_PATCH_PIX,
    reso_arcmin: float = DEFAULT_RESO_ARCMIN,
):
    center = patch_center_pixel(npix)
    scale = float(reso_arcmin) / 60.0
    dx_deg = (float(x_pix) - center) * scale
    dy_deg = (float(y_pix) - center) * scale
    return dx_deg, dy_deg


def patch_offsets_deg_to_pixel(
    dx_deg: float,
    dy_deg: float,
    npix: int = DEFAULT_PATCH_PIX,
    reso_arcmin: float = DEFAULT_RESO_ARCMIN,
):
    center = patch_center_pixel(npix)
    inv_scale = 60.0 / float(reso_arcmin)
    x_pix = center + float(dx_deg) * inv_scale
    y_pix = center + float(dy_deg) * inv_scale
    return x_pix, y_pix


def patch_offsets_deg_to_sky(glon_center_deg: float, glat_center_deg: float, dx_deg: float, dy_deg: float):
    lon0 = math.radians(float(glon_center_deg))
    lat0 = math.radians(float(glat_center_deg))

    x = math.tan(math.radians(float(dx_deg)))
    y = math.tan(math.radians(float(dy_deg)))
    rho = math.hypot(x, y)
    if rho <= 1e-15:
        return float(glon_center_deg) % 360.0, float(glat_center_deg)

    c = math.atan(rho)
    sin_c = math.sin(c)
    cos_c = math.cos(c)

    lat = math.asin(cos_c * math.sin(lat0) + (y * sin_c * math.cos(lat0) / rho))
    lon = lon0 + math.atan2(
        x * sin_c,
        rho * math.cos(lat0) * cos_c - y * math.sin(lat0) * sin_c,
    )
    return math.degrees(lon) % 360.0, math.degrees(lat)


def make_plane_coordinate_grids(npix: int, reso_arcmin: float):
    center = patch_center_pixel(npix)
    iy, ix = np.mgrid[0:npix, 0:npix]
    dx_angle_rad = np.radians((ix - center) * float(reso_arcmin) / 60.0)
    dy_angle_rad = np.radians((iy - center) * float(reso_arcmin) / 60.0)
    return np.tan(dx_angle_rad), np.tan(dy_angle_rad)


def make_angular_distance_grid(
    npix: int,
    reso_arcmin: float,
    center_x_pix: float | None = None,
    center_y_pix: float | None = None,
):
    center = patch_center_pixel(npix)
    if center_x_pix is None:
        center_x_pix = center
    if center_y_pix is None:
        center_y_pix = center

    x_plane, y_plane = make_plane_coordinate_grids(npix, reso_arcmin)
    center_dx_angle = np.radians((float(center_x_pix) - center) * float(reso_arcmin) / 60.0)
    center_dy_angle = np.radians((float(center_y_pix) - center) * float(reso_arcmin) / 60.0)
    center_x_plane = np.tan(center_dx_angle)
    center_y_plane = np.tan(center_dy_angle)

    pixel_vec = np.stack((x_plane, y_plane, np.ones_like(x_plane)), axis=0)
    pixel_vec /= np.linalg.norm(pixel_vec, axis=0, keepdims=True)

    center_vec = np.array([center_x_plane, center_y_plane, 1.0], dtype=np.float64)
    center_vec /= np.linalg.norm(center_vec)

    cos_theta = np.clip(np.sum(pixel_vec * center_vec[:, None, None], axis=0), -1.0, 1.0)
    return np.arccos(cos_theta)


def stable_group_id(*parts):
    encoded = bytearray()
    for part in parts:
        if isinstance(part, (float, np.floating)):
            encoded.extend(b"f64:")
            encoded.extend(struct.pack("!d", float(part)))
        elif isinstance(part, (int, np.integer)):
            encoded.extend(b"u64:")
            encoded.extend(struct.pack("!Q", int(part) & ((1 << 64) - 1)))
        else:
            encoded.extend(b"s:")
            encoded.extend(str(part).encode("utf-8"))
        encoded.extend(b"|")
    digest = hashlib.sha256(bytes(encoded)).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _open_h5_if_needed(h5_or_path, mode="r"):
    if isinstance(h5_or_path, (str, Path)):
        return h5py.File(h5_or_path, mode), True
    return h5_or_path, False


def get_truth_group(h5):
    if "truth" in h5:
        return h5["truth"]
    if "metadata" in h5:
        return h5["metadata"]
    raise KeyError("Dataset does not contain a `truth` or `metadata` group.")


def load_truth_array(h5_or_path, name, dtype=None):
    h5, should_close = _open_h5_if_needed(h5_or_path)
    try:
        group = get_truth_group(h5)
        aliases = []
        if name == "has_signal":
            aliases.append("is_positive")
        if name == "coord_mask_fraction":
            aliases.append("mask_fraction")
        if name not in group:
            for alias in aliases:
                if alias in group:
                    name = alias
                    break
            else:
                raise KeyError(f"Truth field not found: {name}")
        value = np.asarray(group[name][:])
        if dtype is not None:
            value = value.astype(dtype)
        return value
    finally:
        if should_close:
            h5.close()


def load_metadata_array(h5_or_path, name, dtype=None):
    h5, should_close = _open_h5_if_needed(h5_or_path)
    try:
        if "metadata" not in h5 or name not in h5["metadata"]:
            raise KeyError(f"Metadata field not found: {name}")
        value = np.asarray(h5["metadata"][name][:])
        if dtype is not None:
            value = value.astype(dtype)
        return value
    finally:
        if should_close:
            h5.close()


def load_optional_metadata_array(h5_or_path, name, dtype=None, default_value=0):
    try:
        return load_metadata_array(h5_or_path, name, dtype=dtype)
    except KeyError:
        h5, should_close = _open_h5_if_needed(h5_or_path)
        try:
            size = int(h5["labels"].shape[0])
        finally:
            if should_close:
                h5.close()
        value = np.full(size, default_value)
        if dtype is not None:
            value = value.astype(dtype)
        return value


def load_signal_strength(h5_path):
    z0 = load_truth_array(h5_path, "z0", dtype=np.float32)
    zcrit = load_truth_array(h5_path, "zcrit", dtype=np.float32)
    return np.maximum(np.abs(z0), np.abs(zcrit))


def load_predefined_split_indices(h5_path):
    with h5py.File(h5_path, "r") as h5:
        if "splits" not in h5:
            return None
        split_group = h5["splits"]
        if "train_idx" not in split_group:
            return None
        if "calibration_idx" in split_group:
            calibration_key = "calibration_idx"
        elif "val_idx" in split_group:
            calibration_key = "val_idx"
        else:
            return None
        out = {
            "train_idx": np.asarray(split_group["train_idx"][:], dtype=np.int64),
            "val_idx": np.asarray(split_group[calibration_key][:], dtype=np.int64),
            "calibration_idx": np.asarray(split_group[calibration_key][:], dtype=np.int64),
        }
        if "test_idx" in split_group:
            out["test_idx"] = np.asarray(split_group["test_idx"][:], dtype=np.int64)
        return out


def select_candidate_component(pred_mask, prob_map):
    labels, count = ndi.label(pred_mask.astype(bool))
    if count <= 0:
        return None

    best_label = None
    best_score = None
    for label_id in range(1, count + 1):
        component = labels == label_id
        area = int(component.sum())
        max_prob = float(prob_map[component].max())
        mean_prob = float(prob_map[component].mean())
        score = (max_prob, mean_prob, area)
        if best_score is None or score > best_score:
            best_label = label_id
            best_score = score
    return labels == best_label


def build_patch_candidate(
    prob_map,
    threshold,
    patch_center_glon_deg,
    patch_center_glat_deg,
    sample_index,
    reso_arcmin: float = DEFAULT_RESO_ARCMIN,
):
    prob_map = np.asarray(prob_map, dtype=np.float32)
    pred_mask = prob_map >= float(threshold)
    component = select_candidate_component(pred_mask, prob_map)

    candidate = {
        "sample_index": int(sample_index),
        "threshold": float(threshold),
        "patch_center_glon_deg": float(patch_center_glon_deg),
        "patch_center_glat_deg": float(patch_center_glat_deg),
        "has_candidate": False,
        "score_max": 0.0,
        "score_mean": 0.0,
        "positive_fraction": 0.0,
        "candidate_x_pix": None,
        "candidate_y_pix": None,
        "candidate_dx_deg": None,
        "candidate_dy_deg": None,
        "candidate_glon_deg": None,
        "candidate_glat_deg": None,
        "radius_est_deg": 0.0,
        "area_pixels": 0,
    }
    if component is None:
        return candidate, pred_mask.astype(np.uint8)

    ys, xs = np.nonzero(component)
    weights = np.clip(prob_map[component], 1e-8, None)
    x_pix = float(np.average(xs, weights=weights))
    y_pix = float(np.average(ys, weights=weights))
    dx_deg, dy_deg = pixel_to_patch_offsets_deg(
        x_pix=x_pix,
        y_pix=y_pix,
        npix=prob_map.shape[-1],
        reso_arcmin=reso_arcmin,
    )
    glon_deg, glat_deg = patch_offsets_deg_to_sky(
        glon_center_deg=patch_center_glon_deg,
        glat_center_deg=patch_center_glat_deg,
        dx_deg=dx_deg,
        dy_deg=dy_deg,
    )

    theta_grid = make_angular_distance_grid(
        npix=prob_map.shape[-1],
        reso_arcmin=reso_arcmin,
        center_x_pix=x_pix,
        center_y_pix=y_pix,
    )
    theta_component = np.degrees(theta_grid[component])
    radius_est_deg = float(np.percentile(theta_component, 95.0)) if theta_component.size else 0.0

    candidate.update(
        {
            "has_candidate": True,
            "score_max": float(prob_map[component].max()),
            "score_mean": float(prob_map[component].mean()),
            "positive_fraction": float(component.mean()),
            "candidate_x_pix": x_pix,
            "candidate_y_pix": y_pix,
            "candidate_dx_deg": dx_deg,
            "candidate_dy_deg": dy_deg,
            "candidate_glon_deg": glon_deg,
            "candidate_glat_deg": glat_deg,
            "radius_est_deg": radius_est_deg,
            "area_pixels": int(component.sum()),
        }
    )
    return candidate, component.astype(np.uint8)
