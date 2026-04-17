"""
Generate an independent, stratified Phase 2 validation set.

This is intentionally separate from `phase2_generate_training.py`.  The training
generator samples from a broad design distribution; this script builds a
statistical evaluation product with balanced positive cells over:

    |z0| amplitude bin x |zcrit| edge-strength bin x theta_crit bin x edge-sigma bin

The output is an all-validation HDF5 compatible with Phase 3 datasets, with
extra `stratification/*` fields for matched-FPR and bootstrap reporting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

from phase2_generate_training import (
    GEOMETRY_MODE_CODES,
    GEOMETRY_MODES,
    MASK_THRESHOLD,
    NSIDE_WORKING,
    SPLIT_VAL,
    apply_observing_model_to_patch,
    build_balanced_sign_pairs,
    build_coordinate_pool,
    generate_camb_realizations,
    load_mask,
    project_patch,
    sample_actual_geometry_mode,
    sample_log_uniform,
    sample_signal_geometry,
    split_index_pool,
)
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN, inject_signal_into_patch
from phase_dataset_utils import patch_center_pixel, stable_group_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "validation_stratified_v1"

Z0_AMP_BINS = (
    ("z0_1e-6_to_1e-5", 1e-6, 1e-5),
    ("z0_1e-5_to_3e-5", 1e-5, 3e-5),
    ("z0_3e-5_to_1e-4", 3e-5, 1e-4),
)
ZCRIT_ABS_BINS = (
    ("zcrit_smooth_1e-6_to_5e-6", 1e-6, 5e-6),
    ("zcrit_weak_5e-6_to_3e-5", 5e-6, 3e-5),
    ("zcrit_strong_3e-5_to_1e-4", 3e-5, 1e-4),
)
THETA_BINS_DEG = (
    ("theta_5_to_10", 5.0, 10.0),
    ("theta_10_to_15", 10.0, 15.0),
    ("theta_15_to_20", 15.0, 20.0),
    ("theta_20_to_25", 20.0, 25.0),
)
EDGE_SIGMA_BINS_DEG = (
    ("edge_sigma_0p3_to_0p65", 0.3, 0.65),
    ("edge_sigma_0p65_to_1p0", 0.65, 1.0),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an independent stratified validation HDF5 for Phase 3 model comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--samples-per-cell", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--split-seed", type=int, default=271828)
    parser.add_argument("--pool-size", type=int, default=9000)
    parser.add_argument("--num-cmb-realizations", type=int, default=256)
    parser.add_argument("--geometry-mode", type=str, default="contained", choices=GEOMETRY_MODES)
    parser.add_argument(
        "--truncated-positive-fraction",
        type=float,
        default=0.0,
        help="Fraction of positives drawn from edge-crossing geometry when --geometry-mode=mixed.",
    )
    parser.add_argument(
        "--truncated-visible-fraction-min",
        type=float,
        default=0.15,
        help="Minimum visible fraction of the full causal disc for truncated positives.",
    )
    parser.add_argument(
        "--truncated-visible-fraction-max",
        type=float,
        default=0.95,
        help="Maximum visible fraction of the full causal disc for truncated positives.",
    )
    parser.add_argument("--truncated-max-center-draws", type=int, default=256)
    parser.add_argument("--signal-center-edge-margin-pix", type=float, default=0.0)
    parser.add_argument("--contained-margin-deg", type=float, default=0.5)
    parser.add_argument("--beam-fwhm-arcmin", type=float, default=15.0)
    parser.add_argument("--noise-sigma-uk-arcmin", type=float, default=30.0)
    parser.add_argument("--noise-corr-fwhm-arcmin", type=float, default=0.0)
    parser.add_argument(
        "--exclude-h5",
        type=str,
        default=str(PROJECT_ROOT / "data" / "training_v4" / "training_data.h5"),
        help="Optional existing dataset whose coordinate pool should be avoided.",
    )
    parser.add_argument(
        "--exclude-radius-deg",
        type=float,
        default=0.25,
        help="Reject new patch centers within this angular distance of --exclude-h5 coordinate-pool centers.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.samples_per_cell <= 0:
        raise ValueError("--samples-per-cell must be positive.")
    if args.pool_size <= 1:
        raise ValueError("--pool-size must be at least 2.")
    if args.num_cmb_realizations <= 1:
        raise ValueError("--num-cmb-realizations must be at least 2.")
    if args.contained_margin_deg < 0.0:
        raise ValueError("--contained-margin-deg must be non-negative.")
    if args.signal_center_edge_margin_pix < 0.0:
        raise ValueError("--signal-center-edge-margin-pix must be non-negative.")
    if not (0.0 <= args.truncated_positive_fraction <= 1.0):
        raise ValueError("--truncated-positive-fraction must be between 0 and 1.")
    if not (0.0 < args.truncated_visible_fraction_min <= 1.0):
        raise ValueError("--truncated-visible-fraction-min must be in (0, 1].")
    if not (0.0 < args.truncated_visible_fraction_max <= 1.0):
        raise ValueError("--truncated-visible-fraction-max must be in (0, 1].")
    if args.truncated_visible_fraction_min > args.truncated_visible_fraction_max:
        raise ValueError("--truncated-visible-fraction-min must be <= --truncated-visible-fraction-max.")
    if args.truncated_max_center_draws <= 0:
        raise ValueError("--truncated-max-center-draws must be positive.")
    if args.exclude_radius_deg < 0.0:
        raise ValueError("--exclude-radius-deg must be non-negative.")

    positive_cells = len(Z0_AMP_BINS) * len(ZCRIT_ABS_BINS) * len(THETA_BINS_DEG) * len(EDGE_SIGMA_BINS_DEG)
    positive_count = positive_cells * args.samples_per_cell
    if positive_count >= args.num_samples:
        raise ValueError(
            f"Requested {positive_count} positives from stratified cells, leaving no negatives. "
            "Increase --num-samples or reduce --samples-per-cell."
        )


def load_exclusion_vectors(exclude_h5):
    path = Path(exclude_h5)
    if not exclude_h5 or not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        if "coordinate_pool" not in h5:
            return None
        lon = np.asarray(h5["coordinate_pool"]["glon_deg"][:], dtype=np.float64)
        lat = np.asarray(h5["coordinate_pool"]["glat_deg"][:], dtype=np.float64)
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    return np.stack(
        (
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ),
        axis=1,
    )


def filter_excluded_coordinates(coords, mask_fractions, exclusion_vectors, min_distance_deg):
    if exclusion_vectors is None or min_distance_deg <= 0.0:
        return coords, mask_fractions
    lon_rad = np.radians(coords[:, 0].astype(np.float64))
    lat_rad = np.radians(coords[:, 1].astype(np.float64))
    vectors = np.stack(
        (
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ),
        axis=1,
    )
    min_cos = math.cos(math.radians(float(min_distance_deg)))
    keep = np.ones(len(coords), dtype=bool)
    chunk = 512
    for start in range(0, len(coords), chunk):
        stop = min(start + chunk, len(coords))
        max_dot = vectors[start:stop] @ exclusion_vectors.T
        keep[start:stop] = np.max(max_dot, axis=1) < min_cos
    return coords[keep], mask_fractions[keep]


def sample_uniform_in_bin(rng, low, high):
    return float(rng.uniform(float(low), float(high)))


def sample_log_uniform_in_bin(rng, low, high):
    return sample_log_uniform(rng, float(low), float(high))


def iter_positive_cells():
    for z0_idx, z0_bin in enumerate(Z0_AMP_BINS):
        for zcrit_idx, zcrit_bin in enumerate(ZCRIT_ABS_BINS):
            for theta_idx, theta_bin in enumerate(THETA_BINS_DEG):
                for edge_idx, edge_bin in enumerate(EDGE_SIGMA_BINS_DEG):
                    cell_id = (((z0_idx * len(ZCRIT_ABS_BINS)) + zcrit_idx) * len(THETA_BINS_DEG) + theta_idx)
                    cell_id = cell_id * len(EDGE_SIGMA_BINS_DEG) + edge_idx
                    yield cell_id, z0_idx, z0_bin, zcrit_idx, zcrit_bin, theta_idx, theta_bin, edge_idx, edge_bin


def allocate_arrays(num_samples):
    patches = np.empty((num_samples, PATCH_PIX, PATCH_PIX), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.uint8)
    masks = np.zeros((num_samples, PATCH_PIX, PATCH_PIX), dtype=np.uint8)
    metadata = {
        "sample_index": np.arange(num_samples, dtype=np.int32),
        "split_tag": np.full(num_samples, SPLIT_VAL, dtype=np.uint8),
        "glon_deg": np.empty(num_samples, dtype=np.float32),
        "glat_deg": np.empty(num_samples, dtype=np.float32),
        "coord_pool_idx": np.empty(num_samples, dtype=np.int32),
        "coord_mask_fraction": np.empty(num_samples, dtype=np.float32),
        "cmb_realization_idx": np.empty(num_samples, dtype=np.int32),
        "background_id": np.zeros(num_samples, dtype=np.uint64),
        "split_group_id": np.zeros(num_samples, dtype=np.uint64),
    }
    truth = {
        "has_signal": np.zeros(num_samples, dtype=np.uint8),
        "event_id": np.zeros(num_samples, dtype=np.uint64),
        "theta_crit_deg": np.zeros(num_samples, dtype=np.float32),
        "z0": np.zeros(num_samples, dtype=np.float32),
        "zcrit": np.zeros(num_samples, dtype=np.float32),
        "edge_sigma_deg": np.zeros(num_samples, dtype=np.float32),
        "signal_center_x_pix": np.full(num_samples, patch_center_pixel(PATCH_PIX), dtype=np.float32),
        "signal_center_y_pix": np.full(num_samples, patch_center_pixel(PATCH_PIX), dtype=np.float32),
        "signal_center_dx_deg": np.zeros(num_samples, dtype=np.float32),
        "signal_center_dy_deg": np.zeros(num_samples, dtype=np.float32),
        "geometry_mode_code": np.zeros(num_samples, dtype=np.uint8),
        "fully_contained": np.zeros(num_samples, dtype=np.uint8),
        "target_touches_edge": np.zeros(num_samples, dtype=np.uint8),
        "visible_target_fraction": np.zeros(num_samples, dtype=np.float32),
        "visible_target_pixels": np.zeros(num_samples, dtype=np.int32),
        "full_disc_pixels_est": np.zeros(num_samples, dtype=np.int32),
        "target_edge_contact_pixels": np.zeros(num_samples, dtype=np.int32),
        "disc_edge_margin_pix": np.zeros(num_samples, dtype=np.float32),
        "signal_center_in_patch": np.zeros(num_samples, dtype=np.uint8),
    }
    stratification = {
        "positive_cell_id": np.full(num_samples, -1, dtype=np.int32),
        "z0_amp_bin": np.full(num_samples, -1, dtype=np.int16),
        "zcrit_abs_bin": np.full(num_samples, -1, dtype=np.int16),
        "theta_bin": np.full(num_samples, -1, dtype=np.int16),
        "edge_sigma_bin": np.full(num_samples, -1, dtype=np.int16),
    }
    return patches, labels, masks, metadata, truth, stratification


def write_h5(output_dir, patches, labels, masks, metadata, truth, stratification, coord_pool, coord_mask_fractions, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "validation_data.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("patches", data=patches, compression="gzip", shuffle=True)
        h5.create_dataset("labels", data=labels, compression="gzip", shuffle=True)
        h5.create_dataset("masks", data=masks, compression="gzip", shuffle=True)
        for group_name, payload in (
            ("metadata", metadata),
            ("truth", truth),
            ("stratification", stratification),
        ):
            group = h5.create_group(group_name)
            for key, value in payload.items():
                group.create_dataset(key, data=value, compression="gzip", shuffle=True)
        splits = h5.create_group("splits")
        splits.create_dataset("train_idx", data=np.zeros(0, dtype=np.int64), compression="gzip", shuffle=True)
        splits.create_dataset("val_idx", data=np.arange(len(labels), dtype=np.int64), compression="gzip", shuffle=True)
        coord_group = h5.create_group("coordinate_pool")
        coord_group.create_dataset("glon_deg", data=coord_pool[:, 0].astype(np.float32), compression="gzip", shuffle=True)
        coord_group.create_dataset("glat_deg", data=coord_pool[:, 1].astype(np.float32), compression="gzip", shuffle=True)
        coord_group.create_dataset("mask_fraction", data=coord_mask_fractions.astype(np.float32), compression="gzip", shuffle=True)
        summary_group = h5.create_group("summary")
        for key, value in summary.items():
            summary_group.attrs[key] = value

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return h5_path, summary_path


def main():
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)

    mask_256, sky_fraction = load_mask()
    coord_pool, coord_mask_fractions = build_coordinate_pool(mask_256, args.pool_size, rng)
    exclusion_vectors = load_exclusion_vectors(args.exclude_h5)
    coord_pool, coord_mask_fractions = filter_excluded_coordinates(
        coord_pool,
        coord_mask_fractions,
        exclusion_vectors,
        args.exclude_radius_deg,
    )
    if len(coord_pool) < 2:
        raise RuntimeError("Coordinate pool empty after exclusion filtering.")
    print(f"  Coordinate pool after exclusion: {len(coord_pool)}")

    cmb_realizations, camb_params = generate_camb_realizations(args.num_cmb_realizations, rng)

    positive_cells = list(iter_positive_cells())
    num_positive = len(positive_cells) * args.samples_per_cell
    num_negative = args.num_samples - num_positive
    print("\n=== Stratified validation design ===")
    print(f"  Positive cells:      {len(positive_cells)}")
    print(f"  Samples per cell:    {args.samples_per_cell}")
    print(f"  Positive samples:    {num_positive}")
    print(f"  Negative samples:    {num_negative}")
    print(f"  Total samples:       {args.num_samples}")
    print(f"  Geometry mode:       {args.geometry_mode}")

    patches, labels, masks, metadata, truth, stratification = allocate_arrays(args.num_samples)
    sign_pairs = build_balanced_sign_pairs(num_positive, rng)
    positive_counter = 0
    sample_idx = 0
    center_pix = patch_center_pixel(PATCH_PIX)

    for cell in positive_cells:
        cell_id, z0_idx, z0_bin, zcrit_idx, zcrit_bin, theta_idx, theta_bin, edge_idx, edge_bin = cell
        for _ in range(args.samples_per_cell):
            coord_idx = int(rng.integers(0, len(coord_pool)))
            realization_idx = int(rng.integers(0, len(cmb_realizations)))
            lon_i, lat_i = coord_pool[coord_idx]
            clean_patch = np.asarray(project_patch(cmb_realizations[realization_idx], float(lon_i), float(lat_i)), dtype=np.float32)

            sign_z0_i, sign_zcrit_i = sign_pairs[positive_counter]
            theta_i = sample_uniform_in_bin(rng, theta_bin[1], theta_bin[2])
            z0_i = float(sign_z0_i) * sample_log_uniform_in_bin(rng, z0_bin[1], z0_bin[2])
            zcrit_i = float(sign_zcrit_i) * sample_log_uniform_in_bin(rng, zcrit_bin[1], zcrit_bin[2])
            edge_sigma_i = sample_uniform_in_bin(rng, edge_bin[1], edge_bin[2])
            actual_geometry_mode = sample_actual_geometry_mode(
                rng,
                args.geometry_mode,
                args.truncated_positive_fraction,
            )
            geometry_i = sample_signal_geometry(
                rng=rng,
                npix=PATCH_PIX,
                theta_crit_deg=theta_i,
                geometry_mode=actual_geometry_mode,
                edge_margin_pix=args.signal_center_edge_margin_pix,
                contained_margin_deg=args.contained_margin_deg,
                truncated_visible_fraction_min=args.truncated_visible_fraction_min,
                truncated_visible_fraction_max=args.truncated_visible_fraction_max,
                truncated_max_center_draws=args.truncated_max_center_draws,
            )
            center_x_i = geometry_i["center_x_pix"]
            center_y_i = geometry_i["center_y_pix"]
            center_dx_i = (center_x_i - center_pix) * RESO_ARCMIN / 60.0
            center_dy_i = (center_y_i - center_pix) * RESO_ARCMIN / 60.0

            injected_patch, _ = inject_signal_into_patch(
                clean_patch,
                z0_i,
                zcrit_i,
                theta_i,
                edge_sigma_deg=edge_sigma_i,
                center_x_pix=center_x_i,
                center_y_pix=center_y_i,
            )
            mask_i = geometry_i["mask"]
            observed_patch = apply_observing_model_to_patch(
                injected_patch,
                rng=rng,
                beam_fwhm_arcmin=args.beam_fwhm_arcmin,
                noise_sigma_uk_arcmin=args.noise_sigma_uk_arcmin,
                noise_corr_fwhm_arcmin=args.noise_corr_fwhm_arcmin,
            )

            background_id = stable_group_id("stratified_val_camb", args.seed, realization_idx, coord_idx)
            event_id = stable_group_id(
                background_id,
                cell_id,
                f"{theta_i:.6f}",
                f"{z0_i:.6e}",
                f"{zcrit_i:.6e}",
                f"{edge_sigma_i:.6f}",
                f"{center_x_i:.6f}",
                f"{center_y_i:.6f}",
                actual_geometry_mode,
            )

            patches[sample_idx] = observed_patch
            labels[sample_idx] = 1
            masks[sample_idx] = mask_i
            metadata["glon_deg"][sample_idx] = lon_i
            metadata["glat_deg"][sample_idx] = lat_i
            metadata["coord_pool_idx"][sample_idx] = coord_idx
            metadata["coord_mask_fraction"][sample_idx] = coord_mask_fractions[coord_idx]
            metadata["cmb_realization_idx"][sample_idx] = realization_idx
            metadata["background_id"][sample_idx] = np.uint64(background_id)
            metadata["split_group_id"][sample_idx] = np.uint64(background_id)
            truth["has_signal"][sample_idx] = 1
            truth["event_id"][sample_idx] = np.uint64(event_id)
            truth["theta_crit_deg"][sample_idx] = theta_i
            truth["z0"][sample_idx] = z0_i
            truth["zcrit"][sample_idx] = zcrit_i
            truth["edge_sigma_deg"][sample_idx] = edge_sigma_i
            truth["signal_center_x_pix"][sample_idx] = center_x_i
            truth["signal_center_y_pix"][sample_idx] = center_y_i
            truth["signal_center_dx_deg"][sample_idx] = center_dx_i
            truth["signal_center_dy_deg"][sample_idx] = center_dy_i
            truth["geometry_mode_code"][sample_idx] = GEOMETRY_MODE_CODES[actual_geometry_mode]
            truth["fully_contained"][sample_idx] = geometry_i["fully_contained"]
            truth["target_touches_edge"][sample_idx] = geometry_i["target_touches_edge"]
            truth["visible_target_fraction"][sample_idx] = geometry_i["visible_target_fraction"]
            truth["visible_target_pixels"][sample_idx] = geometry_i["visible_target_pixels"]
            truth["full_disc_pixels_est"][sample_idx] = geometry_i["full_disc_pixels_est"]
            truth["target_edge_contact_pixels"][sample_idx] = geometry_i["target_edge_contact_pixels"]
            truth["disc_edge_margin_pix"][sample_idx] = geometry_i["disc_edge_margin_pix"]
            truth["signal_center_in_patch"][sample_idx] = geometry_i["signal_center_in_patch"]
            stratification["positive_cell_id"][sample_idx] = cell_id
            stratification["z0_amp_bin"][sample_idx] = z0_idx
            stratification["zcrit_abs_bin"][sample_idx] = zcrit_idx
            stratification["theta_bin"][sample_idx] = theta_idx
            stratification["edge_sigma_bin"][sample_idx] = edge_idx

            sample_idx += 1
            positive_counter += 1
            if sample_idx % 250 == 0:
                print(f"  Generated positives: {sample_idx:4d} / {num_positive}")

    for _ in range(num_negative):
        coord_idx = int(rng.integers(0, len(coord_pool)))
        realization_idx = int(rng.integers(0, len(cmb_realizations)))
        lon_i, lat_i = coord_pool[coord_idx]
        clean_patch = np.asarray(project_patch(cmb_realizations[realization_idx], float(lon_i), float(lat_i)), dtype=np.float32)
        observed_patch = apply_observing_model_to_patch(
            clean_patch,
            rng=rng,
            beam_fwhm_arcmin=args.beam_fwhm_arcmin,
            noise_sigma_uk_arcmin=args.noise_sigma_uk_arcmin,
            noise_corr_fwhm_arcmin=args.noise_corr_fwhm_arcmin,
        )
        background_id = stable_group_id("stratified_val_camb", args.seed, realization_idx, coord_idx)
        patches[sample_idx] = observed_patch
        metadata["glon_deg"][sample_idx] = lon_i
        metadata["glat_deg"][sample_idx] = lat_i
        metadata["coord_pool_idx"][sample_idx] = coord_idx
        metadata["coord_mask_fraction"][sample_idx] = coord_mask_fractions[coord_idx]
        metadata["cmb_realization_idx"][sample_idx] = realization_idx
        metadata["background_id"][sample_idx] = np.uint64(background_id)
        metadata["split_group_id"][sample_idx] = np.uint64(background_id)
        sample_idx += 1
        if (sample_idx - num_positive) % 250 == 0 or sample_idx == args.num_samples:
            print(f"  Generated negatives: {sample_idx - num_positive:4d} / {num_negative}")

    permutation = rng.permutation(args.num_samples)
    patches = patches[permutation]
    labels = labels[permutation]
    masks = masks[permutation]
    for payload in (metadata, truth, stratification):
        for key, value in payload.items():
            payload[key] = value[permutation]
    metadata["sample_index"] = np.arange(args.num_samples, dtype=np.int32)

    bin_schema = {
        "z0_amp_bins": [{"id": idx, "name": name, "low": low, "high": high} for idx, (name, low, high) in enumerate(Z0_AMP_BINS)],
        "zcrit_abs_bins": [{"id": idx, "name": name, "low": low, "high": high} for idx, (name, low, high) in enumerate(ZCRIT_ABS_BINS)],
        "theta_bins_deg": [{"id": idx, "name": name, "low": low, "high": high} for idx, (name, low, high) in enumerate(THETA_BINS_DEG)],
        "edge_sigma_bins_deg": [{"id": idx, "name": name, "low": low, "high": high} for idx, (name, low, high) in enumerate(EDGE_SIGMA_BINS_DEG)],
    }
    summary = {
        "num_samples": int(args.num_samples),
        "num_positive": int(labels.sum()),
        "num_negative": int(args.num_samples - labels.sum()),
        "num_positive_cells": int(len(positive_cells)),
        "samples_per_cell": int(args.samples_per_cell),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "pool_size_requested": int(args.pool_size),
        "pool_size_after_exclusion": int(len(coord_pool)),
        "num_cmb_realizations": int(args.num_cmb_realizations),
        "nside": int(NSIDE_WORKING),
        "patch_pixels": int(PATCH_PIX),
        "reso_arcmin": float(RESO_ARCMIN),
        "mask_threshold": float(MASK_THRESHOLD),
        "sky_fraction": float(sky_fraction),
        "geometry_mode": args.geometry_mode,
        "truncated_positive_fraction_requested": float(args.truncated_positive_fraction),
        "truncated_visible_fraction_min": float(args.truncated_visible_fraction_min),
        "truncated_visible_fraction_max": float(args.truncated_visible_fraction_max),
        "truncated_max_center_draws": int(args.truncated_max_center_draws),
        "signal_center_edge_margin_pix": float(args.signal_center_edge_margin_pix),
        "num_positive_fully_contained": int(truth["fully_contained"][labels == 1].sum()),
        "num_positive_touching_edge": int(truth["target_touches_edge"][labels == 1].sum()),
        "mean_positive_visible_target_fraction": float(np.mean(truth["visible_target_fraction"][labels == 1])),
        "min_positive_visible_target_fraction": float(np.min(truth["visible_target_fraction"][labels == 1])),
        "contained_margin_deg": float(args.contained_margin_deg),
        "beam_fwhm_arcmin": float(args.beam_fwhm_arcmin),
        "noise_sigma_uk_arcmin": float(args.noise_sigma_uk_arcmin),
        "noise_corr_fwhm_arcmin": float(args.noise_corr_fwhm_arcmin),
        "exclude_h5": str(Path(args.exclude_h5).resolve()) if args.exclude_h5 else "",
        "exclude_radius_deg": float(args.exclude_radius_deg),
        "bin_schema": json.dumps(bin_schema, sort_keys=True),
        "camb_params": json.dumps(camb_params, sort_keys=True),
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }

    h5_path, summary_path = write_h5(
        args.output_dir,
        patches,
        labels,
        masks,
        metadata,
        truth,
        stratification,
        coord_pool,
        coord_mask_fractions,
        summary,
    )
    print("\n=== Saved stratified validation ===")
    print(f"  HDF5:    {h5_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
