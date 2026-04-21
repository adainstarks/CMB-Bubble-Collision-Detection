"""
Build fixed-FPR sensitivity curves for Phase 3 screeners.

This script generates an independent sensitivity grid over signal amplitude and
angular radius, calibrates every method on the same negative patches at a fixed
FPR, and reports P_det(A, theta_c) with binomial confidence intervals.

The injected signal is the hard-boundary Feeney PRD 84 linear cap reference
signal with edge_sigma_deg = 0.0.  That keeps the comparison against the
circular template screen interpretable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.stats import binomtest

import phase3_train_unet as p3
from phase2_generate_stratified_validation import filter_excluded_coordinates, load_exclusion_vectors
from phase2_generate_training import (
    MASK_THRESHOLD,
    NSIDE_WORKING,
    apply_observing_model_to_patch,
    build_balanced_sign_pairs,
    build_coordinate_pool,
    coordinate_cluster_ids,
    coordinate_cluster_pixels,
    generate_camb_realizations,
    fwhm_arcmin_to_sigma_pixels,
    load_mask,
    make_disk_mask,
    project_patch,
    sample_signal_center_pixels,
    target_touches_patch_edge,
)
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN, bubble_collision_signal, inject_signal_into_patch
from phase3_evaluate_run import load_json, resolve_checkpoint_path
from phase_dataset_utils import make_angular_distance_grid, patch_center_pixel, stable_group_id
from phase3_method_registry import CIRCULAR_TEMPLATE_SCREEN, method_metadata
from phase3_thresholds import conformal_threshold_from_scores, threshold_tuple_from_scores
from phase_config import (
    DEFAULTS,
    DEFAULT_INJECTION_CONVENTION,
    INJECTION_CONVENTIONS,
    INJECTION_CONVENTION_MCEWEN2012,
    INJECTION_CONVENTION_NOTES,
    PROVENANCE_SCHEMA_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve"
DEFAULT_MODELS = (
    "random_b64_aux:runs/phase3_unet/remediated_v1_unet_random_b64_aux:best",
    "imagenet_b64_aux:runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux:best",
)
DEFAULT_AMPLITUDE_GRID = (1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4)
DEFAULT_THETA_GRID_DEG = (5.0, 10.0, 15.0, 20.0, 25.0)
SIGN_QUADRANTS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    run_dir: Path
    checkpoint: str


def parse_float_list(text):
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_model_spec(text):
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --model spec: {text}. Expected name:run_dir:checkpoint")
    name, run_dir, checkpoint = parts
    return ModelSpec(name=name, run_dir=Path(run_dir), checkpoint=checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate matched-FPR sensitivity curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--amplitude-grid", type=str, default=",".join(f"{x:g}" for x in DEFAULT_AMPLITUDE_GRID))
    parser.add_argument("--theta-grid-deg", type=str, default=",".join(f"{x:g}" for x in DEFAULT_THETA_GRID_DEG))
    parser.add_argument(
        "--zcrit-ratio-grid",
        type=str,
        default="0,0.5,1,2",
        help="Grid of |zcrit| / |z0| profile ratios to marginalize within each amplitude-radius cell.",
    )
    parser.add_argument("--num-per-cell", type=int, default=200)
    parser.add_argument("--num-negatives", type=int, default=5000)
    parser.add_argument("--fpr-target", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--pool-size", type=int, default=16000)
    parser.add_argument("--coord-cluster-nside", type=int, default=4)
    parser.add_argument("--num-cmb-realizations", type=int, default=384)
    parser.add_argument("--contained-margin-deg", type=float, default=0.5)
    parser.add_argument("--beam-fwhm-arcmin", type=float, default=DEFAULTS.beam_fwhm_arcmin)
    parser.add_argument("--noise-sigma-uk-arcmin", type=float, default=30.0)
    parser.add_argument("--noise-corr-fwhm-arcmin", type=float, default=0.0)
    parser.add_argument(
        "--injection-convention",
        type=str,
        default=DEFAULT_INJECTION_CONVENTION,
        choices=INJECTION_CONVENTIONS,
        help=(
            "Signal convention for generated positives. Use "
            "mcewen2012_first_order_additive for additive same-grid classical "
            "benchmark products."
        ),
    )
    parser.add_argument(
        "--exclude-h5",
        action="append",
        default=[],
        help="Existing HDF5 coordinate pool to avoid. Can be repeated.",
    )
    parser.add_argument("--exclude-radius-deg", type=float, default=0.25)
    parser.add_argument("--model", action="append", default=[], help="Model as name:run_dir:checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--classical-workers", type=int, default=0, help="Thread workers for classical FFT scoring. Use 0 to match --num-workers.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--cache-ml-data",
        action="store_true",
        help=(
            "Cache the sensitivity HDF5 arrays in RAM for ML scoring. This avoids "
            "HDF5 multiprocessing stalls on large remediated grids and forces a "
            "single-process PyTorch loader for deterministic inference."
        ),
    )
    parser.add_argument("--skip-existing-data", action="store_true", help="Reuse output sensitivity_data.h5 if present.")
    parser.add_argument("--skip-ml", action="store_true", help="Only run classical baselines.")
    parser.add_argument("--reuse-ml-scores", action="store_true", help="Reuse ML score arrays from an existing sensitivity_scores.npz.")
    return parser.parse_args()


def validate_args(args):
    args.amplitude_grid = parse_float_list(args.amplitude_grid)
    args.theta_grid_deg = parse_float_list(args.theta_grid_deg)
    args.zcrit_ratio_grid = parse_float_list(args.zcrit_ratio_grid)
    if not args.amplitude_grid or any(value <= 0.0 for value in args.amplitude_grid):
        raise ValueError("--amplitude-grid must contain positive values.")
    if not args.theta_grid_deg or any(value <= 0.0 for value in args.theta_grid_deg):
        raise ValueError("--theta-grid-deg must contain positive values.")
    if not args.zcrit_ratio_grid or any(value < 0.0 for value in args.zcrit_ratio_grid):
        raise ValueError("--zcrit-ratio-grid must contain non-negative values.")
    if args.num_per_cell <= 0:
        raise ValueError("--num-per-cell must be positive.")
    if args.num_negatives <= 0:
        raise ValueError("--num-negatives must be positive.")
    if not (0.0 < args.fpr_target < 1.0):
        raise ValueError("--fpr-target must be in (0, 1).")
    if args.pool_size <= 1:
        raise ValueError("--pool-size must be at least 2.")
    if args.coord_cluster_nside <= 0:
        raise ValueError("--coord-cluster-nside must be positive.")
    if args.num_cmb_realizations <= 1:
        raise ValueError("--num-cmb-realizations must be at least 2.")
    if args.exclude_radius_deg < 0.0:
        raise ValueError("--exclude-radius-deg must be non-negative.")
    if args.contained_margin_deg < 0.0:
        raise ValueError("--contained-margin-deg must be non-negative.")
    if args.beam_fwhm_arcmin < 0.0:
        raise ValueError("--beam-fwhm-arcmin must be non-negative.")
    if args.noise_sigma_uk_arcmin < 0.0:
        raise ValueError("--noise-sigma-uk-arcmin must be non-negative.")
    if args.noise_corr_fwhm_arcmin < 0.0:
        raise ValueError("--noise-corr-fwhm-arcmin must be non-negative.")
    if args.classical_workers < 0:
        raise ValueError("--classical-workers must be non-negative.")


def default_exclusion_h5s():
    paths = [
        PROJECT_ROOT / "data" / "remediated_v1" / "training_data.h5",
        PROJECT_ROOT / "data" / "remediated_v1" / "calibration_data.h5",
        PROJECT_ROOT / "data" / "remediated_v1" / "test_data.h5",
    ]
    return [str(path) for path in paths if path.exists()]


def combined_exclusion_vectors(paths):
    vectors = []
    for path in paths:
        loaded = load_exclusion_vectors(path)
        if loaded is not None:
            vectors.append(loaded)
    if not vectors:
        return None
    return np.concatenate(vectors, axis=0)


def allocate_arrays(num_samples):
    patches = np.empty((num_samples, PATCH_PIX, PATCH_PIX), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.uint8)
    masks = np.zeros((num_samples, PATCH_PIX, PATCH_PIX), dtype=np.uint8)
    metadata = {
        "sample_index": np.arange(num_samples, dtype=np.int32),
        "glon_deg": np.empty(num_samples, dtype=np.float32),
        "glat_deg": np.empty(num_samples, dtype=np.float32),
        "coord_pool_idx": np.empty(num_samples, dtype=np.int32),
        "coord_cluster_id": np.zeros(num_samples, dtype=np.uint64),
        "coord_mask_fraction": np.empty(num_samples, dtype=np.float32),
        "cmb_realization_idx": np.empty(num_samples, dtype=np.int32),
        "background_id": np.zeros(num_samples, dtype=np.uint64),
        "split_group_id": np.zeros(num_samples, dtype=np.uint64),
    }
    truth = {
        "has_signal": np.zeros(num_samples, dtype=np.uint8),
        "event_id": np.zeros(num_samples, dtype=np.uint64),
        "amplitude": np.zeros(num_samples, dtype=np.float32),
        "theta_crit_deg": np.zeros(num_samples, dtype=np.float32),
        "z0": np.zeros(num_samples, dtype=np.float32),
        "zcrit": np.zeros(num_samples, dtype=np.float32),
        "zcrit_ratio": np.zeros(num_samples, dtype=np.float32),
        "edge_sigma_deg": np.zeros(num_samples, dtype=np.float32),
        "signal_center_x_pix": np.full(num_samples, patch_center_pixel(PATCH_PIX), dtype=np.float32),
        "signal_center_y_pix": np.full(num_samples, patch_center_pixel(PATCH_PIX), dtype=np.float32),
        "fully_contained": np.zeros(num_samples, dtype=np.uint8),
        "target_touches_edge": np.zeros(num_samples, dtype=np.uint8),
    }
    stratification = {
        "amplitude_idx": np.full(num_samples, -1, dtype=np.int16),
        "theta_idx": np.full(num_samples, -1, dtype=np.int16),
        "zcrit_ratio_idx": np.full(num_samples, -1, dtype=np.int16),
        "sign_quadrant": np.full(num_samples, -1, dtype=np.int16),
    }
    return patches, labels, masks, metadata, truth, stratification


def fill_common_metadata(
    metadata,
    sample_idx,
    coord_idx,
    realization_idx,
    coord_pool,
    coord_mask_fractions,
    seed,
    coord_cluster_id=None,
):
    lon_i, lat_i = coord_pool[coord_idx]
    metadata["glon_deg"][sample_idx] = lon_i
    metadata["glat_deg"][sample_idx] = lat_i
    metadata["coord_pool_idx"][sample_idx] = coord_idx
    if coord_cluster_id is not None and "coord_cluster_id" in metadata:
        metadata["coord_cluster_id"][sample_idx] = np.uint64(coord_cluster_id[coord_idx])
    metadata["coord_mask_fraction"][sample_idx] = coord_mask_fractions[coord_idx]
    metadata["cmb_realization_idx"][sample_idx] = realization_idx
    background_id = stable_group_id("sensitivity_camb", seed, realization_idx, coord_idx)
    metadata["background_id"][sample_idx] = np.uint64(background_id)
    metadata["split_group_id"][sample_idx] = np.uint64(background_id)
    return float(lon_i), float(lat_i), background_id


def write_sensitivity_h5(
    path,
    patches,
    labels,
    masks,
    metadata,
    truth,
    stratification,
    coord_pool,
    coord_mask_fractions,
    summary,
    coord_cluster_pix=None,
    coord_cluster_id=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("patches", data=patches, compression="gzip", shuffle=True)
        h5.create_dataset("labels", data=labels, compression="gzip", shuffle=True)
        h5.create_dataset("masks", data=masks, compression="gzip", shuffle=True)
        for group_name, payload in (("metadata", metadata), ("truth", truth), ("stratification", stratification)):
            group = h5.create_group(group_name)
            for key, value in payload.items():
                group.create_dataset(key, data=value, compression="gzip", shuffle=True)
        coord_group = h5.create_group("coordinate_pool")
        coord_group.create_dataset("glon_deg", data=coord_pool[:, 0].astype(np.float32), compression="gzip", shuffle=True)
        coord_group.create_dataset("glat_deg", data=coord_pool[:, 1].astype(np.float32), compression="gzip", shuffle=True)
        coord_group.create_dataset("mask_fraction", data=coord_mask_fractions.astype(np.float32), compression="gzip", shuffle=True)
        if coord_cluster_pix is not None:
            coord_group.create_dataset("coord_cluster_pix", data=np.asarray(coord_cluster_pix, dtype=np.int32), compression="gzip", shuffle=True)
        if coord_cluster_id is not None:
            coord_group.create_dataset("coord_cluster_id", data=np.asarray(coord_cluster_id, dtype=np.uint64), compression="gzip", shuffle=True)
        summary_group = h5.create_group("summary")
        for key, value in summary.items():
            summary_group.attrs[key] = value


def generate_sensitivity_dataset(args, h5_path):
    rng = np.random.default_rng(args.seed)
    amplitude_grid = tuple(float(x) for x in args.amplitude_grid)
    theta_grid_deg = tuple(float(x) for x in args.theta_grid_deg)
    zcrit_ratio_grid = tuple(float(x) for x in args.zcrit_ratio_grid)
    num_positive = (
        len(amplitude_grid)
        * len(theta_grid_deg)
        * len(zcrit_ratio_grid)
        * int(args.num_per_cell)
    )
    num_samples = num_positive + int(args.num_negatives)

    mask_256, sky_fraction = load_mask()
    coord_pool, coord_mask_fractions = build_coordinate_pool(mask_256, args.pool_size, rng)
    exclusion_paths = args.exclude_h5 or default_exclusion_h5s()
    exclusion_vectors = combined_exclusion_vectors(exclusion_paths)
    coord_pool, coord_mask_fractions = filter_excluded_coordinates(
        coord_pool,
        coord_mask_fractions,
        exclusion_vectors,
        args.exclude_radius_deg,
    )
    if len(coord_pool) < 2:
        raise RuntimeError("Coordinate pool empty after training/validation exclusion.")
    coord_cluster_pix = coordinate_cluster_pixels(coord_pool, args.coord_cluster_nside)
    coord_cluster_id = coordinate_cluster_ids(coord_cluster_pix, args.coord_cluster_nside)
    print(f"  Coordinate pool after exclusions: {len(coord_pool)}")
    print(f"  Coordinate clusters: {len(np.unique(coord_cluster_pix))} at Nside={args.coord_cluster_nside}")
    camb_realizations, camb_params = generate_camb_realizations(args.num_cmb_realizations, rng)

    patches, labels, masks, metadata, truth, stratification = allocate_arrays(num_samples)
    center_pix = patch_center_pixel(PATCH_PIX)
    sample_idx = 0

    print("\n=== Generating sensitivity positives ===")
    for amp_idx, amplitude in enumerate(amplitude_grid):
        for theta_idx, theta_deg in enumerate(theta_grid_deg):
            for ratio_idx, zcrit_ratio in enumerate(zcrit_ratio_grid):
                sign_pairs = build_balanced_sign_pairs(args.num_per_cell, rng)
                for cell_row in range(args.num_per_cell):
                    coord_idx = int(rng.integers(0, len(coord_pool)))
                    realization_idx = int(rng.integers(0, len(camb_realizations)))
                    lon_i, lat_i, background_id = fill_common_metadata(
                        metadata,
                        sample_idx,
                        coord_idx,
                        realization_idx,
                        coord_pool,
                        coord_mask_fractions,
                        args.seed,
                        coord_cluster_id=coord_cluster_id,
                    )
                    clean_patch = np.asarray(
                        project_patch(camb_realizations[realization_idx], lon_i, lat_i),
                        dtype=np.float32,
                    )
                    center_x_i, center_y_i = sample_signal_center_pixels(
                        rng=rng,
                        npix=PATCH_PIX,
                        theta_crit_deg=theta_deg,
                        geometry_mode="contained",
                        edge_margin_pix=0.0,
                        contained_margin_deg=args.contained_margin_deg,
                    )
                    sign_z0, sign_zcrit = sign_pairs[cell_row]
                    z0_i = float(sign_z0) * float(amplitude)
                    zcrit_i = float(sign_zcrit) * float(amplitude) * float(zcrit_ratio)
                    injected_patch, _ = inject_signal_into_patch(
                        clean_patch,
                        z0_i,
                        zcrit_i,
                        theta_deg,
                        edge_sigma_deg=0.0,
                        center_x_pix=center_x_i,
                        center_y_pix=center_y_i,
                        injection_convention=args.injection_convention,
                    )
                    theta_grid_i = make_angular_distance_grid(
                        PATCH_PIX,
                        RESO_ARCMIN,
                        center_x_pix=center_x_i,
                        center_y_pix=center_y_i,
                    )
                    mask_i = make_disk_mask(theta_grid_i, theta_deg)
                    touches_edge = target_touches_patch_edge(mask_i)
                    if touches_edge:
                        raise RuntimeError("Contained sensitivity target touched the patch edge.")
                    observed_patch = apply_observing_model_to_patch(
                        injected_patch,
                        rng=rng,
                        beam_fwhm_arcmin=args.beam_fwhm_arcmin,
                        noise_sigma_uk_arcmin=args.noise_sigma_uk_arcmin,
                        noise_corr_fwhm_arcmin=args.noise_corr_fwhm_arcmin,
                    )
                    event_id = stable_group_id(
                        background_id,
                        "sensitivity_signal",
                        amp_idx,
                        theta_idx,
                        ratio_idx,
                        cell_row,
                        f"{center_x_i:.6f}",
                        f"{center_y_i:.6f}",
                        f"{z0_i:.6e}",
                        f"{zcrit_i:.6e}",
                    )

                    patches[sample_idx] = observed_patch
                    labels[sample_idx] = 1
                    masks[sample_idx] = mask_i
                    truth["has_signal"][sample_idx] = 1
                    truth["event_id"][sample_idx] = np.uint64(event_id)
                    truth["amplitude"][sample_idx] = amplitude
                    truth["theta_crit_deg"][sample_idx] = theta_deg
                    truth["z0"][sample_idx] = z0_i
                    truth["zcrit"][sample_idx] = zcrit_i
                    truth["zcrit_ratio"][sample_idx] = float(zcrit_ratio)
                    truth["signal_center_x_pix"][sample_idx] = center_x_i
                    truth["signal_center_y_pix"][sample_idx] = center_y_i
                    truth["fully_contained"][sample_idx] = 1
                    stratification["amplitude_idx"][sample_idx] = amp_idx
                    stratification["theta_idx"][sample_idx] = theta_idx
                    stratification["zcrit_ratio_idx"][sample_idx] = ratio_idx
                    stratification["sign_quadrant"][sample_idx] = int(
                        np.where((np.asarray(SIGN_QUADRANTS) == (sign_z0, sign_zcrit)).all(axis=1))[0][0]
                    )

                    sample_idx += 1
                    if sample_idx % 500 == 0 or sample_idx == num_positive:
                        print(f"  Positives: {sample_idx:5d} / {num_positive}")

    print("\n=== Generating sensitivity negatives ===")
    for neg_idx in range(args.num_negatives):
        coord_idx = int(rng.integers(0, len(coord_pool)))
        realization_idx = int(rng.integers(0, len(camb_realizations)))
        lon_i, lat_i, _ = fill_common_metadata(
            metadata,
            sample_idx,
            coord_idx,
            realization_idx,
            coord_pool,
            coord_mask_fractions,
            args.seed,
            coord_cluster_id=coord_cluster_id,
        )
        clean_patch = np.asarray(project_patch(camb_realizations[realization_idx], lon_i, lat_i), dtype=np.float32)
        observed_patch = apply_observing_model_to_patch(
            clean_patch,
            rng=rng,
            beam_fwhm_arcmin=args.beam_fwhm_arcmin,
            noise_sigma_uk_arcmin=args.noise_sigma_uk_arcmin,
            noise_corr_fwhm_arcmin=args.noise_corr_fwhm_arcmin,
        )
        patches[sample_idx] = observed_patch
        sample_idx += 1
        if (neg_idx + 1) % 500 == 0 or neg_idx + 1 == args.num_negatives:
            print(f"  Negatives: {neg_idx + 1:5d} / {args.num_negatives}")

    permutation = rng.permutation(num_samples)
    patches = patches[permutation]
    labels = labels[permutation]
    masks = masks[permutation]
    for payload in (metadata, truth, stratification):
        for key, value in payload.items():
            payload[key] = value[permutation]
    metadata["sample_index"] = np.arange(num_samples, dtype=np.int32)

    summary = {
        "num_samples": int(num_samples),
        "num_positive": int(num_positive),
        "num_negative": int(args.num_negatives),
        "amplitude_grid": json.dumps(amplitude_grid),
        "theta_grid_deg": json.dumps(theta_grid_deg),
        "zcrit_ratio_grid": json.dumps(zcrit_ratio_grid),
        "num_per_cell": int(args.num_per_cell),
        "fpr_target": float(args.fpr_target),
        "seed": int(args.seed),
        "pool_size_requested": int(args.pool_size),
        "pool_size_after_exclusion": int(len(coord_pool)),
        "coord_cluster_nside": int(args.coord_cluster_nside),
        "num_coordinate_clusters": int(len(np.unique(coord_cluster_pix))),
        "num_cmb_realizations": int(args.num_cmb_realizations),
        "nside": int(NSIDE_WORKING),
        "patch_pixels": int(PATCH_PIX),
        "reso_arcmin": float(RESO_ARCMIN),
        "mask_threshold": float(MASK_THRESHOLD),
        "sky_fraction": float(sky_fraction),
        "geometry_mode": "contained",
        "contained_margin_deg": float(args.contained_margin_deg),
        "edge_sigma_deg": 0.0,
        "amplitude_definition": "|z0| = A; |zcrit|/|z0| is scanned over zcrit_ratio_grid with balanced sign quadrants",
        "signal_strength_definition": "reported amplitude grid is the absolute z0 reference amplitude, not max(|z0|, |zcrit|)",
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "injection_convention": args.injection_convention,
        "injection_convention_note": INJECTION_CONVENTION_NOTES[args.injection_convention],
        "matched_filter_approximation_convention": INJECTION_CONVENTION_MCEWEN2012,
        "matched_filter_approximation_note": INJECTION_CONVENTION_NOTES[INJECTION_CONVENTION_MCEWEN2012],
        "beam_fwhm_arcmin": float(args.beam_fwhm_arcmin),
        "noise_sigma_uk_arcmin": float(args.noise_sigma_uk_arcmin),
        "noise_corr_fwhm_arcmin": float(args.noise_corr_fwhm_arcmin),
        "exclude_h5": json.dumps([str(Path(p).resolve()) for p in exclusion_paths]),
        "exclude_radius_deg": float(args.exclude_radius_deg),
        "camb_params": json.dumps(camb_params, sort_keys=True),
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    write_sensitivity_h5(
        h5_path,
        patches,
        labels,
        masks,
        metadata,
        truth,
        stratification,
        coord_pool,
        coord_mask_fractions,
        summary,
        coord_cluster_pix=coord_cluster_pix,
        coord_cluster_id=coord_cluster_id,
    )
    with (h5_path.parent / "sensitivity_data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return h5_path


def make_feeney_template_kernel(theta_crit_deg, z0_sign, zcrit_sign, beam_fwhm_arcmin):
    center = patch_center_pixel(PATCH_PIX)
    theta = make_angular_distance_grid(PATCH_PIX, RESO_ARCMIN, center_x_pix=center, center_y_pix=center)
    template = bubble_collision_signal(theta, float(z0_sign), float(zcrit_sign), np.radians(theta_crit_deg), edge_sigma_deg=0.0)
    beam_sigma_pix = fwhm_arcmin_to_sigma_pixels(beam_fwhm_arcmin)
    if beam_sigma_pix > 0.0:
        template = gaussian_filter(template, sigma=beam_sigma_pix, mode="reflect")
    support = theta <= np.radians(theta_crit_deg)
    template = np.asarray(template, dtype=np.float32)
    template = template - float(template[support].mean() if support.any() else template.mean())
    template = template - float(template.mean())
    norm = float(np.linalg.norm(template))
    if norm > 0.0:
        template = template / norm
    return template.astype(np.float32)


def make_centered_disc_kernel(theta_crit_deg):
    center = patch_center_pixel(PATCH_PIX)
    theta = make_angular_distance_grid(PATCH_PIX, RESO_ARCMIN, center_x_pix=center, center_y_pix=center)
    disc = theta <= np.radians(theta_crit_deg)
    kernel = np.zeros_like(theta, dtype=np.float32)
    if disc.any():
        kernel[disc] = 1.0 / float(disc.sum())
    return kernel.astype(np.float32)


def standardize_patch(patch):
    patch = np.asarray(patch, dtype=np.float32)
    patch = patch - float(np.mean(patch))
    std = float(np.std(patch))
    if std > 0.0:
        patch = patch / std
    return patch


def score_circular_template_patch(patch, kernels):
    """Score a patch with the circular Feeney-template correlation screen."""

    patch = standardize_patch(patch)
    best = -np.inf
    for kernel in kernels:
        response = fftconvolve(patch, kernel[::-1, ::-1], mode="same")
        best = max(best, float(np.max(response)))
    return best


# Compatibility alias for older scripts. This is not a Wiener matched filter.
score_matched_template_patch = score_circular_template_patch


def score_centered_disc_patch(patch, kernels):
    patch = standardize_patch(patch)
    return float(max(abs(float(np.sum(patch * kernel))) for kernel in kernels))


def score_classical_methods(h5_path, theta_grid_deg, classical_workers, beam_fwhm_arcmin):
    with h5py.File(h5_path, "r") as h5:
        patches = np.asarray(h5["patches"][:], dtype=np.float32)
    n = int(patches.shape[0])
    matched_kernels = [
        make_feeney_template_kernel(theta, z0_sign, zcrit_sign, beam_fwhm_arcmin=beam_fwhm_arcmin)
        for theta in theta_grid_deg
        for z0_sign, zcrit_sign in SIGN_QUADRANTS
    ]
    centered_kernels = [make_centered_disc_kernel(theta) for theta in theta_grid_deg]
    scores = {
        CIRCULAR_TEMPLATE_SCREEN: np.zeros(n, dtype=np.float32),
        "centered_disc": np.zeros(n, dtype=np.float32),
    }

    def score_one(idx):
        patch = patches[idx]
        return (
            idx,
            score_circular_template_patch(patch, matched_kernels),
            score_centered_disc_patch(patch, centered_kernels),
        )

    workers = max(1, int(classical_workers))
    progress = p3.ProgressPrinter(n, f"Classical scores ({workers} threads)")
    completed = 0
    if workers == 1:
        for idx in range(n):
            row_idx, matched_score, centered_score = score_one(idx)
            scores[CIRCULAR_TEMPLATE_SCREEN][row_idx] = matched_score
            scores["centered_disc"][row_idx] = centered_score
            completed += 1
            if completed % 250 == 0 or completed == n:
                progress.update(completed)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(score_one, idx) for idx in range(n)]
            for future in as_completed(futures):
                row_idx, matched_score, centered_score = future.result()
                scores[CIRCULAR_TEMPLATE_SCREEN][row_idx] = matched_score
                scores["centered_disc"][row_idx] = centered_score
                completed += 1
                if completed % 250 == 0 or completed == n:
                    progress.update(completed)
    return scores


def build_model_from_run(run_dir, checkpoint_arg, device):
    run_config = load_json(run_dir / "run_config.json")
    model_args = p3.model_args_from_run_config(run_config)
    model = p3.build_model(model_args).to(device)
    checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, checkpoint_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model, run_config, str(checkpoint_path), checkpoint_label


def score_ml_model(spec, h5_path, args, device):
    model, run_config, checkpoint_path, checkpoint_label = build_model_from_run(spec.run_dir.resolve(), spec.checkpoint, device)
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["labels"].shape[0])
    dataset = p3.H5BubbleDataset(
        h5_path=h5_path,
        indices=np.arange(n, dtype=np.int64),
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 101,
        max_translate_pixels=0,
        cache_data=bool(args.cache_ml_data),
    )
    effective_num_workers = 0 if args.cache_ml_data else args.num_workers
    loader = p3.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=effective_num_workers > 0,
    )
    scores = np.zeros(n, dtype=np.float32)
    offset = 0
    progress = p3.ProgressPrinter(len(loader), f"ML scores {spec.name}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            mask_logits, _ = p3.unpack_model_output(model(images))
            score = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values
            batch_size = int(images.shape[0])
            scores[offset : offset + batch_size] = score.detach().cpu().numpy()
            offset += batch_size
            progress.update(batch_idx)
    return scores, {"checkpoint_path": checkpoint_path, "checkpoint_label": checkpoint_label}


def threshold_from_negatives(scores, labels, fpr_target):
    return threshold_tuple_from_scores(scores, labels, fpr_target)


def binomial_ci(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def summarize_sensitivity(scores_by_method, h5_path, fpr_target):
    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        amplitude = np.asarray(h5["truth"]["amplitude"][:], dtype=np.float64)
        theta = np.asarray(h5["truth"]["theta_crit_deg"][:], dtype=np.float64)
        amplitude_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amplitude_grid = json.loads(h5["summary"].attrs["amplitude_grid"])
        theta_grid_deg = json.loads(h5["summary"].attrs["theta_grid_deg"])

    rows = []
    thresholds = {}
    for method_name, scores in scores_by_method.items():
        threshold_record = conformal_threshold_from_scores(scores, labels, fpr_target)
        threshold = float(threshold_record["threshold"])
        neg_fp = int(threshold_record["negative_fp"])
        neg_fpr = float(threshold_record["negative_fpr"])
        thresholds[method_name] = threshold_record
        for amp_i, amp_value in enumerate(amplitude_grid):
            for theta_i, theta_value in enumerate(theta_grid_deg):
                mask = (labels == 1) & (amplitude_idx == amp_i) & (theta_idx == theta_i)
                n = int(mask.sum())
                k = int(np.sum(scores[mask] > threshold))
                low, high = binomial_ci(k, n)
                rows.append(
                    {
                        "method": method_name,
                        "amplitude": float(amp_value),
                        "theta_crit_deg": float(theta_value),
                        "num_positive": n,
                        "detected": k,
                        "p_det": float(k / max(n, 1)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "threshold": threshold,
                        "negative_fp": neg_fp,
                        "negative_fpr": neg_fpr,
                    }
                )
    return rows, thresholds


def write_csv(path, rows):
    columns = [
        "method",
        "amplitude",
        "theta_crit_deg",
        "num_positive",
        "detected",
        "p_det",
        "ci95_low",
        "ci95_high",
        "threshold",
        "negative_fp",
        "negative_fpr",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row[col]) for col in columns) + "\n")


def plot_sensitivity(path, rows, method_order, theta_grid_deg):
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(method_order), 3)))
    color_for = {method: colors[idx] for idx, method in enumerate(method_order)}
    fig, axes = plt.subplots(1, len(theta_grid_deg), figsize=(4.2 * len(theta_grid_deg), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, theta_value in zip(axes, theta_grid_deg):
        theta_rows = [row for row in rows if math.isclose(row["theta_crit_deg"], float(theta_value))]
        for method in method_order:
            method_rows = sorted([row for row in theta_rows if row["method"] == method], key=lambda row: row["amplitude"])
            if not method_rows:
                continue
            x = np.asarray([row["amplitude"] for row in method_rows], dtype=np.float64)
            y = np.asarray([row["p_det"] for row in method_rows], dtype=np.float64)
            ylo = np.asarray([row["ci95_low"] for row in method_rows], dtype=np.float64)
            yhi = np.asarray([row["ci95_high"] for row in method_rows], dtype=np.float64)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - ylo, yhi - y)),
                marker="o",
                linewidth=1.4,
                capsize=2.5,
                markersize=3.5,
                label=method,
                color=color_for[method],
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.set_title(rf"$\theta_c={theta_value:g}^\circ$")
        ax.set_xlabel(r"$A=|z_0|$")
    axes[0].set_ylabel(r"$P_{\rm det}$ at fixed FPR")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(method_order), 4), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, report):
    lines = [
        "# Phase 3 Sensitivity Curve",
        "",
        f"- Dataset: `{report['data_h5']}`",
        f"- FPR target: `{report['fpr_target']}`",
        f"- Positives per (A, theta, zcrit-ratio) cell: `{report['num_per_cell']}`",
        f"- zcrit ratio grid: `{report['zcrit_ratio_grid']}`",
        f"- Negatives for threshold calibration: `{report['num_negative']}`",
        f"- Signal definition: `{report['amplitude_definition']}`",
        f"- Edge smoothing: `{report['edge_sigma_deg']}`",
        "",
        "## Thresholds",
        "",
        "| method | threshold | negative FP | realized negative FPR |",
        "|---|---:|---:|---:|",
    ]
    for method, row in report["thresholds"].items():
        lines.append(f"| {method} | {row['threshold']:.8g} | {row['negative_fp']} | {row['negative_fpr']:.4f} |")
    lines.extend(
        [
            "",
            "## Sensitivity Rows",
            "",
            "| method | A | theta_deg | detected / n | P_det | 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            f"| {row['method']} | {row['amplitude']:.3g} | {row['theta_crit_deg']:.1f} | "
            f"{row['detected']} / {row['num_positive']} | {row['p_det']:.3f} | "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "sensitivity_data.h5"

    if args.skip_existing_data and h5_path.exists():
        print(f"Reusing existing dataset: {h5_path}")
    else:
        generate_sensitivity_dataset(args, h5_path)

    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        data_summary = dict(h5["summary"].attrs)
        theta_grid_deg = tuple(float(x) for x in json.loads(data_summary["theta_grid_deg"]))

    classical_workers = args.classical_workers or args.num_workers or 1
    scores_path = output_dir / "sensitivity_scores.npz"
    old_scores = {}
    if args.reuse_ml_scores and scores_path.exists():
        with np.load(scores_path) as loaded:
            old_scores = {key.removeprefix("score__"): np.asarray(loaded[key], dtype=np.float32) for key in loaded.files if key.startswith("score__")}

    beam_fwhm_arcmin = float(data_summary.get("beam_fwhm_arcmin", args.beam_fwhm_arcmin))
    scores_by_method = score_classical_methods(
        h5_path,
        theta_grid_deg,
        classical_workers=classical_workers,
        beam_fwhm_arcmin=beam_fwhm_arcmin,
    )
    model_metadata = {}
    if not args.skip_ml:
        device = p3.resolve_device(args.device)
        specs = [parse_model_spec(text) for text in (args.model or DEFAULT_MODELS)]
        for spec in specs:
            if spec.name in old_scores:
                scores_by_method[spec.name] = old_scores[spec.name]
                model_metadata[spec.name] = {"score_source": str(scores_path), "reused": True}
            else:
                scores, metadata = score_ml_model(spec, h5_path, args, device)
                scores_by_method[spec.name] = scores
                model_metadata[spec.name] = metadata

    score_payload = {f"score__{name}": values.astype(np.float32) for name, values in scores_by_method.items()}
    if CIRCULAR_TEMPLATE_SCREEN in scores_by_method:
        score_payload["score__matched_template"] = scores_by_method[CIRCULAR_TEMPLATE_SCREEN].astype(np.float32)
    np.savez_compressed(
        scores_path,
        labels=labels,
        **score_payload,
    )
    rows, thresholds = summarize_sensitivity(scores_by_method, h5_path, args.fpr_target)
    if CIRCULAR_TEMPLATE_SCREEN in thresholds:
        thresholds["matched_template"] = {
            **thresholds[CIRCULAR_TEMPLATE_SCREEN],
            "legacy_alias_for": CIRCULAR_TEMPLATE_SCREEN,
            "naming_warning": method_metadata("matched_template").get("naming_warning", ""),
        }
    csv_path = output_dir / "sensitivity_curve.csv"
    write_csv(csv_path, rows)
    method_order = list(scores_by_method.keys())
    plot_path = output_dir / "sensitivity_curve.png"
    plot_sensitivity(plot_path, rows, method_order, theta_grid_deg)

    report = {
        "data_h5": str(h5_path),
        "scores_npz": str(scores_path),
        "csv": str(csv_path),
        "plot_png": str(plot_path),
        "fpr_target": float(args.fpr_target),
        "num_positive": int(data_summary["num_positive"]),
        "num_negative": int(data_summary["num_negative"]),
        "num_per_cell": int(data_summary["num_per_cell"]),
        "amplitude_grid": json.loads(data_summary["amplitude_grid"]),
        "theta_grid_deg": json.loads(data_summary["theta_grid_deg"]),
        "zcrit_ratio_grid": json.loads(data_summary.get("zcrit_ratio_grid", "[1.0]")),
        "amplitude_definition": data_summary["amplitude_definition"],
        "signal_strength_definition": data_summary.get("signal_strength_definition", ""),
        "edge_sigma_deg": float(data_summary["edge_sigma_deg"]),
        "thresholds": thresholds,
        "method_metadata": {name: method_metadata(name) for name in scores_by_method},
        "model_metadata": model_metadata,
        "rows": rows,
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    report_path = output_dir / "sensitivity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = output_dir / "sensitivity_report.md"
    write_markdown(md_path, report)
    print("\n=== Sensitivity curve complete ===")
    print(f"  Dataset: {h5_path}")
    print(f"  Scores:  {scores_path}")
    print(f"  CSV:     {csv_path}")
    print(f"  Plot:    {plot_path}")
    print(f"  Report:  {report_path}")


if __name__ == "__main__":
    main()
