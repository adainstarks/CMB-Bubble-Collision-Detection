"""
Inject Feeney Eq. 1 signals into real Planck cleaned-map patches and compare
deployment-policy sensitivity against the CAMB-background sensitivity grid.

This is the real-background validation gate before full Phase 5 sky screening.
It does not train. It:

    1. Samples clean, mask-buffered real-map patch centers independent of prior
       train/validation/sensitivity coordinate pools.
    2. Extracts real cleaned-map background patches.
    3. Injects hard-boundary Feeney Eq. 1 signals over the sensitivity grid.
    4. Scores the frozen Phase 3 policy:
           v5_consensus AND (score_avg OR circular_template_screen)
    5. Compares P_det(A, theta_c) to the existing CAMB sensitivity curve.

Default map is Planck 2018 SMICA. Other cleaned maps can be selected if already
supported by phase2_extract_smica_null_controls.py.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve
from scipy.stats import binomtest

import phase3_train_unet as p3
from phase2_extract_smica_null_controls import PLANCK_CLEANED_MAPS, ensure_map_input
from phase2_generate_stratified_validation import filter_excluded_coordinates, load_exclusion_vectors
from phase_config import CANONICAL_MASK_THRESHOLD, DEFAULTS
from phase2_generate_training import (
    GEOMETRY_MODE_CODES,
    GEOMETRY_MODES,
    NSIDE_WORKING,
    build_coordinate_pool,
    coordinate_cluster_ids,
    coordinate_cluster_pixels,
    fwhm_arcmin_to_sigma_pixels,
    load_mask,
    project_patch,
    sample_actual_geometry_mode,
    sample_signal_geometry,
)
from phase2_observing_model import remove_real_map_low_modes
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN, T_CMB_K, bubble_collision_signal
from phase3_ensemble_evaluate import DEFAULT_MODELS, load_model, parse_model_spec
from phase3_method_registry import CIRCULAR_TEMPLATE_SCREEN, method_metadata
from phase3_sensitivity_curve import (
    SIGN_QUADRANTS,
    make_feeney_template_kernel,
    score_circular_template_patch,
)
from phase_dataset_utils import patch_center_pixel, stable_group_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_injection_v1"
DEFAULT_SENS_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "sensitivity_curve_v1" / "sensitivity_report.json"
DEFAULT_SENS_SCORES = PROJECT_ROOT / "runs" / "phase3_unet" / "sensitivity_curve_v1" / "sensitivity_scores.npz"
DEFAULT_SENS_H5 = PROJECT_ROOT / "runs" / "phase3_unet" / "sensitivity_curve_v1" / "sensitivity_data.h5"
DEFAULT_ENSEMBLE_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "ensemble_eval_v1" / "ensemble_eval.json"
DEFAULT_AMPLITUDE_GRID = (1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4)
DEFAULT_THETA_GRID_DEG = (5.0, 10.0, 15.0, 20.0, 25.0)
DEFAULT_ZCRIT_RATIO_GRID = (0.0, 0.5, 1.0, 2.0)
POLICIES = (
    "v5_only",
    "score_avg_only",
    f"{CIRCULAR_TEMPLATE_SCREEN}_only",
    "normal_candidate",
    "all_candidates",
    "union_or",
)
ML_METHODS = ("original_v4", "boundary_v4", "v5_consensus", "v6_aux_only", "v6_hard_w15")


def parse_float_list(text):
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate frozen Phase 3 policy on signals injected into real cleaned-map backgrounds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--map-name", type=str, default="smica", choices=sorted(PLANCK_CLEANED_MAPS))
    parser.add_argument("--num-backgrounds", type=int, default=500)
    parser.add_argument("--pool-size", type=int, default=6000)
    parser.add_argument("--mask-threshold", type=float, default=CANONICAL_MASK_THRESHOLD)
    parser.add_argument(
        "--coord-cluster-nside",
        type=int,
        default=4,
        help="HEALPix Nside used for coordinate-cluster metadata on real-map backgrounds.",
    )
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--amplitude-grid", type=str, default=",".join(f"{x:g}" for x in DEFAULT_AMPLITUDE_GRID))
    parser.add_argument("--theta-grid-deg", type=str, default=",".join(f"{x:g}" for x in DEFAULT_THETA_GRID_DEG))
    parser.add_argument("--zcrit-ratio-grid", type=str, default=",".join(f"{x:g}" for x in DEFAULT_ZCRIT_RATIO_GRID))
    parser.add_argument("--geometry-mode", type=str, default="contained", choices=GEOMETRY_MODES)
    parser.add_argument(
        "--truncated-positive-fraction",
        type=float,
        default=0.0,
        help="Fraction of positives drawn from edge-crossing geometry when --geometry-mode=mixed.",
    )
    parser.add_argument("--truncated-visible-fraction-min", type=float, default=0.15)
    parser.add_argument("--truncated-visible-fraction-max", type=float, default=0.95)
    parser.add_argument("--truncated-max-center-draws", type=int, default=256)
    parser.add_argument("--signal-center-edge-margin-pix", type=float, default=0.0)
    parser.add_argument("--contained-margin-deg", type=float, default=0.5)
    parser.add_argument("--edge-sigma-deg", type=float, default=0.0)
    parser.add_argument(
        "--signal-beam-fwhm-arcmin",
        type=float,
        default=0.0,
        help=(
            "Optional legacy patch-space smoothing applied only to the injected signal delta. "
            "Keep 0 for remediated real-sky injections to avoid double-smoothing Planck maps."
        ),
    )
    parser.add_argument("--exclude-h5", action="append", default=[])
    parser.add_argument("--exclude-radius-deg", type=float, default=0.25)
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_SENS_REPORT))
    parser.add_argument("--sensitivity-scores", type=str, default=str(DEFAULT_SENS_SCORES))
    parser.add_argument("--sensitivity-h5", type=str, default=str(DEFAULT_SENS_H5))
    parser.add_argument("--ensemble-report", type=str, default=str(DEFAULT_ENSEMBLE_REPORT))
    parser.add_argument("--model", action="append", default=[], help="Optional model specs; defaults to Phase 3 five-branch set.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cache-ml-data",
        action="store_true",
        help=(
            "Load the real-sky injection HDF5 tensors into host memory before ML scoring. "
            "This avoids slow repeated HDF5 reads for score-only validation runs."
        ),
    )
    parser.add_argument("--classical-workers", type=int, default=8)
    parser.add_argument("--classical-chunk-size", type=int, default=256)
    parser.add_argument("--classical-beam-fwhm-arcmin", type=float, default=DEFAULTS.beam_fwhm_arcmin)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def validate_args(args):
    args.amplitude_grid = parse_float_list(args.amplitude_grid)
    args.theta_grid_deg = parse_float_list(args.theta_grid_deg)
    args.zcrit_ratio_grid = parse_float_list(args.zcrit_ratio_grid)
    if args.num_backgrounds <= 0:
        raise ValueError("--num-backgrounds must be positive.")
    if args.num_backgrounds % len(SIGN_QUADRANTS) != 0:
        raise ValueError("--num-backgrounds must be divisible by 4 for balanced sign quadrants per cell.")
    if args.pool_size < args.num_backgrounds:
        raise ValueError("--pool-size must be >= --num-backgrounds.")
    if not (0.0 < args.mask_threshold <= 1.0):
        raise ValueError("--mask-threshold must lie in (0, 1].")
    if args.coord_cluster_nside <= 0 or not hp.isnsideok(args.coord_cluster_nside):
        raise ValueError("--coord-cluster-nside must be a valid positive HEALPix Nside.")
    if any(value <= 0.0 for value in args.amplitude_grid):
        raise ValueError("--amplitude-grid values must be positive.")
    if any(value <= 0.0 for value in args.theta_grid_deg):
        raise ValueError("--theta-grid-deg values must be positive.")
    if not args.zcrit_ratio_grid or any(value < 0.0 for value in args.zcrit_ratio_grid):
        raise ValueError("--zcrit-ratio-grid must contain non-negative values.")
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
    if args.edge_sigma_deg < 0.0:
        raise ValueError("--edge-sigma-deg must be non-negative.")
    if args.signal_beam_fwhm_arcmin < 0.0:
        raise ValueError("--signal-beam-fwhm-arcmin must be non-negative.")
    if args.exclude_radius_deg < 0.0:
        raise ValueError("--exclude-radius-deg must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.classical_workers <= 0:
        raise ValueError("--classical-workers must be positive.")
    if args.classical_chunk_size <= 0:
        raise ValueError("--classical-chunk-size must be positive.")
    if args.classical_beam_fwhm_arcmin < 0.0:
        raise ValueError("--classical-beam-fwhm-arcmin must be non-negative.")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive.")


def default_exclusion_h5s():
    paths = [
        PROJECT_ROOT / "data" / "training_v4" / "training_data.h5",
        PROJECT_ROOT / "data" / "validation_stratified_v1" / "validation_data.h5",
        PROJECT_ROOT / "runs" / "phase3_unet" / "sensitivity_curve_v1" / "sensitivity_data.h5",
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


def exact_ci(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_sensitivity_thresholds(sensitivity_report, ensemble_report, ml_methods):
    sens = load_json(sensitivity_report)
    thresholds = {
        method: float(sens["thresholds"][method]["threshold"])
        for method in ml_methods
        if method in sens["thresholds"]
    }
    missing = [method for method in ml_methods if method not in thresholds]
    if missing:
        raise KeyError(f"Sensitivity report is missing thresholds for requested ML methods: {missing}")
    classical_key = (
        CIRCULAR_TEMPLATE_SCREEN
        if CIRCULAR_TEMPLATE_SCREEN in sens["thresholds"]
        else "matched_template"
    )
    thresholds[CIRCULAR_TEMPLATE_SCREEN] = float(sens["thresholds"][classical_key]["threshold"])
    thresholds["matched_template"] = thresholds[CIRCULAR_TEMPLATE_SCREEN]
    ensemble_path = Path(ensemble_report)
    if ensemble_path.exists():
        ensemble = load_json(ensemble_path)
        if "ensemble_thresholds" in ensemble and "score_avg" in ensemble["ensemble_thresholds"]:
            thresholds["score_avg"] = float(ensemble["ensemble_thresholds"]["score_avg"])
    return thresholds


def allocate_h5(path, num_samples, chunk_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(path, "w")
    patch_chunk = (min(chunk_size, num_samples), PATCH_PIX, PATCH_PIX)
    row_chunk = (min(max(chunk_size, 1), num_samples),)
    h5.create_dataset("patches", shape=(num_samples, PATCH_PIX, PATCH_PIX), dtype=np.float32, chunks=patch_chunk, compression="gzip", shuffle=True)
    h5.create_dataset("masks", shape=(num_samples, PATCH_PIX, PATCH_PIX), dtype=np.uint8, chunks=patch_chunk, compression="gzip", shuffle=True)
    h5.create_dataset("labels", shape=(num_samples,), dtype=np.uint8, chunks=row_chunk, compression="gzip", shuffle=True)
    return h5


def write_array_group(h5, name, arrays):
    group = h5.create_group(name)
    for key, value in arrays.items():
        group.create_dataset(key, data=value, compression="gzip", shuffle=True)
    return group


def generate_dataset(args, h5_path):
    rng = np.random.default_rng(args.seed)
    mask_256, sky_fraction = load_mask(threshold=args.mask_threshold)
    coord_pool, mask_fractions = build_coordinate_pool(
        mask_256,
        args.pool_size,
        rng,
        min_unmasked_fraction=args.mask_threshold,
    )
    exclusions = args.exclude_h5 or default_exclusion_h5s()
    exclusion_vectors = combined_exclusion_vectors(exclusions)
    coord_pool, mask_fractions = filter_excluded_coordinates(
        coord_pool,
        mask_fractions,
        exclusion_vectors,
        args.exclude_radius_deg,
    )
    if len(coord_pool) < args.num_backgrounds:
        raise RuntimeError(
            f"Only {len(coord_pool)} coordinates remain after exclusions; need {args.num_backgrounds}."
        )
    coord_pool = coord_pool[: args.num_backgrounds]
    mask_fractions = mask_fractions[: args.num_backgrounds]
    coord_cluster_pix = coordinate_cluster_pixels(coord_pool, args.coord_cluster_nside)
    coord_cluster_id = coordinate_cluster_ids(coord_cluster_pix, args.coord_cluster_nside)

    product = ensure_map_input(args.map_name)
    cleaned_map = hp.read_map(product["path"], field=0)
    cleaned_map_256 = hp.ud_grade(cleaned_map, NSIDE_WORKING)
    cleaned_map_256 = remove_real_map_low_modes(cleaned_map_256, mask=mask_256)
    cleaned_map_256[np.asarray(mask_256) <= 0] = 0.0

    num_neg = int(args.num_backgrounds)
    num_pos = int(
        args.num_backgrounds
        * len(args.amplitude_grid)
        * len(args.theta_grid_deg)
        * len(args.zcrit_ratio_grid)
    )
    num_samples = num_neg + num_pos
    chunk_size = min(max(args.classical_chunk_size, 1), num_samples)
    base_patches = np.empty((num_neg, PATCH_PIX, PATCH_PIX), dtype=np.float32)

    metadata = {
        "sample_index": np.arange(num_samples, dtype=np.int32),
        "background_index": np.zeros(num_samples, dtype=np.int32),
        "glon_deg": np.zeros(num_samples, dtype=np.float32),
        "glat_deg": np.zeros(num_samples, dtype=np.float32),
        "coord_pool_idx": np.zeros(num_samples, dtype=np.int32),
        "coord_cluster_id": np.zeros(num_samples, dtype=np.uint64),
        "coord_mask_fraction": np.zeros(num_samples, dtype=np.float32),
        "cmb_realization_idx": np.full(num_samples, -1, dtype=np.int32),
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
    strat = {
        "amplitude_idx": np.full(num_samples, -1, dtype=np.int16),
        "theta_idx": np.full(num_samples, -1, dtype=np.int16),
        "zcrit_ratio_idx": np.full(num_samples, -1, dtype=np.int16),
        "sign_quadrant": np.full(num_samples, -1, dtype=np.int16),
    }

    signal_beam_sigma_pix = fwhm_arcmin_to_sigma_pixels(args.signal_beam_fwhm_arcmin)
    center_pix = patch_center_pixel(PATCH_PIX)
    h5 = allocate_h5(h5_path, num_samples, chunk_size)
    try:
        patches = h5["patches"]
        masks = h5["masks"]
        labels = h5["labels"]

        print(f"Extracting {num_neg} real {args.map_name.upper()} background patches...", flush=True)
        for bg_idx, (lon, lat) in enumerate(coord_pool):
            patch = np.asarray(project_patch(cleaned_map_256, float(lon), float(lat)), dtype=np.float32)
            base_patches[bg_idx] = patch
            background_id = stable_group_id("real_sky_injection", args.map_name, bg_idx, float(lon), float(lat))
            patches[bg_idx] = patch
            masks[bg_idx] = np.zeros((PATCH_PIX, PATCH_PIX), dtype=np.uint8)
            labels[bg_idx] = 0
            metadata["background_index"][bg_idx] = bg_idx
            metadata["glon_deg"][bg_idx] = float(lon)
            metadata["glat_deg"][bg_idx] = float(lat)
            metadata["coord_pool_idx"][bg_idx] = bg_idx
            metadata["coord_cluster_id"][bg_idx] = coord_cluster_id[bg_idx]
            metadata["coord_mask_fraction"][bg_idx] = float(mask_fractions[bg_idx])
            metadata["background_id"][bg_idx] = np.uint64(background_id)
            metadata["split_group_id"][bg_idx] = np.uint64(background_id)
            if (bg_idx + 1) % 100 == 0 or bg_idx + 1 == num_neg:
                print(f"  Backgrounds {bg_idx + 1:5d} / {num_neg}", flush=True)

        sample_idx = num_neg
        sign_table = np.asarray(SIGN_QUADRANTS, dtype=np.float32)
        for theta_i, theta_deg in enumerate(args.theta_grid_deg):
            # Reuse one geometry draw per real background across all
            # amplitudes for this radius. This preserves the geometry
            # distribution while avoiding seven redundant angular-grid builds.
            center_x = np.empty(args.num_backgrounds, dtype=np.float32)
            center_y = np.empty(args.num_backgrounds, dtype=np.float32)
            theta_grids = np.empty((args.num_backgrounds, PATCH_PIX, PATCH_PIX), dtype=np.float32)
            target_masks = np.empty((args.num_backgrounds, PATCH_PIX, PATCH_PIX), dtype=np.uint8)
            geometry_codes = np.zeros(args.num_backgrounds, dtype=np.uint8)
            fully_contained = np.zeros(args.num_backgrounds, dtype=np.uint8)
            touches_edge = np.zeros(args.num_backgrounds, dtype=np.uint8)
            visible_target_fraction = np.zeros(args.num_backgrounds, dtype=np.float32)
            visible_target_pixels = np.zeros(args.num_backgrounds, dtype=np.int32)
            full_disc_pixels_est = np.zeros(args.num_backgrounds, dtype=np.int32)
            target_edge_contact_pixels = np.zeros(args.num_backgrounds, dtype=np.int32)
            disc_edge_margin_pix = np.zeros(args.num_backgrounds, dtype=np.float32)
            signal_center_in_patch = np.zeros(args.num_backgrounds, dtype=np.uint8)
            for bg_idx in range(args.num_backgrounds):
                actual_geometry_mode = sample_actual_geometry_mode(
                    rng,
                    args.geometry_mode,
                    args.truncated_positive_fraction,
                )
                geometry_i = sample_signal_geometry(
                    rng=rng,
                    npix=PATCH_PIX,
                    theta_crit_deg=theta_deg,
                    geometry_mode=actual_geometry_mode,
                    edge_margin_pix=args.signal_center_edge_margin_pix,
                    contained_margin_deg=args.contained_margin_deg,
                    truncated_visible_fraction_min=args.truncated_visible_fraction_min,
                    truncated_visible_fraction_max=args.truncated_visible_fraction_max,
                    truncated_max_center_draws=args.truncated_max_center_draws,
                )
                cx = geometry_i["center_x_pix"]
                cy = geometry_i["center_y_pix"]
                center_x[bg_idx] = float(cx)
                center_y[bg_idx] = float(cy)
                theta_grids[bg_idx] = geometry_i["theta_grid"].astype(np.float32)
                target_masks[bg_idx] = geometry_i["mask"]
                geometry_codes[bg_idx] = GEOMETRY_MODE_CODES[actual_geometry_mode]
                fully_contained[bg_idx] = geometry_i["fully_contained"]
                touches_edge[bg_idx] = geometry_i["target_touches_edge"]
                visible_target_fraction[bg_idx] = geometry_i["visible_target_fraction"]
                visible_target_pixels[bg_idx] = geometry_i["visible_target_pixels"]
                full_disc_pixels_est[bg_idx] = geometry_i["full_disc_pixels_est"]
                target_edge_contact_pixels[bg_idx] = geometry_i["target_edge_contact_pixels"]
                disc_edge_margin_pix[bg_idx] = geometry_i["disc_edge_margin_pix"]
                signal_center_in_patch[bg_idx] = geometry_i["signal_center_in_patch"]

            for amp_i, amp in enumerate(args.amplitude_grid):
                for ratio_i, zcrit_ratio in enumerate(args.zcrit_ratio_grid):
                    quadrants = np.tile(
                        np.arange(len(SIGN_QUADRANTS), dtype=np.int16),
                        args.num_backgrounds // len(SIGN_QUADRANTS),
                    )
                    rng.shuffle(quadrants)
                    signs = sign_table[quadrants]
                    z0_values = (signs[:, 0] * float(amp)).astype(np.float32)
                    zcrit_values = (signs[:, 1] * float(amp) * float(zcrit_ratio)).astype(np.float32)
                    injected_batch = np.empty_like(base_patches)
                    for bg_idx in range(args.num_backgrounds):
                        signal = bubble_collision_signal(
                            theta_grids[bg_idx],
                            float(z0_values[bg_idx]),
                            float(zcrit_values[bg_idx]),
                            np.radians(float(theta_deg)),
                            edge_sigma_deg=args.edge_sigma_deg,
                        )
                        signal_delta = np.asarray(signal * (T_CMB_K + base_patches[bg_idx]), dtype=np.float32)
                        if signal_beam_sigma_pix > 0.0:
                            signal_delta = gaussian_filter(signal_delta, sigma=signal_beam_sigma_pix, mode="reflect")
                        injected_batch[bg_idx] = (base_patches[bg_idx] + signal_delta).astype(np.float32)

                    stop_idx = sample_idx + args.num_backgrounds
                    row_slice = slice(sample_idx, stop_idx)
                    sample_rows = np.arange(sample_idx, stop_idx, dtype=np.int64)
                    patches[row_slice] = injected_batch
                    masks[row_slice] = target_masks
                    labels[row_slice] = 1
                    metadata["background_index"][row_slice] = np.arange(args.num_backgrounds, dtype=np.int32)
                    metadata["glon_deg"][row_slice] = metadata["glon_deg"][: args.num_backgrounds]
                    metadata["glat_deg"][row_slice] = metadata["glat_deg"][: args.num_backgrounds]
                    metadata["coord_pool_idx"][row_slice] = np.arange(args.num_backgrounds, dtype=np.int32)
                    metadata["coord_cluster_id"][row_slice] = coord_cluster_id
                    metadata["coord_mask_fraction"][row_slice] = metadata["coord_mask_fraction"][: args.num_backgrounds]
                    metadata["background_id"][row_slice] = metadata["background_id"][: args.num_backgrounds]
                    metadata["split_group_id"][row_slice] = np.asarray(
                        [
                            stable_group_id("real_sky_injected_event", args.seed, int(row), int(bg_id))
                            for row, bg_id in zip(sample_rows, metadata["background_id"][: args.num_backgrounds])
                        ],
                        dtype=np.uint64,
                    )
                    truth["has_signal"][row_slice] = 1
                    truth["event_id"][row_slice] = np.asarray(
                        [stable_group_id("real_sky_injection_event", args.seed, int(row)) for row in sample_rows],
                        dtype=np.uint64,
                    )
                    truth["amplitude"][row_slice] = float(amp)
                    truth["theta_crit_deg"][row_slice] = float(theta_deg)
                    truth["z0"][row_slice] = z0_values
                    truth["zcrit"][row_slice] = zcrit_values
                    truth["zcrit_ratio"][row_slice] = float(zcrit_ratio)
                    truth["edge_sigma_deg"][row_slice] = float(args.edge_sigma_deg)
                    truth["signal_center_x_pix"][row_slice] = center_x
                    truth["signal_center_y_pix"][row_slice] = center_y
                    truth["signal_center_dx_deg"][row_slice] = (center_x - center_pix) * RESO_ARCMIN / 60.0
                    truth["signal_center_dy_deg"][row_slice] = (center_y - center_pix) * RESO_ARCMIN / 60.0
                    truth["geometry_mode_code"][row_slice] = geometry_codes
                    truth["fully_contained"][row_slice] = fully_contained
                    truth["target_touches_edge"][row_slice] = touches_edge
                    truth["visible_target_fraction"][row_slice] = visible_target_fraction
                    truth["visible_target_pixels"][row_slice] = visible_target_pixels
                    truth["full_disc_pixels_est"][row_slice] = full_disc_pixels_est
                    truth["target_edge_contact_pixels"][row_slice] = target_edge_contact_pixels
                    truth["disc_edge_margin_pix"][row_slice] = disc_edge_margin_pix
                    truth["signal_center_in_patch"][row_slice] = signal_center_in_patch
                    strat["amplitude_idx"][row_slice] = amp_i
                    strat["theta_idx"][row_slice] = theta_i
                    strat["zcrit_ratio_idx"][row_slice] = ratio_i
                    strat["sign_quadrant"][row_slice] = quadrants
                    sample_idx = stop_idx
                    print(
                        f"  Injected cell A={amp:.1e}, theta={theta_deg:g}, "
                        f"|zcrit|/|z0|={zcrit_ratio:g}: {sample_idx - num_neg:6d} / {num_pos}",
                        flush=True,
                    )
                    h5.flush()

        write_array_group(h5, "metadata", metadata)
        write_array_group(h5, "truth", truth)
        write_array_group(h5, "stratification", strat)
        coord_group = h5.create_group("coordinate_pool")
        coord_group.create_dataset("glon_deg", data=coord_pool[:, 0].astype(np.float32), compression="gzip", shuffle=True)
        coord_group.create_dataset("glat_deg", data=coord_pool[:, 1].astype(np.float32), compression="gzip", shuffle=True)
        coord_group.create_dataset("mask_fraction", data=mask_fractions.astype(np.float32), compression="gzip", shuffle=True)
        coord_group.create_dataset("coord_cluster_pix", data=coord_cluster_pix.astype(np.int32), compression="gzip", shuffle=True)
        coord_group.create_dataset("coord_cluster_id", data=coord_cluster_id.astype(np.uint64), compression="gzip", shuffle=True)
        summary = {
            "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "map_name": args.map_name,
            "source_map": Path(product["path"]).name,
            "source_map_url": product["url"],
            "source_map_product": product["product"],
            "num_samples": int(num_samples),
            "num_backgrounds": int(args.num_backgrounds),
            "num_positive": int(num_pos),
            "num_negative": int(num_neg),
            "amplitude_grid": json.dumps(tuple(float(x) for x in args.amplitude_grid)),
            "theta_grid_deg": json.dumps(tuple(float(x) for x in args.theta_grid_deg)),
            "zcrit_ratio_grid": json.dumps(tuple(float(x) for x in args.zcrit_ratio_grid)),
            "amplitude_definition": "|z0| = A; |zcrit|/|z0| is scanned over zcrit_ratio_grid with balanced sign quadrants",
            "geometry_mode": args.geometry_mode,
            "truncated_positive_fraction_requested": float(args.truncated_positive_fraction),
            "truncated_visible_fraction_min": float(args.truncated_visible_fraction_min),
            "truncated_visible_fraction_max": float(args.truncated_visible_fraction_max),
            "truncated_max_center_draws": int(args.truncated_max_center_draws),
            "signal_center_edge_margin_pix": float(args.signal_center_edge_margin_pix),
            "num_positive_fully_contained": int(truth["fully_contained"][num_neg:].sum()),
            "num_positive_touching_edge": int(truth["target_touches_edge"][num_neg:].sum()),
            "mean_positive_visible_target_fraction": float(np.mean(truth["visible_target_fraction"][num_neg:])),
            "min_positive_visible_target_fraction": float(np.min(truth["visible_target_fraction"][num_neg:])),
            "edge_sigma_deg": float(args.edge_sigma_deg),
            "signal_beam_fwhm_arcmin": float(args.signal_beam_fwhm_arcmin),
            "beam_fwhm_arcmin": float(DEFAULTS.beam_fwhm_arcmin),
            "injection_beam_policy": (
                "no_patch_space_injection_smoothing"
                if args.signal_beam_fwhm_arcmin == 0.0
                else "legacy_signal_delta_only_patch_space_smoothing"
            ),
            "background_beam_note": "real cleaned-map background is not re-smoothed; default injection path avoids patch-space signal smoothing",
            "nside": int(NSIDE_WORKING),
            "patch_pixels": int(PATCH_PIX),
            "reso_arcmin": float(RESO_ARCMIN),
            "mask_threshold": float(args.mask_threshold),
            "sky_fraction": float(sky_fraction),
            "low_mode_policy": "remove_monopole_dipole_with_mask",
            "coord_cluster_nside": int(args.coord_cluster_nside),
            "num_coordinate_clusters": int(len(np.unique(coord_cluster_pix))),
            "exclude_h5": json.dumps([str(Path(p).resolve()) for p in exclusions]),
            "exclude_radius_deg": float(args.exclude_radius_deg),
            "contained_margin_deg": float(args.contained_margin_deg),
            "seed": int(args.seed),
        }
        summary_group = h5.create_group("summary")
        for key, value in summary.items():
            summary_group.attrs[key] = value
    finally:
        h5.close()

    summary_path = h5_path.with_name(h5_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved real-sky injection HDF5: {h5_path}")
    print(f"Saved summary: {summary_path}")
    return h5_path


def score_circular_template_batch(patches, kernels):
    """Vectorized circular-template screen matching the per-patch FFT score.

    The score remains a translation scan of the linear Feeney cap templates
    used by :func:`score_circular_template_patch`; this only batches the FFT
    convolutions across patches to avoid per-sample Python overhead.
    """

    patches = np.asarray(patches, dtype=np.float32).copy()
    patches -= patches.mean(axis=(1, 2), keepdims=True)
    std = patches.std(axis=(1, 2), keepdims=True)
    np.divide(patches, std, out=patches, where=std > 0.0)
    best = np.full(patches.shape[0], -np.inf, dtype=np.float32)
    for kernel in kernels:
        response = fftconvolve(
            patches,
            np.asarray(kernel[::-1, ::-1], dtype=np.float32)[None, :, :],
            mode="same",
            axes=(-2, -1),
        )
        best = np.maximum(best, response.reshape(response.shape[0], -1).max(axis=1).astype(np.float32))
    return best


def score_circular_template_batch_torch(patches, kernels, device):
    """GPU FFT implementation of the circular-template screen.

    This computes the same full linear convolution and centered ``same`` crop
    as SciPy's :func:`fftconvolve`; the only change is batching over patches
    and template kernels on the accelerator.
    """

    if device.type != "cuda":
        return score_circular_template_batch(patches, kernels)
    patches = np.asarray(patches, dtype=np.float32).copy()
    patches -= patches.mean(axis=(1, 2), keepdims=True)
    std = patches.std(axis=(1, 2), keepdims=True)
    np.divide(patches, std, out=patches, where=std > 0.0)
    patch_tensor = torch.from_numpy(patches).to(device=device, non_blocking=True)
    kernel_array = np.stack(
        [np.asarray(kernel[::-1, ::-1], dtype=np.float32) for kernel in kernels],
        axis=0,
    )
    kernel_tensor = torch.from_numpy(kernel_array).to(device=device, non_blocking=True)
    _, height, width = patch_tensor.shape
    _, kernel_height, kernel_width = kernel_tensor.shape
    full_height = height + kernel_height - 1
    full_width = width + kernel_width - 1
    fft_height = next_fast_len(full_height)
    fft_width = next_fast_len(full_width)
    patch_fft = torch.fft.rfftn(patch_tensor[:, None, :, :], s=(fft_height, fft_width), dim=(-2, -1))
    kernel_fft = torch.fft.rfftn(kernel_tensor[None, :, :, :], s=(fft_height, fft_width), dim=(-2, -1))
    conv = torch.fft.irfftn(patch_fft * kernel_fft, s=(fft_height, fft_width), dim=(-2, -1))
    start_y = (full_height - height) // 2
    start_x = (full_width - width) // 2
    same = conv[:, :, start_y: start_y + height, start_x: start_x + width]
    scores = same.flatten(2).amax(dim=2).amax(dim=1).detach().cpu().numpy()
    del patch_tensor, kernel_tensor, patch_fft, kernel_fft, conv, same
    return scores.astype(np.float32, copy=False)


def score_circular_template_h5(h5_path, output_dir, theta_grid_deg, beam_fwhm_arcmin, workers, chunk_size, reuse_scores, device=None):
    cache_dir = output_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"real_sky_{CIRCULAR_TEMPLATE_SCREEN}_scores.npz"
    if reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            return np.asarray(loaded["scores"], dtype=np.float32)

    kernels = [
        make_feeney_template_kernel(theta, z0_sign, zcrit_sign, beam_fwhm_arcmin=beam_fwhm_arcmin)
        for theta in theta_grid_deg
        for z0_sign, zcrit_sign in SIGN_QUADRANTS
    ]
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["labels"].shape[0])
        scores = np.zeros(n, dtype=np.float32)
        progress = p3.ProgressPrinter(n, f"Real-sky {CIRCULAR_TEMPLATE_SCREEN} ({workers} threads)")
        completed = 0
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            patches = np.asarray(h5["patches"][start:stop], dtype=np.float32)
            if device is not None and device.type == "cuda":
                scores[start:stop] = score_circular_template_batch_torch(patches, kernels, device)
            else:
                scores[start:stop] = score_circular_template_batch(patches, kernels)
            completed = stop
            progress.update(completed)
    np.savez_compressed(
        cache_path,
        scores=scores,
        method=CIRCULAR_TEMPLATE_SCREEN,
        method_metadata=json.dumps(method_metadata(CIRCULAR_TEMPLATE_SCREEN), sort_keys=True),
    )
    return scores


def score_model_batched(spec, h5_path, output_dir, args, device):
    cache_dir = output_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"real_sky_{spec.name}_scores.npz"
    if args.reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            return np.asarray(loaded["scores"], dtype=np.float32)

    model, run_config = load_model(spec.run_dir.resolve(), spec.checkpoint, device)
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["labels"].shape[0])
    dataset = p3.H5BubbleDataset(
        h5_path=h5_path,
        indices=np.arange(n, dtype=np.int64),
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 10009,
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
        worker_init_fn=p3.seed_worker if effective_num_workers > 0 else None,
        generator=torch.Generator().manual_seed(int(run_config["args"]["seed"]) + 12009),
    )
    scores = np.zeros(n, dtype=np.float32)
    offset = 0
    progress = p3.ProgressPrinter(len(loader), f"Real-sky ML {spec.name}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            mask_logits, _ = p3.unpack_model_output(model(images))
            batch_scores = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values.detach().cpu().numpy()
            batch_size = int(images.shape[0])
            scores[offset: offset + batch_size] = batch_scores.astype(np.float32)
            offset += batch_size
            progress.update(batch_idx)
    np.savez_compressed(cache_path, scores=scores)
    return scores


def score_real_sky_dataset(args, h5_path, output_dir):
    specs = [parse_model_spec(text) for text in (args.model or DEFAULT_MODELS)]
    device = p3.resolve_device(args.device)
    scores = {}
    for spec in specs:
        scores[spec.name] = score_model_batched(spec, h5_path, output_dir, args, device)
    scores[CIRCULAR_TEMPLATE_SCREEN] = score_circular_template_h5(
        h5_path=h5_path,
        output_dir=output_dir,
        theta_grid_deg=args.theta_grid_deg,
        beam_fwhm_arcmin=args.classical_beam_fwhm_arcmin,
        workers=args.classical_workers,
        chunk_size=args.classical_chunk_size,
        reuse_scores=args.reuse_scores,
        device=device,
    )
    scores["matched_template"] = scores[CIRCULAR_TEMPLATE_SCREEN]
    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    score_path = output_dir / "real_sky_scores.npz"
    np.savez_compressed(
        score_path,
        labels=labels,
        **{f"score__{name}": value for name, value in scores.items()},
    )
    print(f"Saved score cache: {score_path}")
    return labels, scores


def load_real_scores(score_path):
    with np.load(score_path) as loaded:
        labels = np.asarray(loaded["labels"], dtype=np.uint8)
        scores = {}
        for name in loaded.files:
            if name.startswith("score__"):
                scores[name.removeprefix("score__")] = np.asarray(loaded[name], dtype=np.float32)
    if CIRCULAR_TEMPLATE_SCREEN not in scores and "matched_template" in scores:
        scores[CIRCULAR_TEMPLATE_SCREEN] = scores["matched_template"]
    if "matched_template" not in scores and CIRCULAR_TEMPLATE_SCREEN in scores:
        scores["matched_template"] = scores[CIRCULAR_TEMPLATE_SCREEN]
    return labels, scores


def policy_order_for_methods(ml_methods, thresholds):
    policies = [f"{method}_only" for method in ml_methods]
    policies.append(f"{CIRCULAR_TEMPLATE_SCREEN}_only")
    if len(ml_methods) > 1:
        policies.extend(["ml_score_mean", f"ml_vote_{(len(ml_methods) // 2) + 1}of{len(ml_methods)}", "union_or"])
    historical = {"v5_consensus", "score_avg"}.issubset(set(ml_methods) | set(thresholds))
    if historical and "v5_consensus" in ml_methods and "score_avg" in thresholds:
        policies.extend(["v5_only", "score_avg_only", "normal_candidate", "all_candidates"])
    return tuple(dict.fromkeys(policies))


def active_policies(scores, thresholds, ml_methods):
    ml_methods = tuple(method for method in ml_methods if method in scores and method in thresholds)
    if not ml_methods:
        raise ValueError("No requested ML methods are available in score arrays and thresholds.")
    ml_matrix = np.vstack([scores[method] for method in ml_methods])
    ml_hits = np.vstack([scores[method] > thresholds[method] for method in ml_methods])
    avg = ml_matrix.mean(axis=0)
    votes = ml_hits.sum(axis=0)
    circular = scores[CIRCULAR_TEMPLATE_SCREEN] > thresholds[CIRCULAR_TEMPLATE_SCREEN]
    policies = {f"{method}_only": scores[method] > thresholds[method] for method in ml_methods}
    policies[f"{CIRCULAR_TEMPLATE_SCREEN}_only"] = circular
    if len(ml_methods) > 1:
        mean_threshold = float(np.mean([thresholds[method] for method in ml_methods]))
        policies["ml_score_mean"] = avg > mean_threshold
        policies[f"ml_vote_{(len(ml_methods) // 2) + 1}of{len(ml_methods)}"] = votes >= ((len(ml_methods) // 2) + 1)
        policies["union_or"] = votes > 0
    if "v5_consensus" in ml_methods and "v5_consensus" in thresholds and "score_avg" in thresholds:
        v5 = scores["v5_consensus"] > thresholds["v5_consensus"]
        score_avg = avg > thresholds["score_avg"]
        policies.update(
            {
                "v5_only": v5,
                "score_avg_only": score_avg,
                "normal_candidate": v5 & (score_avg | circular),
                "all_candidates": (v5 & (score_avg | circular)) | (circular & ~v5),
            }
        )
    return policies


def binary_metrics(active, labels):
    labels = np.asarray(labels, dtype=np.uint8)
    active = np.asarray(active, dtype=bool)
    tp = int(np.logical_and(active, labels == 1).sum())
    fp = int(np.logical_and(active, labels == 0).sum())
    tn = int(np.logical_and(~active, labels == 0).sum())
    fn = int(np.logical_and(~active, labels == 1).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "fpr": fpr, "f1": f1}


def load_camb_scores(sensitivity_scores, sensitivity_h5, ml_methods):
    with np.load(sensitivity_scores) as loaded:
        labels = np.asarray(loaded["labels"], dtype=np.uint8)
        scores = {
            method: np.asarray(loaded[f"score__{method}"], dtype=np.float32)
            for method in ml_methods
            if f"score__{method}" in loaded
        }
        missing = [method for method in ml_methods if method not in scores]
        if missing:
            raise KeyError(f"Sensitivity scores are missing requested ML score arrays: {missing}")
        classical_key = (
            f"score__{CIRCULAR_TEMPLATE_SCREEN}"
            if f"score__{CIRCULAR_TEMPLATE_SCREEN}" in loaded
            else "score__matched_template"
        )
        scores[CIRCULAR_TEMPLATE_SCREEN] = np.asarray(loaded[classical_key], dtype=np.float32)
        scores["matched_template"] = scores[CIRCULAR_TEMPLATE_SCREEN]
    with h5py.File(sensitivity_h5, "r") as h5:
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amp_grid = [float(x) for x in json.loads(h5["summary"].attrs["amplitude_grid"])]
        theta_grid = [float(x) for x in json.loads(h5["summary"].attrs["theta_grid_deg"])]
        zcrit_ratio_grid = [
            float(x)
            for x in json.loads(h5["summary"].attrs.get("zcrit_ratio_grid", "[1.0]"))
        ]
    return labels, scores, amp_idx, theta_idx, amp_grid, theta_grid, zcrit_ratio_grid


def load_real_stratification(h5_path):
    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amp_grid = [float(x) for x in json.loads(h5["summary"].attrs["amplitude_grid"])]
        theta_grid = [float(x) for x in json.loads(h5["summary"].attrs["theta_grid_deg"])]
        zcrit_ratio_grid = [
            float(x)
            for x in json.loads(h5["summary"].attrs.get("zcrit_ratio_grid", "[1.0]"))
        ]
    return labels, amp_idx, theta_idx, amp_grid, theta_grid, zcrit_ratio_grid


def bootstrap_delta_ci(real_hits, camb_hits, resamples, rng):
    real_hits = np.asarray(real_hits, dtype=np.float32)
    camb_hits = np.asarray(camb_hits, dtype=np.float32)
    if real_hits.size == 0 or camb_hits.size == 0:
        return [float("nan"), float("nan")]
    deltas = np.empty(resamples, dtype=np.float32)
    for idx in range(resamples):
        real_sample = real_hits[rng.integers(0, real_hits.size, size=real_hits.size)]
        camb_sample = camb_hits[rng.integers(0, camb_hits.size, size=camb_hits.size)]
        deltas[idx] = float(real_sample.mean() - camb_sample.mean())
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def compare_real_to_camb(args, h5_path, real_labels, real_scores, thresholds, output_dir, ml_methods):
    real_labels_h5, real_amp_idx, real_theta_idx, real_amp_grid, real_theta_grid, real_zcrit_ratio_grid = load_real_stratification(h5_path)
    if not np.array_equal(real_labels, real_labels_h5):
        raise RuntimeError("Score labels do not match real-sky HDF5 labels.")
    camb_labels, camb_scores, camb_amp_idx, camb_theta_idx, camb_amp_grid, camb_theta_grid, camb_zcrit_ratio_grid = load_camb_scores(
        args.sensitivity_scores,
        args.sensitivity_h5,
        ml_methods,
    )
    if [float(x) for x in real_amp_grid] != [float(x) for x in camb_amp_grid]:
        raise RuntimeError("Real-sky amplitude grid does not match CAMB sensitivity grid.")
    if [float(x) for x in real_theta_grid] != [float(x) for x in camb_theta_grid]:
        raise RuntimeError("Real-sky theta grid does not match CAMB sensitivity grid.")
    if [float(x) for x in real_zcrit_ratio_grid] != [float(x) for x in camb_zcrit_ratio_grid]:
        raise RuntimeError("Real-sky zcrit-ratio grid does not match CAMB sensitivity grid.")

    rng = np.random.default_rng(args.seed + 991)
    real_active = active_policies(real_scores, thresholds, ml_methods)
    camb_active = active_policies(camb_scores, thresholds, ml_methods)
    policy_order = policy_order_for_methods(ml_methods, thresholds)
    policy_order = tuple(policy for policy in policy_order if policy in real_active and policy in camb_active)
    comparison_policy = (
        "normal_candidate"
        if "normal_candidate" in policy_order
        else (f"{ml_methods[-1]}_only" if ml_methods else f"{CIRCULAR_TEMPLATE_SCREEN}_only")
    )
    rows = []
    for policy in policy_order:
        for amp_i, amp in enumerate(real_amp_grid):
            for theta_i, theta in enumerate(real_theta_grid):
                real_mask = (real_labels == 1) & (real_amp_idx == amp_i) & (real_theta_idx == theta_i)
                camb_mask = (camb_labels == 1) & (camb_amp_idx == amp_i) & (camb_theta_idx == theta_i)
                real_hits = real_active[policy][real_mask]
                camb_hits = camb_active[policy][camb_mask]
                real_k = int(real_hits.sum())
                camb_k = int(camb_hits.sum())
                real_n = int(real_hits.size)
                camb_n = int(camb_hits.size)
                real_p = real_k / max(real_n, 1)
                camb_p = camb_k / max(camb_n, 1)
                delta_ci = bootstrap_delta_ci(real_hits, camb_hits, args.bootstrap_resamples, rng)
                rows.append(
                    {
                        "policy": policy,
                        "amplitude": float(amp),
                        "theta_crit_deg": float(theta),
                        "real_detected": real_k,
                        "real_n": real_n,
                        "real_p_det": float(real_p),
                        "real_ci95_low": exact_ci(real_k, real_n)[0],
                        "real_ci95_high": exact_ci(real_k, real_n)[1],
                        "camb_detected": camb_k,
                        "camb_n": camb_n,
                        "camb_p_det": float(camb_p),
                        "camb_ci95_low": exact_ci(camb_k, camb_n)[0],
                        "camb_ci95_high": exact_ci(camb_k, camb_n)[1],
                        "delta_real_minus_camb": float(real_p - camb_p),
                        "delta_ci95_low": delta_ci[0],
                        "delta_ci95_high": delta_ci[1],
                        "delta_significant": bool(delta_ci[0] > 0.0 or delta_ci[1] < 0.0),
                    }
                )
    report = {
        "real_sky_h5": str(Path(h5_path).resolve()),
        "sensitivity_scores": str(Path(args.sensitivity_scores).resolve()),
        "sensitivity_h5": str(Path(args.sensitivity_h5).resolve()),
        "thresholds": thresholds,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "ml_methods": list(ml_methods),
        "policy_order": list(policy_order),
        "comparison_policy": comparison_policy,
        "policies": {
            policy: {
                "real": binary_metrics(real_active[policy], real_labels),
                "camb": binary_metrics(camb_active[policy], camb_labels),
            }
            for policy in policy_order
        },
        "rows": rows,
    }
    return report


def write_csv(path, rows):
    columns = [
        "policy",
        "amplitude",
        "theta_crit_deg",
        "real_detected",
        "real_n",
        "real_p_det",
        "real_ci95_low",
        "real_ci95_high",
        "camb_detected",
        "camb_n",
        "camb_p_det",
        "camb_ci95_low",
        "camb_ci95_high",
        "delta_real_minus_camb",
        "delta_ci95_low",
        "delta_ci95_high",
        "delta_significant",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, report):
    comparison_policy = report.get("comparison_policy", "normal_candidate")
    normal_rows = [row for row in report["rows"] if row["policy"] == comparison_policy]
    sig = [row for row in normal_rows if row["delta_significant"]]
    lines = ["# Real-Sky Injection Validation", ""]
    lines.append(f"Real-sky HDF5: `{report['real_sky_h5']}`")
    lines.append(f"Bootstrap resamples: `{report['bootstrap_resamples']}`")
    lines.append("")
    lines.append("## Global Policy Metrics")
    lines.append("")
    lines.append("| policy | real recall | real FPR | CAMB recall | CAMB FPR |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy in report.get("policy_order", list(report["policies"])):
        real = report["policies"][policy]["real"]
        camb = report["policies"][policy]["camb"]
        lines.append(f"| `{policy}` | {real['recall']:.3f} | {real['fpr']:.3f} | {camb['recall']:.3f} | {camb['fpr']:.3f} |")
    lines.append("")
    lines.append(f"## `{comparison_policy}` Cell Comparison")
    lines.append("")
    lines.append("| A | theta_deg | real P_det | CAMB P_det | delta | delta 95% CI | significant |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in normal_rows:
        lines.append(
            f"| {row['amplitude']:.0e} | {row['theta_crit_deg']:.1f} | "
            f"{row['real_p_det']:.3f} | {row['camb_p_det']:.3f} | "
            f"{row['delta_real_minus_camb']:+.3f} | "
            f"[{row['delta_ci95_low']:+.3f}, {row['delta_ci95_high']:+.3f}] | "
            f"{row['delta_significant']} |"
        )
    lines.append("")
    lines.append(f"Significant `{comparison_policy}` CAMB-vs-real cells: `{len(sig)} / {len(normal_rows)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_normal_policy(report, output_dir):
    comparison_policy = report.get("comparison_policy", "normal_candidate")
    rows = [row for row in report["rows"] if row["policy"] == comparison_policy]
    if not rows:
        return
    amplitudes = sorted({row["amplitude"] for row in rows})
    thetas = sorted({row["theta_crit_deg"] for row in rows})
    fig, axes = plt.subplots(1, len(thetas), figsize=(4.0 * len(thetas), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, theta in zip(axes, thetas):
        theta_rows = sorted([row for row in rows if row["theta_crit_deg"] == theta], key=lambda row: row["amplitude"])
        x = np.asarray([row["amplitude"] for row in theta_rows], dtype=np.float64)
        real_y = np.asarray([row["real_p_det"] for row in theta_rows], dtype=np.float64)
        camb_y = np.asarray([row["camb_p_det"] for row in theta_rows], dtype=np.float64)
        ax.plot(x, camb_y, marker="o", label="CAMB")
        ax.plot(x, real_y, marker="s", label="real map")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"theta={theta:g} deg")
        ax.set_xlabel("A")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("P_det")
    axes[-1].legend(loc="lower right")
    fig.suptitle(f"{comparison_policy}: real cleaned-map backgrounds vs CAMB")
    fig.tight_layout()
    path = output_dir / f"real_vs_camb_{comparison_policy}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{args.map_name}_real_sky_injection.h5"
    score_path = output_dir / "real_sky_scores.npz"

    if args.score_only:
        if not h5_path.exists():
            raise FileNotFoundError(f"--score-only requested but HDF5 is missing: {h5_path}")
    elif args.reuse_data and h5_path.exists():
        print(f"Reusing existing real-sky injection HDF5: {h5_path}")
    else:
        generate_dataset(args, h5_path)

    if args.generate_only:
        return

    specs = [parse_model_spec(text) for text in (args.model or DEFAULT_MODELS)]
    ml_methods = tuple(spec.name for spec in specs)
    thresholds = read_sensitivity_thresholds(args.sensitivity_report, args.ensemble_report, ml_methods)
    if args.reuse_scores and score_path.exists():
        real_labels, real_scores = load_real_scores(score_path)
    else:
        real_labels, real_scores = score_real_sky_dataset(args, h5_path, output_dir)
    report = compare_real_to_camb(args, h5_path, real_labels, real_scores, thresholds, output_dir, ml_methods)

    json_path = output_dir / "real_sky_injection_report.json"
    csv_path = output_dir / "real_sky_injection_cells.csv"
    md_path = output_dir / "real_sky_injection_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, report["rows"])
    write_markdown(md_path, report)
    if not args.skip_plot:
        plot_normal_policy(report, output_dir)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
