"""
Phase 2 remediated generator for bubble-collision candidate screening.

Assumptions
-----------
* The project is a candidate-screening front end, not a cosmological detection
  or Bayesian evidence pipeline.
* `z0` and `zcrit` are dimensionless fractional Delta T / T amplitudes.
* CAMB maps are generated in Kelvin, with HEALPix pixel-window convolution
  enabled and the canonical Planck cleaned-map beam represented by a 5 arcmin
  harmonic-space Gaussian.
* The Feeney template source is Phys. Rev. D 84, 043507 (2011),
  arXiv:1012.3667; arXiv:1012.1995 is only the short companion summary.

The generator builds synthetic Planck-era patch observables from three pieces:
    1. CAMB CMB realizations at the working resolution
    2. Feeney et al. (2011) collision templates with multiplicative injection
    3. A remediated observing model with source-backed beam/pixel-window policy

This version makes three policy choices explicit:
    - the training size distribution is an Eq. 2-motivated design choice, not the later inference prior
    - train/calibration/test splits are created from disjoint coordinate pools and disjoint CMB realizations
    - the default geometry regime is fully contained signals; truncated/mixed signals must be requested explicitly
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os

import healpy as hp
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from phase1_explore import DATA_DIR, MASK_FILE, MASK_URL, download_file
from phase_config import DEFAULTS, CANONICAL_MASK_THRESHOLD
from phase2_observing_model import synthesize_cmb_maps, write_observing_model_provenance
from phase2_physics_checks import (
    check_eq1_special_cases,
    check_multiplicative_injection,
    check_patch_geometry,
    check_smooth_window_bounds,
)
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN, inject_signal_into_patch
from phase2_audit_dataset import run_audit
from phase_dataset_utils import make_angular_distance_grid, patch_center_pixel, stable_group_id

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, DEFAULTS.output_subdir)
NSIDE_WORKING = DEFAULTS.nside
MASK_THRESHOLD = CANONICAL_MASK_THRESHOLD
GEOMETRY_MODES = ("contained", "truncated", "mixed")
GEOMETRY_MODE_CODES = {"contained": 0, "truncated": 1}
SPLIT_TRAIN = 0
SPLIT_CALIBRATION = 1
SPLIT_TEST = 2
SPLIT_VAL = SPLIT_CALIBRATION


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a provenance-clean CAMB-based training dataset for bubble-collision segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=314159)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument(
        "--coord-cluster-nside",
        type=int,
        default=4,
        help=(
            "HEALPix Nside used to assign sky-coordinate clusters for "
            "train/calibration/test splitting and block-level accounting."
        ),
    )
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--num-cmb-realizations", type=int, default=192)
    parser.add_argument("--edge-sigma-min-deg", type=float, default=0.3)
    parser.add_argument("--edge-sigma-max-deg", type=float, default=1.0)
    parser.add_argument("--geometry-mode", type=str, default="contained", choices=GEOMETRY_MODES)
    parser.add_argument(
        "--truncated-positive-fraction",
        type=float,
        default=0.0,
        help="Fraction of positive samples drawn from edge-crossing geometry when --geometry-mode=mixed.",
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
    parser.add_argument(
        "--truncated-max-center-draws",
        type=int,
        default=256,
        help="Maximum rejection-sampling attempts when drawing a valid truncated positive.",
    )
    parser.add_argument(
        "--signal-center-edge-margin-pix",
        type=float,
        default=16.0,
        help="Only used when --geometry-mode=truncated.",
    )
    parser.add_argument(
        "--contained-margin-deg",
        type=float,
        default=0.5,
        help="Minimum sky margin between a contained target boundary and the patch edge.",
    )
    parser.add_argument(
        "--beam-fwhm-arcmin",
        type=float,
        default=DEFAULTS.beam_fwhm_arcmin,
        help="Canonical harmonic-space beam used when synthesizing CAMB backgrounds.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=MASK_THRESHOLD,
        help="Minimum gnomonic patch unmasked fraction. Canonical science default is 0.9.",
    )
    parser.add_argument(
        "--legacy-patch-beam",
        action="store_true",
        help="Use the historical patch-space Gaussian beam after injection. Not for remediated products.",
    )
    parser.add_argument(
        "--noise-sigma-uk-arcmin",
        type=float,
        default=30.0,
        help="Approximate white-noise level of the working observable.",
    )
    parser.add_argument(
        "--noise-corr-fwhm-arcmin",
        type=float,
        default=0.0,
        help="Optional Gaussian correlation scale for the noise field. Set 0 for white noise.",
    )
    parser.add_argument(
        "--skip-post-audit",
        action="store_true",
        help="Skip the automatic post-generation dataset audit. Intended only for debugging.",
    )
    parser.add_argument(
        "--skip-physics-checks",
        action="store_true",
        help="Skip pre-generation signal/injection physics checks. Intended only for debugging.",
    )
    return parser.parse_args()


def ensure_even_sample_count(num_samples):
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if num_samples % 2 != 0:
        raise ValueError("--num-samples must be even so the dataset is exactly 50/50 positive and negative.")


def validate_args(args):
    if args.num_cmb_realizations <= 1:
        raise ValueError("--num-cmb-realizations must be at least 2.")
    if args.pool_size <= 1:
        raise ValueError("--pool-size must be at least 2.")
    if args.coord_cluster_nside <= 0 or not hp.isnsideok(args.coord_cluster_nside):
        raise ValueError("--coord-cluster-nside must be a valid positive HEALPix Nside.")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1.")
    if not (0.0 <= args.calibration_fraction < 1.0):
        raise ValueError("--calibration-fraction must be in [0, 1).")
    if args.train_fraction + args.calibration_fraction >= 1.0:
        raise ValueError("--train-fraction + --calibration-fraction must be < 1.")
    if not (0.0 < args.mask_threshold <= 1.0):
        raise ValueError("--mask-threshold must be in (0, 1].")
    if args.edge_sigma_min_deg < 0.0 or args.edge_sigma_max_deg < 0.0:
        raise ValueError("--edge sigma bounds must be non-negative.")
    if args.edge_sigma_min_deg > args.edge_sigma_max_deg:
        raise ValueError("--edge-sigma-min-deg must be <= --edge-sigma-max-deg.")
    if args.signal_center_edge_margin_pix < 0.0:
        raise ValueError("--signal-center-edge-margin-pix must be non-negative.")
    if args.signal_center_edge_margin_pix >= PATCH_PIX / 2.0:
        raise ValueError("--signal-center-edge-margin-pix must be smaller than half the patch width.")
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
    if args.contained_margin_deg < 0.0:
        raise ValueError("--contained-margin-deg must be non-negative.")
    if args.beam_fwhm_arcmin < 0.0:
        raise ValueError("--beam-fwhm-arcmin must be non-negative.")
    if args.noise_sigma_uk_arcmin < 0.0:
        raise ValueError("--noise-sigma-uk-arcmin must be non-negative.")
    if args.noise_corr_fwhm_arcmin < 0.0:
        raise ValueError("--noise-corr-fwhm-arcmin must be non-negative.")


def ensure_planck_inputs():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_file(MASK_URL, MASK_FILE)


def load_mask(threshold=MASK_THRESHOLD):
    print("\n=== Loading galactic mask ===")
    ensure_planck_inputs()

    mask = hp.read_map(MASK_FILE, field=0)
    mask_256 = hp.ud_grade(mask, NSIDE_WORKING)
    mask_256 = np.where(mask_256 >= float(threshold), 1.0, 0.0)

    sky_fraction = float(np.mean(mask_256))
    print(f"  Mask threshold: {float(threshold):.2f}")
    print(f"  Mask sky fraction: {sky_fraction:.1%}")
    return mask_256, sky_fraction


def planck2018_bestfit_params():
    return {
        "H0": 67.36,
        "ombh2": 0.02237,
        "omch2": 0.1200,
        "tau": 0.0544,
        "ns": 0.9649,
        "As": math.exp(3.044) / 1e10,
    }


def generate_camb_realizations(num_realizations, rng, beam_fwhm_arcmin=DEFAULTS.beam_fwhm_arcmin):
    print("\n=== Generating CAMB realizations ===")
    realizations, provenance = synthesize_cmb_maps(
        num_realizations=int(num_realizations),
        rng=rng,
        nside=NSIDE_WORKING,
        beam_fwhm_arcmin=float(beam_fwhm_arcmin),
        pixwin=True,
    )
    seeds = provenance["seeds"]
    for idx in range(num_realizations):
        if (idx + 1) % 16 == 0 or idx + 1 == num_realizations:
            print(f"  Generated {idx + 1:4d} / {num_realizations} CAMB skies")

    return realizations, provenance


def sample_random_galactic_coordinate(rng):
    glon = rng.uniform(0.0, 360.0)
    sin_glat = rng.uniform(-1.0, 1.0)
    glat = np.degrees(np.arcsin(sin_glat))
    return float(glon), float(glat)


def project_patch(hp_map, glon_deg, glat_deg):
    return hp.gnomview(
        hp_map,
        rot=(glon_deg, glat_deg),
        reso=RESO_ARCMIN,
        xsize=PATCH_PIX,
        return_projected_map=True,
        no_plot=True,
    )


def is_center_unmasked(mask_256, glon_deg, glat_deg):
    theta = np.radians(90.0 - glat_deg)
    phi = np.radians(glon_deg)
    pix = hp.ang2pix(NSIDE_WORKING, theta, phi)
    return bool(mask_256[pix] > 0.5)


def projected_unmasked_fraction(mask_patch):
    mask_patch = np.asarray(mask_patch)
    usable = np.isfinite(mask_patch) & (mask_patch > -1e20)
    if not np.any(usable):
        return 0.0
    return float(np.mean(mask_patch[usable] > 0.5))


def build_coordinate_pool(mask_256, pool_size, rng, min_unmasked_fraction=MASK_THRESHOLD):
    print("\n=== Building valid coordinate pool ===")
    coords = []
    mask_fractions = []
    attempts = 0
    max_attempts = max(pool_size * 200, 1000)

    while len(coords) < pool_size and attempts < max_attempts:
        attempts += 1
        glon_deg, glat_deg = sample_random_galactic_coordinate(rng)
        if not is_center_unmasked(mask_256, glon_deg, glat_deg):
            continue

        mask_patch = project_patch(mask_256, glon_deg, glat_deg)
        mask_fraction = projected_unmasked_fraction(mask_patch)
        if mask_fraction < min_unmasked_fraction:
            continue

        coords.append((glon_deg, glat_deg))
        mask_fractions.append(mask_fraction)
        if len(coords) % 250 == 0 or len(coords) == pool_size:
            print(f"  Accepted {len(coords):4d} / {pool_size} centers after {attempts} attempts")

    if len(coords) < pool_size:
        raise RuntimeError(
            f"Could only build {len(coords)} valid centers after {attempts} attempts. "
            "Try reducing --pool-size or loosening the mask threshold."
        )

    print(f"  Final coordinate pool: {len(coords)} centers")
    return np.asarray(coords, dtype=np.float32), np.asarray(mask_fractions, dtype=np.float32)


def split_index_pool(num_items, train_fraction, seed):
    if num_items < 2:
        raise ValueError("Need at least two items to build a train/validation split.")
    rng = np.random.default_rng(seed)
    indices = np.arange(num_items, dtype=np.int64)
    rng.shuffle(indices)
    train_count = int(round(num_items * train_fraction))
    train_count = min(max(train_count, 1), num_items - 1)
    train_idx = np.sort(indices[:train_count])
    val_idx = np.sort(indices[train_count:])
    return train_idx, val_idx


def split_index_pool_three(num_items, train_fraction, calibration_fraction, seed):
    """Split an index pool into train/calibration/test partitions."""

    if num_items < 3:
        raise ValueError("Need at least three items to build train/calibration/test splits.")
    rng = np.random.default_rng(seed)
    indices = np.arange(num_items, dtype=np.int64)
    rng.shuffle(indices)
    train_count = int(round(num_items * train_fraction))
    calibration_count = int(round(num_items * calibration_fraction))
    train_count = min(max(train_count, 1), num_items - 2)
    calibration_count = min(max(calibration_count, 1), num_items - train_count - 1)
    train_idx = np.sort(indices[:train_count])
    calibration_idx = np.sort(indices[train_count : train_count + calibration_count])
    test_idx = np.sort(indices[train_count + calibration_count :])
    return train_idx, calibration_idx, test_idx


def coordinate_cluster_pixels(coord_pool, cluster_nside):
    """Return HEALPix super-pixels used as sky-coordinate block IDs.

    The same Planck sky cannot provide IID evidence from thousands of
    overlapping gnomonic patches.  We therefore split and report by coarse
    sky blocks rather than exact center coordinates.
    """

    coords = np.asarray(coord_pool, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coord_pool must have shape (N, 2) with (glon, glat) degrees.")
    cluster_nside = int(cluster_nside)
    if cluster_nside <= 0 or not hp.isnsideok(cluster_nside):
        raise ValueError("cluster_nside must be a valid positive HEALPix Nside.")
    theta = np.radians(90.0 - coords[:, 1])
    phi = np.radians(coords[:, 0] % 360.0)
    return hp.ang2pix(cluster_nside, theta, phi).astype(np.int64)


def coordinate_cluster_ids(cluster_pixels, cluster_nside):
    """Return stable uint64 cluster IDs for HEALPix coordinate blocks."""

    cluster_pixels = np.asarray(cluster_pixels, dtype=np.int64)
    return np.asarray(
        [stable_group_id("healpix_coord_cluster", int(cluster_nside), int(pixel)) for pixel in cluster_pixels],
        dtype=np.uint64,
    )


def split_index_pool_three_by_group(group_ids, train_fraction, calibration_fraction, seed):
    """Split row indices by group, keeping each group in exactly one split."""

    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be one-dimensional.")
    unique_groups = np.unique(group_ids)
    if unique_groups.size < 3:
        raise ValueError("Need at least three coordinate clusters for train/calibration/test splits.")

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    train_count = int(round(shuffled_groups.size * train_fraction))
    calibration_count = int(round(shuffled_groups.size * calibration_fraction))
    train_count = min(max(train_count, 1), shuffled_groups.size - 2)
    calibration_count = min(max(calibration_count, 1), shuffled_groups.size - train_count - 1)
    train_groups = shuffled_groups[:train_count]
    calibration_groups = shuffled_groups[train_count : train_count + calibration_count]
    test_groups = shuffled_groups[train_count + calibration_count :]

    train_idx = np.flatnonzero(np.isin(group_ids, train_groups)).astype(np.int64)
    calibration_idx = np.flatnonzero(np.isin(group_ids, calibration_groups)).astype(np.int64)
    test_idx = np.flatnonzero(np.isin(group_ids, test_groups)).astype(np.int64)
    if not (train_idx.size and calibration_idx.size and test_idx.size):
        raise ValueError("Coordinate cluster split produced an empty partition.")
    return np.sort(train_idx), np.sort(calibration_idx), np.sort(test_idx)


def sample_log_uniform(rng, low, high):
    log_value = rng.uniform(np.log10(low), np.log10(high))
    return float(10.0 ** log_value)


def sample_theta_crit_from_training_prior(rng, low_deg=5.0, high_deg=25.0):
    low_rad = np.radians(low_deg)
    high_rad = np.radians(high_deg)
    u = rng.uniform()
    cos_theta = np.cos(low_rad) - u * (np.cos(low_rad) - np.cos(high_rad))
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


def build_balanced_sign_pairs(num_positive, rng):
    base_pairs = np.array(
        [
            (+1.0, +1.0),
            (+1.0, -1.0),
            (-1.0, +1.0),
            (-1.0, -1.0),
        ],
        dtype=np.float32,
    )
    repeats = int(math.ceil(num_positive / len(base_pairs)))
    sign_pairs = np.tile(base_pairs, (repeats, 1))[:num_positive]
    rng.shuffle(sign_pairs)
    return sign_pairs


def make_disk_mask(theta_grid, theta_crit_deg):
    theta_crit_rad = np.radians(theta_crit_deg)
    return (theta_grid <= theta_crit_rad).astype(np.uint8)


def target_touches_patch_edge(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def estimate_full_disc_pixel_count(npix, reso_arcmin, theta_crit_deg):
    theta_grid = make_angular_distance_grid(
        npix,
        reso_arcmin,
        center_x_pix=patch_center_pixel(npix),
        center_y_pix=patch_center_pixel(npix),
    )
    return int(make_disk_mask(theta_grid, theta_crit_deg).sum())


def target_edge_contact_count(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    edge = np.zeros_like(mask, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    return int(np.count_nonzero(mask & edge))


def approximate_disc_edge_margin_pix(center_x_pix, center_y_pix, theta_crit_deg, npix=PATCH_PIX):
    radius_pix = float(theta_crit_deg) * 60.0 / RESO_ARCMIN
    nearest_patch_edge_pix = min(
        float(center_x_pix),
        float(center_y_pix),
        float(npix - 1) - float(center_x_pix),
        float(npix - 1) - float(center_y_pix),
    )
    return float(nearest_patch_edge_pix - radius_pix)


def sample_actual_geometry_mode(rng, geometry_mode, truncated_positive_fraction):
    if geometry_mode == "mixed":
        return "truncated" if rng.random() < float(truncated_positive_fraction) else "contained"
    return geometry_mode


def sample_signal_center_pixels(
    rng,
    npix,
    theta_crit_deg,
    geometry_mode,
    edge_margin_pix,
    contained_margin_deg,
):
    if geometry_mode == "truncated":
        low = float(edge_margin_pix)
        high = float(npix - 1 - edge_margin_pix)
    else:
        contained_margin_pix = float(contained_margin_deg) * 60.0 / RESO_ARCMIN
        radius_pix = float(theta_crit_deg) * 60.0 / RESO_ARCMIN
        low = radius_pix + contained_margin_pix
        high = float(npix - 1) - low
        if low > high:
            raise RuntimeError(
                "Contained geometry is not feasible for the current patch size and target radius. "
                "Increase the patch size or switch to --geometry-mode=truncated."
            )
    center_x_pix = rng.uniform(low, high)
    center_y_pix = rng.uniform(low, high)
    return center_x_pix, center_y_pix


def sample_truncated_signal_center_pixels(rng, npix, theta_crit_deg, edge_margin_pix):
    radius_pix = float(theta_crit_deg) * 60.0 / RESO_ARCMIN
    low = -radius_pix + float(edge_margin_pix)
    high = float(npix - 1) + radius_pix - float(edge_margin_pix)
    if low > high:
        raise RuntimeError("Truncated geometry center sampling is infeasible for the current target radius.")
    return float(rng.uniform(low, high)), float(rng.uniform(low, high))


def sample_signal_geometry(
    rng,
    npix,
    theta_crit_deg,
    geometry_mode,
    edge_margin_pix,
    contained_margin_deg,
    truncated_visible_fraction_min,
    truncated_visible_fraction_max,
    truncated_max_center_draws,
):
    full_disc_pixels = max(1, estimate_full_disc_pixel_count(npix, RESO_ARCMIN, theta_crit_deg))

    if geometry_mode == "contained":
        center_x_pix, center_y_pix = sample_signal_center_pixels(
            rng=rng,
            npix=npix,
            theta_crit_deg=theta_crit_deg,
            geometry_mode="contained",
            edge_margin_pix=edge_margin_pix,
            contained_margin_deg=contained_margin_deg,
        )
        theta_grid = make_angular_distance_grid(
            npix,
            RESO_ARCMIN,
            center_x_pix=center_x_pix,
            center_y_pix=center_y_pix,
        )
        mask = make_disk_mask(theta_grid, theta_crit_deg)
        touches_edge = target_touches_patch_edge(mask)
        if touches_edge:
            raise RuntimeError("Contained geometry produced a target that touches the patch edge.")
        visible_pixels = int(mask.sum())
        return {
            "center_x_pix": float(center_x_pix),
            "center_y_pix": float(center_y_pix),
            "theta_grid": theta_grid,
            "mask": mask,
            "visible_target_fraction": float(min(1.0, visible_pixels / full_disc_pixels)),
            "visible_target_pixels": visible_pixels,
            "full_disc_pixels_est": int(full_disc_pixels),
            "target_touches_edge": 0,
            "fully_contained": 1,
            "target_edge_contact_pixels": 0,
            "disc_edge_margin_pix": approximate_disc_edge_margin_pix(center_x_pix, center_y_pix, theta_crit_deg, npix=npix),
            "signal_center_in_patch": 1,
        }

    last_visible_fraction = 0.0
    for _ in range(int(truncated_max_center_draws)):
        center_x_pix, center_y_pix = sample_truncated_signal_center_pixels(
            rng,
            npix=npix,
            theta_crit_deg=theta_crit_deg,
            edge_margin_pix=edge_margin_pix,
        )
        theta_grid = make_angular_distance_grid(
            npix,
            RESO_ARCMIN,
            center_x_pix=center_x_pix,
            center_y_pix=center_y_pix,
        )
        mask = make_disk_mask(theta_grid, theta_crit_deg)
        visible_pixels = int(mask.sum())
        if visible_pixels <= 0:
            continue
        touches_edge = target_touches_patch_edge(mask)
        visible_fraction = float(min(1.0, visible_pixels / full_disc_pixels))
        last_visible_fraction = visible_fraction
        if not touches_edge:
            continue
        if not (truncated_visible_fraction_min <= visible_fraction <= truncated_visible_fraction_max):
            continue
        center_in_patch = 0 <= center_x_pix <= npix - 1 and 0 <= center_y_pix <= npix - 1
        return {
            "center_x_pix": float(center_x_pix),
            "center_y_pix": float(center_y_pix),
            "theta_grid": theta_grid,
            "mask": mask,
            "visible_target_fraction": visible_fraction,
            "visible_target_pixels": visible_pixels,
            "full_disc_pixels_est": int(full_disc_pixels),
            "target_touches_edge": 1,
            "fully_contained": 0,
            "target_edge_contact_pixels": target_edge_contact_count(mask),
            "disc_edge_margin_pix": approximate_disc_edge_margin_pix(center_x_pix, center_y_pix, theta_crit_deg, npix=npix),
            "signal_center_in_patch": int(center_in_patch),
        }

    raise RuntimeError(
        "Could not draw a valid truncated signal center after "
        f"{truncated_max_center_draws} attempts for theta_crit={theta_crit_deg:.3f} deg. "
        f"Last visible fraction was {last_visible_fraction:.3f}. "
        "Loosen --truncated-visible-fraction-* or increase --truncated-max-center-draws."
    )


def fwhm_arcmin_to_sigma_pixels(fwhm_arcmin):
    if fwhm_arcmin <= 0.0:
        return 0.0
    return float(fwhm_arcmin) / (2.0 * np.sqrt(2.0 * np.log(2.0)) * RESO_ARCMIN)


def noise_sigma_k_per_pixel(noise_sigma_uk_arcmin):
    if noise_sigma_uk_arcmin <= 0.0:
        return 0.0
    return float(noise_sigma_uk_arcmin) / RESO_ARCMIN * 1e-6


def draw_patch_noise(rng, shape, noise_sigma_uk_arcmin, noise_corr_fwhm_arcmin):
    sigma_k = noise_sigma_k_per_pixel(noise_sigma_uk_arcmin)
    if sigma_k <= 0.0:
        return np.zeros(shape, dtype=np.float32)

    noise = rng.normal(loc=0.0, scale=sigma_k, size=shape).astype(np.float32)
    corr_sigma_pix = fwhm_arcmin_to_sigma_pixels(noise_corr_fwhm_arcmin)
    if corr_sigma_pix > 0.0:
        noise = gaussian_filter(noise, sigma=corr_sigma_pix, mode="reflect")
        std = float(np.std(noise))
        if std > 0.0:
            noise = noise * (sigma_k / std)
    return noise.astype(np.float32)


def apply_observing_model_to_patch(
    patch,
    rng,
    beam_fwhm_arcmin,
    noise_sigma_uk_arcmin,
    noise_corr_fwhm_arcmin,
    legacy_patch_beam=False,
):
    observed = np.asarray(patch, dtype=np.float32)
    beam_sigma_pix = fwhm_arcmin_to_sigma_pixels(beam_fwhm_arcmin) if legacy_patch_beam else 0.0
    if beam_sigma_pix > 0.0:
        observed = gaussian_filter(observed, sigma=beam_sigma_pix, mode="reflect")
    observed = observed + draw_patch_noise(
        rng=rng,
        shape=observed.shape,
        noise_sigma_uk_arcmin=noise_sigma_uk_arcmin,
        noise_corr_fwhm_arcmin=noise_corr_fwhm_arcmin,
    )
    return observed.astype(np.float32)


def make_preview_grid(indices, patches, labels, metadata, truth, output_path):
    if len(indices) == 0:
        return

    ncols = min(4, len(indices))
    nrows = int(math.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, indices):
        patch = patches[idx]
        ax.imshow(patch, cmap="RdBu_r", origin="lower")
        label = "pos" if labels[idx] == 1 else "neg"
        glon = metadata["glon_deg"][idx]
        glat = metadata["glat_deg"][idx]
        split_name = "train" if metadata["split_tag"][idx] == SPLIT_TRAIN else "val"
        if labels[idx] == 1:
            theta_crit = truth["theta_crit_deg"][idx]
            dx_deg = truth["signal_center_dx_deg"][idx]
            dy_deg = truth["signal_center_dy_deg"][idx]
            ax.set_title(
                f"{label}/{split_name} lon={glon:.1f}, lat={glat:.1f}\nR={theta_crit:.1f} deg",
                fontsize=10,
            )
            ax.set_xlabel(f"dx={dx_deg:.1f} deg, dy={dy_deg:.1f} deg", fontsize=9)
        else:
            ax.set_title(f"{label}/{split_name} lon={glon:.1f}, lat={glat:.1f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(indices):]:
        ax.set_visible(False)

    fig.suptitle("Random training samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_positive_preview(indices, patches, masks, metadata, truth, output_path):
    if len(indices) == 0:
        return

    fig, axes = plt.subplots(len(indices), 3, figsize=(14, 4 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(indices):
        patch_ax = axes[row, 0]
        signal_ax = axes[row, 1]
        mask_ax = axes[row, 2]
        visible_fraction = float(truth["visible_target_fraction"][idx]) if "visible_target_fraction" in truth else 1.0

        patch_ax.imshow(patches[idx], cmap="RdBu_r", origin="lower")
        patch_ax.set_title(
            "Patch\n"
            f"split={'train' if metadata['split_tag'][idx] == SPLIT_TRAIN else 'val'} "
            f"lon={metadata['glon_deg'][idx]:.1f}, lat={metadata['glat_deg'][idx]:.1f}\n"
            f"R={truth['theta_crit_deg'][idx]:.1f} deg, "
            f"z0={truth['z0'][idx]:.2e}, zcrit={truth['zcrit'][idx]:.2e}, "
            f"sigma={truth['edge_sigma_deg'][idx]:.2f}\n"
            f"dx={truth['signal_center_dx_deg'][idx]:.1f} deg, "
            f"dy={truth['signal_center_dy_deg'][idx]:.1f} deg, "
            f"visible={visible_fraction:.2f}",
            fontsize=10,
        )
        patch_ax.set_xticks([])
        patch_ax.set_yticks([])

        theta_grid = make_angular_distance_grid(
            PATCH_PIX,
            RESO_ARCMIN,
            center_x_pix=float(truth["signal_center_x_pix"][idx]),
            center_y_pix=float(truth["signal_center_y_pix"][idx]),
        )
        signal = np.where(
            theta_grid <= np.radians(float(truth["theta_crit_deg"][idx])),
            1.0,
            0.0,
        )
        signal_ax.imshow(signal, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        signal_ax.set_title("Causal disc support", fontsize=10)
        signal_ax.set_xticks([])
        signal_ax.set_yticks([])

        mask_ax.imshow(masks[idx], cmap="gray", origin="lower", vmin=0, vmax=1)
        mask_ax.set_title("Target mask", fontsize=10)
        mask_ax.set_xticks([])
        mask_ax.set_yticks([])

    fig.suptitle("Positive patches and target masks", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_validation_histograms(labels, metadata, truth, output_path):
    pos = labels == 1
    fig, axes = plt.subplots(2, 5, figsize=(21, 8.8))
    axes = axes.ravel()

    axes[0].hist(metadata["glon_deg"], bins=30, color="#2563eb", alpha=0.85)
    axes[0].set_title("Galactic longitude")

    axes[1].hist(metadata["glat_deg"], bins=30, color="#0891b2", alpha=0.85)
    axes[1].set_title("Galactic latitude")

    axes[2].hist(metadata["coord_mask_fraction"], bins=30, color="#0f766e", alpha=0.85)
    axes[2].set_title("Projected mask fraction")

    axes[3].hist(truth["theta_crit_deg"][pos], bins=30, color="#7c3aed", alpha=0.85)
    axes[3].set_title(r"$\theta_{\rm crit}$")

    axes[4].hist(truth["z0"][pos], bins=30, color="#dc2626", alpha=0.85)
    axes[4].set_title(r"$z_0$")

    axes[5].hist(truth["zcrit"][pos], bins=30, color="#ea580c", alpha=0.85)
    axes[5].set_title(r"$z_{\rm crit}$")

    axes[6].hist(truth["edge_sigma_deg"][pos], bins=30, color="#16a34a", alpha=0.85)
    axes[6].set_title("Edge sigma (deg)")

    axes[7].hist(truth["signal_center_dx_deg"][pos], bins=30, color="#7c2d12", alpha=0.85)
    axes[7].set_title("Signal center dx (deg)")

    axes[8].hist(truth["signal_center_dy_deg"][pos], bins=30, color="#1d4ed8", alpha=0.85)
    axes[8].set_title("Signal center dy (deg)")

    if "visible_target_fraction" in truth:
        axes[9].hist(truth["visible_target_fraction"][pos], bins=30, color="#111827", alpha=0.85)
        axes[9].set_title("Visible target fraction")
    else:
        axes[9].hist(truth["target_touches_edge"][pos], bins=np.array([-0.5, 0.5, 1.5]), color="#111827", alpha=0.85)
        axes[9].set_title("Target touches edge")
        axes[9].set_xticks([0, 1])

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.suptitle("Phase 2 parameter validation", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    output_dir,
    patches,
    labels,
    masks,
    metadata,
    truth,
    summary,
    preview_count,
    rng,
    split_payload,
    coordinate_pool_payload,
):
    os.makedirs(output_dir, exist_ok=True)

    h5_path = os.path.join(output_dir, "training_data.h5")
    summary_path = os.path.join(output_dir, "summary.json")
    preview_samples_path = os.path.join(output_dir, "preview_samples.png")
    preview_positives_path = os.path.join(output_dir, "preview_positives.png")
    validation_hist_path = os.path.join(output_dir, "validation_histograms.png")

    def content_sha256(index_array, chunk_size=64):
        digest = hashlib.sha256()
        index_array = np.asarray(index_array, dtype=np.int64)
        for name, array in (
            ("patches", patches),
            ("labels", labels),
            ("masks", masks),
        ):
            subset_shape = (len(index_array), *array.shape[1:])
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(np.asarray(subset_shape, dtype=np.int64).tobytes())
            for start in range(0, len(index_array), int(chunk_size)):
                rows = index_array[start : start + int(chunk_size)]
                digest.update(np.ascontiguousarray(array[rows]).tobytes())
        for group_name, payload in (("metadata", metadata), ("truth", truth)):
            for key in sorted(payload):
                array = payload[key]
                subset_shape = (len(index_array), *array.shape[1:])
                digest.update(group_name.encode("utf-8"))
                digest.update(key.encode("utf-8"))
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(np.asarray(subset_shape, dtype=np.int64).tobytes())
                for start in range(0, len(index_array), int(chunk_size)):
                    rows = index_array[start : start + int(chunk_size)]
                    digest.update(np.ascontiguousarray(array[rows]).tobytes())
        return digest.hexdigest()

    all_indices = np.arange(len(labels), dtype=np.int64)
    summary = {
        **summary,
        "config_json": json.dumps(summary, sort_keys=True, default=str),
        "dataset_sha256": content_sha256(all_indices),
        "dataset_sha256_policy": "sha256_over_core_hdf5_arrays_before_compression",
    }

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("patches", data=patches.astype(np.float32), compression="gzip", shuffle=True)
        h5.create_dataset("labels", data=labels.astype(np.uint8), compression="gzip", shuffle=True)
        h5.create_dataset("masks", data=masks.astype(np.uint8), compression="gzip", shuffle=True)

        metadata_group = h5.create_group("metadata")
        for key, value in metadata.items():
            metadata_group.create_dataset(key, data=value, compression="gzip", shuffle=True)

        truth_group = h5.create_group("truth")
        for key, value in truth.items():
            truth_group.create_dataset(key, data=value, compression="gzip", shuffle=True)

        splits_group = h5.create_group("splits")
        for key, value in split_payload.items():
            splits_group.create_dataset(key, data=value, compression="gzip", shuffle=True)

        coord_group = h5.create_group("coordinate_pool")
        for key, value in coordinate_pool_payload.items():
            coord_group.create_dataset(key, data=value, compression="gzip", shuffle=True)

        summary_group = h5.create_group("summary")
        for key, value in summary.items():
            summary_group.attrs[key] = value

    def write_split_file(filename, split_name, indices):
        path = os.path.join(output_dir, filename)
        indices = np.asarray(indices, dtype=np.int64)
        split_summary = {
            **summary,
            "source_dataset_path": os.path.abspath(h5_path),
            "split_name": split_name,
            "num_samples": int(len(indices)),
            "dataset_sha256": content_sha256(indices),
        }
        with h5py.File(path, "w") as split_h5:
            split_h5.create_dataset("patches", data=patches[indices].astype(np.float32), compression="gzip", shuffle=True)
            split_h5.create_dataset("labels", data=labels[indices].astype(np.uint8), compression="gzip", shuffle=True)
            split_h5.create_dataset("masks", data=masks[indices].astype(np.uint8), compression="gzip", shuffle=True)
            metadata_group = split_h5.create_group("metadata")
            for key, value in metadata.items():
                metadata_group.create_dataset(key, data=value[indices], compression="gzip", shuffle=True)
            truth_group = split_h5.create_group("truth")
            for key, value in truth.items():
                truth_group.create_dataset(key, data=value[indices], compression="gzip", shuffle=True)
            splits_group = split_h5.create_group("splits")
            split_idx = np.arange(len(indices), dtype=np.int64)
            splits_group.create_dataset(f"{split_name}_idx", data=split_idx, compression="gzip", shuffle=True)
            if split_name == "calibration":
                splits_group.create_dataset("val_idx", data=split_idx, compression="gzip", shuffle=True)
            coord_group = split_h5.create_group("coordinate_pool")
            for key, value in coordinate_pool_payload.items():
                coord_group.create_dataset(key, data=value, compression="gzip", shuffle=True)
            summary_group = split_h5.create_group("summary")
            for key, value in split_summary.items():
                summary_group.attrs[key] = value
        return path

    split_files = {}
    for split_name, filename, key in (
        ("calibration", "calibration_data.h5", "calibration_idx"),
        ("test", "test_data.h5", "test_idx"),
    ):
        if key in split_payload and len(split_payload[key]) > 0:
            split_files[split_name] = write_split_file(filename, split_name, split_payload[key])

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    sample_preview_count = min(preview_count, len(labels))
    sample_indices = rng.choice(len(labels), size=sample_preview_count, replace=False)
    make_preview_grid(sample_indices, patches, labels, metadata, truth, preview_samples_path)

    positive_indices = np.flatnonzero(labels == 1)
    positive_preview_count = min(preview_count, len(positive_indices))
    positive_preview_indices = rng.choice(positive_indices, size=positive_preview_count, replace=False)
    make_positive_preview(positive_preview_indices, patches, masks, metadata, truth, preview_positives_path)
    make_validation_histograms(labels, metadata, truth, validation_hist_path)

    print("\n=== Saved outputs ===")
    print(f"  {h5_path}")
    print(f"  {summary_path}")
    print(f"  {preview_samples_path}")
    print(f"  {preview_positives_path}")
    print(f"  {validation_hist_path}")
    for split_name, split_path in split_files.items():
        print(f"  {split_name}: {split_path}")
    return h5_path


def split_class_counts(total_count, train_fraction):
    train_count = int(round(total_count * train_fraction))
    train_count = min(max(train_count, 1), total_count - 1)
    val_count = total_count - train_count
    return train_count, val_count


def split_class_counts_three(total_count, train_fraction, calibration_fraction):
    train_count = int(round(total_count * train_fraction))
    calibration_count = int(round(total_count * calibration_fraction))
    train_count = min(max(train_count, 1), total_count - 2)
    calibration_count = min(max(calibration_count, 1), total_count - train_count - 1)
    test_count = total_count - train_count - calibration_count
    return train_count, calibration_count, test_count


def main():
    args = parse_args()
    ensure_even_sample_count(args.num_samples)
    validate_args(args)

    if not args.skip_physics_checks:
        print("\n=== Running pre-generation physics checks ===")
        check_eq1_special_cases()
        check_smooth_window_bounds()
        check_multiplicative_injection()
        check_patch_geometry()
        print("  Physics checks: pass")

    rng = np.random.default_rng(args.seed)
    mask_256, sky_fraction = load_mask(threshold=args.mask_threshold)
    coord_pool, coord_mask_fractions = build_coordinate_pool(
        mask_256,
        args.pool_size,
        rng,
        min_unmasked_fraction=args.mask_threshold,
    )
    coord_cluster_pix = coordinate_cluster_pixels(coord_pool, args.coord_cluster_nside)
    coord_cluster_ids = coordinate_cluster_ids(coord_cluster_pix, args.coord_cluster_nside)
    coord_train_idx, coord_calibration_idx, coord_test_idx = split_index_pool_three_by_group(
        coord_cluster_pix,
        args.train_fraction,
        args.calibration_fraction,
        args.split_seed,
    )
    cmb_realizations, observing_provenance = generate_camb_realizations(
        args.num_cmb_realizations,
        rng,
        beam_fwhm_arcmin=args.beam_fwhm_arcmin,
    )
    realization_train_idx, realization_calibration_idx, realization_test_idx = split_index_pool_three(
        len(cmb_realizations),
        args.train_fraction,
        args.calibration_fraction,
        args.split_seed + 1,
    )

    num_samples = args.num_samples
    num_positive = num_samples // 2
    num_negative = num_samples - num_positive
    train_positive, calibration_positive, test_positive = split_class_counts_three(
        num_positive,
        args.train_fraction,
        args.calibration_fraction,
    )
    train_negative, calibration_negative, test_negative = split_class_counts_three(
        num_negative,
        args.train_fraction,
        args.calibration_fraction,
    )
    train_samples = train_positive + train_negative
    calibration_samples = calibration_positive + calibration_negative
    test_samples = test_positive + test_negative

    print("\n=== Split design ===")
    print(f"  Train samples:       {train_samples} ({train_positive} pos / {train_negative} neg)")
    print(
        f"  Calibration samples: {calibration_samples} "
        f"({calibration_positive} pos / {calibration_negative} neg)"
    )
    print(f"  Test samples:        {test_samples} ({test_positive} pos / {test_negative} neg)")
    print(f"  Train coordinates:   {len(coord_train_idx)}")
    print(f"  Calibration coords:  {len(coord_calibration_idx)}")
    print(f"  Test coordinates:    {len(coord_test_idx)}")
    print(f"  Coordinate clusters: {len(np.unique(coord_cluster_pix))} at Nside={args.coord_cluster_nside}")
    print(f"  Train clusters:      {len(np.unique(coord_cluster_pix[coord_train_idx]))}")
    print(f"  Calibration clusters:{len(np.unique(coord_cluster_pix[coord_calibration_idx]))}")
    print(f"  Test clusters:       {len(np.unique(coord_cluster_pix[coord_test_idx]))}")
    print(f"  Train realizations:  {len(realization_train_idx)}")
    print(f"  Calibration skies:   {len(realization_calibration_idx)}")
    print(f"  Test skies:          {len(realization_test_idx)}")

    patches = np.empty((num_samples, PATCH_PIX, PATCH_PIX), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.uint8)
    masks = np.zeros((num_samples, PATCH_PIX, PATCH_PIX), dtype=np.uint8)

    metadata = {
        "sample_index": np.arange(num_samples, dtype=np.int32),
        "split_tag": np.zeros(num_samples, dtype=np.uint8),
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

    train_idx = np.arange(train_samples, dtype=np.int64)
    calibration_idx = np.arange(train_samples, train_samples + calibration_samples, dtype=np.int64)
    test_idx = np.arange(train_samples + calibration_samples, num_samples, dtype=np.int64)
    val_idx = calibration_idx

    split_specs = [
        {
            "name": "train",
            "sample_indices": train_idx,
            "split_tag": SPLIT_TRAIN,
            "num_positive": train_positive,
            "coord_indices": coord_train_idx,
            "realization_indices": realization_train_idx,
            "seed_offset": 1000,
        },
        {
            "name": "calibration",
            "sample_indices": calibration_idx,
            "split_tag": SPLIT_CALIBRATION,
            "num_positive": calibration_positive,
            "coord_indices": coord_calibration_idx,
            "realization_indices": realization_calibration_idx,
            "seed_offset": 2000,
        },
        {
            "name": "test",
            "sample_indices": test_idx,
            "split_tag": SPLIT_TEST,
            "num_positive": test_positive,
            "coord_indices": coord_test_idx,
            "realization_indices": realization_test_idx,
            "seed_offset": 3000,
        },
    ]

    for split_spec in split_specs:
        split_rng = np.random.default_rng(args.seed + split_spec["seed_offset"])
        sample_indices = split_spec["sample_indices"]
        positive_flags = np.zeros(len(sample_indices), dtype=bool)
        positive_flags[: split_spec["num_positive"]] = True
        split_rng.shuffle(positive_flags)
        sign_pairs = build_balanced_sign_pairs(split_spec["num_positive"], split_rng)
        positive_counter = 0

        for local_offset, sample_idx in enumerate(sample_indices):
            coord_idx = int(split_rng.choice(split_spec["coord_indices"]))
            realization_idx = int(split_rng.choice(split_spec["realization_indices"]))
            lon_i, lat_i = coord_pool[coord_idx]
            mask_fraction_i = float(coord_mask_fractions[coord_idx])

            clean_patch = np.asarray(
                project_patch(cmb_realizations[realization_idx], float(lon_i), float(lat_i)),
                dtype=np.float32,
            )

            metadata["split_tag"][sample_idx] = split_spec["split_tag"]
            metadata["glon_deg"][sample_idx] = lon_i
            metadata["glat_deg"][sample_idx] = lat_i
            metadata["coord_pool_idx"][sample_idx] = coord_idx
            metadata["coord_cluster_id"][sample_idx] = coord_cluster_ids[coord_idx]
            metadata["coord_mask_fraction"][sample_idx] = mask_fraction_i
            metadata["cmb_realization_idx"][sample_idx] = realization_idx

            background_id = stable_group_id("camb", realization_idx, coord_idx)
            metadata["background_id"][sample_idx] = np.uint64(background_id)
            metadata["split_group_id"][sample_idx] = np.uint64(background_id)

            if positive_flags[local_offset]:
                theta_i = sample_theta_crit_from_training_prior(split_rng, 5.0, 25.0)
                sign_z0_i, sign_zcrit_i = sign_pairs[positive_counter]
                z0_i = sign_z0_i * sample_log_uniform(split_rng, 1e-6, 1e-4)
                zcrit_i = sign_zcrit_i * sample_log_uniform(split_rng, 1e-6, 1e-4)
                edge_sigma_i = split_rng.uniform(args.edge_sigma_min_deg, args.edge_sigma_max_deg)
                actual_geometry_mode = sample_actual_geometry_mode(
                    split_rng,
                    args.geometry_mode,
                    args.truncated_positive_fraction,
                )
                geometry_i = sample_signal_geometry(
                    rng=split_rng,
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
                center_pix = patch_center_pixel(PATCH_PIX)
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
                touches_edge = int(geometry_i["target_touches_edge"])
                fully_contained = int(geometry_i["fully_contained"])

                observed_patch = apply_observing_model_to_patch(
                    injected_patch,
                    rng=split_rng,
                    beam_fwhm_arcmin=args.beam_fwhm_arcmin,
                    noise_sigma_uk_arcmin=args.noise_sigma_uk_arcmin,
                    noise_corr_fwhm_arcmin=args.noise_corr_fwhm_arcmin,
                    legacy_patch_beam=args.legacy_patch_beam,
                )

                event_id = stable_group_id(
                    background_id,
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
                truth["fully_contained"][sample_idx] = fully_contained
                truth["target_touches_edge"][sample_idx] = int(touches_edge)
                truth["visible_target_fraction"][sample_idx] = geometry_i["visible_target_fraction"]
                truth["visible_target_pixels"][sample_idx] = geometry_i["visible_target_pixels"]
                truth["full_disc_pixels_est"][sample_idx] = geometry_i["full_disc_pixels_est"]
                truth["target_edge_contact_pixels"][sample_idx] = geometry_i["target_edge_contact_pixels"]
                truth["disc_edge_margin_pix"][sample_idx] = geometry_i["disc_edge_margin_pix"]
                truth["signal_center_in_patch"][sample_idx] = geometry_i["signal_center_in_patch"]
                positive_counter += 1
            else:
                observed_patch = apply_observing_model_to_patch(
                    clean_patch,
                    rng=split_rng,
                    beam_fwhm_arcmin=args.beam_fwhm_arcmin,
                    noise_sigma_uk_arcmin=args.noise_sigma_uk_arcmin,
                    noise_corr_fwhm_arcmin=args.noise_corr_fwhm_arcmin,
                    legacy_patch_beam=args.legacy_patch_beam,
                )
                patches[sample_idx] = observed_patch

            if (local_offset + 1) % 50 == 0 or (local_offset + 1) == len(sample_indices):
                positives_so_far = int(labels[sample_indices[: local_offset + 1]].sum())
                print(
                    f"  {split_spec['name']}: generated {local_offset + 1:4d} / {len(sample_indices)} "
                    f"samples (positives so far: {positives_so_far})"
                )

    summary = {
        "num_samples": int(num_samples),
        "num_positive": int(labels.sum()),
        "num_negative": int(num_samples - labels.sum()),
        "num_train_samples": int(len(train_idx)),
        "num_calibration_samples": int(len(calibration_idx)),
        "num_test_samples": int(len(test_idx)),
        "num_val_samples": int(len(val_idx)),
        "num_train_positive": int(labels[train_idx].sum()),
        "num_train_negative": int(len(train_idx) - labels[train_idx].sum()),
        "num_calibration_positive": int(labels[calibration_idx].sum()),
        "num_calibration_negative": int(len(calibration_idx) - labels[calibration_idx].sum()),
        "num_test_positive": int(labels[test_idx].sum()),
        "num_test_negative": int(len(test_idx) - labels[test_idx].sum()),
        "num_val_positive": int(labels[val_idx].sum()),
        "num_val_negative": int(len(val_idx) - labels[val_idx].sum()),
        "pool_size": int(len(coord_pool)),
        "num_train_coordinates": int(len(coord_train_idx)),
        "num_calibration_coordinates": int(len(coord_calibration_idx)),
        "num_test_coordinates": int(len(coord_test_idx)),
        "num_val_coordinates": int(len(coord_calibration_idx)),
        "coord_cluster_nside": int(args.coord_cluster_nside),
        "num_coordinate_clusters": int(len(np.unique(coord_cluster_pix))),
        "num_train_coordinate_clusters": int(len(np.unique(coord_cluster_pix[coord_train_idx]))),
        "num_calibration_coordinate_clusters": int(len(np.unique(coord_cluster_pix[coord_calibration_idx]))),
        "num_test_coordinate_clusters": int(len(np.unique(coord_cluster_pix[coord_test_idx]))),
        "num_cmb_realizations": int(args.num_cmb_realizations),
        "num_train_realizations": int(len(realization_train_idx)),
        "num_calibration_realizations": int(len(realization_calibration_idx)),
        "num_test_realizations": int(len(realization_test_idx)),
        "num_val_realizations": int(len(realization_calibration_idx)),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "train_fraction": float(args.train_fraction),
        "calibration_fraction": float(args.calibration_fraction),
        "test_fraction": float(1.0 - args.train_fraction - args.calibration_fraction),
        "preview_count": int(args.preview_count),
        "nside": int(NSIDE_WORKING),
        "patch_pixels": int(PATCH_PIX),
        "reso_arcmin": float(RESO_ARCMIN),
        "mask_threshold": float(args.mask_threshold),
        "edge_sigma_min_deg": float(args.edge_sigma_min_deg),
        "edge_sigma_max_deg": float(args.edge_sigma_max_deg),
        "geometry_mode": args.geometry_mode,
        "truncated_positive_fraction_requested": float(args.truncated_positive_fraction),
        "truncated_visible_fraction_min": float(args.truncated_visible_fraction_min),
        "truncated_visible_fraction_max": float(args.truncated_visible_fraction_max),
        "truncated_max_center_draws": int(args.truncated_max_center_draws),
        "num_positive_fully_contained": int(truth["fully_contained"][labels == 1].sum()),
        "num_positive_touching_edge": int(truth["target_touches_edge"][labels == 1].sum()),
        "mean_positive_visible_target_fraction": float(np.mean(truth["visible_target_fraction"][labels == 1])),
        "min_positive_visible_target_fraction": float(np.min(truth["visible_target_fraction"][labels == 1])),
        "contained_margin_deg": float(args.contained_margin_deg),
        "signal_center_edge_margin_pix": float(args.signal_center_edge_margin_pix),
        "sky_fraction": float(sky_fraction),
        "theta_training_distribution": "sin(theta_crit)",
        "theta_distribution_note": "Eq. 2-motivated training design choice; not the downstream inference prior.",
        "z0_sign_sampling": "balanced",
        "zcrit_sign_sampling": "balanced",
        "split_method": "healpix_coordinate_cluster_and_realization_disjoint_train_calibration_test",
        "beam_fwhm_arcmin": float(args.beam_fwhm_arcmin),
        "beam_domain": "harmonic_sphere" if not args.legacy_patch_beam else "legacy_patch_space",
        "pixel_window_policy": "synfast_pixwin_true",
        "legacy_patch_beam": bool(args.legacy_patch_beam),
        "noise_sigma_uk_arcmin": float(args.noise_sigma_uk_arcmin),
        "noise_corr_fwhm_arcmin": float(args.noise_corr_fwhm_arcmin),
        "camb_params": json.dumps(observing_provenance["camb"]["params"], sort_keys=True),
        "observing_model": json.dumps(observing_provenance, sort_keys=True),
        "output_dir": os.path.abspath(args.output_dir),
        "dataset_path": os.path.abspath(os.path.join(args.output_dir, "training_data.h5")),
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }

    split_payload = {
        "train_idx": train_idx.astype(np.int64),
        "calibration_idx": calibration_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "coord_train_idx": coord_train_idx.astype(np.int32),
        "coord_calibration_idx": coord_calibration_idx.astype(np.int32),
        "coord_test_idx": coord_test_idx.astype(np.int32),
        "coord_val_idx": coord_calibration_idx.astype(np.int32),
        "coord_cluster_train_pix": np.unique(coord_cluster_pix[coord_train_idx]).astype(np.int32),
        "coord_cluster_calibration_pix": np.unique(coord_cluster_pix[coord_calibration_idx]).astype(np.int32),
        "coord_cluster_test_pix": np.unique(coord_cluster_pix[coord_test_idx]).astype(np.int32),
        "coord_cluster_val_pix": np.unique(coord_cluster_pix[coord_calibration_idx]).astype(np.int32),
        "realization_train_idx": realization_train_idx.astype(np.int32),
        "realization_calibration_idx": realization_calibration_idx.astype(np.int32),
        "realization_test_idx": realization_test_idx.astype(np.int32),
        "realization_val_idx": realization_calibration_idx.astype(np.int32),
    }
    coordinate_pool_payload = {
        "glon_deg": coord_pool[:, 0].astype(np.float32),
        "glat_deg": coord_pool[:, 1].astype(np.float32),
        "mask_fraction": coord_mask_fractions.astype(np.float32),
        "coord_cluster_pix": coord_cluster_pix.astype(np.int32),
        "coord_cluster_id": coord_cluster_ids.astype(np.uint64),
    }

    print("\n=== Final class counts ===")
    print(f"  Positive: {summary['num_positive']}")
    print(f"  Negative: {summary['num_negative']}")
    print(f"  Geometry mode: {args.geometry_mode}")
    print(
        f"  Positive targets touching edge: "
        f"{int(truth['target_touches_edge'][labels == 1].sum())} / {int(labels.sum())}"
    )
    print(
        f"  Positive visible fraction: "
        f"mean={float(np.mean(truth['visible_target_fraction'][labels == 1])):.3f}, "
        f"min={float(np.min(truth['visible_target_fraction'][labels == 1])):.3f}"
    )

    h5_path = save_outputs(
        output_dir=args.output_dir,
        patches=patches,
        labels=labels,
        masks=masks,
        metadata=metadata,
        truth=truth,
        summary=summary,
        preview_count=args.preview_count,
        rng=rng,
        split_payload=split_payload,
        coordinate_pool_payload=coordinate_pool_payload,
    )
    write_observing_model_provenance(
        os.path.join(args.output_dir, "observing_model_provenance.json"),
        observing_provenance,
    )

    if not args.skip_post_audit:
        print("\n=== Running post-generation audit ===")
        audit_report = run_audit(h5_path, allow_legacy=False, sample_patch_count=0)
        audit_path = os.path.join(args.output_dir, "audit_report.json")
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(audit_report, handle, indent=2)
        print(f"  Audit status: {audit_report['status']}")
        print(f"  Audit report: {audit_path}")
        if audit_report["status"] != "pass":
            raise RuntimeError(f"Generated dataset failed audit. See {audit_path}")


if __name__ == "__main__":
    main()
