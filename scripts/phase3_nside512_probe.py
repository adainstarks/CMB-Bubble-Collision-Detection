"""
Historical Nside=512 probe utilities for the small-radius question.

This is intentionally isolated from the production Nside=256 pipeline. It
generates a small 512x512 CAMB training dataset, a focused real-SMICA contested
cell evaluation dataset, and scores that evaluation dataset with a trained
fully-convolutional U-Net checkpoint.

The current remediated-v1 baseline remains Nside=256. Use this script only as
provenance or for a new focused resolution probe, not as the active pipeline
default.
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
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase3_train_unet as p3
from phase1_explore import DATA_DIR, MASK_FILE, MASK_URL, SMICA_FILE, download_file
from phase_config import (
    DEFAULT_INJECTION_CONVENTION,
    INJECTION_CONVENTIONS,
    INJECTION_CONVENTION_MCEWEN2012,
    INJECTION_CONVENTION_NOTES,
    PROVENANCE_SCHEMA_VERSION,
)
from phase2_generate_training import (
    MASK_THRESHOLD,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_balanced_sign_pairs,
    planck2018_bestfit_params,
    projected_unmasked_fraction,
    sample_log_uniform,
    sample_random_galactic_coordinate,
    sample_theta_crit_from_training_prior,
    split_class_counts,
    split_index_pool,
    stable_group_id,
    target_touches_patch_edge,
)
from phase2_signal_model import bubble_collision_signal, fractional_signal_delta
from phase3_evaluate_run import load_json, resolve_checkpoint_path
from phase_dataset_utils import make_angular_distance_grid, patch_center_pixel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "training_v4_nside512_probe"
DEFAULT_EVAL_H5 = PROJECT_ROOT / "runs" / "phase3_unet" / "nside512_probe_eval" / "real_smica_contested_512.h5"
DEFAULT_SCORE_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "nside512_probe_eval"


def add_common_geometry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--patch-pix", type=int, default=512)
    parser.add_argument("--reso-arcmin", type=float, default=6.5)
    parser.add_argument("--mask-threshold", type=float, default=MASK_THRESHOLD)
    parser.add_argument("--contained-margin-deg", type=float, default=0.5)
    parser.add_argument("--beam-fwhm-arcmin", type=float, default=15.0)
    parser.add_argument("--noise-sigma-uk-arcmin", type=float, default=30.0)
    parser.add_argument("--noise-corr-fwhm-arcmin", type=float, default=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and score a focused Nside=512 probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("generate-train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_geometry_args(train)
    train.add_argument("--num-samples", type=int, default=1000)
    train.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    train.add_argument("--seed", type=int, default=20260422)
    train.add_argument("--split-seed", type=int, default=20260423)
    train.add_argument("--train-fraction", type=float, default=0.9)
    train.add_argument("--pool-size", type=int, default=1000)
    train.add_argument("--num-cmb-realizations", type=int, default=64)
    train.add_argument("--lmax", type=int, default=0, help="Default is 3*nside.")
    train.add_argument("--edge-sigma-min-deg", type=float, default=0.3)
    train.add_argument("--edge-sigma-max-deg", type=float, default=1.0)
    train.add_argument("--preview-count", type=int, default=6)
    train.add_argument("--injection-convention", type=str, default=DEFAULT_INJECTION_CONVENTION, choices=INJECTION_CONVENTIONS)

    evalp = sub.add_parser("generate-eval", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_geometry_args(evalp)
    evalp.add_argument("--output-h5", type=str, default=str(DEFAULT_EVAL_H5))
    evalp.add_argument("--num-positive", type=int, default=200)
    evalp.add_argument("--num-negative", type=int, default=0)
    evalp.add_argument("--seed", type=int, default=20260424)
    evalp.add_argument("--pool-size", type=int, default=300)
    evalp.add_argument("--amplitude", type=float, default=2e-5)
    evalp.add_argument("--theta-crit-deg", type=float, default=5.0)
    evalp.add_argument("--edge-sigma-min-deg", type=float, default=0.3)
    evalp.add_argument("--edge-sigma-max-deg", type=float, default=1.0)
    evalp.add_argument("--injection-convention", type=str, default=DEFAULT_INJECTION_CONVENTION, choices=INJECTION_CONVENTIONS)

    score = sub.add_parser("score-eval", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    score.add_argument("--eval-h5", type=str, default=str(DEFAULT_EVAL_H5))
    score.add_argument("--run-dir", type=str, required=True)
    score.add_argument("--checkpoint", type=str, default="best")
    score.add_argument("--output-dir", type=str, default=str(DEFAULT_SCORE_DIR))
    score.add_argument("--thresholds", type=str, default="0.75,0.80,0.85,0.90")
    score.add_argument("--batch-size", type=int, default=2)
    score.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def validate_common(args: argparse.Namespace) -> None:
    if args.nside <= 0 or args.patch_pix <= 0 or args.reso_arcmin <= 0.0:
        raise ValueError("nside, patch-pix, and reso-arcmin must be positive.")
    if args.beam_fwhm_arcmin < 0.0 or args.noise_sigma_uk_arcmin < 0.0 or args.noise_corr_fwhm_arcmin < 0.0:
        raise ValueError("Observing-model parameters must be non-negative.")


def ensure_inputs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    download_file(MASK_URL, MASK_FILE)


def load_mask(nside: int) -> tuple[np.ndarray, float]:
    ensure_inputs()
    mask = hp.read_map(MASK_FILE, field=0)
    mask_work = hp.ud_grade(mask, nside)
    mask_work = np.where(mask_work > 0.5, 1.0, 0.0).astype(np.float32)
    return mask_work, float(np.mean(mask_work))


def load_smica(nside: int) -> np.ndarray:
    ensure_inputs()
    smica = hp.read_map(SMICA_FILE, field=0)
    return hp.ud_grade(smica, nside).astype(np.float32)


def project_patch(hp_map: np.ndarray, glon_deg: float, glat_deg: float, patch_pix: int, reso_arcmin: float) -> np.ndarray:
    return hp.gnomview(
        hp_map,
        rot=(float(glon_deg), float(glat_deg)),
        reso=float(reso_arcmin),
        xsize=int(patch_pix),
        return_projected_map=True,
        no_plot=True,
    )


def is_center_unmasked(mask: np.ndarray, nside: int, glon_deg: float, glat_deg: float) -> bool:
    theta = np.radians(90.0 - float(glat_deg))
    phi = np.radians(float(glon_deg))
    pix = hp.ang2pix(nside, theta, phi)
    return bool(mask[pix] > 0.5)


def build_coordinate_pool(args: argparse.Namespace, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    coords: list[tuple[float, float]] = []
    mask_fractions: list[float] = []
    attempts = 0
    max_attempts = max(int(args.pool_size) * 250, 1000)
    print("\n=== Building Nside=512 coordinate pool ===")
    while len(coords) < int(args.pool_size) and attempts < max_attempts:
        attempts += 1
        lon, lat = sample_random_galactic_coordinate(rng)
        if not is_center_unmasked(mask, args.nside, lon, lat):
            continue
        patch_mask = project_patch(mask, lon, lat, args.patch_pix, args.reso_arcmin)
        frac = projected_unmasked_fraction(patch_mask)
        if frac < float(args.mask_threshold):
            continue
        coords.append((lon, lat))
        mask_fractions.append(frac)
        if len(coords) % 100 == 0 or len(coords) == int(args.pool_size):
            print(f"  Accepted {len(coords):4d} / {args.pool_size} centers after {attempts} attempts")
    if len(coords) < int(args.pool_size):
        raise RuntimeError(f"Could only build {len(coords)} coordinates after {attempts} attempts.")
    return np.asarray(coords, dtype=np.float32), np.asarray(mask_fractions, dtype=np.float32)


def generate_camb_realizations(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, dict, int]:
    print("\n=== Generating Nside=512 CAMB realizations ===")
    try:
        import camb
    except ImportError as exc:
        raise RuntimeError("CAMB is required for this probe.") from exc
    params = planck2018_bestfit_params()
    lmax = int(args.lmax) if int(args.lmax) > 0 else int(3 * args.nside)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params["H0"], ombh2=params["ombh2"], omch2=params["omch2"], tau=params["tau"])
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    cls_tt = results.get_cmb_power_spectra(pars, CMB_unit="K", raw_cl=True)["lensed_scalar"][:, 0]
    cls_tt = cls_tt[: lmax + 1]
    realizations = np.empty((int(args.num_cmb_realizations), hp.nside2npix(args.nside)), dtype=np.float32)
    seeds = rng.integers(0, 2**32 - 1, size=int(args.num_cmb_realizations), dtype=np.uint32)
    for idx, seed_i in enumerate(seeds):
        np.random.seed(int(seed_i))
        realizations[idx] = hp.synfast(cls_tt, args.nside, lmax=lmax, new=True, pixwin=False).astype(np.float32)
        if (idx + 1) % 8 == 0 or (idx + 1) == int(args.num_cmb_realizations):
            print(f"  Generated {idx + 1:4d} / {args.num_cmb_realizations} skies")
    return realizations, params, lmax


def fwhm_arcmin_to_sigma_pixels(fwhm_arcmin: float, reso_arcmin: float) -> float:
    if fwhm_arcmin <= 0.0:
        return 0.0
    return float(fwhm_arcmin) / (2.0 * np.sqrt(2.0 * np.log(2.0)) * float(reso_arcmin))


def noise_sigma_k_per_pixel(noise_sigma_uk_arcmin: float, reso_arcmin: float) -> float:
    if noise_sigma_uk_arcmin <= 0.0:
        return 0.0
    return float(noise_sigma_uk_arcmin) / float(reso_arcmin) * 1e-6


def draw_patch_noise(args: argparse.Namespace, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    sigma = noise_sigma_k_per_pixel(args.noise_sigma_uk_arcmin, args.reso_arcmin)
    if sigma <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    noise = rng.normal(0.0, sigma, size=shape).astype(np.float32)
    corr_sigma = fwhm_arcmin_to_sigma_pixels(args.noise_corr_fwhm_arcmin, args.reso_arcmin)
    if corr_sigma > 0.0:
        noise = gaussian_filter(noise, sigma=corr_sigma, mode="reflect")
        std = float(noise.std())
        if std > 0.0:
            noise *= sigma / std
    return noise.astype(np.float32)


def observe_patch(args: argparse.Namespace, patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    observed = np.asarray(patch, dtype=np.float32)
    beam_sigma = fwhm_arcmin_to_sigma_pixels(args.beam_fwhm_arcmin, args.reso_arcmin)
    if beam_sigma > 0.0:
        observed = gaussian_filter(observed, sigma=beam_sigma, mode="reflect")
    observed = observed + draw_patch_noise(args, rng, observed.shape)
    return observed.astype(np.float32)


def sample_signal_center_pixels(
    rng: np.random.Generator,
    patch_pix: int,
    reso_arcmin: float,
    theta_crit_deg: float,
    contained_margin_deg: float,
) -> tuple[float, float]:
    margin_pix = float(contained_margin_deg) * 60.0 / float(reso_arcmin)
    radius_pix = float(theta_crit_deg) * 60.0 / float(reso_arcmin)
    low = radius_pix + margin_pix
    high = float(patch_pix - 1) - low
    if low > high:
        raise RuntimeError("Contained signal is not feasible for this patch geometry.")
    return float(rng.uniform(low, high)), float(rng.uniform(low, high))


def inject_signal(
    patch: np.ndarray,
    patch_pix: int,
    reso_arcmin: float,
    z0: float,
    zcrit: float,
    theta_crit_deg: float,
    edge_sigma_deg: float,
    center_x_pix: float,
    center_y_pix: float,
    injection_convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    theta_grid = make_angular_distance_grid(
        patch_pix,
        reso_arcmin,
        center_x_pix=center_x_pix,
        center_y_pix=center_y_pix,
    )
    signal = bubble_collision_signal(
        theta_grid,
        z0,
        zcrit,
        np.radians(theta_crit_deg),
        edge_sigma_deg=edge_sigma_deg,
    )
    injected = np.asarray(patch, dtype=np.float64) + fractional_signal_delta(
        patch,
        signal,
        injection_convention=injection_convention,
    )
    mask = (theta_grid <= np.radians(theta_crit_deg)).astype(np.uint8)
    return injected.astype(np.float32), mask


def empty_arrays(num_samples: int, patch_pix: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    patches = np.empty((num_samples, patch_pix, patch_pix), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.uint8)
    masks = np.zeros((num_samples, patch_pix, patch_pix), dtype=np.uint8)
    metadata = {
        "sample_index": np.arange(num_samples, dtype=np.int32),
        "split_tag": np.zeros(num_samples, dtype=np.uint8),
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
        "signal_center_x_pix": np.full(num_samples, patch_center_pixel(patch_pix), dtype=np.float32),
        "signal_center_y_pix": np.full(num_samples, patch_center_pixel(patch_pix), dtype=np.float32),
        "signal_center_dx_deg": np.zeros(num_samples, dtype=np.float32),
        "signal_center_dy_deg": np.zeros(num_samples, dtype=np.float32),
        "geometry_mode_code": np.zeros(num_samples, dtype=np.uint8),
        "fully_contained": np.zeros(num_samples, dtype=np.uint8),
        "target_touches_edge": np.zeros(num_samples, dtype=np.uint8),
    }
    return patches, labels, masks, metadata, truth


def fill_common_metadata(
    metadata: dict,
    sample_idx: int,
    split_tag: int,
    lon: float,
    lat: float,
    coord_idx: int,
    mask_fraction: float,
    realization_idx: int,
    background_id: int,
) -> None:
    metadata["split_tag"][sample_idx] = split_tag
    metadata["glon_deg"][sample_idx] = lon
    metadata["glat_deg"][sample_idx] = lat
    metadata["coord_pool_idx"][sample_idx] = coord_idx
    metadata["coord_mask_fraction"][sample_idx] = mask_fraction
    metadata["cmb_realization_idx"][sample_idx] = realization_idx
    metadata["background_id"][sample_idx] = np.uint64(background_id)
    metadata["split_group_id"][sample_idx] = np.uint64(background_id)


def fill_truth(
    truth: dict,
    sample_idx: int,
    patch_pix: int,
    reso_arcmin: float,
    theta_crit_deg: float,
    z0: float,
    zcrit: float,
    edge_sigma_deg: float,
    center_x: float,
    center_y: float,
    mask: np.ndarray,
    background_id: int,
) -> None:
    center_pix = patch_center_pixel(patch_pix)
    dx_deg = (center_x - center_pix) * reso_arcmin / 60.0
    dy_deg = (center_y - center_pix) * reso_arcmin / 60.0
    touches = target_touches_patch_edge(mask)
    event_id = stable_group_id(
        background_id,
        f"{theta_crit_deg:.6f}",
        f"{z0:.6e}",
        f"{zcrit:.6e}",
        f"{edge_sigma_deg:.6f}",
        f"{center_x:.6f}",
        f"{center_y:.6f}",
    )
    truth["has_signal"][sample_idx] = 1
    truth["event_id"][sample_idx] = np.uint64(event_id)
    truth["theta_crit_deg"][sample_idx] = theta_crit_deg
    truth["z0"][sample_idx] = z0
    truth["zcrit"][sample_idx] = zcrit
    truth["edge_sigma_deg"][sample_idx] = edge_sigma_deg
    truth["signal_center_x_pix"][sample_idx] = center_x
    truth["signal_center_y_pix"][sample_idx] = center_y
    truth["signal_center_dx_deg"][sample_idx] = dx_deg
    truth["signal_center_dy_deg"][sample_idx] = dy_deg
    truth["fully_contained"][sample_idx] = int(not touches)
    truth["target_touches_edge"][sample_idx] = int(touches)


def write_h5(
    path: Path,
    patches: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    metadata: dict,
    truth: dict,
    summary: dict,
    splits: dict,
    coord_pool: np.ndarray,
    coord_mask_fractions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
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
        for key, value in splits.items():
            splits_group.create_dataset(key, data=value, compression="gzip", shuffle=True)
        coord_group = h5.create_group("coordinate_pool")
        coord_group.create_dataset("glon_deg", data=coord_pool[:, 0].astype(np.float32), compression="gzip", shuffle=True)
        coord_group.create_dataset("glat_deg", data=coord_pool[:, 1].astype(np.float32), compression="gzip", shuffle=True)
        coord_group.create_dataset("mask_fraction", data=coord_mask_fractions.astype(np.float32), compression="gzip", shuffle=True)
        summary_group = h5.create_group("summary")
        for key, value in summary.items():
            summary_group.attrs[key] = value


def save_preview(path: Path, patches: np.ndarray, masks: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> None:
    count = min(6, len(labels))
    indices = rng.choice(len(labels), size=count, replace=False)
    fig, axes = plt.subplots(count, 2, figsize=(7, 3.2 * count))
    axes = np.atleast_2d(axes)
    for row, idx in enumerate(indices):
        vmin, vmax = np.percentile(patches[idx], [1, 99])
        axes[row, 0].imshow(patches[idx], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"patch {idx}, label={int(labels[idx])}")
        axes[row, 1].imshow(masks[idx], origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("mask")
        for col in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def generate_train(args: argparse.Namespace) -> None:
    validate_common(args)
    if args.num_samples <= 0 or args.num_samples % 2 != 0:
        raise ValueError("--num-samples must be positive and even.")
    rng = np.random.default_rng(args.seed)
    mask, sky_fraction = load_mask(args.nside)
    coord_pool, coord_mask_fractions = build_coordinate_pool(args, mask, rng)
    coord_train_idx, coord_val_idx = split_index_pool(len(coord_pool), args.train_fraction, args.split_seed)
    cmb_maps, camb_params, lmax = generate_camb_realizations(args, rng)
    realization_train_idx, realization_val_idx = split_index_pool(len(cmb_maps), args.train_fraction, args.split_seed + 1)

    num_samples = int(args.num_samples)
    num_positive = num_samples // 2
    train_positive, val_positive = split_class_counts(num_positive, args.train_fraction)
    train_negative, val_negative = split_class_counts(num_samples - num_positive, args.train_fraction)
    train_samples = train_positive + train_negative
    val_samples = val_positive + val_negative
    patches, labels, masks, metadata, truth = empty_arrays(num_samples, args.patch_pix)
    train_idx = np.arange(train_samples, dtype=np.int64)
    val_idx = np.arange(train_samples, num_samples, dtype=np.int64)

    split_specs = [
        ("train", train_idx, SPLIT_TRAIN, train_positive, coord_train_idx, realization_train_idx, 1000),
        ("val", val_idx, SPLIT_VAL, val_positive, coord_val_idx, realization_val_idx, 2000),
    ]
    for split_name, sample_indices, split_tag, split_positive, coord_indices, realization_indices, seed_offset in split_specs:
        split_rng = np.random.default_rng(args.seed + seed_offset)
        positive_flags = np.zeros(len(sample_indices), dtype=bool)
        positive_flags[:split_positive] = True
        split_rng.shuffle(positive_flags)
        sign_pairs = build_balanced_sign_pairs(split_positive, split_rng)
        positive_counter = 0
        for local_offset, sample_idx in enumerate(sample_indices):
            coord_idx = int(split_rng.choice(coord_indices))
            realization_idx = int(split_rng.choice(realization_indices))
            lon, lat = coord_pool[coord_idx]
            clean = np.asarray(project_patch(cmb_maps[realization_idx], lon, lat, args.patch_pix, args.reso_arcmin), dtype=np.float32)
            background_id = stable_group_id("nside512-camb", realization_idx, coord_idx)
            fill_common_metadata(
                metadata,
                int(sample_idx),
                split_tag,
                float(lon),
                float(lat),
                coord_idx,
                float(coord_mask_fractions[coord_idx]),
                realization_idx,
                background_id,
            )
            if positive_flags[local_offset]:
                theta = sample_theta_crit_from_training_prior(split_rng, 5.0, 25.0)
                sign_z0, sign_zcrit = sign_pairs[positive_counter]
                z0 = float(sign_z0) * sample_log_uniform(split_rng, 1e-6, 1e-4)
                zcrit = float(sign_zcrit) * sample_log_uniform(split_rng, 1e-6, 1e-4)
                edge_sigma = float(split_rng.uniform(args.edge_sigma_min_deg, args.edge_sigma_max_deg))
                center_x, center_y = sample_signal_center_pixels(
                    split_rng,
                    args.patch_pix,
                    args.reso_arcmin,
                    theta,
                    args.contained_margin_deg,
                )
                injected, mask_i = inject_signal(
                    clean,
                    args.patch_pix,
                    args.reso_arcmin,
                    z0,
                    zcrit,
                    theta,
                    edge_sigma,
                    center_x,
                    center_y,
                    args.injection_convention,
                )
                if target_touches_patch_edge(mask_i):
                    raise RuntimeError("Contained 512 training target touches patch edge.")
                patches[sample_idx] = observe_patch(args, injected, split_rng)
                labels[sample_idx] = 1
                masks[sample_idx] = mask_i
                fill_truth(
                    truth,
                    int(sample_idx),
                    args.patch_pix,
                    args.reso_arcmin,
                    theta,
                    z0,
                    zcrit,
                    edge_sigma,
                    center_x,
                    center_y,
                    mask_i,
                    background_id,
                )
                positive_counter += 1
            else:
                patches[sample_idx] = observe_patch(args, clean, split_rng)
            if (local_offset + 1) % 50 == 0 or (local_offset + 1) == len(sample_indices):
                print(f"  {split_name}: generated {local_offset + 1:4d} / {len(sample_indices)}")

    summary = {
        "probe_type": "nside512_training",
        "num_samples": int(num_samples),
        "num_positive": int(labels.sum()),
        "num_negative": int(num_samples - labels.sum()),
        "num_train_samples": int(len(train_idx)),
        "num_val_samples": int(len(val_idx)),
        "nside": int(args.nside),
        "lmax": int(lmax),
        "patch_pixels": int(args.patch_pix),
        "reso_arcmin": float(args.reso_arcmin),
        "sky_fraction": float(sky_fraction),
        "num_cmb_realizations": int(args.num_cmb_realizations),
        "beam_fwhm_arcmin": float(args.beam_fwhm_arcmin),
        "noise_sigma_uk_arcmin": float(args.noise_sigma_uk_arcmin),
        "noise_corr_fwhm_arcmin": float(args.noise_corr_fwhm_arcmin),
        "edge_sigma_min_deg": float(args.edge_sigma_min_deg),
        "edge_sigma_max_deg": float(args.edge_sigma_max_deg),
        "theta_training_distribution": "sin(theta_crit)",
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "injection_convention": args.injection_convention,
        "injection_convention_note": INJECTION_CONVENTION_NOTES[args.injection_convention],
        "matched_filter_approximation_convention": INJECTION_CONVENTION_MCEWEN2012,
        "matched_filter_approximation_note": INJECTION_CONVENTION_NOTES[INJECTION_CONVENTION_MCEWEN2012],
        "split_method": "coordinate_and_realization_disjoint",
        "camb_params": json.dumps(camb_params, sort_keys=True),
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    splits = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "coord_train_idx": coord_train_idx.astype(np.int32),
        "coord_val_idx": coord_val_idx.astype(np.int32),
        "realization_train_idx": realization_train_idx.astype(np.int32),
        "realization_val_idx": realization_val_idx.astype(np.int32),
    }
    output_dir = Path(args.output_dir).resolve()
    h5_path = output_dir / "training_data.h5"
    write_h5(h5_path, patches, labels, masks, metadata, truth, summary, splits, coord_pool, coord_mask_fractions)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_preview(output_dir / "preview.png", patches, masks, labels, rng)
    print(json.dumps({"training_h5": str(h5_path), "summary": str(output_dir / "summary.json")}, indent=2))


def generate_eval(args: argparse.Namespace) -> None:
    validate_common(args)
    if args.num_positive < 0 or args.num_negative < 0:
        raise ValueError("--num-positive and --num-negative must be non-negative.")
    if args.num_positive + args.num_negative <= 0:
        raise ValueError("At least one eval sample is required.")
    rng = np.random.default_rng(args.seed)
    mask, sky_fraction = load_mask(args.nside)
    coord_pool, coord_mask_fractions = build_coordinate_pool(args, mask, rng)
    smica = load_smica(args.nside)
    num_positive = int(args.num_positive)
    num_negative = int(args.num_negative)
    n = num_positive + num_negative
    patches, labels, masks, metadata, truth = empty_arrays(n, args.patch_pix)
    labels[:num_positive] = 1
    sign_pairs = build_balanced_sign_pairs(num_positive, rng) if num_positive else []
    for idx in range(n):
        coord_idx = int(rng.integers(0, len(coord_pool)))
        lon, lat = coord_pool[coord_idx]
        clean = np.asarray(project_patch(smica, lon, lat, args.patch_pix, args.reso_arcmin), dtype=np.float32)
        background_id = stable_group_id("nside512-smica", coord_idx)
        if idx < num_positive:
            z0_sign, zcrit_sign = sign_pairs[idx]
            z0 = float(z0_sign) * float(args.amplitude)
            zcrit = float(zcrit_sign) * float(args.amplitude)
            edge_sigma = float(rng.uniform(args.edge_sigma_min_deg, args.edge_sigma_max_deg))
            center_x, center_y = sample_signal_center_pixels(
                rng,
                args.patch_pix,
                args.reso_arcmin,
                float(args.theta_crit_deg),
                float(args.contained_margin_deg),
            )
            injected, mask_i = inject_signal(
                clean,
                args.patch_pix,
                args.reso_arcmin,
                z0,
                zcrit,
                float(args.theta_crit_deg),
                edge_sigma,
                center_x,
                center_y,
                args.injection_convention,
            )
            if target_touches_patch_edge(mask_i):
                raise RuntimeError("Contained 512 eval target touches patch edge.")
            # Match the existing real-SMICA injection convention: do not re-smooth
            # the real map background; smooth only the injected signal delta.
            theta_grid = make_angular_distance_grid(args.patch_pix, args.reso_arcmin, center_x_pix=center_x, center_y_pix=center_y)
            signal = bubble_collision_signal(theta_grid, z0, zcrit, np.radians(float(args.theta_crit_deg)), edge_sigma_deg=edge_sigma)
            signal_delta = np.asarray(
                fractional_signal_delta(
                    clean,
                    signal,
                    injection_convention=args.injection_convention,
                ),
                dtype=np.float32,
            )
            beam_sigma = fwhm_arcmin_to_sigma_pixels(args.beam_fwhm_arcmin, args.reso_arcmin)
            if beam_sigma > 0.0:
                signal_delta = gaussian_filter(signal_delta, sigma=beam_sigma, mode="reflect")
            patches[idx] = (clean + signal_delta).astype(np.float32)
            masks[idx] = mask_i
            fill_truth(
                truth,
                idx,
                args.patch_pix,
                args.reso_arcmin,
                float(args.theta_crit_deg),
                z0,
                zcrit,
                edge_sigma,
                center_x,
                center_y,
                mask_i,
                background_id,
            )
        else:
            patches[idx] = clean.astype(np.float32)
        fill_common_metadata(
            metadata,
            idx,
            SPLIT_VAL,
            float(lon),
            float(lat),
            coord_idx,
            float(coord_mask_fractions[coord_idx]),
            -1,
            background_id,
        )
        if (idx + 1) % 25 == 0 or idx + 1 == n:
            print(f"  Eval samples: {idx + 1:4d} / {n}")
    summary = {
        "probe_type": "nside512_real_smica_contested_eval",
        "num_samples": n,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "nside": int(args.nside),
        "patch_pixels": int(args.patch_pix),
        "reso_arcmin": float(args.reso_arcmin),
        "source_map": str(SMICA_FILE),
        "amplitude": float(args.amplitude),
        "theta_crit_deg": float(args.theta_crit_deg),
        "edge_sigma_min_deg": float(args.edge_sigma_min_deg),
        "edge_sigma_max_deg": float(args.edge_sigma_max_deg),
        "beam_fwhm_arcmin": float(args.beam_fwhm_arcmin),
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "injection_convention": args.injection_convention,
        "injection_convention_note": INJECTION_CONVENTION_NOTES[args.injection_convention],
        "matched_filter_approximation_convention": INJECTION_CONVENTION_MCEWEN2012,
        "matched_filter_approximation_note": INJECTION_CONVENTION_NOTES[INJECTION_CONVENTION_MCEWEN2012],
        "sky_fraction": float(sky_fraction),
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    idx = np.arange(n, dtype=np.int64)
    splits = {
        "train_idx": np.empty(0, dtype=np.int64),
        "val_idx": idx,
        "coord_train_idx": np.empty(0, dtype=np.int32),
        "coord_val_idx": np.arange(len(coord_pool), dtype=np.int32),
    }
    output_h5 = Path(args.output_h5).resolve()
    write_h5(output_h5, patches, labels, masks, metadata, truth, summary, splits, coord_pool, coord_mask_fractions)
    (output_h5.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_preview(output_h5.parent / "preview.png", patches, masks, labels, rng)
    print(json.dumps({"eval_h5": str(output_h5), "summary": str(output_h5.parent / "summary.json")}, indent=2))


def sigmoid_max_scores(run_config: dict, model, patches: np.ndarray, device: "torch.device") -> tuple[np.ndarray, np.ndarray]:
    import torch

    norm = p3.dataset_kwargs_from_run_config(run_config)
    means = np.asarray(norm["channel_means"], dtype=np.float32)
    stds = np.maximum(np.asarray(norm["channel_stds"], dtype=np.float32), 1e-8)
    if len(norm["extra_channel_datasets"]) != 0:
        raise RuntimeError("512 probe scorer currently expects one-channel models.")
    batch = ((patches[:, None, :, :] - means[None, :, None, None]) / stds[None, :, None, None]).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(batch).to(device, non_blocking=True)
        logits, _ = p3.unpack_model_output(model(x))
        probs = torch.sigmoid(logits)
        return (
            probs.flatten(1).max(dim=1).values.detach().cpu().numpy().astype(np.float32),
            probs.detach().cpu().numpy().astype(np.float32),
        )


def score_eval(args: argparse.Namespace) -> None:
    import torch

    thresholds = parse_float_list(args.thresholds)
    run_dir = Path(args.run_dir).resolve()
    run_config = load_json(run_dir / "run_config.json")
    device = p3.resolve_device(args.device)
    model = p3.build_model(p3.model_args_from_run_config(run_config)).to(device)
    checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    eval_h5 = Path(args.eval_h5).resolve()
    scores = []
    preview = None
    labels = None
    with h5py.File(eval_h5, "r") as h5:
        n = int(h5["patches"].shape[0])
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        for start in range(0, n, int(args.batch_size)):
            stop = min(start + int(args.batch_size), n)
            patch_batch = np.asarray(h5["patches"][start:stop], dtype=np.float32)
            batch_scores, batch_probs = sigmoid_max_scores(run_config, model, patch_batch, device)
            scores.append(batch_scores)
            if preview is None:
                preview = {
                    "patches": patch_batch,
                    "probs": batch_probs[:, 0],
                    "masks": np.asarray(h5["masks"][start:stop], dtype=np.uint8),
                }
    scores_arr = np.concatenate(scores).astype(np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    pos = labels == 1
    neg = labels == 0
    num_pos = int(pos.sum())
    num_neg = int(neg.sum())
    rows = []
    for threshold in thresholds:
        selected = scores_arr > float(threshold)
        detected = int(np.sum(selected))
        tp = int(np.sum(selected & pos))
        fp = int(np.sum(selected & neg))
        rows.append(
            {
                "threshold": float(threshold),
                "detected": detected,
                "n": int(scores_arr.size),
                "num_positive": num_pos,
                "num_negative": num_neg,
                "true_positive": tp,
                "false_positive": fp,
                "recall": float(tp / num_pos) if num_pos else None,
                "fpr": float(fp / num_neg) if num_neg else None,
                "score_min": float(scores_arr.min()),
                "score_median": float(np.median(scores_arr)),
                "score_max": float(scores_arr.max()),
            }
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "nside512_probe_scores.npz", scores=scores_arr)
    report = {
        "eval_h5": str(eval_h5),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_label": checkpoint_label,
        "rows": rows,
        "score_quantiles": {str(q): float(np.quantile(scores_arr, q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)},
    }
    (output_dir / "nside512_probe_score_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Nside=512 Probe Score Report",
        "",
        f"- Eval HDF5: `{eval_h5}`",
        f"- Run dir: `{run_dir}`",
        f"- Checkpoint: `{checkpoint_path}`",
        "",
        "| threshold | selected / n | TP / pos | recall | FP / neg | FPR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        recall = "n/a" if row["recall"] is None else f"{row['recall']:.3f}"
        fpr = "n/a" if row["fpr"] is None else f"{row['fpr']:.3f}"
        lines.append(
            f"| {row['threshold']:.2f} | {row['detected']} / {row['n']} | "
            f"{row['true_positive']} / {row['num_positive']} | {recall} | "
            f"{row['false_positive']} / {row['num_negative']} | {fpr} |"
        )
    lines.extend(["", "## Score Quantiles", ""])
    for key, value in report["score_quantiles"].items():
        lines.append(f"- q={key}: `{value:.6f}`")
    (output_dir / "nside512_probe_score_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if preview is not None:
        count = min(4, len(preview["patches"]))
        fig, axes = plt.subplots(count, 3, figsize=(10.5, 3.2 * count))
        axes = np.atleast_2d(axes)
        for row in range(count):
            patch = preview["patches"][row]
            vmin, vmax = np.percentile(patch, [1, 99])
            axes[row, 0].imshow(patch, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
            axes[row, 0].set_title("patch")
            axes[row, 1].imshow(preview["masks"][row], origin="lower", cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("truth")
            axes[row, 2].imshow(preview["probs"][row], origin="lower", cmap="magma", vmin=0, vmax=1)
            axes[row, 2].set_title("probability")
            for col in range(3):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        fig.tight_layout()
        fig.savefig(output_dir / "nside512_probe_preview.png", dpi=150)
        plt.close(fig)
    print(json.dumps({"report": str(output_dir / "nside512_probe_score_report.md")}, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "generate-train":
        generate_train(args)
    elif args.command == "generate-eval":
        generate_eval(args)
    elif args.command == "score-eval":
        score_eval(args)
    else:  # pragma: no cover
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
