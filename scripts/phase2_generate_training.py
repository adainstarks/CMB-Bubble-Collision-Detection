"""
Phase 2 V1: Generate inspectable training data from real Planck SMICA patches.

This script reuses the Phase 1 data-loading flow and the Phase 2 multiplicative
signal injection code to build a simple training set:
    - 50% clean negative patches
    - 50% positive patches with a centered bubble-collision injection

Outputs are saved as an HDF5 file plus quick-look preview PNGs so the first run
can be sanity-checked before scaling up.

Usage (from project root, with cmb conda env activated):
    python scripts/phase2_generate_training.py
    python scripts/phase2_generate_training.py --num-samples 20 --pool-size 200
"""

import argparse
import datetime as dt
import json
import math
import os

import healpy as hp
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase1_explore import DATA_DIR, MASK_FILE, MASK_URL, SMICA_FILE, SMICA_URL, download_file
from phase2_signal_model import (
    PATCH_PIX,
    RESO_ARCMIN,
    inject_signal_into_patch,
    make_angular_distance_grid,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "training_v1")
NSIDE_WORKING = 256
MASK_THRESHOLD = 0.95


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a simple SMICA-based training dataset for bubble-collision segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--preview-count", type=int, default=8)
    return parser.parse_args()


def ensure_even_sample_count(num_samples):
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if num_samples % 2 != 0:
        raise ValueError("--num-samples must be even so the dataset is exactly 50/50 positive and negative.")


def ensure_planck_inputs():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_file(SMICA_URL, SMICA_FILE)
    download_file(MASK_URL, MASK_FILE)


def load_smica_and_mask():
    print("\n=== Loading Planck inputs ===")
    ensure_planck_inputs()

    smica = hp.read_map(SMICA_FILE, field=0, verbose=False)
    smica_256 = hp.ud_grade(smica, NSIDE_WORKING)

    mask = hp.read_map(MASK_FILE, field=0, verbose=False)
    mask_256 = hp.ud_grade(mask, NSIDE_WORKING)
    mask_256 = np.where(mask_256 > 0.5, 1.0, 0.0)

    sky_fraction = float(np.mean(mask_256))
    print(f"  SMICA degraded to Nside={NSIDE_WORKING}")
    print(f"  Mask sky fraction: {sky_fraction:.1%}")
    return smica_256, mask_256, sky_fraction


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
    attempts = 0
    max_attempts = max(pool_size * 200, 1000)

    while len(coords) < pool_size and attempts < max_attempts:
        attempts += 1
        glon_deg, glat_deg = sample_random_galactic_coordinate(rng)
        if not is_center_unmasked(mask_256, glon_deg, glat_deg):
            continue

        mask_patch = project_patch(mask_256, glon_deg, glat_deg)
        if projected_unmasked_fraction(mask_patch) < min_unmasked_fraction:
            continue

        coords.append((glon_deg, glat_deg))
        if len(coords) % 250 == 0 or len(coords) == pool_size:
            print(f"  Accepted {len(coords):4d} / {pool_size} centers after {attempts} attempts")

    if len(coords) < pool_size:
        raise RuntimeError(
            f"Could only build {len(coords)} valid centers after {attempts} attempts. "
            "Try reducing --pool-size or loosening the mask threshold."
        )

    print(f"  Final coordinate pool: {len(coords)} centers")
    return np.asarray(coords, dtype=np.float32)


def sample_log_uniform(rng, low, high):
    log_value = rng.uniform(np.log10(low), np.log10(high))
    return float(10.0 ** log_value)


def max_fully_contained_radius_deg():
    center = (PATCH_PIX - 1) / 2.0
    axis_offset_rad = np.radians(center * RESO_ARCMIN / 60.0)
    return float(np.degrees(np.arctan(axis_offset_rad)))


def make_centered_disk_mask(theta_grid, theta_crit_deg):
    theta_crit_rad = np.radians(theta_crit_deg)
    return (theta_grid <= theta_crit_rad).astype(np.uint8)


def make_preview_grid(indices, patches, labels, metadata, output_path):
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
        if labels[idx] == 1:
            theta_crit = metadata["theta_crit_deg"][idx]
            ax.set_title(f"{label}  lon={glon:.1f}, lat={glat:.1f}\nR={theta_crit:.1f} deg", fontsize=10)
        else:
            ax.set_title(f"{label}  lon={glon:.1f}, lat={glat:.1f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(indices):]:
        ax.set_visible(False)

    fig.suptitle("Random training samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_positive_preview(indices, patches, masks, metadata, output_path):
    if len(indices) == 0:
        return

    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 4 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(indices):
        patch_ax = axes[row, 0]
        mask_ax = axes[row, 1]

        patch_ax.imshow(patches[idx], cmap="RdBu_r", origin="lower")
        patch_ax.set_title(
            "Patch\n"
            f"lon={metadata['glon_deg'][idx]:.1f}, lat={metadata['glat_deg'][idx]:.1f}\n"
            f"R={metadata['theta_crit_deg'][idx]:.1f} deg, "
            f"z0={metadata['z0'][idx]:.2e}, zcrit={metadata['zcrit'][idx]:.2e}",
            fontsize=10,
        )
        patch_ax.set_xticks([])
        patch_ax.set_yticks([])

        mask_ax.imshow(masks[idx], cmap="gray", origin="lower", vmin=0, vmax=1)
        mask_ax.set_title("Target mask", fontsize=10)
        mask_ax.set_xticks([])
        mask_ax.set_yticks([])

    fig.suptitle("Positive samples and masks", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outputs(output_dir, patches, labels, masks, metadata, summary, preview_count, rng):
    os.makedirs(output_dir, exist_ok=True)

    h5_path = os.path.join(output_dir, "training_data.h5")
    summary_path = os.path.join(output_dir, "summary.json")
    preview_samples_path = os.path.join(output_dir, "preview_samples.png")
    preview_positives_path = os.path.join(output_dir, "preview_positives.png")

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("patches", data=patches.astype(np.float32), compression="gzip", shuffle=True)
        h5.create_dataset("labels", data=labels.astype(np.uint8), compression="gzip", shuffle=True)
        h5.create_dataset("masks", data=masks.astype(np.uint8), compression="gzip", shuffle=True)

        metadata_group = h5.create_group("metadata")
        for key, value in metadata.items():
            metadata_group.create_dataset(key, data=value, compression="gzip", shuffle=True)

        summary_group = h5.create_group("summary")
        for key, value in summary.items():
            summary_group.attrs[key] = value

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sample_preview_count = min(preview_count, len(labels))
    sample_indices = rng.choice(len(labels), size=sample_preview_count, replace=False)
    make_preview_grid(sample_indices, patches, labels, metadata, preview_samples_path)

    positive_indices = np.flatnonzero(labels == 1)
    positive_preview_count = min(preview_count, len(positive_indices))
    positive_preview_indices = rng.choice(
        positive_indices, size=positive_preview_count, replace=False
    )
    make_positive_preview(
        positive_preview_indices,
        patches,
        masks,
        metadata,
        preview_positives_path,
    )

    print("\n=== Saved outputs ===")
    print(f"  {h5_path}")
    print(f"  {summary_path}")
    print(f"  {preview_samples_path}")
    print(f"  {preview_positives_path}")


def main():
    args = parse_args()
    ensure_even_sample_count(args.num_samples)

    rng = np.random.default_rng(args.seed)
    smica_256, mask_256, sky_fraction = load_smica_and_mask()
    coord_pool = build_coordinate_pool(mask_256, args.pool_size, rng)

    theta_grid = make_angular_distance_grid(PATCH_PIX, RESO_ARCMIN)
    contained_radius_deg = max_fully_contained_radius_deg()
    requested_max_radius_deg = 25.0
    if requested_max_radius_deg > contained_radius_deg:
        print(
            "\n=== Geometry warning ===\n"
            f"  The current patch geometry fully contains centered disks only up to about "
            f"{contained_radius_deg:.2f} deg.\n"
            f"  Requested injections extend to {requested_max_radius_deg:.1f} deg, so the "
            "largest disks will be clipped by the patch boundaries."
        )

    num_samples = args.num_samples
    num_positive = num_samples // 2
    num_negative = num_samples - num_positive

    print("\n=== Generating samples ===")
    patches = np.empty((num_samples, PATCH_PIX, PATCH_PIX), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.uint8)
    masks = np.zeros((num_samples, PATCH_PIX, PATCH_PIX), dtype=np.uint8)

    glon_deg = np.empty(num_samples, dtype=np.float32)
    glat_deg = np.empty(num_samples, dtype=np.float32)
    theta_crit_deg = np.full(num_samples, np.nan, dtype=np.float32)
    z0 = np.full(num_samples, np.nan, dtype=np.float32)
    zcrit = np.full(num_samples, np.nan, dtype=np.float32)

    positive_flags = np.zeros(num_samples, dtype=bool)
    positive_flags[:num_positive] = True
    rng.shuffle(positive_flags)

    for idx in range(num_samples):
        coord_idx = rng.integers(0, len(coord_pool))
        lon_i, lat_i = coord_pool[coord_idx]
        clean_patch = np.asarray(project_patch(smica_256, float(lon_i), float(lat_i)), dtype=np.float32)

        glon_deg[idx] = lon_i
        glat_deg[idx] = lat_i

        if positive_flags[idx]:
            label = 1
            theta_i = rng.uniform(5.0, 25.0)
            z0_i = sample_log_uniform(rng, 1e-6, 1e-4)
            zcrit_i = sample_log_uniform(rng, 1e-6, 1e-4)
            zcrit_i *= -1.0 if rng.random() < 0.5 else 1.0

            patch_i, _ = inject_signal_into_patch(clean_patch, z0_i, zcrit_i, theta_i)
            mask_i = make_centered_disk_mask(theta_grid, theta_i)

            patches[idx] = np.asarray(patch_i, dtype=np.float32)
            labels[idx] = label
            masks[idx] = mask_i
            theta_crit_deg[idx] = theta_i
            z0[idx] = z0_i
            zcrit[idx] = zcrit_i
        else:
            patches[idx] = clean_patch

        if (idx + 1) % 50 == 0 or idx + 1 == num_samples:
            positives_so_far = int(labels[:idx + 1].sum())
            print(
                f"  Generated {idx + 1:4d} / {num_samples} samples "
                f"(positives so far: {positives_so_far})"
            )

    metadata = {
        "glon_deg": glon_deg,
        "glat_deg": glat_deg,
        "theta_crit_deg": theta_crit_deg,
        "z0": z0,
        "zcrit": zcrit,
        "is_positive": labels.astype(np.uint8),
    }
    summary = {
        "num_samples": int(num_samples),
        "num_positive": int(labels.sum()),
        "num_negative": int(num_samples - labels.sum()),
        "pool_size": int(len(coord_pool)),
        "seed": int(args.seed),
        "preview_count": int(args.preview_count),
        "nside": int(NSIDE_WORKING),
        "patch_pixels": int(PATCH_PIX),
        "reso_arcmin": float(RESO_ARCMIN),
        "mask_threshold": float(MASK_THRESHOLD),
        "sky_fraction": float(sky_fraction),
        "output_dir": os.path.abspath(args.output_dir),
        "dataset_path": os.path.abspath(os.path.join(args.output_dir, "training_data.h5")),
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }

    print("\n=== Final class counts ===")
    print(f"  Positive: {summary['num_positive']}")
    print(f"  Negative: {summary['num_negative']}")

    save_outputs(
        args.output_dir,
        patches,
        labels,
        masks,
        metadata,
        summary,
        args.preview_count,
        rng,
    )


if __name__ == "__main__":
    main()
