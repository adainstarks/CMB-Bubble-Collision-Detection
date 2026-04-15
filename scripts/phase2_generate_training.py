"""
Phase 2 V3: Generate training data for bubble-collision detection.

The training set is built from CAMB realizations, not from the single real SMICA
sky. This matters because the model should learn the bubble-collision pattern on
top of generic CMB fluctuations, not overfit to one particular realization of
the sky. The real Planck mask is still used to choose clean sky coordinates so
the patch geometry matches the final inference setup on SMICA.

Physics choices:
    - Feeney et al. (2011) Eq. 1 for the radial collision template
    - Feeney et al. (2011) Eq. 15 for multiplicative injection
    - Feeney et al. (2011) Eq. 2 motivation for a sin(theta_crit) size prior
    - always-on edge smoothing to reflect the paper's discussion that the
      causal boundary may be smeared on sub-degree scales

Outputs are saved as an HDF5 file plus validation plots so the dataset can be
checked before moving on to model training.
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

from phase1_explore import DATA_DIR, MASK_FILE, MASK_URL, download_file
from phase2_signal_model import (
    PATCH_PIX,
    RESO_ARCMIN,
    bubble_collision_signal,
    inject_signal_into_patch,
    make_angular_distance_grid,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "training_v3")
NSIDE_WORKING = 256
MASK_THRESHOLD = 0.95


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a CAMB-based training dataset for bubble-collision segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--num-cmb-realizations", type=int, default=192)
    parser.add_argument("--edge-sigma-min-deg", type=float, default=0.3)
    parser.add_argument("--edge-sigma-max-deg", type=float, default=1.0)
    parser.add_argument("--signal-center-edge-margin-pix", type=float, default=16.0)
    return parser.parse_args()


def ensure_even_sample_count(num_samples):
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if num_samples % 2 != 0:
        raise ValueError("--num-samples must be even so the dataset is exactly 50/50 positive and negative.")


def validate_smoothing_args(args):
    if args.num_cmb_realizations <= 0:
        raise ValueError("--num-cmb-realizations must be positive.")
    if args.edge_sigma_min_deg < 0.0 or args.edge_sigma_max_deg < 0.0:
        raise ValueError("--edge sigma bounds must be non-negative.")
    if args.edge_sigma_min_deg > args.edge_sigma_max_deg:
        raise ValueError("--edge-sigma-min-deg must be <= --edge-sigma-max-deg.")
    if args.signal_center_edge_margin_pix < 0.0:
        raise ValueError("--signal-center-edge-margin-pix must be non-negative.")
    if args.signal_center_edge_margin_pix >= PATCH_PIX / 2.0:
        raise ValueError("--signal-center-edge-margin-pix must be smaller than half the patch width.")


def ensure_planck_inputs():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_file(MASK_URL, MASK_FILE)


def load_mask():
    print("\n=== Loading galactic mask ===")
    ensure_planck_inputs()

    mask = hp.read_map(MASK_FILE, field=0, verbose=False)
    mask_256 = hp.ud_grade(mask, NSIDE_WORKING)
    mask_256 = np.where(mask_256 > 0.5, 1.0, 0.0)

    sky_fraction = float(np.mean(mask_256))
    print(f"  Mask sky fraction: {sky_fraction:.1%}")
    return mask_256, sky_fraction


def planck2018_bestfit_params():
    """
    Planck 2018 base-LambdaCDM best-fit values used for CAMB realizations.

    Values correspond to the standard Planck 2018 TT,TE,EE+lowE+lensing fit:
        ombh2 = 0.02237
        omch2 = 0.1200
        H0 = 67.36
        tau = 0.0544
        ns = 0.9649
        ln(1e10 As) = 3.044
    """
    return {
        "H0": 67.36,
        "ombh2": 0.02237,
        "omch2": 0.1200,
        "tau": 0.0544,
        "ns": 0.9649,
        "As": math.exp(3.044) / 1e10,
    }


def generate_camb_realizations(num_realizations, rng):
    print("\n=== Generating CAMB realizations ===")
    try:
        import camb
    except ImportError as exc:
        raise RuntimeError(
            "CAMB is required for Phase 2 V3. Install it in the active environment "
            "before running this script."
        ) from exc

    params = planck2018_bestfit_params()
    lmax = 3 * NSIDE_WORKING - 1

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=params["H0"],
        ombh2=params["ombh2"],
        omch2=params["omch2"],
        tau=params["tau"],
    )
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    results = camb.get_results(pars)
    # synfast expects raw C_ell, not D_ell = l(l+1)C_ell/2pi.
    cls_tt = results.get_cmb_power_spectra(
        pars,
        CMB_unit="K",
        raw_cl=True,
    )["lensed_scalar"][:, 0]
    cls_tt = cls_tt[: lmax + 1]

    realizations = np.empty((num_realizations, hp.nside2npix(NSIDE_WORKING)), dtype=np.float32)
    seeds = rng.integers(0, 2**32 - 1, size=num_realizations, dtype=np.uint32)

    for idx, seed_i in enumerate(seeds):
        np.random.seed(int(seed_i))
        realizations[idx] = hp.synfast(
            cls_tt,
            NSIDE_WORKING,
            lmax=lmax,
            new=True,
            pixwin=False,
            verbose=False,
        ).astype(np.float32)
        if (idx + 1) % 16 == 0 or idx + 1 == num_realizations:
            print(f"  Generated {idx + 1:4d} / {num_realizations} CAMB skies")

    return realizations, params


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


def sample_theta_crit_from_sin_prior(rng, low_deg=5.0, high_deg=25.0):
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


def max_fully_contained_radius_deg():
    center = (PATCH_PIX - 1) / 2.0
    axis_offset_rad = np.radians(center * RESO_ARCMIN / 60.0)
    return float(np.degrees(np.arctan(axis_offset_rad)))


def make_disk_mask(theta_grid, theta_crit_deg):
    theta_crit_rad = np.radians(theta_crit_deg)
    return (theta_grid <= theta_crit_rad).astype(np.uint8)


def sample_signal_center_pixels(rng, npix, edge_margin_pix):
    low = float(edge_margin_pix)
    high = float(npix - 1 - edge_margin_pix)
    center_x_pix = rng.uniform(low, high)
    center_y_pix = rng.uniform(low, high)
    return center_x_pix, center_y_pix


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
            dx_deg = metadata["signal_center_dx_deg"][idx]
            dy_deg = metadata["signal_center_dy_deg"][idx]
            ax.set_title(f"{label}  lon={glon:.1f}, lat={glat:.1f}\nR={theta_crit:.1f} deg", fontsize=10)
            ax.set_xlabel(f"dx={dx_deg:.1f} deg, dy={dy_deg:.1f} deg", fontsize=9)
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

    fig, axes = plt.subplots(len(indices), 3, figsize=(14, 4 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(indices):
        patch_ax = axes[row, 0]
        signal_ax = axes[row, 1]
        mask_ax = axes[row, 2]

        patch_ax.imshow(patches[idx], cmap="RdBu_r", origin="lower")
        patch_ax.set_title(
            "Patch\n"
            f"lon={metadata['glon_deg'][idx]:.1f}, lat={metadata['glat_deg'][idx]:.1f}\n"
            f"R={metadata['theta_crit_deg'][idx]:.1f} deg, "
            f"z0={metadata['z0'][idx]:.2e}, zcrit={metadata['zcrit'][idx]:.2e}, "
            f"sigma={metadata['edge_sigma_deg'][idx]:.2f}\n"
            f"dx={metadata['signal_center_dx_deg'][idx]:.1f} deg, "
            f"dy={metadata['signal_center_dy_deg'][idx]:.1f} deg",
            fontsize=10,
        )
        patch_ax.set_xticks([])
        patch_ax.set_yticks([])

        theta_grid = make_angular_distance_grid(
            PATCH_PIX,
            RESO_ARCMIN,
            center_x_pix=float(metadata["signal_center_x_pix"][idx]),
            center_y_pix=float(metadata["signal_center_y_pix"][idx]),
        )
        signal = bubble_collision_signal(
            theta_grid,
            float(metadata["z0"][idx]),
            float(metadata["zcrit"][idx]),
            np.radians(float(metadata["theta_crit_deg"][idx])),
            edge_sigma_deg=float(metadata["edge_sigma_deg"][idx]),
        )
        signal_ax.imshow(signal, cmap="RdBu_r", origin="lower")
        signal_ax.set_title("Injected signal template", fontsize=10)
        signal_ax.set_xticks([])
        signal_ax.set_yticks([])

        mask_ax.imshow(masks[idx], cmap="gray", origin="lower", vmin=0, vmax=1)
        mask_ax.set_title("Target mask", fontsize=10)
        mask_ax.set_xticks([])
        mask_ax.set_yticks([])

    fig.suptitle("Positive patches, injected templates, and masks", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_validation_histograms(labels, metadata, output_path):
    pos = labels == 1
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    axes[0].hist(metadata["glon_deg"], bins=30, color="#2563eb", alpha=0.85)
    axes[0].set_title("Galactic longitude")

    axes[1].hist(metadata["glat_deg"], bins=30, color="#0891b2", alpha=0.85)
    axes[1].set_title("Galactic latitude")

    axes[2].hist(metadata["theta_crit_deg"][pos], bins=30, color="#7c3aed", alpha=0.85)
    axes[2].set_title(r"$\theta_{\rm crit}$")

    axes[3].hist(metadata["z0"][pos], bins=30, color="#dc2626", alpha=0.85)
    axes[3].set_title(r"$z_0$")

    axes[4].hist(metadata["zcrit"][pos], bins=30, color="#ea580c", alpha=0.85)
    axes[4].set_title(r"$z_{\rm crit}$")

    axes[5].hist(metadata["edge_sigma_deg"][pos], bins=30, color="#16a34a", alpha=0.85)
    axes[5].set_title("Edge sigma (deg)")

    axes[6].hist(metadata["signal_center_dx_deg"][pos], bins=30, color="#0f766e", alpha=0.85)
    axes[6].set_title("Signal center dx (deg)")

    axes[7].hist(metadata["signal_center_dy_deg"][pos], bins=30, color="#7c2d12", alpha=0.85)
    axes[7].set_title("Signal center dy (deg)")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.suptitle("Phase 2 parameter validation", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outputs(output_dir, patches, labels, masks, metadata, summary, preview_count, rng):
    os.makedirs(output_dir, exist_ok=True)

    h5_path = os.path.join(output_dir, "training_data.h5")
    summary_path = os.path.join(output_dir, "summary.json")
    preview_samples_path = os.path.join(output_dir, "preview_samples.png")
    preview_positives_path = os.path.join(output_dir, "preview_positives.png")
    validation_hist_path = os.path.join(output_dir, "validation_histograms.png")

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
    make_validation_histograms(labels, metadata, validation_hist_path)

    print("\n=== Saved outputs ===")
    print(f"  {h5_path}")
    print(f"  {summary_path}")
    print(f"  {preview_samples_path}")
    print(f"  {preview_positives_path}")
    print(f"  {validation_hist_path}")


def main():
    args = parse_args()
    ensure_even_sample_count(args.num_samples)
    validate_smoothing_args(args)

    rng = np.random.default_rng(args.seed)
    mask_256, sky_fraction = load_mask()
    coord_pool = build_coordinate_pool(mask_256, args.pool_size, rng)
    cmb_realizations, camb_params = generate_camb_realizations(args.num_cmb_realizations, rng)

    contained_radius_deg = max_fully_contained_radius_deg()
    requested_max_radius_deg = 25.0
    if requested_max_radius_deg > contained_radius_deg:
        print(
            "\n=== Geometry warning ===\n"
            f"  The current patch geometry fully contains centered disks only up to about "
            f"{contained_radius_deg:.2f} deg.\n"
            f"  Requested injections extend to {requested_max_radius_deg:.1f} deg.\n"
            "  Positive injections are now randomized within the patch, so partial disks "
            "near the patch boundaries are expected by design."
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
    cmb_realization_idx = np.full(num_samples, -1, dtype=np.int32)
    theta_crit_deg = np.full(num_samples, np.nan, dtype=np.float32)
    z0 = np.full(num_samples, np.nan, dtype=np.float32)
    zcrit = np.full(num_samples, np.nan, dtype=np.float32)
    edge_sigma_deg = np.full(num_samples, np.nan, dtype=np.float32)
    signal_center_x_pix = np.full(num_samples, np.nan, dtype=np.float32)
    signal_center_y_pix = np.full(num_samples, np.nan, dtype=np.float32)
    signal_center_dx_deg = np.full(num_samples, np.nan, dtype=np.float32)
    signal_center_dy_deg = np.full(num_samples, np.nan, dtype=np.float32)

    positive_flags = np.zeros(num_samples, dtype=bool)
    positive_flags[:num_positive] = True
    rng.shuffle(positive_flags)
    sign_pairs = build_balanced_sign_pairs(num_positive, rng)
    positive_counter = 0

    for idx in range(num_samples):
        coord_idx = rng.integers(0, len(coord_pool))
        lon_i, lat_i = coord_pool[coord_idx]
        realization_idx_i = int(rng.integers(0, len(cmb_realizations)))
        clean_patch = np.asarray(
            project_patch(cmb_realizations[realization_idx_i], float(lon_i), float(lat_i)),
            dtype=np.float32,
        )

        glon_deg[idx] = lon_i
        glat_deg[idx] = lat_i
        cmb_realization_idx[idx] = realization_idx_i

        if positive_flags[idx]:
            label = 1
            theta_i = sample_theta_crit_from_sin_prior(rng, 5.0, 25.0)
            sign_z0_i, sign_zcrit_i = sign_pairs[positive_counter]
            z0_i = sign_z0_i * sample_log_uniform(rng, 1e-6, 1e-4)
            zcrit_i = sign_zcrit_i * sample_log_uniform(rng, 1e-6, 1e-4)
            edge_sigma_i = rng.uniform(args.edge_sigma_min_deg, args.edge_sigma_max_deg)
            center_x_i, center_y_i = sample_signal_center_pixels(
                rng,
                PATCH_PIX,
                edge_margin_pix=args.signal_center_edge_margin_pix,
            )
            center_pix = (PATCH_PIX - 1) / 2.0
            center_dx_i = (center_x_i - center_pix) * RESO_ARCMIN / 60.0
            center_dy_i = (center_y_i - center_pix) * RESO_ARCMIN / 60.0

            patch_i, _ = inject_signal_into_patch(
                clean_patch,
                z0_i,
                zcrit_i,
                theta_i,
                edge_sigma_deg=edge_sigma_i,
                center_x_pix=center_x_i,
                center_y_pix=center_y_i,
            )
            theta_grid_i = make_angular_distance_grid(
                PATCH_PIX,
                RESO_ARCMIN,
                center_x_pix=center_x_i,
                center_y_pix=center_y_i,
            )
            mask_i = make_disk_mask(theta_grid_i, theta_i)

            patches[idx] = np.asarray(patch_i, dtype=np.float32)
            labels[idx] = label
            masks[idx] = mask_i
            theta_crit_deg[idx] = theta_i
            z0[idx] = z0_i
            zcrit[idx] = zcrit_i
            edge_sigma_deg[idx] = edge_sigma_i
            signal_center_x_pix[idx] = center_x_i
            signal_center_y_pix[idx] = center_y_i
            signal_center_dx_deg[idx] = center_dx_i
            signal_center_dy_deg[idx] = center_dy_i
            positive_counter += 1
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
        "cmb_realization_idx": cmb_realization_idx,
        "theta_crit_deg": theta_crit_deg,
        "z0": z0,
        "zcrit": zcrit,
        "edge_sigma_deg": edge_sigma_deg,
        "signal_center_x_pix": signal_center_x_pix,
        "signal_center_y_pix": signal_center_y_pix,
        "signal_center_dx_deg": signal_center_dx_deg,
        "signal_center_dy_deg": signal_center_dy_deg,
        "is_positive": labels.astype(np.uint8),
    }
    summary = {
        "num_samples": int(num_samples),
        "num_positive": int(labels.sum()),
        "num_negative": int(num_samples - labels.sum()),
        "pool_size": int(len(coord_pool)),
        "num_cmb_realizations": int(args.num_cmb_realizations),
        "seed": int(args.seed),
        "preview_count": int(args.preview_count),
        "nside": int(NSIDE_WORKING),
        "patch_pixels": int(PATCH_PIX),
        "reso_arcmin": float(RESO_ARCMIN),
        "mask_threshold": float(MASK_THRESHOLD),
        "edge_sigma_min_deg": float(args.edge_sigma_min_deg),
        "edge_sigma_max_deg": float(args.edge_sigma_max_deg),
        "signal_center_edge_margin_pix": float(args.signal_center_edge_margin_pix),
        "sky_fraction": float(sky_fraction),
        "theta_prior": "sin(theta_crit)",
        "z0_sign_sampling": "balanced",
        "zcrit_sign_sampling": "balanced",
        "positive_injection_geometry": "randomized within patch",
        "camb_params": json.dumps(camb_params, sort_keys=True),
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
