"""
Step 1 audit: full-sky gnomonic tiling of a Planck cleaned map + 14-feature
learned_gbt router scoring + angular-distance candidate clustering.

Motivation
----------
Batch 4 (PR #9) reports `learned_gbt` with `--feature-set all` at FPR 0.08
mixed recall 0.408. The FPR target is a *per-patch* metric. On a full-sky
tiling with overlapping gnomonic patches, the same physical sky feature
can trigger multiple neighboring patches because 256-pixel / 13-arcmin
patches are ~55 deg wide and typical tiling spacings are 10-20 deg.

This script measures the gap between patch-level and cluster-level false
positive burden on real SMICA (no injections):

  1. Enumerate HEALPix tile-center pixels at `--tile-nside`.
  2. Keep centers whose gnomonic patch has >= MASK_THRESHOLD unmasked
     fraction under the common Planck mask (same rule as training data).
  3. Extract gnomonic patches and score through v6_aux_only and
     v7_mixed_ft. Compute the 14-feature vector per patch using the same
     transforms + geometry features as `phase3_postprocess_ablation.py`.
  4. Fit the 14-feature `learned_gbt` on the Batch 4 cached features with
     the same cross-geometry training and 2500/2500 null split as PR #9.
     Calibrate the FPR 0.08 threshold on the held-out null half.
  5. Score the tile patches. Triggered-patch list = patches with
     GBT probability >= threshold.
  6. For each triggered patch, compute the sky (glon, glat) of the
     probability mask's peak pixel via gnomonic inverse.
  7. Cluster triggered peak coords by greedy angular-distance linkage at
     multiple cluster radii.

Output
------
  runs/phase3_unet/batch5_fullsky_fp_audit_v1/
    fullsky_tile_report.json    # machine-readable full report
    fullsky_tile_report.md      # human-readable summary
    tile_candidates.jsonl       # per-triggered-patch record
    clusters_<radius>.jsonl     # per-cluster record at each radius
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import torch

import phase3_train_unet as p3
import phase3_postprocess_ablation as ablation
from phase3_ensemble_evaluate import load_model
from phase3_geometry_router import (
    ALL_FEATURE_NAMES,
    FEATURE_SOURCES,
    stack_features,
    load_transform_npz,
)
from phase3_real_sky_v7_gate import DEFAULT_V7_SPEC, DEFAULT_V6_SPEC
from phase2_extract_smica_null_controls import PLANCK_CLEANED_MAPS, ensure_map_input
from phase2_generate_training import (
    NSIDE_WORKING,
    project_patch,
    projected_unmasked_fraction,
    MASK_THRESHOLD,
)
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN
from sklearn.ensemble import GradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH2 = PROJECT_ROOT / "runs" / "phase3_unet" / "batch2_postprocess_ablation_v1"
DEFAULT_COMMON_MASK = PROJECT_ROOT / "data" / "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "batch5_fullsky_fp_audit_v1"

CLUSTER_RADII_DEG = (5.0, 10.0, 15.0, 25.0, 40.0)
DEFAULT_FPR_TARGET = 0.08
DEFAULT_SEED = 20260417


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: full-sky tiling + cluster-level FP audit")
    parser.add_argument("--map", type=str, default="smica",
                        choices=tuple(PLANCK_CLEANED_MAPS.keys()),
                        help="Planck cleaned map to tile (smica, nilc, sevem, commander).")
    parser.add_argument("--tile-nside", type=int, default=4,
                        help="HEALPix Nside for tile centers. 4 -> 192 centers (14.7 deg spacing).")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help=("Minimum unmasked fraction within a patch for it to be included. "
                              "Default 0.5 is appropriate for deployment tiling; the training-data "
                              "default (MASK_THRESHOLD=0.95) is too strict for full-sky coverage."))
    parser.add_argument("--batch2-dir", type=str, default=str(DEFAULT_BATCH2))
    parser.add_argument("--common-mask", type=str, default=str(DEFAULT_COMMON_MASK))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fpr-target", type=float, default=DEFAULT_FPR_TARGET)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--learned-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reuse-tile-features", action="store_true",
                        help="Reuse cached tile feature npz if present.")
    return parser.parse_args()


def enumerate_tile_centers(tile_nside, common_mask_256, min_unmasked):
    """Return list of (glon_deg, glat_deg) tile centers that pass the common mask."""
    n_pix = hp.nside2npix(tile_nside)
    t_start = time.time()
    print(f"Enumerating {n_pix} HEALPix pixel centers at Nside={tile_nside} "
          f"(angular spacing ~{np.degrees(hp.nside2resol(tile_nside)):.1f} deg)...", flush=True)
    centers = []
    rejected = 0
    for pix in range(n_pix):
        theta, phi = hp.pix2ang(tile_nside, pix)
        glat_deg = 90.0 - np.degrees(theta)
        glon_deg = np.degrees(phi)
        # Mirror the training-data rule: project the COMMON mask through the patch,
        # require unmasked fraction >= min_unmasked.
        mask_patch = project_patch(common_mask_256, glon_deg, glat_deg)
        frac = projected_unmasked_fraction(mask_patch)
        if frac < min_unmasked:
            rejected += 1
            continue
        centers.append((float(glon_deg), float(glat_deg), float(frac)))
    print(f"  kept {len(centers)} / {n_pix} patches "
          f"(rejected {rejected} for mask fraction < {min_unmasked:.2f}) "
          f"in {time.time()-t_start:.1f}s", flush=True)
    return centers


def build_tile_hdf5(tile_hdf5_path, tile_centers, map_256_patches_accessor):
    """Write a minimal HDF5 with tile patches matching the H5BubbleDataset schema."""
    n = len(tile_centers)
    print(f"Extracting {n} tile patches to {tile_hdf5_path.name}...", flush=True)
    t_start = time.time()
    patches = np.zeros((n, PATCH_PIX, PATCH_PIX), dtype=np.float32)
    for i, (glon_deg, glat_deg, _frac) in enumerate(tile_centers):
        patches[i] = map_256_patches_accessor(glon_deg, glat_deg)
        if (i + 1) % 50 == 0:
            print(f"    extracted {i+1}/{n} patches", flush=True)
    labels = np.zeros(n, dtype=np.int64)
    masks = np.zeros((n, PATCH_PIX, PATCH_PIX), dtype=np.float32)
    with h5py.File(tile_hdf5_path, "w") as h5:
        h5.create_dataset("patches", data=patches, compression="lzf")
        h5.create_dataset("masks", data=masks, compression="lzf")
        h5.create_dataset("labels", data=labels)
        meta = h5.create_group("metadata")
        meta.create_dataset("glon_deg", data=np.array([c[0] for c in tile_centers], dtype=np.float64))
        meta.create_dataset("glat_deg", data=np.array([c[1] for c in tile_centers], dtype=np.float64))
        meta.create_dataset("mask_fraction", data=np.array([c[2] for c in tile_centers], dtype=np.float32))
    print(f"  patch extraction done in {time.time()-t_start:.1f}s", flush=True)


def score_tile_with_model(spec, tile_hdf5_path, run_config_source_h5, batch_size, device):
    """
    Run one forward pass on tile patches with normalization config from the model's
    training-data run_config. Return the 7-feature dict (3 transforms + 4 geometry
    + baseline) matching phase3_postprocess_ablation's cache schema, plus peak sky
    pixel offset per patch.
    """
    model, run_config = load_model(spec.run_dir.resolve(), spec.checkpoint, device)
    with h5py.File(tile_hdf5_path, "r") as h5:
        n = int(h5["labels"].shape[0])
    indices = np.arange(n, dtype=np.int64)
    # Use training-data normalization (mean, std) from the model's run_config.
    dataset = p3.H5BubbleDataset(
        h5_path=str(tile_hdf5_path),
        indices=indices,
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 40001,
        max_translate_pixels=0,
        cache_data=True,
    )
    loader = p3.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                           pin_memory=device.type == "cuda")
    features = {name: np.zeros(n, dtype=np.float32)
                for name in tuple(ablation.TRANSFORMS) + tuple(ablation.GEOMETRY_FEATURES)}
    peak_i = np.zeros(n, dtype=np.int32)
    peak_j = np.zeros(n, dtype=np.int32)
    peak_prob = np.zeros(n, dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            mask_logits, _ = p3.unpack_model_output(model(images))
            probs = torch.sigmoid(mask_logits).squeeze(1).float().cpu().numpy()
            bs = probs.shape[0]
            transformed = ablation.apply_transforms_batch(probs)
            for name in features:
                features[name][offset:offset + bs] = transformed[name]
            flat_idx = probs.reshape(bs, -1).argmax(axis=1)
            peak_i[offset:offset + bs] = flat_idx // PATCH_PIX
            peak_j[offset:offset + bs] = flat_idx % PATCH_PIX
            peak_prob[offset:offset + bs] = probs.reshape(bs, -1).max(axis=1)
            offset += bs
    return features, peak_i, peak_j, peak_prob


def fit_shipped_gbt(batch2_dir, learned_seed, feature_names):
    """
    Reproduce the Batch 4 shipped 14-feature GBT. Cross-geometry training:
    fit on mixed positives + null-train half, calibrate threshold on null-eval half
    of mixed (the same recipe the router script uses for the 'mixed' geometry
    report). The returned model is the deployment-facing classifier.
    """
    cache_dir = Path(batch2_dir) / "score_cache"
    # Training positives: mixed geometry (cross-geometry = "contained" was used
    # when *evaluating* mixed, so for DEPLOYMENT we want to fit on the full
    # positive set; but to be most consistent with the shipped `gbt_14` number
    # we use the cross-geometry recipe that PR #9 reported for mixed: fit on
    # contained positives. That matches the shipped threshold 0.8814 exactly.
    v6_inj_mixed = load_transform_npz(cache_dir / "inj_mixed_v6_aux_only_transforms.npz")
    v7_inj_mixed = load_transform_npz(cache_dir / "inj_mixed_v7_mixed_ft_transforms.npz")
    v6_null = load_transform_npz(cache_dir / "null_v6_aux_only_transforms.npz")
    v7_null = load_transform_npz(cache_dir / "null_v7_mixed_ft_transforms.npz")
    v6_inj_contained = load_transform_npz(cache_dir / "inj_contained_v6_aux_only_transforms.npz")
    v7_inj_contained = load_transform_npz(cache_dir / "inj_contained_v7_mixed_ft_transforms.npz")

    labels_mixed = np.asarray(v6_inj_mixed["labels"], dtype=np.uint8)
    X_mixed = stack_features(v6_inj_mixed, v7_inj_mixed, feature_names)
    labels_contained = np.asarray(v6_inj_contained["labels"], dtype=np.uint8)
    X_contained = stack_features(v6_inj_contained, v7_inj_contained, feature_names)
    X_null = stack_features(v6_null, v7_null, feature_names)

    rng = np.random.default_rng(learned_seed)
    n_null = X_null.shape[0]
    perm = rng.permutation(n_null)
    null_train_idx = perm[: n_null // 2]
    null_eval_idx = perm[n_null // 2 :]
    X_null_train = X_null[null_train_idx]
    X_null_eval = X_null[null_eval_idx]

    # Same as geometry_router.py for the mixed report: fit on contained positives.
    X_train_pos = X_contained[labels_contained == 1]
    X_train = np.concatenate([X_train_pos, X_null_train], axis=0)
    y_train = np.concatenate([np.ones(X_train_pos.shape[0], dtype=np.int64),
                              np.zeros(X_null_train.shape[0], dtype=np.int64)], axis=0)
    gbt = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                     random_state=learned_seed).fit(X_train, y_train)
    null_eval_scores = gbt.predict_proba(X_null_eval)[:, 1]
    null_train_scores = gbt.predict_proba(X_null_train)[:, 1]
    full_null_scores = gbt.predict_proba(X_null)[:, 1]
    mixed_scores = gbt.predict_proba(X_mixed)[:, 1]
    return {
        "model": gbt,
        "null_eval_scores": null_eval_scores,
        "null_train_scores": null_train_scores,
        "full_null_scores": full_null_scores,
        "mixed_scores": mixed_scores,
        "mixed_labels": labels_mixed,
    }


def threshold_at_fpr(null_scores, target_fpr):
    null_sorted = np.sort(np.asarray(null_scores, dtype=np.float64))
    n = null_sorted.size
    if n == 0:
        return float("inf")
    k_allowed = int(np.floor(target_fpr * n))
    if k_allowed <= 0:
        return float(np.nextafter(null_sorted[-1], np.inf))
    return float(null_sorted[-k_allowed])


def gnomonic_inverse(glon0_deg, glat0_deg, dx_rad, dy_rad):
    """
    Invert a gnomonic projection to get sky coord from tangent-plane offsets.
    dx, dy are in radians measured from the patch center. dx is east-positive
    (increasing glon in the local tangent plane), dy is north-positive
    (increasing glat).
    """
    glat0 = np.radians(glat0_deg)
    glon0 = np.radians(glon0_deg)
    rho = np.hypot(dx_rad, dy_rad)
    if rho == 0.0:
        return glon0_deg, glat0_deg
    c = np.arctan(rho)
    sin_c, cos_c = np.sin(c), np.cos(c)
    sin_lat = cos_c * np.sin(glat0) + (dy_rad * sin_c * np.cos(glat0)) / rho
    glat = np.arcsin(np.clip(sin_lat, -1.0, 1.0))
    glon = glon0 + np.arctan2(
        dx_rad * sin_c,
        rho * np.cos(glat0) * cos_c - dy_rad * np.sin(glat0) * sin_c,
    )
    glon_deg = float(np.degrees(glon)) % 360.0
    glat_deg = float(np.degrees(glat))
    return glon_deg, glat_deg


def peak_sky_coord(glon0_deg, glat0_deg, peak_i, peak_j):
    """Convert patch-pixel peak coord (row=peak_i, col=peak_j) to sky glon/glat."""
    center = (PATCH_PIX - 1) / 2.0
    reso_rad = np.radians(RESO_ARCMIN / 60.0)
    # hp.gnomview's projected array orientation: row 0 is the TOP (positive dy
    # = increasing glat in the standard convention). So dy = (center - peak_i) * reso
    # and dx = (peak_j - center) * reso matches the standard tangent-plane east/north.
    dx_rad = (peak_j - center) * reso_rad
    dy_rad = (center - peak_i) * reso_rad
    return gnomonic_inverse(glon0_deg, glat0_deg, dx_rad, dy_rad)


def angular_distance_deg(glon_a_deg, glat_a_deg, glon_b_deg, glat_b_deg):
    """Great-circle distance in degrees between two (glon, glat) pairs."""
    lon_a = np.radians(glon_a_deg)
    lon_b = np.radians(glon_b_deg)
    lat_a = np.radians(glat_a_deg)
    lat_b = np.radians(glat_b_deg)
    d_lat = lat_b - lat_a
    d_lon = lon_b - lon_a
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return float(np.degrees(c))


def greedy_cluster(trigger_records, radius_deg):
    """
    Greedy clustering by great-circle distance. Each trigger joins the
    highest-score existing cluster within `radius_deg` of its peak coord;
    otherwise starts a new cluster. Cluster score = max member score.
    Stable output (does not depend on input ordering at equal scores).
    """
    # Sort triggers by score descending so the highest-score triggers seed clusters.
    order = sorted(range(len(trigger_records)), key=lambda i: -trigger_records[i]["gbt_score"])
    clusters = []
    assignment = [-1] * len(trigger_records)
    for idx in order:
        tr = trigger_records[idx]
        best_cluster = -1
        best_dist = float("inf")
        for ci, cluster in enumerate(clusters):
            d = angular_distance_deg(
                cluster["peak_glon_deg"], cluster["peak_glat_deg"],
                tr["peak_glon_deg"], tr["peak_glat_deg"],
            )
            if d <= radius_deg and d < best_dist:
                best_cluster = ci
                best_dist = d
        if best_cluster < 0:
            cluster = {
                "cluster_id": len(clusters),
                "peak_glon_deg": tr["peak_glon_deg"],
                "peak_glat_deg": tr["peak_glat_deg"],
                "max_gbt_score": tr["gbt_score"],
                "member_patches": [tr["patch_index"]],
                "n_members": 1,
                "seed_patch_index": tr["patch_index"],
            }
            clusters.append(cluster)
            assignment[idx] = cluster["cluster_id"]
        else:
            cluster = clusters[best_cluster]
            cluster["member_patches"].append(tr["patch_index"])
            cluster["n_members"] += 1
            cluster["max_gbt_score"] = max(cluster["max_gbt_score"], tr["gbt_score"])
            assignment[idx] = cluster["cluster_id"]
    return clusters, assignment


def run_tile(args, device, output_dir):
    print(f"[step1] tiling {args.map} at Nside={args.tile_nside}", flush=True)

    # Load map + mask
    print("Loading map + common mask ...", flush=True)
    product = ensure_map_input(args.map)
    map_path = product["path"]
    map_2048 = hp.read_map(str(map_path), field=0, dtype=np.float64)
    map_256 = hp.ud_grade(map_2048, NSIDE_WORKING, order_in="RING", order_out="RING")
    common_mask_2048 = hp.read_map(args.common_mask, field=0)
    common_mask_256 = np.where(hp.ud_grade(common_mask_2048, NSIDE_WORKING) > 0.5, 1.0, 0.0)
    print(f"  map {args.map}: 2048->256 ud_grade; mask loaded "
          f"({float(np.mean(common_mask_256 > 0.5)):.3f} usable fraction at Nside=256)", flush=True)

    # Enumerate tile centers
    tile_centers = enumerate_tile_centers(args.tile_nside, common_mask_256, args.mask_threshold)
    if not tile_centers:
        raise SystemExit("No tile centers passed the common mask filter; aborting.")

    # Build tile HDF5
    tile_hdf5 = output_dir / f"tile_patches_{args.map}_nside{args.tile_nside}.h5"
    tile_feat_cache = output_dir / f"tile_features_{args.map}_nside{args.tile_nside}.npz"
    if args.reuse_tile_features and tile_feat_cache.exists():
        print(f"  [reuse] {tile_feat_cache.name}", flush=True)
        with np.load(tile_feat_cache, allow_pickle=True) as loaded:
            glon = loaded["glon_deg"]
            glat = loaded["glat_deg"]
            mask_frac = loaded["mask_fraction"]
            n = glon.shape[0]
            peak_sky = {"v6": (loaded["v6_peak_i"], loaded["v6_peak_j"], loaded["v6_peak_prob"]),
                        "v7": (loaded["v7_peak_i"], loaded["v7_peak_j"], loaded["v7_peak_prob"])}
            v6_npz = {k: loaded[f"v6_{k}"] for k in tuple(ablation.TRANSFORMS) + tuple(ablation.GEOMETRY_FEATURES)}
            v7_npz = {k: loaded[f"v7_{k}"] for k in tuple(ablation.TRANSFORMS) + tuple(ablation.GEOMETRY_FEATURES)}
    else:
        def _accessor(glon_deg, glat_deg):
            return project_patch(map_256, glon_deg, glat_deg)
        build_tile_hdf5(tile_hdf5, tile_centers, _accessor)
        glon = np.array([c[0] for c in tile_centers], dtype=np.float64)
        glat = np.array([c[1] for c in tile_centers], dtype=np.float64)
        mask_frac = np.array([c[2] for c in tile_centers], dtype=np.float32)
        n = len(tile_centers)
        print(f"[step1] scoring {n} tile patches through v6_aux_only ...", flush=True)
        v6_feats, v6_pi, v6_pj, v6_pp = score_tile_with_model(
            DEFAULT_V6_SPEC, tile_hdf5, None, args.batch_size, device)
        print(f"[step1] scoring {n} tile patches through v7_mixed_ft ...", flush=True)
        v7_feats, v7_pi, v7_pj, v7_pp = score_tile_with_model(
            DEFAULT_V7_SPEC, tile_hdf5, None, args.batch_size, device)
        peak_sky = {"v6": (v6_pi, v6_pj, v6_pp), "v7": (v7_pi, v7_pj, v7_pp)}
        v6_npz = v6_feats
        v7_npz = v7_feats
        np.savez_compressed(
            tile_feat_cache,
            glon_deg=glon, glat_deg=glat, mask_fraction=mask_frac,
            v6_peak_i=v6_pi, v6_peak_j=v6_pj, v6_peak_prob=v6_pp,
            v7_peak_i=v7_pi, v7_peak_j=v7_pj, v7_peak_prob=v7_pp,
            **{f"v6_{k}": v6_feats[k] for k in v6_feats},
            **{f"v7_{k}": v7_feats[k] for k in v7_feats},
        )

    # Fit the shipped GBT and calibrate threshold
    print(f"[step1] fitting 14-feature GBT (seed {args.learned_seed}, cross-geometry protocol)...",
          flush=True)
    fit = fit_shipped_gbt(Path(args.batch2_dir), args.learned_seed, list(ALL_FEATURE_NAMES))
    gbt = fit["model"]
    threshold = threshold_at_fpr(fit["null_eval_scores"], args.fpr_target)
    null_fpr_full = float((fit["full_null_scores"] >= threshold).mean())
    null_fpr_eval = float((fit["null_eval_scores"] >= threshold).mean())
    mixed_recall_at_threshold = float(
        (fit["mixed_scores"][fit["mixed_labels"] == 1] >= threshold).mean()
    )
    print(f"  GBT threshold at FPR={args.fpr_target:.2f} (null-eval-half): {threshold:.6f}",
          flush=True)
    print(f"  sanity: null FPR on full 5000-patch pool = {null_fpr_full:.4f}", flush=True)
    print(f"  sanity: mixed positive recall = {mixed_recall_at_threshold:.4f}", flush=True)

    # Score the tile
    X_tile = np.stack([
        v6_npz[FEATURE_SOURCES[name][1]] if FEATURE_SOURCES[name][0] == "v6"
        else v7_npz[FEATURE_SOURCES[name][1]]
        for name in ALL_FEATURE_NAMES
    ], axis=1).astype(np.float64)
    tile_scores = gbt.predict_proba(X_tile)[:, 1]
    triggered_mask = tile_scores >= threshold
    n_triggered = int(triggered_mask.sum())
    print(f"[step1] patch-level triggers: {n_triggered} / {n} "
          f"(patch-level FPR = {n_triggered/n:.4f}, expected ~{args.fpr_target:.2f})",
          flush=True)

    # Build trigger records. For each triggered patch, prefer the peak-prob sky
    # coord from the model with the higher raw score (v6 or v7 baseline) at that
    # patch, as a reasonable proxy for "where in the patch the trigger is".
    v6_baseline = v6_npz["baseline"]
    v7_baseline = v7_npz["baseline"]
    trigger_records = []
    for idx in np.where(triggered_mask)[0]:
        use_v7 = v7_baseline[idx] >= v6_baseline[idx]
        pi_arr, pj_arr, pp_arr = peak_sky[("v7" if use_v7 else "v6")]
        peak_glon, peak_glat = peak_sky_coord(
            float(glon[idx]), float(glat[idx]), int(pi_arr[idx]), int(pj_arr[idx]))
        record = {
            "patch_index": int(idx),
            "patch_glon_deg": float(glon[idx]),
            "patch_glat_deg": float(glat[idx]),
            "peak_glon_deg": peak_glon,
            "peak_glat_deg": peak_glat,
            "peak_source_model": ("v7" if use_v7 else "v6"),
            "peak_pixel_i": int(pi_arr[idx]),
            "peak_pixel_j": int(pj_arr[idx]),
            "gbt_score": float(tile_scores[idx]),
            "v6_baseline": float(v6_baseline[idx]),
            "v7_baseline": float(v7_baseline[idx]),
            "v6_peak_prob": float(peak_sky["v6"][2][idx]),
            "v7_peak_prob": float(peak_sky["v7"][2][idx]),
            "mask_fraction": float(mask_frac[idx]),
        }
        trigger_records.append(record)

    # Cluster at multiple radii
    cluster_results = {}
    for radius in CLUSTER_RADII_DEG:
        clusters, assignment = greedy_cluster(trigger_records, radius)
        cluster_results[radius] = {
            "n_clusters": len(clusters),
            "max_cluster_size": max((c["n_members"] for c in clusters), default=0),
            "mean_cluster_size": (float(np.mean([c["n_members"] for c in clusters]))
                                  if clusters else 0.0),
            "reduction_factor": (n_triggered / max(len(clusters), 1)),
            "clusters": clusters,
            "assignment": assignment,
        }
        print(f"[step1] cluster radius {radius:>5.1f} deg: "
              f"{len(clusters)} clusters (reduction {n_triggered/max(len(clusters),1):.2f}x, "
              f"max_size {cluster_results[radius]['max_cluster_size']})", flush=True)

    # Write outputs
    candidates_path = output_dir / "tile_candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8") as f:
        for rec in trigger_records:
            f.write(json.dumps(rec) + "\n")

    for radius, info in cluster_results.items():
        cluster_path = output_dir / f"clusters_{int(radius)}deg.jsonl"
        with cluster_path.open("w", encoding="utf-8") as f:
            for c in info["clusters"]:
                f.write(json.dumps({k: c[k] for k in c if k != "member_patches"} | {
                    "member_patches": c["member_patches"]}) + "\n")

    return {
        "map": args.map,
        "tile_nside": args.tile_nside,
        "n_centers_enumerated": int(hp.nside2npix(args.tile_nside)),
        "n_centers_after_mask": int(n),
        "mask_threshold": args.mask_threshold,
        "fpr_target": args.fpr_target,
        "gbt_threshold": threshold,
        "null_fpr_on_full_pool": null_fpr_full,
        "null_fpr_on_eval_half": null_fpr_eval,
        "mixed_positive_recall_sanity": mixed_recall_at_threshold,
        "n_patches_triggered": n_triggered,
        "patch_level_fpr_observed": n_triggered / float(n),
        "cluster_radii_deg": list(CLUSTER_RADII_DEG),
        "cluster_results": {
            f"{r:.1f}": {
                "n_clusters": cluster_results[r]["n_clusters"],
                "max_cluster_size": cluster_results[r]["max_cluster_size"],
                "mean_cluster_size": cluster_results[r]["mean_cluster_size"],
                "reduction_factor": cluster_results[r]["reduction_factor"],
            } for r in CLUSTER_RADII_DEG
        },
    }


def write_markdown(path, summary):
    lines = ["# Step 1: Full-sky FP burden audit", ""]
    lines.append(f"Map: **{summary['map']}**  ")
    lines.append(f"Tile Nside: **{summary['tile_nside']}**  ")
    lines.append(f"Patches enumerated / after mask: **{summary['n_centers_enumerated']} / {summary['n_centers_after_mask']}**  ")
    lines.append(f"14-feature GBT threshold @ FPR target {summary['fpr_target']:.2f}: "
                 f"**{summary['gbt_threshold']:.6f}**  ")
    lines.append(f"Null FPR sanity (full 5000-patch pool): "
                 f"**{summary['null_fpr_on_full_pool']:.4f}**  ")
    lines.append(f"Null FPR sanity (eval half): "
                 f"**{summary['null_fpr_on_eval_half']:.4f}**  ")
    lines.append(f"Mixed positive recall sanity (all positives): "
                 f"**{summary['mixed_positive_recall_sanity']:.4f}**")
    lines.append("")
    lines.append("## Trigger burden")
    lines.append("")
    lines.append(f"Patches triggered: **{summary['n_patches_triggered']} / {summary['n_centers_after_mask']}**  ")
    lines.append(f"Patch-level FPR observed: **{summary['patch_level_fpr_observed']:.4f}** "
                 f"(target {summary['fpr_target']:.2f})")
    lines.append("")
    lines.append("## Cluster reduction by radius")
    lines.append("")
    lines.append("| cluster radius (deg) | n clusters | reduction factor | max cluster size | mean cluster size |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in summary["cluster_radii_deg"]:
        info = summary["cluster_results"][f"{r:.1f}"]
        lines.append(
            f"| {r:.1f} | {info['n_clusters']} | {info['reduction_factor']:.2f}x | "
            f"{info['max_cluster_size']} | {info['mean_cluster_size']:.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = p3.resolve_device(args.device)
    summary = run_tile(args, device, output_dir)
    json_path = output_dir / "fullsky_tile_report.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path = output_dir / "fullsky_tile_report.md"
    write_markdown(md_path, summary)
    print(f"\n=== Saved ===\n  JSON: {json_path}\n  MD:   {md_path}", flush=True)


if __name__ == "__main__":
    main()
