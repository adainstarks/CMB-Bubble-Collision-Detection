"""
Batch 2 post-processing ablation on the real-SMICA gate HDF5s.

Transforms applied to the v7_mixed_ft and v6_aux_only probability masks before
scoring:

    baseline                       score = max(mask)
    +A (smooth_multi)              score = max over sigma in SMOOTH_SIGMAS_PIX
                                            of max(gaussian_smooth(mask, sigma))
    +A+D (mf_on_mask)              score = max over theta in THETA_GRID_DEG
                                            of max(fftcorr(smoothed_mask, disc_kernel(theta)))

Outputs per-model x per-geometry ablation tables with recall at real-SMICA-
calibrated FPR 0.05, 0.08, 0.10 plus per-geometry group breakdown.

The harness runs a single GPU forward pass per model per gate HDF5, emits
three transformed score columns per patch, then calibrates thresholds on the
null score distribution. Each ablation row is fully attributable.

Usage::

    python scripts/phase3_postprocess_ablation.py \\
        --gate-root runs/phase3_unet/real_sky_v7_gate_v1 \\
        --output-dir runs/phase3_unet/batch2_postprocess_ablation_v1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.signal import fftconvolve
from scipy.stats import binomtest

import phase3_train_unet as p3
from phase3_ensemble_evaluate import load_model
from phase3_real_sky_v7_gate import DEFAULT_V7_SPEC, DEFAULT_V6_SPEC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NULL_H5 = PROJECT_ROOT / "data" / "training_v4" / "smica_null_controls_all.h5"

DEFAULT_SMOOTH_SIGMAS_PIX = (4.0, 8.0, 16.0)
THETA_GRID_DEG = (5.0, 10.0, 15.0, 20.0, 25.0)
RESO_ARCMIN = 13.0
PATCH_PIX = 256
FPR_TARGETS = (0.05, 0.08, 0.10)

TRANSFORMS = ("baseline", "smooth_multi", "mf_on_mask")

# Batch 4: truth-free geometry proxies computed per patch from the frozen
# probability mask. Stored alongside transform scores in the same cache npz
# so the learned-router feature set can mix scalars and shape features.
GEOMETRY_FEATURES = (
    "mask_area_at_0.5",
    "centroid_offset_px",
    "compactness",
    "edge_touching_fraction",
)
GEOMETRY_MASK_THRESHOLD = 0.5
GEOMETRY_EDGE_DISTANCE_PIX = 4

# Populated from CLI at main(); apply_transforms_batch reads from this module-level name.
SMOOTH_SIGMAS_PIX = DEFAULT_SMOOTH_SIGMAS_PIX


@dataclass(frozen=True)
class ModelCfg:
    name: str
    run_dir: Path
    checkpoint: str


def parse_args():
    parser = argparse.ArgumentParser(description="Batch 2 post-processing ablation")
    parser.add_argument(
        "--gate-root",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_v7_gate_v1"),
        help="Directory containing contained/ and mixed/ subdirs produced by phase3_real_sky_v7_gate.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "phase3_unet" / "batch2_postprocess_ablation_v1"),
    )
    parser.add_argument("--null-h5", type=str, default=str(NULL_H5))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--geometry", type=str, action="append", default=[],
                        help="contained or mixed. Repeat to evaluate both. Default: both.")
    parser.add_argument("--model", type=str, action="append", default=[],
                        help="v6 or v7. Repeat for multiple. Default: both.")
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument(
        "--smooth-sigmas-pix",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SMOOTH_SIGMAS_PIX),
        help="Comma-separated Gaussian sigmas (pixels) for smooth_multi and base sigma for mf_on_mask (first entry).",
    )
    return parser.parse_args()


def exact_ci(k, n):
    if n <= 0:
        return [float("nan"), float("nan")]
    r = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(r.low), float(r.high)]


def theta_to_radius_pix(theta_deg):
    return float(theta_deg) / (RESO_ARCMIN / 60.0)


def build_disc_kernel(theta_deg, npix=PATCH_PIX):
    """Zero-mean L2-normalized positive-disc kernel for matched filtering on a probability mask."""
    radius_pix = theta_to_radius_pix(theta_deg)
    y, x = np.mgrid[:npix, :npix]
    cy = cx = (npix - 1) / 2.0
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    kernel = (r <= radius_pix).astype(np.float32)
    kernel -= float(kernel.mean())
    norm = float(np.linalg.norm(kernel))
    if norm > 0.0:
        kernel = kernel / norm
    return kernel.astype(np.float32)


DISC_KERNELS = None


def get_disc_kernels():
    global DISC_KERNELS
    if DISC_KERNELS is None:
        DISC_KERNELS = [build_disc_kernel(t) for t in THETA_GRID_DEG]
    return DISC_KERNELS


def _edge_distance_grid(h, w):
    y, x = np.mgrid[:h, :w]
    return np.minimum(np.minimum(y, h - 1 - y), np.minimum(x, w - 1 - x))


_EDGE_DIST_CACHE = None


def _get_edge_dist(h, w):
    global _EDGE_DIST_CACHE
    if _EDGE_DIST_CACHE is None or _EDGE_DIST_CACHE.shape != (h, w):
        _EDGE_DIST_CACHE = _edge_distance_grid(h, w)
    return _EDGE_DIST_CACHE


def apply_transforms_batch(masks_np):
    """
    Apply all transforms + compute truth-free geometry proxies for a batch.

    Args:
        masks_np: (N, H, W) float32 in [0, 1]

    Returns:
        dict mapping feature name -> (N,) float32 scores/features. Keys include
        both TRANSFORMS and GEOMETRY_FEATURES.
    """
    n = masks_np.shape[0]
    h, w = masks_np.shape[1], masks_np.shape[2]
    out = {name: np.zeros(n, dtype=np.float32) for name in TRANSFORMS}
    for name in GEOMETRY_FEATURES:
        out[name] = np.zeros(n, dtype=np.float32)
    kernels = get_disc_kernels()
    edge_dist = _get_edge_dist(h, w)
    near_edge = edge_dist <= GEOMETRY_EDGE_DISTANCE_PIX
    y_grid, x_grid = np.mgrid[:h, :w].astype(np.float32)
    cy_patch = (h - 1) / 2.0
    cx_patch = (w - 1) / 2.0
    total_pix = float(h * w)
    for i, mask in enumerate(masks_np):
        out["baseline"][i] = float(mask.max())
        smoothed_stack = [gaussian_filter(mask, sigma=s, mode="reflect") for s in SMOOTH_SIGMAS_PIX]
        out["smooth_multi"][i] = max(float(s.max()) for s in smoothed_stack)
        # Match filter on the "base-sigma" smoothed mask: single sigma per theta
        # keeps cost O(T) rather than O(T*S). sigma_pix chosen scale-free.
        smoothed_for_mf = smoothed_stack[0]
        best_mf = -np.inf
        for kernel in kernels:
            response = fftconvolve(smoothed_for_mf, kernel[::-1, ::-1], mode="same")
            best_mf = max(best_mf, float(response.max()))
        out["mf_on_mask"][i] = best_mf

        binary = mask >= GEOMETRY_MASK_THRESHOLD
        area_pix = int(binary.sum())
        out["mask_area_at_0.5"][i] = float(area_pix) / total_pix

        total_w = float(mask.sum())
        if total_w > 0.0:
            cy = float((mask * y_grid).sum()) / total_w
            cx = float((mask * x_grid).sum()) / total_w
            out["centroid_offset_px"][i] = float(np.hypot(cy - cy_patch, cx - cx_patch))
        else:
            out["centroid_offset_px"][i] = 0.0

        if area_pix > 0:
            # Perimeter via 4-connected erosion; border_value=0 treats patch edge as outside
            # so mask pixels against the patch boundary are counted as boundary pixels.
            eroded = binary_erosion(binary, border_value=0)
            perim = int(area_pix - int(eroded.sum()))
            out["compactness"][i] = perim / float(np.sqrt(float(area_pix)))
            edge_touch = int((binary & near_edge).sum())
            out["edge_touching_fraction"][i] = edge_touch / float(area_pix)
        else:
            out["compactness"][i] = 0.0
            out["edge_touching_fraction"][i] = 0.0
    return out


def score_model_with_masks(spec, h5_path, batch_size, device):
    """Run one forward pass, return dict[name] -> (N,) scores plus label array."""
    model, run_config = load_model(spec.run_dir.resolve(), spec.checkpoint, device)
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["labels"].shape[0])
    import time as _time
    print(f"  [{spec.name}] preload {h5_path.name} ...", flush=True)
    t0 = _time.time()
    dataset = p3.H5BubbleDataset(
        h5_path=str(h5_path),
        indices=np.arange(n, dtype=np.int64),
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 10013,
        max_translate_pixels=0,
        cache_data=True,
    )
    print(f"  [{spec.name}] preload done in {_time.time() - t0:.1f}s; forward + transforms + geometry features...", flush=True)
    loader = p3.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    feature_names = tuple(TRANSFORMS) + tuple(GEOMETRY_FEATURES)
    score_cols = {name: np.zeros(n, dtype=np.float32) for name in feature_names}
    labels = np.zeros(n, dtype=np.uint8)
    offset = 0
    progress = p3.ProgressPrinter(len(loader), f"{spec.name}:{h5_path.name}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            mask_logits, _ = p3.unpack_model_output(model(images))
            probs = torch.sigmoid(mask_logits).squeeze(1).float().cpu().numpy()
            bs = probs.shape[0]
            transformed = apply_transforms_batch(probs)
            for name in feature_names:
                score_cols[name][offset:offset + bs] = transformed[name]
            labels[offset:offset + bs] = batch["label"].detach().cpu().numpy().astype(np.uint8)
            offset += bs
            progress.update(batch_idx)
    return score_cols, labels


def load_gate_strat(gate_h5):
    with h5py.File(gate_h5, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amp_grid = [float(x) for x in json.loads(h5["summary"].attrs["amplitude_grid"])]
        theta_grid = [float(x) for x in json.loads(h5["summary"].attrs["theta_grid_deg"])]
        truth = {
            "fully_contained": np.asarray(h5["truth"]["fully_contained"][:], dtype=np.uint8),
            "signal_center_in_patch": np.asarray(h5["truth"]["signal_center_in_patch"][:], dtype=np.uint8),
            "visible_target_fraction": np.asarray(h5["truth"]["visible_target_fraction"][:], dtype=np.float32),
        }
    return labels, amp_idx, theta_idx, amp_grid, theta_grid, truth


def threshold_at_fpr(null_scores, target_fpr):
    null_scores = np.sort(np.asarray(null_scores, dtype=np.float64))
    n = null_scores.size
    if n == 0:
        return float("inf"), 0.0
    k_allowed = int(np.floor(target_fpr * n))
    if k_allowed <= 0:
        threshold = float(np.nextafter(null_scores[-1], np.inf))
    else:
        threshold = float(null_scores[-k_allowed])
    actual_fpr = float((null_scores >= threshold).sum()) / float(n)
    return threshold, actual_fpr


def group_breakdown(pos_mask, scores, threshold, truth):
    fully_contained = truth["fully_contained"][pos_mask].astype(bool)
    center_in_patch = truth["signal_center_in_patch"][pos_mask].astype(bool)
    visible = truth["visible_target_fraction"][pos_mask]
    detected = scores[pos_mask] >= threshold
    total = int(pos_mask.sum())
    groups = {}
    def add(name, mask):
        n = int(mask.sum())
        if n == 0:
            return
        hits = int(detected[mask].sum())
        groups[name] = {"n": n, "detected": hits, "recall": hits / float(n), "recall_ci95": exact_ci(hits, n)}
    add("all_positive", np.ones(total, dtype=bool))
    add("geometry_contained", fully_contained)
    add("geometry_truncated", ~fully_contained)
    add("center_inside_patch", center_in_patch)
    add("center_outside_patch", ~center_in_patch)
    add("visible_fraction_low", (visible > 0) & (visible < 0.35))
    add("visible_fraction_mid", (visible >= 0.35) & (visible < 0.70))
    add("visible_fraction_high", visible >= 0.70)
    return groups


def evaluate_transform(transform_name, inj_scores, null_scores, labels, truth, fpr_targets):
    pos_mask = labels == 1
    rows = []
    for fpr_target in fpr_targets:
        threshold, actual_fpr = threshold_at_fpr(null_scores, fpr_target)
        detected = inj_scores[pos_mask] >= threshold
        total = int(pos_mask.sum())
        hits = int(detected.sum())
        row = {
            "transform": transform_name,
            "fpr_target": float(fpr_target),
            "threshold": float(threshold),
            "actual_null_fpr": float(actual_fpr),
            "recall_global": hits / float(max(total, 1)),
            "recall_ci95": exact_ci(hits, total),
            "groups": group_breakdown(pos_mask, inj_scores, threshold, truth),
        }
        rows.append(row)
    return rows


def run_model_geometry(spec, geometry, gate_root, null_h5, batch_size, device, cache_dir):
    geom_dir = gate_root / geometry
    inj_h5 = geom_dir / f"smica_real_sky_injection_{geometry}.h5"
    if not inj_h5.exists():
        raise FileNotFoundError(f"Missing gate injection HDF5: {inj_h5}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    inj_cache = cache_dir / f"inj_{geometry}_{spec.name}_transforms.npz"
    null_cache = cache_dir / f"null_{spec.name}_transforms.npz"

    required_keys = set(TRANSFORMS) | set(GEOMETRY_FEATURES)

    def cache_is_complete(path, need_labels):
        if not path.exists():
            return False
        with np.load(path) as loaded:
            keys = set(loaded.files)
        missing = required_keys - keys
        if need_labels and "labels" not in keys:
            missing.add("labels")
        return not missing

    if cache_is_complete(inj_cache, need_labels=True):
        print(f"  [reuse] {inj_cache.name}", flush=True)
        with np.load(inj_cache) as loaded:
            inj_scores = {k: loaded[k] for k in required_keys}
            inj_labels = np.asarray(loaded["labels"], dtype=np.uint8)
    else:
        if inj_cache.exists():
            print(f"  [rebuild] {inj_cache.name} missing new geometry features", flush=True)
        inj_scores, inj_labels = score_model_with_masks(spec, inj_h5, batch_size, device)
        np.savez_compressed(inj_cache, labels=inj_labels, **inj_scores)

    if cache_is_complete(null_cache, need_labels=False):
        print(f"  [reuse] {null_cache.name}", flush=True)
        with np.load(null_cache) as loaded:
            null_scores = {k: loaded[k] for k in required_keys}
    else:
        if null_cache.exists():
            print(f"  [rebuild] {null_cache.name} missing new geometry features", flush=True)
        null_all, _ = score_model_with_masks(spec, null_h5, batch_size, device)
        null_scores = {k: null_all[k] for k in required_keys}
        np.savez_compressed(null_cache, **null_scores)

    _, _, _, _, _, truth = load_gate_strat(inj_h5)
    report = {"model": spec.name, "geometry": geometry, "transforms": {}}
    for transform in TRANSFORMS:
        report["transforms"][transform] = evaluate_transform(
            transform_name=transform,
            inj_scores=inj_scores[transform],
            null_scores=null_scores[transform],
            labels=inj_labels,
            truth=truth,
            fpr_targets=FPR_TARGETS,
        )
    return report


def write_summary_markdown(path, reports):
    lines = ["# Batch 2 Post-Processing Ablation", ""]
    lines.append("Transforms:")
    lines.append("")
    lines.append("- `baseline`: current operating score `max(sigmoid(logits))`.")
    lines.append(f"- `smooth_multi`: max over sigma in {SMOOTH_SIGMAS_PIX} pixels of `max(gaussian_smooth(mask, sigma))`.")
    lines.append(f"- `mf_on_mask`: max over theta in {THETA_GRID_DEG} deg of `max(fftcorr(smoothed_mask(sigma={SMOOTH_SIGMAS_PIX[0]}pix), disc_kernel(theta)))`.")
    lines.append("")
    lines.append("Calibration: thresholds picked at real-SMICA null FPR targets 0.05, 0.08, 0.10 per transform.")
    lines.append("")
    lines.append("## Global recall by model x geometry x transform")
    lines.append("")
    lines.append("| model | geometry | transform | FPR target | threshold | actual FPR | recall | Δ vs baseline |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for report in reports:
        for transform in TRANSFORMS:
            rows = report["transforms"][transform]
            baseline_rows = {r["fpr_target"]: r for r in report["transforms"]["baseline"]}
            for row in rows:
                base_recall = baseline_rows[row["fpr_target"]]["recall_global"]
                delta = row["recall_global"] - base_recall
                delta_sign = "+" if delta >= 0 else ""
                lines.append(
                    f"| `{report['model']}` | {report['geometry']} | `{transform}` | "
                    f"{row['fpr_target']:.2f} | {row['threshold']:.4f} | "
                    f"{row['actual_null_fpr']:.3f} | {row['recall_global']:.3f} | {delta_sign}{delta:.3f} |"
                )
    lines.append("")
    lines.append("## Per-geometry recall at FPR 0.08 (mixed geometry only)")
    lines.append("")
    lines.append("| model | transform | contained | truncated | center_out | vis_low |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for report in reports:
        if report["geometry"] != "mixed":
            continue
        for transform in TRANSFORMS:
            row = [r for r in report["transforms"][transform] if abs(r["fpr_target"] - 0.08) < 1e-9][0]
            g = row["groups"]
            c = g.get("geometry_contained", {}).get("recall", float("nan"))
            t = g.get("geometry_truncated", {}).get("recall", float("nan"))
            co = g.get("center_outside_patch", {}).get("recall", float("nan"))
            vl = g.get("visible_fraction_low", {}).get("recall", float("nan"))
            lines.append(f"| `{report['model']}` | `{transform}` | {c:.3f} | {t:.3f} | {co:.3f} | {vl:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global SMOOTH_SIGMAS_PIX
    args = parse_args()
    sigmas = tuple(float(s.strip()) for s in args.smooth_sigmas_pix.split(",") if s.strip())
    if not sigmas:
        raise SystemExit("--smooth-sigmas-pix must contain at least one positive value.")
    if any(s <= 0 for s in sigmas):
        raise SystemExit("--smooth-sigmas-pix values must be positive.")
    SMOOTH_SIGMAS_PIX = sigmas
    print(f"Using smoothing sigmas (pix): {SMOOTH_SIGMAS_PIX}", flush=True)
    gate_root = Path(args.gate_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    null_h5 = Path(args.null_h5).resolve()
    if not null_h5.exists():
        raise FileNotFoundError(null_h5)

    device = p3.resolve_device(args.device)
    cache_dir = output_dir / "score_cache"

    model_map = {"v7": DEFAULT_V7_SPEC, "v6": DEFAULT_V6_SPEC}
    requested_models = args.model or ["v6", "v7"]
    specs = [ModelCfg(name=model_map[m].name, run_dir=model_map[m].run_dir, checkpoint=model_map[m].checkpoint) for m in requested_models]

    geometries = args.geometry or ["contained", "mixed"]

    reports = []
    for spec in specs:
        for geometry in geometries:
            print(f"\n=== {spec.name} x {geometry} ===", flush=True)
            report = run_model_geometry(spec, geometry, gate_root, null_h5, args.batch_size, device, cache_dir)
            reports.append(report)

    json_path = output_dir / "batch2_postprocess_ablation_report.json"
    json_path.write_text(json.dumps({"reports": reports, "smooth_sigmas_pix": SMOOTH_SIGMAS_PIX, "theta_grid_deg": THETA_GRID_DEG, "fpr_targets": FPR_TARGETS}, indent=2), encoding="utf-8")
    md_path = output_dir / "batch2_postprocess_ablation_report.md"
    write_summary_markdown(md_path, reports)
    print(f"\n=== Saved ===\n  JSON: {json_path}\n  MD:   {md_path}", flush=True)


if __name__ == "__main__":
    main()
