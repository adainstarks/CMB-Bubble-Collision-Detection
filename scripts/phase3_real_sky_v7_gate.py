"""
v7 vs v6 real-SMICA validation gate.

Purpose: decide whether v7_mixed_ft remains the deployment model after moving
from synthetic-CAMB backgrounds to real SMICA backgrounds. Produces:

    1. A real-SMICA injection HDF5 (contained or mixed geometry) covering the
       full (amplitude, theta_crit) grid.
    2. ML scores for v7_mixed_ft and v6_aux_only on both the injection and on
       the frozen real-SMICA null cache.
    3. Map-calibrated detection thresholds at target FPRs derived from the
       5000-patch real-SMICA null distribution.
    4. Per-cell and per-geometry recall, head-to-head comparison, and a
       markdown report.

This is a focused alternative to phase3_real_sky_injection.py. It reuses that
script's dataset generation and model scoring helpers but bypasses its
v5_consensus-hardcoded policy logic so we can swap in v7 cleanly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import binomtest

import phase3_train_unet as p3
import phase3_real_sky_injection as real_inj
from phase3_ensemble_evaluate import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_real_sky_v7_gate"
DEFAULT_NULL_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "null_controls_smica_mask090.h5"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    run_dir: Path
    checkpoint: str


DEFAULT_V7_SPEC = ModelSpec(
    name="v7_mixed_ft",
    run_dir=PROJECT_ROOT / "runs" / "phase3_unet" / "phase3_v7_mixed_ft",
    checkpoint="best",
)
DEFAULT_V6_SPEC = ModelSpec(
    name="v6_aux_only",
    run_dir=PROJECT_ROOT / "runs" / "phase3_unet" / "phase3_v6_aux_only_w4",
    checkpoint="best",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="v7 vs v6 real-SMICA validation gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--map-name", type=str, default="smica")
    parser.add_argument("--num-backgrounds", type=int, default=500)
    parser.add_argument("--pool-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument(
        "--amplitude-grid",
        type=str,
        default="1e-06,2e-06,5e-06,1e-05,2e-05,5e-05,1e-04",
    )
    parser.add_argument("--theta-grid-deg", type=str, default="5,10,15,20,25")
    parser.add_argument("--geometry-mode", type=str, default="contained", choices=("contained", "truncated", "mixed"))
    parser.add_argument("--truncated-positive-fraction", type=float, default=0.0)
    parser.add_argument("--truncated-visible-fraction-min", type=float, default=0.15)
    parser.add_argument("--truncated-visible-fraction-max", type=float, default=0.95)
    parser.add_argument("--truncated-max-center-draws", type=int, default=256)
    parser.add_argument("--signal-center-edge-margin-pix", type=float, default=0.0)
    parser.add_argument("--contained-margin-deg", type=float, default=0.5)
    parser.add_argument("--edge-sigma-deg", type=float, default=0.0)
    parser.add_argument("--signal-beam-fwhm-arcmin", type=float, default=15.0)
    parser.add_argument("--exclude-h5", action="append", default=[])
    parser.add_argument("--exclude-radius-deg", type=float, default=0.25)
    parser.add_argument("--null-h5", type=str, default=str(DEFAULT_NULL_H5))
    parser.add_argument("--fpr-targets", type=str, default="0.05,0.08,0.10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--classical-workers", type=int, default=8)
    parser.add_argument("--classical-chunk-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default="v7,v6",
        help="Comma-separated subset of models to score. Options: v7, v6.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip generation and scoring; just assemble the head-to-head report from cached scores.",
    )
    return parser.parse_args()


def parse_float_list(text):
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def exact_ci(k, n):
    if n <= 0:
        return [float("nan"), float("nan")]
    result = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(result.low), float(result.high)]


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


def build_injection_args(args, h5_path):
    """Shim: construct the argparse.Namespace expected by real_inj.generate_dataset."""
    ns = argparse.Namespace(
        map_name=args.map_name,
        num_backgrounds=args.num_backgrounds,
        pool_size=args.pool_size,
        seed=args.seed,
        amplitude_grid=parse_float_list(args.amplitude_grid),
        theta_grid_deg=parse_float_list(args.theta_grid_deg),
        geometry_mode=args.geometry_mode,
        truncated_positive_fraction=args.truncated_positive_fraction,
        truncated_visible_fraction_min=args.truncated_visible_fraction_min,
        truncated_visible_fraction_max=args.truncated_visible_fraction_max,
        truncated_max_center_draws=args.truncated_max_center_draws,
        signal_center_edge_margin_pix=args.signal_center_edge_margin_pix,
        contained_margin_deg=args.contained_margin_deg,
        edge_sigma_deg=args.edge_sigma_deg,
        signal_beam_fwhm_arcmin=args.signal_beam_fwhm_arcmin,
        exclude_h5=args.exclude_h5,
        exclude_radius_deg=args.exclude_radius_deg,
        classical_chunk_size=args.classical_chunk_size,
    )
    return ns, Path(h5_path)


def build_score_args(args):
    """Shim: construct the argparse.Namespace expected by real_inj.score_model_batched."""
    return argparse.Namespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        reuse_scores=args.reuse_scores,
    )


def score_model(spec, h5_path, cache_dir, args, device, cache_label):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_label}_{spec.name}_scores.npz"
    if args.reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            return np.asarray(loaded["scores"], dtype=np.float32)

    model, run_config = load_model(spec.run_dir.resolve(), spec.checkpoint, device)
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["labels"].shape[0])
    print(f"  Preloading {cache_label}:{spec.name} HDF5 patches into RAM (one-time decompression)...", flush=True)
    t_preload = p3.time.time() if hasattr(p3, "time") else __import__("time").time()
    dataset = p3.H5BubbleDataset(
        h5_path=str(h5_path),
        indices=np.arange(n, dtype=np.int64),
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 10011,
        max_translate_pixels=0,
        cache_data=True,
    )
    t_preload_done = __import__("time").time()
    print(f"  Preload complete in {t_preload_done - t_preload:.1f}s; running GPU forward pass...", flush=True)
    loader = p3.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    scores = np.zeros(n, dtype=np.float32)
    offset = 0
    progress = p3.ProgressPrinter(len(loader), f"Score {cache_label}:{spec.name}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            mask_logits, _ = p3.unpack_model_output(model(images))
            batch_scores = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values.detach().cpu().numpy()
            batch_size = int(images.shape[0])
            scores[offset:offset + batch_size] = batch_scores.astype(np.float32)
            offset += batch_size
            progress.update(batch_idx)
    np.savez_compressed(cache_path, scores=scores)
    return scores


def load_strat(h5_path):
    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amp_grid = [float(x) for x in json.loads(h5["summary"].attrs["amplitude_grid"])]
        theta_grid = [float(x) for x in json.loads(h5["summary"].attrs["theta_grid_deg"])]
        truth = {
            "fully_contained": np.asarray(h5["truth"]["fully_contained"][:], dtype=np.uint8),
            "signal_center_in_patch": np.asarray(h5["truth"]["signal_center_in_patch"][:], dtype=np.uint8),
            "visible_target_fraction": np.asarray(h5["truth"]["visible_target_fraction"][:], dtype=np.float32),
            "target_touches_edge": np.asarray(h5["truth"]["target_touches_edge"][:], dtype=np.uint8),
        }
    return labels, amp_idx, theta_idx, amp_grid, theta_grid, truth


def evaluate_model(name, inj_scores, inj_labels, null_scores, fpr_targets, amp_idx, theta_idx, amp_grid, theta_grid, truth):
    results = {"name": name, "n_positive": int((inj_labels == 1).sum()), "n_null": int(null_scores.size), "fpr_points": []}
    for fpr_target in fpr_targets:
        threshold, actual_fpr = threshold_at_fpr(null_scores, fpr_target)
        pos_mask = inj_labels == 1
        pos_scores = inj_scores[pos_mask]
        detected = pos_scores >= threshold
        total = int(pos_mask.sum())
        recall = float(detected.sum()) / float(max(total, 1))

        cell_rows = []
        for t_i, t_deg in enumerate(theta_grid):
            for a_i, amp in enumerate(amp_grid):
                mask = pos_mask & (amp_idx == a_i) & (theta_idx == t_i)
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                hit = int((inj_scores[idx] >= threshold).sum())
                cell_rows.append({
                    "amplitude": float(amp),
                    "theta_deg": float(t_deg),
                    "n": int(idx.size),
                    "detected": hit,
                    "recall": hit / float(idx.size),
                    "recall_ci95": exact_ci(hit, int(idx.size)),
                })

        groups = {}
        fully_contained = truth["fully_contained"][pos_mask]
        center_in_patch = truth["signal_center_in_patch"][pos_mask]
        visible = truth["visible_target_fraction"][pos_mask]
        group_defs = {
            "all_positive": np.ones(total, dtype=bool),
            "geometry_contained": fully_contained.astype(bool),
            "geometry_truncated": ~fully_contained.astype(bool),
            "center_inside_patch": center_in_patch.astype(bool),
            "center_outside_patch": ~center_in_patch.astype(bool),
            "visible_fraction_low": (visible > 0) & (visible < 0.35),
            "visible_fraction_mid": (visible >= 0.35) & (visible < 0.70),
            "visible_fraction_high": visible >= 0.70,
        }
        for group_name, group_mask in group_defs.items():
            n_grp = int(group_mask.sum())
            if n_grp == 0:
                continue
            hits_grp = int(detected[group_mask].sum())
            groups[group_name] = {
                "n": n_grp,
                "detected": hits_grp,
                "recall": hits_grp / float(n_grp),
                "recall_ci95": exact_ci(hits_grp, n_grp),
            }

        results["fpr_points"].append({
            "fpr_target": float(fpr_target),
            "threshold": float(threshold),
            "actual_null_fpr": float(actual_fpr),
            "recall_global": recall,
            "recall_ci95": exact_ci(int(detected.sum()), total),
            "cells": cell_rows,
            "groups": groups,
        })
    return results


def format_recall_line(value, n):
    if n == 0:
        return "n/a"
    ci = exact_ci(int(round(value * n)), n)
    return f"{value:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"


def write_markdown(output_dir, geometry, v7, v6, fpr_targets, amp_grid, theta_grid):
    lines = [
        f"# Real-SMICA Gate: v7_mixed_ft vs v6_aux_only  ({geometry} geometry)",
        "",
        f"Backgrounds per cell: {v7['n_positive'] // (len(amp_grid) * len(theta_grid))}",
        f"Null samples: {v7['n_null']}",
        "",
        "## Global recall at map-calibrated thresholds",
        "",
        "| FPR target | model | threshold | actual null FPR | recall | n_pos |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for fp_i, fpr in enumerate(fpr_targets):
        for row_model, row in ((v7["name"], v7["fpr_points"][fp_i]), (v6["name"], v6["fpr_points"][fp_i])):
            ci = row["recall_ci95"]
            lines.append(
                f"| {fpr:.2f} | `{row_model}` | {row['threshold']:.4f} | {row['actual_null_fpr']:.3f} | {row['recall_global']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] | {v7['n_positive']} |"
            )
    lines.append("")
    lines.append("## Per-geometry recall at FPR 0.08")
    lines.append("")
    if len(fpr_targets) >= 2:
        fpr_idx = 1
    else:
        fpr_idx = 0
    group_keys = list(v7["fpr_points"][fpr_idx]["groups"].keys())
    lines.append("| group | v7 recall | v6 recall | delta |")
    lines.append("|---|---:|---:|---:|")
    for g in group_keys:
        v7g = v7["fpr_points"][fpr_idx]["groups"].get(g)
        v6g = v6["fpr_points"][fpr_idx]["groups"].get(g)
        if v7g is None or v6g is None:
            continue
        delta = v7g["recall"] - v6g["recall"]
        lines.append(f"| `{g}` (n={v7g['n']}) | {v7g['recall']:.3f} | {v6g['recall']:.3f} | {delta:+.3f} |")
    lines.append("")
    lines.append("## Per-cell recall at FPR 0.08 (v7 minus v6)")
    lines.append("")
    lines.append("| A | theta_deg | v7 recall | v6 recall | delta |")
    lines.append("|---:|---:|---:|---:|---:|")
    v7_cells = {(row["amplitude"], row["theta_deg"]): row for row in v7["fpr_points"][fpr_idx]["cells"]}
    v6_cells = {(row["amplitude"], row["theta_deg"]): row for row in v6["fpr_points"][fpr_idx]["cells"]}
    for key in sorted(v7_cells.keys()):
        a, t = key
        v7c = v7_cells[key]
        v6c = v6_cells.get(key)
        if v6c is None:
            continue
        delta = v7c["recall"] - v6c["recall"]
        lines.append(f"| {a:.1e} | {t:g} | {v7c['recall']:.3f} | {v6c['recall']:.3f} | {delta:+.3f} |")
    out_path = output_dir / f"v7_vs_v6_{geometry}_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = output_dir / args.geometry_mode
    geometry_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = geometry_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    inj_h5 = geometry_dir / f"smica_real_sky_injection_{args.geometry_mode}.h5"
    device = p3.resolve_device(args.device)
    fpr_targets = parse_float_list(args.fpr_targets)

    if not (args.score_only or args.report_only):
        if inj_h5.exists() and args.reuse_data:
            print(f"Reusing existing injection HDF5: {inj_h5}", flush=True)
        else:
            if inj_h5.exists():
                print(f"Overwriting existing injection HDF5: {inj_h5}", flush=True)
                inj_h5.unlink()
            inj_args, inj_path = build_injection_args(args, inj_h5)
            real_inj.generate_dataset(inj_args, inj_path)

    if args.generate_only:
        print("--generate-only set; skipping scoring.", flush=True)
        return

    model_map = {"v7": DEFAULT_V7_SPEC, "v6": DEFAULT_V6_SPEC}
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested if m not in model_map]
    if unknown:
        raise ValueError(f"Unknown --models entries: {unknown}. Valid: v7, v6.")
    active_specs = [model_map[m] for m in requested]
    all_specs = [DEFAULT_V7_SPEC, DEFAULT_V6_SPEC]

    inj_scores = {}
    null_h5 = Path(args.null_h5).resolve()
    if not null_h5.exists():
        raise FileNotFoundError(f"Null HDF5 not found: {null_h5}")

    if not args.report_only:
        for spec in active_specs:
            inj_scores[spec.name] = score_model(spec, inj_h5, cache_dir, args, device, cache_label="inj")
        for spec in active_specs:
            score_model(spec, null_h5, cache_dir, args, device, cache_label="null")

    missing_for_report = []
    for spec in all_specs:
        inj_cache = cache_dir / f"inj_{spec.name}_scores.npz"
        null_cache = cache_dir / f"null_{spec.name}_scores.npz"
        if not inj_cache.exists() or not null_cache.exists():
            missing_for_report.append(spec.name)
    if missing_for_report:
        print(
            f"Score caches missing for {missing_for_report}; skipping report assembly. "
            "Rerun with --report-only after all models are scored.",
            flush=True,
        )
        return

    inj_scores = {}
    null_scores = {}
    for spec in all_specs:
        with np.load(cache_dir / f"inj_{spec.name}_scores.npz") as loaded:
            inj_scores[spec.name] = np.asarray(loaded["scores"], dtype=np.float32)
        with np.load(cache_dir / f"null_{spec.name}_scores.npz") as loaded:
            null_scores[spec.name] = np.asarray(loaded["scores"], dtype=np.float32)
    specs = all_specs

    labels, amp_idx, theta_idx, amp_grid, theta_grid, truth = load_strat(inj_h5)

    reports = {}
    for spec in specs:
        reports[spec.name] = evaluate_model(
            name=spec.name,
            inj_scores=inj_scores[spec.name],
            inj_labels=labels,
            null_scores=null_scores[spec.name],
            fpr_targets=fpr_targets,
            amp_idx=amp_idx,
            theta_idx=theta_idx,
            amp_grid=amp_grid,
            theta_grid=theta_grid,
            truth=truth,
        )

    combined = {
        "geometry_mode": args.geometry_mode,
        "injection_h5": str(inj_h5),
        "null_h5": str(null_h5),
        "fpr_targets": list(fpr_targets),
        "amp_grid": amp_grid,
        "theta_grid": theta_grid,
        "models": reports,
    }
    json_path = geometry_dir / f"v7_vs_v6_{args.geometry_mode}_report.json"
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    md_path = write_markdown(geometry_dir, args.geometry_mode, reports[DEFAULT_V7_SPEC.name], reports[DEFAULT_V6_SPEC.name], fpr_targets, amp_grid, theta_grid)
    print(f"\n=== Saved ===", flush=True)
    print(f"  JSON: {json_path}", flush=True)
    print(f"  MD:   {md_path}", flush=True)


if __name__ == "__main__":
    main()
