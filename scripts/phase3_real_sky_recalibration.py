"""
Recalibrate Phase 3 detection thresholds on real SMICA null backgrounds.

This script separates two failure modes in the real-sky injection gate:

1. Threshold miscalibration: CAMB-negative thresholds are too high for real
   SMICA score distributions.
2. Residual domain gap: even after SMICA-null recalibration, real-background
   injected signals remain below CAMB-background sensitivity.

It intentionally keeps only the two retained methods requested for this pass:
v6_aux_only and matched_template. Composite policies are restricted to those
two methods only.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

import phase3_train_unet as p3
from phase3_sensitivity_curve import (
    SIGN_QUADRANTS,
    make_feeney_template_kernel,
    score_matched_template_patch,
    threshold_from_negatives,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_real_sky_recalibration"
DEFAULT_REAL_H5 = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_real_sky_injection_smica_mask090"
    / "smica_real_sky_injection.h5"
)
DEFAULT_REAL_SCORES = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_real_sky_injection_smica_mask090"
    / "real_sky_scores.npz"
)
DEFAULT_REAL_NULL_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "null_controls_smica_mask090.h5"
DEFAULT_REAL_NULL_V6 = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_ensemble_eval"
    / "score_cache"
    / "null_v6_aux_only_scores.npz"
)
DEFAULT_SENS_H5 = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_data.h5"
DEFAULT_SENS_SCORES = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_scores.npz"
DEFAULT_SENS_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_report.json"

METHODS = ("v6_aux_only", "matched_template")
POLICIES = ("v6_aux_only", "matched_template", "either_v6_or_matched", "both_v6_and_matched")


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-SMICA threshold recalibration for v6_aux_only and matched_template.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-h5", type=str, default=str(DEFAULT_REAL_H5))
    parser.add_argument("--real-scores", type=str, default=str(DEFAULT_REAL_SCORES))
    parser.add_argument("--real-null-h5", type=str, default=str(DEFAULT_REAL_NULL_H5))
    parser.add_argument("--real-null-v6-scores", type=str, default=str(DEFAULT_REAL_NULL_V6))
    parser.add_argument("--sensitivity-h5", type=str, default=str(DEFAULT_SENS_H5))
    parser.add_argument("--sensitivity-scores", type=str, default=str(DEFAULT_SENS_SCORES))
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_SENS_REPORT))
    parser.add_argument("--fpr-targets", type=str, default="0.05,0.08,0.10")
    parser.add_argument("--classical-workers", type=int, default=8)
    parser.add_argument("--classical-chunk-size", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--full-sky-independent-patches",
        type=float,
        default=3000.0,
        help="Reference number of independent full-sky patches for expected false-positive burden.",
    )
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--reuse-null-scores", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.fpr_targets = parse_float_list(args.fpr_targets)
    if not args.fpr_targets:
        raise ValueError("--fpr-targets must contain at least one target.")
    if any(target <= 0.0 or target >= 1.0 for target in args.fpr_targets):
        raise ValueError("--fpr-targets entries must be in (0, 1).")
    if args.classical_workers <= 0:
        raise ValueError("--classical-workers must be positive.")
    if args.classical_chunk_size <= 0:
        raise ValueError("--classical-chunk-size must be positive.")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive.")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def exact_ci(k: int, n: int) -> list[float]:
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def binary_metrics(active: np.ndarray, labels: np.ndarray) -> dict:
    active = np.asarray(active, dtype=bool)
    labels = np.asarray(labels, dtype=np.uint8)
    tp = int(np.logical_and(active, labels == 1).sum())
    fp = int(np.logical_and(active, labels == 0).sum())
    tn = int(np.logical_and(~active, labels == 0).sum())
    fn = int(np.logical_and(~active, labels == 1).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
        "recall_ci95": exact_ci(tp, tp + fn) if tp + fn else [0.0, 0.0],
        "fpr_ci95": exact_ci(fp, fp + tn) if fp + tn else [0.0, 0.0],
    }


def score_summary(scores: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    quantiles = {f"q{int(q * 100):02d}": float(np.quantile(scores, q)) for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)}
    return {
        "n": int(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        **quantiles,
    }


def load_score_npz(path: Path, methods: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path) as loaded:
        labels = np.asarray(loaded["labels"], dtype=np.uint8)
        scores = {method: np.asarray(loaded[f"score__{method}"], dtype=np.float32) for method in methods}
    return labels, scores


def load_stratification(path: Path) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    with h5py.File(path, "r") as h5:
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amp_grid = [float(x) for x in json.loads(h5["summary"].attrs["amplitude_grid"])]
        theta_grid = [float(x) for x in json.loads(h5["summary"].attrs["theta_grid_deg"])]
    return amp_idx, theta_idx, amp_grid, theta_grid


def score_matched_template_null(
    h5_path: Path,
    output_dir: Path,
    theta_grid_deg: list[float],
    beam_fwhm_arcmin: float,
    workers: int,
    chunk_size: int,
    reuse_scores: bool,
) -> np.ndarray:
    cache_dir = output_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "null_matched_template_scores.npz"
    if reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            return np.asarray(loaded["scores"], dtype=np.float32)

    kernels = [
        make_feeney_template_kernel(theta, z0_sign, zcrit_sign, beam_fwhm_arcmin=beam_fwhm_arcmin)
        for theta in theta_grid_deg
        for z0_sign, zcrit_sign in SIGN_QUADRANTS
    ]
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["patches"].shape[0])
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        if int(labels.sum()) != 0:
            raise RuntimeError(f"Expected all-zero null HDF5 labels; found {int(labels.sum())} positives.")
        scores = np.zeros(n, dtype=np.float32)
        progress = p3.ProgressPrinter(n, f"Matched-template real null ({workers} threads)")
        completed = 0
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            patches = np.asarray(h5["patches"][start:stop], dtype=np.float32)

            def score_one(local_idx: int) -> tuple[int, float]:
                return local_idx, score_matched_template_patch(patches[local_idx], kernels)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(score_one, idx) for idx in range(len(patches))]
                for future in as_completed(futures):
                    local_idx, score = future.result()
                    scores[start + local_idx] = score
                    completed += 1
                    if completed % 500 == 0 or completed == n:
                        progress.update(completed)
    np.savez_compressed(cache_path, scores=scores)
    return scores


def load_real_null_scores(args: argparse.Namespace, output_dir: Path, theta_grid: list[float], beam_fwhm_arcmin: float) -> dict[str, np.ndarray]:
    v6_path = Path(args.real_null_v6_scores).resolve()
    if not v6_path.exists():
        raise FileNotFoundError(
            f"Missing v6 real-null score cache: {v6_path}. "
            "Run phase3_ensemble_evaluate.py first or pass --real-null-v6-scores."
        )
    with np.load(v6_path) as loaded:
        v6_scores = np.asarray(loaded["scores"], dtype=np.float32)
    matched_scores = score_matched_template_null(
        h5_path=Path(args.real_null_h5).resolve(),
        output_dir=output_dir,
        theta_grid_deg=theta_grid,
        beam_fwhm_arcmin=beam_fwhm_arcmin,
        workers=args.classical_workers,
        chunk_size=args.classical_chunk_size,
        reuse_scores=args.reuse_null_scores,
    )
    if v6_scores.shape != matched_scores.shape:
        raise RuntimeError(f"Null score shape mismatch: v6={v6_scores.shape}, matched={matched_scores.shape}")
    return {"v6_aux_only": v6_scores, "matched_template": matched_scores}


def method_thresholds(
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
    fpr_target: float,
) -> dict[str, dict]:
    out = {}
    for method, values in scores.items():
        threshold, fp, fpr = threshold_from_negatives(values, labels, fpr_target)
        out[method] = {"threshold": float(threshold), "negative_fp": int(fp), "negative_fpr": float(fpr)}
    return out


def active_policies(scores: dict[str, np.ndarray], thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    v6 = scores["v6_aux_only"] > float(thresholds["v6_aux_only"])
    matched = scores["matched_template"] > float(thresholds["matched_template"])
    return {
        "v6_aux_only": v6,
        "matched_template": matched,
        "either_v6_or_matched": v6 | matched,
        "both_v6_and_matched": v6 & matched,
    }


def bootstrap_delta_ci(real_hits: np.ndarray, camb_hits: np.ndarray, resamples: int, rng: np.random.Generator) -> list[float]:
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


def cell_rows(
    fpr_target: float,
    real_labels: np.ndarray,
    real_amp_idx: np.ndarray,
    real_theta_idx: np.ndarray,
    real_active: dict[str, np.ndarray],
    camb_labels: np.ndarray,
    camb_amp_idx: np.ndarray,
    camb_theta_idx: np.ndarray,
    camb_active: dict[str, np.ndarray],
    amp_grid: list[float],
    theta_grid: list[float],
    resamples: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    for policy in POLICIES:
        for amp_i, amp in enumerate(amp_grid):
            for theta_i, theta in enumerate(theta_grid):
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
                delta_ci = bootstrap_delta_ci(real_hits, camb_hits, resamples, rng)
                rows.append(
                    {
                        "fpr_target": float(fpr_target),
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
    return rows


def summarize_target(rows: list[dict], fpr_target: float, policy: str) -> dict:
    target_rows = [row for row in rows if row["fpr_target"] == fpr_target and row["policy"] == policy]
    deficits = [row for row in target_rows if row["delta_significant"] and row["delta_real_minus_camb"] < 0.0]
    gains = [row for row in target_rows if row["delta_significant"] and row["delta_real_minus_camb"] > 0.0]
    return {
        "policy": policy,
        "fpr_target": float(fpr_target),
        "num_cells": len(target_rows),
        "significant_deficit_cells": len(deficits),
        "significant_gain_cells": len(gains),
        "largest_deficits": sorted(deficits, key=lambda row: row["delta_real_minus_camb"])[:5],
        "largest_gains": sorted(gains, key=lambda row: row["delta_real_minus_camb"], reverse=True)[:5],
    }


def classify_outcome(previous: dict, recalibrated: dict, camb: dict, target_summary: dict) -> str:
    prev_recall = float(previous["recall"])
    real_recall = float(recalibrated["recall"])
    camb_recall = float(camb["recall"])
    lift = real_recall - prev_recall
    gap_after = camb_recall - real_recall
    if target_summary["significant_deficit_cells"] == 0 and gap_after <= 0.03:
        return "a_threshold_miscalibration_dominated"
    if lift > 0.02 and target_summary["significant_deficit_cells"] > 0:
        return "b_threshold_plus_domain_gap"
    if lift <= 0.02:
        return "c_domain_gap_dominated"
    return "mixed_inconclusive"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Real-SMICA Threshold Recalibration",
        "",
        f"- Real injection HDF5: `{report['inputs']['real_h5']}`",
        f"- Real null HDF5: `{report['inputs']['real_null_h5']}`",
        f"- Sensitivity HDF5: `{report['inputs']['sensitivity_h5']}`",
        f"- Methods retained: `{', '.join(METHODS)}`",
        f"- Bootstrap resamples: `{report['bootstrap_resamples']}`",
        "",
        "## Outcome",
        "",
        f"- FPR 0.05 outcome for `v6_aux_only`: `{report['outcome']['v6_aux_only']['classification']}`",
        f"- FPR 0.05 outcome for `matched_template`: `{report['outcome']['matched_template']['classification']}`",
        "",
        "## Global Metrics",
        "",
        "| FPR target | policy | threshold domain | recall | FPR | precision | FP | TP |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["global_rows"]:
        lines.append(
            f"| {row['fpr_target']:.2f} | `{row['policy']}` | {row['domain']} | "
            f"{row['recall']:.3f} | {row['fpr']:.3f} | {row['precision']:.3f} | "
            f"{row['fp']} | {row['tp']} |"
        )
    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            "| FPR target | method | CAMB threshold | real-SMICA threshold | real-null FP | real-null FPR |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["threshold_rows"]:
        lines.append(
            f"| {row['fpr_target']:.2f} | `{row['method']}` | {row['camb_threshold']:.8g} | "
            f"{row['real_threshold']:.8g} | {row['real_negative_fp']} | {row['real_negative_fpr']:.4f} |"
        )
    lines.extend(["", "## FPR 0.05 Cell Summary", ""])
    for method, summary in report["cell_summary_fpr_005"].items():
        lines.append(
            f"- `{method}`: significant real deficits `{summary['significant_deficit_cells']} / {summary['num_cells']}`, "
            f"significant real gains `{summary['significant_gain_cells']} / {summary['num_cells']}`."
        )
    lines.extend(
        [
            "",
            f"## Expected False Positives For {report['full_sky_independent_patches']:.0f} Independent Patches",
            "",
            "| FPR target | policy | real-null FPR | expected FP |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in report["global_rows"]:
        if row["domain"] != "real_null_at_smica_threshold":
            continue
        lines.append(
            f"| {row['fpr_target']:.2f} | `{row['policy']}` | "
            f"{row['fpr']:.4f} | {row['expected_fp_full_sky']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cells(rows: list[dict], output_dir: Path) -> None:
    fpr_rows = [row for row in rows if row["fpr_target"] == 0.05 and row["policy"] in METHODS]
    if not fpr_rows:
        return
    policies = list(METHODS)
    thetas = sorted({row["theta_crit_deg"] for row in fpr_rows})
    fig, axes = plt.subplots(len(policies), len(thetas), figsize=(4.0 * len(thetas), 3.2 * len(policies)), sharey=True)
    axes = np.asarray(axes)
    for r_idx, policy in enumerate(policies):
        for c_idx, theta in enumerate(thetas):
            ax = axes[r_idx, c_idx]
            subset = sorted(
                [row for row in fpr_rows if row["policy"] == policy and row["theta_crit_deg"] == theta],
                key=lambda row: row["amplitude"],
            )
            x = np.asarray([row["amplitude"] for row in subset], dtype=np.float64)
            ax.plot(x, [row["camb_p_det"] for row in subset], marker="o", label="CAMB")
            ax.plot(x, [row["real_p_det"] for row in subset], marker="s", label="SMICA recal")
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25)
            if r_idx == 0:
                ax.set_title(f"theta={theta:g} deg")
            if c_idx == 0:
                ax.set_ylabel(policy)
            if r_idx == len(policies) - 1:
                ax.set_xlabel("A")
    axes[0, -1].legend(loc="lower right")
    fig.suptitle("Real-SMICA recalibrated sensitivity vs CAMB-calibrated sensitivity, FPR=0.05")
    fig.tight_layout()
    fig.savefig(output_dir / "real_sky_recalibrated_vs_camb_fpr005.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_h5 = Path(args.real_h5).resolve()
    sens_h5 = Path(args.sensitivity_h5).resolve()
    real_labels, real_scores = load_score_npz(Path(args.real_scores).resolve(), METHODS)
    camb_labels, camb_scores = load_score_npz(Path(args.sensitivity_scores).resolve(), METHODS)
    real_amp_idx, real_theta_idx, real_amp_grid, real_theta_grid = load_stratification(real_h5)
    camb_amp_idx, camb_theta_idx, camb_amp_grid, camb_theta_grid = load_stratification(sens_h5)
    if real_amp_grid != camb_amp_grid or real_theta_grid != camb_theta_grid:
        raise RuntimeError("Real-sky and CAMB sensitivity grids do not match.")

    sensitivity = load_json(Path(args.sensitivity_report).resolve())
    with h5py.File(sens_h5, "r") as h5:
        beam_fwhm_arcmin = float(h5["summary"].attrs["beam_fwhm_arcmin"])
    real_null_scores = load_real_null_scores(args, output_dir, real_theta_grid, beam_fwhm_arcmin)
    real_null_labels = np.zeros_like(next(iter(real_null_scores.values())), dtype=np.uint8)

    camb_thresholds = {
        method: {
            "threshold": float(sensitivity["thresholds"][method]["threshold"]),
            "negative_fp": int(sensitivity["thresholds"][method]["negative_fp"]),
            "negative_fpr": float(sensitivity["thresholds"][method]["negative_fpr"]),
        }
        for method in METHODS
    }

    rng = np.random.default_rng(args.seed)
    threshold_rows = []
    global_rows = []
    rows = []
    outcomes = {}
    cell_summary_fpr_005 = {}
    score_distributions = {
        "camb_negative": {method: score_summary(camb_scores[method][camb_labels == 0]) for method in METHODS},
        "real_injection_negative": {method: score_summary(real_scores[method][real_labels == 0]) for method in METHODS},
        "real_null_5000": {method: score_summary(real_null_scores[method]) for method in METHODS},
    }

    for fpr_target in args.fpr_targets:
        real_thresholds_full = method_thresholds(real_null_scores, real_null_labels, fpr_target)
        camb_method_threshold_values = {method: camb_thresholds[method]["threshold"] for method in METHODS}
        real_method_threshold_values = {method: real_thresholds_full[method]["threshold"] for method in METHODS}

        real_active_recal = active_policies(real_scores, real_method_threshold_values)
        camb_active = active_policies(camb_scores, camb_method_threshold_values)
        real_active_at_camb = active_policies(real_scores, camb_method_threshold_values)
        real_null_active = active_policies(real_null_scores, real_method_threshold_values)

        for method in METHODS:
            threshold_rows.append(
                {
                    "fpr_target": float(fpr_target),
                    "method": method,
                    "camb_threshold": camb_thresholds[method]["threshold"],
                    "camb_negative_fp": camb_thresholds[method]["negative_fp"],
                    "camb_negative_fpr": camb_thresholds[method]["negative_fpr"],
                    "real_threshold": real_thresholds_full[method]["threshold"],
                    "real_negative_fp": real_thresholds_full[method]["negative_fp"],
                    "real_negative_fpr": real_thresholds_full[method]["negative_fpr"],
                }
            )

        for policy in POLICIES:
            for domain, labels, active in (
                ("real_recalibrated_on_smica_null", real_labels, real_active_recal[policy]),
                ("real_at_camb_threshold", real_labels, real_active_at_camb[policy]),
                ("camb_at_camb_threshold", camb_labels, camb_active[policy]),
                ("real_null_at_smica_threshold", real_null_labels, real_null_active[policy]),
            ):
                metrics = binary_metrics(active, labels)
                global_rows.append(
                    {
                        "fpr_target": float(fpr_target),
                        "policy": policy,
                        "domain": domain,
                        "expected_fp_full_sky": float(metrics["fpr"] * args.full_sky_independent_patches),
                        **metrics,
                    }
                )

        rows.extend(
            cell_rows(
                fpr_target=fpr_target,
                real_labels=real_labels,
                real_amp_idx=real_amp_idx,
                real_theta_idx=real_theta_idx,
                real_active=real_active_recal,
                camb_labels=camb_labels,
                camb_amp_idx=camb_amp_idx,
                camb_theta_idx=camb_theta_idx,
                camb_active=camb_active,
                amp_grid=real_amp_grid,
                theta_grid=real_theta_grid,
                resamples=args.bootstrap_resamples,
                rng=rng,
            )
        )

        if abs(fpr_target - 0.05) < 1e-12:
            for method in METHODS:
                previous = binary_metrics(real_active_at_camb[method], real_labels)
                recalibrated = binary_metrics(real_active_recal[method], real_labels)
                camb = binary_metrics(camb_active[method], camb_labels)
                summary = summarize_target(rows, fpr_target, method)
                outcomes[method] = {
                    "classification": classify_outcome(previous, recalibrated, camb, summary),
                    "previous_real_at_camb_threshold": previous,
                    "real_recalibrated": recalibrated,
                    "camb_reference": camb,
                    "cell_summary": summary,
                }
                cell_summary_fpr_005[method] = summary

    report = {
        "inputs": {
            "real_h5": str(real_h5),
            "real_scores": str(Path(args.real_scores).resolve()),
            "real_null_h5": str(Path(args.real_null_h5).resolve()),
            "real_null_v6_scores": str(Path(args.real_null_v6_scores).resolve()),
            "sensitivity_h5": str(sens_h5),
            "sensitivity_scores": str(Path(args.sensitivity_scores).resolve()),
            "sensitivity_report": str(Path(args.sensitivity_report).resolve()),
        },
        "methods": list(METHODS),
        "policies": list(POLICIES),
        "fpr_targets": [float(x) for x in args.fpr_targets],
        "beam_fwhm_arcmin": beam_fwhm_arcmin,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "full_sky_independent_patches": float(args.full_sky_independent_patches),
        "score_distributions": score_distributions,
        "threshold_rows": threshold_rows,
        "global_rows": global_rows,
        "cell_summary_fpr_005": cell_summary_fpr_005,
        "outcome": outcomes,
        "cell_rows": rows,
    }

    json_path = output_dir / "real_sky_recalibration_report.json"
    md_path = output_dir / "real_sky_recalibration_report.md"
    csv_path = output_dir / "real_sky_recalibration_cells.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    write_csv(csv_path, rows)
    if not args.skip_plot:
        plot_cells(rows, output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
