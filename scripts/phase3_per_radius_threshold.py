"""
Calibrate classical baselines with per-radius thresholds on the sensitivity grid.

The U-Net checkpoints currently emit one scalar image score from the mask logits.
They do not expose a theta_c-conditioned scan score, so a per-radius threshold is
not well-defined for those models without adding a scale head or radius-specific
candidate scorer. This script therefore audits the valid case: classical methods
whose score can be computed for a fixed theta_c.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import binomtest

import phase3_train_unet as p3
from phase3_sensitivity_curve import (
    SIGN_QUADRANTS,
    make_centered_disc_kernel,
    make_feeney_template_kernel,
    score_circular_template_patch,
    standardize_patch,
    threshold_from_negatives,
)
from phase3_method_registry import CIRCULAR_TEMPLATE_SCREEN, canonical_method_name, method_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_H5 = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_data.h5"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_report.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_per_radius_threshold"
METHODS = (CIRCULAR_TEMPLATE_SCREEN, "centered_disc")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-radius threshold calibration for classical sensitivity baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-h5", type=str, default=str(DEFAULT_DATA_H5))
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fpr-target", type=float, default=0.05)
    parser.add_argument("--classical-workers", type=int, default=8)
    return parser.parse_args()


def validate_args(args):
    if not (0.0 < args.fpr_target < 1.0):
        raise ValueError("--fpr-target must be in (0, 1).")
    if args.classical_workers <= 0:
        raise ValueError("--classical-workers must be positive.")


def exact_ci(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def score_fixed_circular_template(patch, kernels):
    """Score a patch with the current circular-template correlation screen."""

    patch = standardize_patch(patch)
    return score_circular_template_patch(patch, kernels)


def score_fixed_centered(patch, kernel):
    patch = standardize_patch(patch)
    return abs(float(np.sum(patch * kernel)))


def score_theta(patches, method, theta_deg, beam_fwhm_arcmin, workers):
    method = canonical_method_name(method)
    if method == CIRCULAR_TEMPLATE_SCREEN:
        kernels = [
            make_feeney_template_kernel(theta_deg, z0_sign, zcrit_sign, beam_fwhm_arcmin=beam_fwhm_arcmin)
            for z0_sign, zcrit_sign in SIGN_QUADRANTS
        ]

        def score_one(idx):
            return idx, score_fixed_circular_template(patches[idx], kernels)

    elif method == "centered_disc":
        kernel = make_centered_disc_kernel(theta_deg)

        def score_one(idx):
            return idx, score_fixed_centered(patches[idx], kernel)

    else:
        raise ValueError(f"Unknown method: {method}")

    scores = np.zeros(int(patches.shape[0]), dtype=np.float32)
    progress = p3.ProgressPrinter(len(scores), f"Per-radius {method} theta={theta_deg:g} ({workers} threads)")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(score_one, idx) for idx in range(len(scores))]
        for future in as_completed(futures):
            idx, score = future.result()
            scores[idx] = score
            completed += 1
            if completed % 500 == 0 or completed == len(scores):
                progress.update(completed)
    return scores


def load_global_rows(report_path):
    with Path(report_path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    lookup = {}
    for row in report["rows"]:
        key = (row["method"], float(row["amplitude"]), float(row["theta_crit_deg"]))
        lookup[key] = row
    thresholds = report.get("thresholds", {})
    return report, lookup, thresholds


def summarize_method(method, theta_scores, labels, amp_idx, theta_idx, amp_grid, theta_grid, fpr_target, global_lookup):
    rows = []
    thresholds = {}
    neg_mask = labels == 0
    familywise_active = np.zeros(int(neg_mask.sum()), dtype=bool)

    for ti, theta in enumerate(theta_grid):
        scores = theta_scores[float(theta)]
        threshold, neg_fp, neg_fpr = threshold_from_negatives(scores, labels, fpr_target)
        thresholds[str(theta)] = {
            "threshold": float(threshold),
            "negative_fp": int(neg_fp),
            "negative_fpr": float(neg_fpr),
        }
        familywise_active |= scores[neg_mask] > threshold

        theta_pos = (labels == 1) & (theta_idx == ti)
        theta_k = int(np.sum(scores[theta_pos] > threshold))
        theta_n = int(theta_pos.sum())
        theta_ci = exact_ci(theta_k, theta_n)
        rows.append(
            {
                "method": method,
                "amplitude": "all",
                "theta_crit_deg": float(theta),
                "num_positive": theta_n,
                "detected": theta_k,
                "p_det": float(theta_k / max(theta_n, 1)),
                "ci95_low": theta_ci[0],
                "ci95_high": theta_ci[1],
                "threshold": float(threshold),
                "negative_fp": int(neg_fp),
                "negative_fpr": float(neg_fpr),
                "global_p_det": None,
                "delta_vs_global": None,
            }
        )

        for ai, amp in enumerate(amp_grid):
            cell = (labels == 1) & (amp_idx == ai) & (theta_idx == ti)
            n = int(cell.sum())
            k = int(np.sum(scores[cell] > threshold))
            low, high = exact_ci(k, n)
            global_row = global_lookup.get((method, float(amp), float(theta)))
            global_p = None if global_row is None else float(global_row["p_det"])
            rows.append(
                {
                    "method": method,
                    "amplitude": float(amp),
                    "theta_crit_deg": float(theta),
                    "num_positive": n,
                    "detected": k,
                    "p_det": float(k / max(n, 1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "threshold": float(threshold),
                    "negative_fp": int(neg_fp),
                    "negative_fpr": float(neg_fpr),
                    "global_p_det": global_p,
                    "delta_vs_global": None if global_p is None else float(k / max(n, 1) - global_p),
                }
            )

    familywise_fp = int(familywise_active.sum())
    familywise_fpr = float(familywise_fp / max(int(neg_mask.sum()), 1))
    return rows, thresholds, {"negative_fp_any_radius": familywise_fp, "negative_fpr_any_radius": familywise_fpr}


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
        "global_p_det",
        "delta_vs_global",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, report):
    lines = ["# Per-Radius Threshold Calibration", ""]
    lines.append(f"Dataset: `{report['data_h5']}`")
    lines.append(f"FPR target per radius: `{report['fpr_target']}`")
    lines.append("")
    lines.append(
        "U-Net rows are intentionally absent: the current ML checkpoints emit a single scalar patch score, "
        "not a theta-conditioned scan score. Applying per-radius thresholds to them would be a false comparison."
    )
    lines.append("")
    lines.append("## Per-Radius Summary")
    lines.append("")
    lines.append("| method | theta_deg | threshold | neg FP | neg FPR | P_det all A | delta vs global mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in METHODS:
        method_rows = [row for row in report["rows"] if row["method"] == method and row["amplitude"] == "all"]
        for row in method_rows:
            cell_rows = [
                r
                for r in report["rows"]
                if r["method"] == method and r["theta_crit_deg"] == row["theta_crit_deg"] and r["amplitude"] != "all"
            ]
            deltas = [float(r["delta_vs_global"]) for r in cell_rows if r["delta_vs_global"] is not None]
            delta_mean = float(np.mean(deltas)) if deltas else float("nan")
            lines.append(
                f"| {method} | {row['theta_crit_deg']:.1f} | {row['threshold']:.6f} | "
                f"{row['negative_fp']} | {row['negative_fpr']:.4f} | {row['p_det']:.3f} | {delta_mean:+.3f} |"
            )
    lines.append("")
    lines.append("## Family-Wise FPR If All Radius Thresholds Are OR-Scanned")
    lines.append("")
    lines.append("| method | neg FP any radius | neg FPR any radius |")
    lines.append("|---|---:|---:|")
    for method, row in report["familywise"].items():
        lines.append(f"| {method} | {row['negative_fp_any_radius']} | {row['negative_fpr_any_radius']:.4f} |")
    lines.append("")
    lines.append(
        "Interpretation: per-radius thresholds are valid for a fixed-radius operating mode. "
        "If all radii are OR-scanned on the same sky, the family-wise FPR must be controlled separately."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    data_h5 = Path(args.data_h5).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_report, global_lookup, global_thresholds = load_global_rows(args.sensitivity_report)
    with h5py.File(data_h5, "r") as h5:
        patches = np.asarray(h5["patches"][:], dtype=np.float32)
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amp_grid = [float(x) for x in json.loads(h5["summary"].attrs["amplitude_grid"])]
        theta_grid = [float(x) for x in json.loads(h5["summary"].attrs["theta_grid_deg"])]
        beam_fwhm_arcmin = float(h5["summary"].attrs.get("beam_fwhm_arcmin", 0.0))

    report = {
        "data_h5": str(data_h5),
        "sensitivity_report": str(Path(args.sensitivity_report).resolve()),
        "fpr_target": float(args.fpr_target),
        "beam_fwhm_arcmin": float(beam_fwhm_arcmin),
        "amplitude_grid": amp_grid,
        "theta_grid_deg": theta_grid,
        "global_thresholds": {method: global_thresholds.get(method, {}) for method in METHODS},
        "ml_applicability": "not_applicable_current_scalar_unet_no_theta_conditioned_score",
        "rows": [],
        "thresholds": {},
        "familywise": {},
        "method_metadata": {method: method_metadata(method) for method in METHODS},
    }

    for method in METHODS:
        theta_scores = {}
        for theta in theta_grid:
            theta_scores[float(theta)] = score_theta(
                patches,
                method=method,
                theta_deg=float(theta),
                beam_fwhm_arcmin=beam_fwhm_arcmin,
                workers=args.classical_workers,
            )
        rows, thresholds, familywise = summarize_method(
            method,
            theta_scores,
            labels,
            amp_idx,
            theta_idx,
            amp_grid,
            theta_grid,
            args.fpr_target,
            global_lookup,
        )
        report["rows"].extend(rows)
        report["thresholds"][method] = thresholds
        report["familywise"][method] = familywise

    json_path = output_dir / "per_radius_threshold.json"
    csv_path = output_dir / "per_radius_threshold.csv"
    md_path = output_dir / "per_radius_threshold.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, report["rows"])
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
