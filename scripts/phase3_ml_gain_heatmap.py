"""
Per-cell ML gain heatmap over the circular-template baseline.

Uses the score arrays and thresholds from phase3_sensitivity_curve.py.  For
each (A, theta_c) cell, it bootstraps paired positive samples and estimates
delta = P_det(ML) - P_det(circular_template_screen).  The default analysis uses one
pre-specified ML method to avoid testing the post-hoc best branch in each cell.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase3_method_registry import CIRCULAR_TEMPLATE_SCREEN, method_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "sensitivity_curve_v1" / "sensitivity_report.json"
DEFAULT_SCORES = PROJECT_ROOT / "runs" / "phase3_unet" / "sensitivity_curve_v1" / "sensitivity_scores.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "ml_gain_heatmap_v1"
ML_METHODS = ("original_v4", "boundary_v4", "v5_consensus", "v6_aux_only", "v6_hard_w15")
ANALYSIS_MODES = ("preselected", "best_per_cell")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap per-cell ML sensitivity gain over the circular-template screen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--scores-npz", type=str, default=str(DEFAULT_SCORES))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--ml-method", action="append", default=[], help="ML method to include. Can be repeated.")
    parser.add_argument(
        "--analysis-mode",
        type=str,
        default="preselected",
        choices=ANALYSIS_MODES,
        help=(
            "Use one pre-specified ML method for significance testing, or reproduce the "
            "historical post-hoc best-per-cell analysis."
        ),
    )
    parser.add_argument(
        "--primary-method",
        type=str,
        default="v6_aux_only",
        help="Pre-specified ML method used when --analysis-mode=preselected.",
    )
    return parser.parse_args()


def ci(values):
    values = np.asarray(values, dtype=np.float64)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_sign_pvalue(values):
    """Two-sided bootstrap sign p-value for a paired delta distribution."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    le_zero = float(np.mean(values <= 0.0))
    ge_zero = float(np.mean(values >= 0.0))
    return min(1.0, max(1.0 / (values.size + 1.0), 2.0 * min(le_zero, ge_zero)))


def holm_adjust(p_values):
    """Return Holm-Bonferroni adjusted p-values."""

    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def benjamini_hochberg_adjust(p_values):
    """Return Benjamini-Hochberg FDR q-values."""

    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 1.0
    m = len(p_values)
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        value = min(running, float(p_values[idx]) * m / max(rank, 1))
        running = value
        adjusted[idx] = value
    return adjusted


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_delta(model_hits, matched_hits, rng, resamples):
    n = int(model_hits.size)
    values = np.empty(resamples, dtype=np.float64)
    for idx in range(resamples):
        sample = rng.integers(0, n, size=n)
        values[idx] = float(model_hits[sample].mean() - matched_hits[sample].mean())
    return values


def write_csv(path, rows):
    columns = [
        "amplitude",
        "theta_crit_deg",
        "best_model",
        "matched_p_det",
        "best_ml_p_det",
        "delta",
        "ci95_low",
        "ci95_high",
        "p_value_raw",
        "p_value_holm",
        "q_value_bh",
        "significant_holm_005",
        "significant_fdr_005",
        "significant",
        "direction",
        "winner_bootstrap_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def plot_heatmap(path, rows, amplitude_grid, theta_grid):
    matrix = np.full((len(theta_grid), len(amplitude_grid)), np.nan, dtype=np.float64)
    labels = [["" for _ in amplitude_grid] for _ in theta_grid]
    row_by_cell = {(row["amplitude"], row["theta_crit_deg"]): row for row in rows}
    for ti, theta in enumerate(theta_grid):
        for ai, amp in enumerate(amplitude_grid):
            row = row_by_cell[(amp, theta)]
            if row["significant_holm_005"] and row["direction"] == "ml_better":
                matrix[ti, ai] = row["delta"]
                labels[ti][ai] = row["best_model"].replace("_", "\n")

    fig, ax = plt.subplots(figsize=(1.25 * len(amplitude_grid) + 3.0, 1.0 * len(theta_grid) + 2.5))
    finite = matrix[np.isfinite(matrix)]
    vmax = max(float(np.max(np.abs(finite))) if finite.size else 0.1, 0.1)
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(amplitude_grid)))
    ax.set_xticklabels([f"{amp:.0e}" for amp in amplitude_grid], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(theta_grid)))
    ax.set_yticklabels([f"{theta:g}" for theta in theta_grid])
    ax.set_xlabel(r"$A=|z_0|=|z_{\rm crit}|$")
    ax.set_ylabel(r"$\theta_c$ (deg)")
    ax.set_title(r"Significant ML Gain: $P_{\rm det}^{ML} - P_{\rm det}^{circular}$")
    for ti in range(len(theta_grid)):
        for ai in range(len(amplitude_grid)):
            row = row_by_cell[(amplitude_grid[ai], theta_grid[ti])]
            if np.isfinite(matrix[ti, ai]):
                ax.text(ai, ti, f"{row['delta']:.2f}\n{labels[ti][ai]}", ha="center", va="center", fontsize=7, color="white")
            else:
                ax.text(ai, ti, "n.s.", ha="center", va="center", fontsize=7, color="#333333")
    fig.colorbar(im, ax=ax, label="delta P_det")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, report):
    model_column = "selected_model" if report["analysis_mode"] == "preselected" else "best_model"
    lines = [
        "# ML Gain Heatmap",
        "",
        f"- Sensitivity report: `{report['sensitivity_report']}`",
        f"- Scores: `{report['scores_npz']}`",
        f"- Bootstrap resamples: `{report['bootstrap_resamples']}`",
        f"- Analysis mode: `{report['analysis_mode']}`",
        f"- Primary method: `{report['primary_method']}`",
        "",
        f"| A | theta_deg | {model_column} | circular | ML | delta | 95% CI | Holm p | BH q | significant |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['amplitude']:.3g} | {row['theta_crit_deg']:.1f} | {row['best_model']} | "
            f"{row['matched_p_det']:.3f} | {row['best_ml_p_det']:.3f} | {row['delta']:.3f} | "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | "
            f"{row['p_value_holm']:.3g} | {row['q_value_bh']:.3g} | {row['significant_holm_005']} |"
        )
    lines.extend(["", "## Winner Counts", ""])
    for method, count in sorted(report["winner_counts"].items()):
        lines.append(f"- `{method}`: `{count}` cells")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_report = Path(args.sensitivity_report).resolve()
    scores_npz = Path(args.scores_npz).resolve()
    report = load_json(sensitivity_report)
    data_h5 = Path(report["data_h5"])
    thresholds = {name: float(row["threshold"]) for name, row in report["thresholds"].items()}
    amplitude_grid = [float(x) for x in report["amplitude_grid"]]
    theta_grid = [float(x) for x in report["theta_grid_deg"]]
    ml_methods = tuple(args.ml_method) if args.ml_method else ML_METHODS
    if args.analysis_mode == "preselected" and args.primary_method not in ml_methods:
        raise ValueError("--primary-method must be included in --ml-method when --analysis-mode=preselected.")

    score_prefix = "score__"
    with np.load(scores_npz) as loaded:
        scores = {
            key[len(score_prefix):]: np.asarray(loaded[key], dtype=np.float64)
            for key in loaded.files
            if key.startswith(score_prefix)
        }
    baseline_method = CIRCULAR_TEMPLATE_SCREEN if CIRCULAR_TEMPLATE_SCREEN in scores else "matched_template"
    missing = [baseline_method, *ml_methods]
    missing = [name for name in missing if name not in scores]
    if missing:
        raise KeyError(f"Missing score arrays: {missing}")

    with h5py.File(data_h5, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        amplitude_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)

    hits = {name: scores[name] > thresholds[name] for name in (baseline_method, *ml_methods)}
    rng = np.random.default_rng(args.seed)
    rows = []
    winner_counts = {name: 0 for name in ml_methods}
    significant_counts = {name: 0 for name in ml_methods}
    for amp_i, amp in enumerate(amplitude_grid):
        for theta_i, theta in enumerate(theta_grid):
            cell = (labels == 1) & (amplitude_idx == amp_i) & (theta_idx == theta_i)
            idx = np.flatnonzero(cell)
            if idx.size == 0:
                continue
            matched_hits = hits[baseline_method][idx].astype(np.float64)
            matched_p = float(matched_hits.mean())

            per_model = []
            bootstrap_values = {}
            for method in ml_methods:
                model_hits = hits[method][idx].astype(np.float64)
                delta_values = bootstrap_delta(model_hits, matched_hits, rng, args.bootstrap_resamples)
                low, high = ci(delta_values)
                delta = float(model_hits.mean() - matched_p)
                per_model.append((delta, method, float(model_hits.mean()), low, high))
                bootstrap_values[method] = delta_values
            per_model.sort(reverse=True, key=lambda item: (item[0], item[1]))
            historical_delta, historical_best_model, historical_best_p, _, _ = per_model[0]
            winner_counts[historical_best_model] += 1
            if args.analysis_mode == "best_per_cell":
                delta, selected_model, selected_p, low, high = per_model[0]
            else:
                selected = [item for item in per_model if item[1] == args.primary_method]
                if not selected:
                    raise RuntimeError(f"Primary method not found in per-cell results: {args.primary_method}")
                delta, selected_model, selected_p, low, high = selected[0]
            raw_p = bootstrap_sign_pvalue(bootstrap_values[selected_model])
            if low > 0.0:
                significant_counts[selected_model] += 1

            stacked = np.stack([bootstrap_values[method] for method in ml_methods], axis=0)
            bootstrap_winners = np.argmax(stacked, axis=0)
            winner_fraction = float(np.mean([ml_methods[i] == selected_model for i in bootstrap_winners]))
            rows.append(
                {
                    "amplitude": float(amp),
                    "theta_crit_deg": float(theta),
                    "best_model": selected_model,
                    "historical_best_model": historical_best_model,
                    "historical_best_delta": historical_delta,
                    "historical_best_p_det": historical_best_p,
                    "matched_p_det": matched_p,
                    "best_ml_p_det": selected_p,
                    "delta": delta,
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value_raw": raw_p,
                    "p_value_holm": float("nan"),
                    "q_value_bh": float("nan"),
                    "significant_holm_005": False,
                    "significant_fdr_005": False,
                    "significant": bool(low > 0.0 or high < 0.0),
                    "direction": "ml_better" if low > 0.0 else ("ml_lower" if high < 0.0 else "not_significant"),
                    "winner_bootstrap_fraction": winner_fraction,
                }
            )

    if rows:
        raw_p_values = np.asarray([row["p_value_raw"] for row in rows], dtype=np.float64)
        holm = holm_adjust(raw_p_values)
        fdr = benjamini_hochberg_adjust(raw_p_values)
        for row, holm_i, fdr_i in zip(rows, holm, fdr):
            row["p_value_holm"] = float(holm_i)
            row["q_value_bh"] = float(fdr_i)
            row["significant_holm_005"] = bool(holm_i <= 0.05)
            row["significant_fdr_005"] = bool(fdr_i <= 0.05)

    csv_path = output_dir / "ml_gain_cells.csv"
    png_path = output_dir / "ml_gain_heatmap.png"
    json_path = output_dir / "ml_gain_heatmap.json"
    md_path = output_dir / "ml_gain_heatmap.md"
    write_csv(csv_path, rows)
    plot_heatmap(png_path, rows, amplitude_grid, theta_grid)
    out = {
        "sensitivity_report": str(sensitivity_report),
        "scores_npz": str(scores_npz),
        "data_h5": str(data_h5.resolve()),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "analysis_mode": args.analysis_mode,
        "baseline_method": baseline_method,
        "baseline_method_metadata": method_metadata(baseline_method),
        "primary_method": args.primary_method if args.analysis_mode == "preselected" else None,
        "significance_note": (
            "pre-specified method CI; no post-hoc model selection"
            if args.analysis_mode == "preselected"
            else "historical post-hoc best-per-cell CI; does not control model-selection error"
        ),
        "ml_methods": list(ml_methods),
        "winner_counts": winner_counts,
        "significant_winner_counts": significant_counts,
        "multiple_testing": {
            "family": "amplitude_theta_cells",
            "raw_p_value": "two_sided_bootstrap_sign_test_on_paired_detection_delta",
            "holm_alpha_005_count": int(sum(row["significant_holm_005"] for row in rows)),
            "bh_fdr_005_count": int(sum(row["significant_fdr_005"] for row in rows)),
            "note": (
                "Holm and Benjamini-Hochberg corrections are applied across cells. "
                "Permutation/max-statistic sensitivity is still a separate appendix."
            ),
        },
        "rows": rows,
        "artifacts": {"csv": str(csv_path), "heatmap_png": str(png_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_markdown(md_path, out)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"PNG:  {png_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
