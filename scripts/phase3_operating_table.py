"""
Build a historical operating table for ML branches and classical baselines.

Assumptions
-----------
* This is a legacy branch-triage utility retained for provenance.
* The mixed ``branch_score`` is an operational heuristic, not a discovery
  statistic, not a Bayesian evidence ratio, and not part of the active
  remediated-v1 headline claims.
* Current science claims should use the remediated sensitivity grid, deployment
  tile calibration, upper-limit calculator, and true classical-filter
  benchmarks instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROW_METRICS = (
    "selected_threshold",
    "synthetic_recall",
    "synthetic_fpr",
    "synthetic_precision",
    "synthetic_f1",
    "positive_dice",
    "contour_f1",
    "strong_edge_contour_f1",
    "smica5000_candidates",
    "smica5000_fpr",
    "template_delta_chi2_pos_median",
    "template_delta_chi2_neg_median",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a compact operating table across ML branches and classical screeners.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--original-eval-dir", type=str, required=True)
    parser.add_argument("--boundary-eval-dir", type=str, required=True)
    parser.add_argument("--classical-dir", type=str, required=True)
    parser.add_argument("--original-null-summary", type=str, required=True)
    parser.add_argument("--boundary-null-summary", type=str, required=True)
    parser.add_argument("--classical-null-summary", type=str, default="")
    parser.add_argument("--null-failure-comparison", type=str, default="")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_or_none(row, key):
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def load_eval_row(label, eval_dir, null_summary_path):
    eval_dir = Path(eval_dir)
    summary = load_json(eval_dir / "evaluation_summary.json")
    selected = summary["selected_threshold_metrics"]
    boundary = load_json(eval_dir / "boundary_analysis_contour.json")
    template_path = eval_dir / "template_fit_summary.json"
    template = load_json(template_path)["metrics"] if template_path.exists() else {}
    null_summary = load_json(null_summary_path)
    null_metrics = null_summary["frozen_threshold_nearest_grid_metrics"]
    strong = boundary["by_edge_strength"].get("strong_|zcrit|>=3e-5", {})
    return {
        "method": label,
        "method_family": "ml",
        "selected_threshold": float(summary["selected_threshold"]),
        "synthetic_recall": metric_or_none(selected, "image_recall"),
        "synthetic_fpr": metric_or_none(selected, "image_false_positive_rate"),
        "synthetic_precision": metric_or_none(selected, "image_precision"),
        "synthetic_f1": metric_or_none(selected, "image_f1"),
        "positive_dice": metric_or_none(selected, "hard_dice_pos"),
        "contour_f1": metric_or_none(boundary["overall"], "contour_f1_mean"),
        "strong_edge_contour_f1": metric_or_none(strong, "contour_f1_mean"),
        "smica5000_candidates": int(null_metrics["false_positive_count"]),
        "smica5000_fpr": float(null_metrics["false_positive_rate"]),
        "template_delta_chi2_pos_median": metric_or_none(template, "truth_positive_delta_chi2_median"),
        "template_delta_chi2_neg_median": metric_or_none(template, "truth_negative_delta_chi2_median"),
        "eval_dir": str(eval_dir.resolve()),
        "null_summary": str(Path(null_summary_path).resolve()),
    }


def load_classical_rows(classical_dir, classical_null_summary=""):
    classical_dir = Path(classical_dir)
    summary = load_json(classical_dir / "summary.json")
    null_methods = {}
    if classical_null_summary:
        null_methods = load_json(classical_null_summary).get("methods", {})
    rows = []
    for method, payload in sorted(summary["methods"].items()):
        selected = payload["selected_threshold_metrics"]
        null_metrics = null_methods.get(method, {})
        boundary_path = Path(payload["artifacts_dir"]) / "boundary_analysis_contour.json"
        boundary = load_json(boundary_path) if boundary_path.exists() else {}
        strong = boundary.get("by_edge_strength", {}).get("strong_|zcrit|>=3e-5", {})
        rows.append(
            {
                "method": method,
                "method_family": "classical",
                "selected_threshold": float(payload["selected_threshold"]),
                "synthetic_recall": metric_or_none(selected, "image_recall"),
                "synthetic_fpr": metric_or_none(selected, "image_false_positive_rate"),
                "synthetic_precision": metric_or_none(selected, "image_precision"),
                "synthetic_f1": metric_or_none(selected, "image_f1"),
                "positive_dice": metric_or_none(selected, "hard_dice_pos"),
                "contour_f1": metric_or_none(boundary.get("overall", {}), "contour_f1_mean"),
                "strong_edge_contour_f1": metric_or_none(strong, "contour_f1_mean"),
                "smica5000_candidates": (
                    int(null_metrics["false_positive_count"]) if "false_positive_count" in null_metrics else None
                ),
                "smica5000_fpr": (
                    float(null_metrics["false_positive_rate"]) if "false_positive_rate" in null_metrics else None
                ),
                "template_delta_chi2_pos_median": None,
                "template_delta_chi2_neg_median": None,
                "eval_dir": str(Path(payload["artifacts_dir"]).resolve()),
                "null_summary": str(Path(classical_null_summary).resolve()) if classical_null_summary else None,
            }
        )
    return rows


def score_row(row):
    """
    A transparent checkpoint/branch score, not a claim of optimality.

    The terms intentionally reward synthetic screening, contour recovery, and
    real-null cleanliness. The real-null penalty is candidate-count based so a
    branch with clustered false positives still pays operational cost.
    """
    if row["method_family"] != "ml":
        return None
    recall = row["synthetic_recall"] or 0.0
    strong_contour = row["strong_edge_contour_f1"] or 0.0
    dice = row["positive_dice"] or 0.0
    null_penalty = min((row["smica5000_candidates"] or 0) / 500.0, 1.0)
    return float(0.35 * recall + 0.35 * strong_contour + 0.15 * dice - 0.15 * null_penalty)


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if abs(value) < 1e-3 and value != 0.0:
        return f"{value:.3e}"
    return f"{value:.4f}"


def write_markdown(path, report):
    lines = [
        "# Phase 3 Operating Table",
        "",
        "This table compares broad screening, Feeney-relevant morphology, real-SMICA null burden, and template-fit handoff behavior under fixed operating artifacts.",
        "",
        "| method | family | branch_score | " + " | ".join(ROW_METRICS) + " |",
        "| --- | --- | --- | " + " | ".join(["---"] * len(ROW_METRICS)) + " |",
    ]
    for row in report["rows"]:
        values = [format_value(row.get(key)) for key in ROW_METRICS]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    row["method_family"],
                    format_value(row.get("branch_score")),
                    *values,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `branch_score` is only applied to ML branches and is a transparent operating score: `0.35*recall + 0.35*strong_edge_contour_f1 + 0.15*positive_dice - 0.15*min(smica5000_candidates/500,1)`.",
            "- Classical rows currently include synthetic screening and Dice only; contour/null/handoff cells are blank unless separate classical artifacts are generated.",
            "- The score is not a Bayesian evidence ratio or discovery statistic.",
        ]
    )
    if report.get("null_failure_comparison"):
        overlap = report["null_failure_comparison"]["overlap"]
        lines.extend(
            [
                "",
                "## Null-Failure Complementarity",
                "",
                f"- Shared null-failure samples: `{overlap['shared_sample_count']}`.",
                f"- Cross-model pairs within 10 deg: `{overlap['cross_model_pairs_within_10deg']}`.",
                f"- First-only candidates: `{overlap['first_only_count']}`.",
                f"- Second-only candidates: `{overlap['second_only_count']}`.",
            ]
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = [
        load_eval_row("original_v4", args.original_eval_dir, args.original_null_summary),
        load_eval_row("boundary_w4_last", args.boundary_eval_dir, args.boundary_null_summary),
    ]
    rows.extend(load_classical_rows(args.classical_dir, args.classical_null_summary))
    for row in rows:
        row["branch_score"] = score_row(row)

    report = {
        "rows": rows,
        "score_note": (
            "branch_score = 0.35*synthetic_recall + 0.35*strong_edge_contour_f1 "
            "+ 0.15*positive_dice - 0.15*min(smica5000_candidates/500,1)."
        ),
        "null_failure_comparison": load_json(args.null_failure_comparison) if args.null_failure_comparison else None,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_markdown(args.output_md, report)
    print(json.dumps({"rows": rows, "score_note": report["score_note"]}, indent=2))
    print(f"Operating table JSON: {args.output_json}")
    print(f"Operating table MD:   {args.output_md}")


if __name__ == "__main__":
    main()
