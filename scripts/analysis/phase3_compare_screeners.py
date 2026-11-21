"""
Compare the ML screener against classical baselines on the same Phase 3 split.

This is the immediate answer to the Feeney-facing question: does the U-Net add
screening value over a circular-template screen and a trivial centered-disc
shortcut under the same operating-point rule?
"""

from __future__ import annotations
# Bootstrap organized script folders for direct CLI execution.
import sys as _cmb_sys
from pathlib import Path as _cmb_Path

_cmb_scripts_root = _cmb_Path(__file__).resolve().parents[1]
_cmb_paths = [_cmb_scripts_root]
_cmb_paths.extend(
    sorted(
        _cmb_path
        for _cmb_path in _cmb_scripts_root.iterdir()
        if _cmb_path.is_dir() and _cmb_path.name != "__pycache__"
    )
)
for _cmb_path in reversed(_cmb_paths):
    _cmb_path_text = str(_cmb_path)
    if _cmb_path_text not in _cmb_sys.path:
        _cmb_sys.path.insert(0, _cmb_path_text)
del _cmb_Path, _cmb_sys, _cmb_scripts_root, _cmb_paths, _cmb_path, _cmb_path_text

import argparse
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = (
    "image_precision",
    "image_recall",
    "image_f1",
    "image_false_positive_rate",
    "hard_dice_pos",
    "iou_pos",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ML and classical candidate screeners on one audited split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ml-eval-dir", type=str, required=True)
    parser.add_argument("--classical-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-md", type=str, default="")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(row, key):
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def summarize_metrics(metrics):
    return {key: safe_float(metrics, key) for key in METRIC_KEYS if key in metrics}


def metric_delta(lhs, rhs):
    delta = {}
    for key in METRIC_KEYS:
        if lhs.get(key) is None or rhs.get(key) is None:
            continue
        delta[key] = float(lhs[key] - rhs[key])
    return delta


def validate_same_split(ml_summary, classical_summary):
    failures = []
    if int(ml_summary["num_samples"]) != int(classical_summary["num_samples"]):
        failures.append(
            f"num_samples mismatch: ML={ml_summary['num_samples']} "
            f"classical={classical_summary['num_samples']}"
        )
    if ml_summary["split"] != classical_summary["split"]:
        failures.append(f"split mismatch: ML={ml_summary['split']} classical={classical_summary['split']}")
    ml_run = str(Path(ml_summary["run_dir"]).resolve())
    classical_run = str(Path(classical_summary["run_dir"]).resolve())
    if ml_run != classical_run:
        failures.append(f"run_dir mismatch: ML={ml_run} classical={classical_run}")
    return failures


def row_to_markdown(name, metrics):
    values = []
    for key in METRIC_KEYS:
        value = metrics.get(key)
        values.append("" if value is None else f"{value:.4f}")
    return "| " + " | ".join([name] + values) + " |"


def write_markdown(path, report):
    headers = ["method", *METRIC_KEYS]
    lines = [
        "# Phase 3 Screener Comparison",
        "",
        f"- ML eval dir: `{report['ml_eval_dir']}`",
        f"- Classical dir: `{report['classical_dir']}`",
        f"- Split: `{report['split']}`",
        f"- Samples: `{report['num_samples']}`",
        f"- Operating rule: `{report['operating_point_rule']}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        row_to_markdown("ml_unet", report["ml_metrics"]),
    ]
    for method_name, method_report in report["classical_methods"].items():
        lines.append(row_to_markdown(method_name, method_report["metrics"]))

    lines.extend(["", "## Deltas vs ML", ""])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for method_name, method_report in report["classical_methods"].items():
        lines.append(row_to_markdown(f"ml_minus_{method_name}", method_report["ml_minus_classical"]))

    if report["validation_failures"]:
        lines.extend(["", "## Validation Failures", ""])
        for failure in report["validation_failures"]:
            lines.append(f"- {failure}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_comparison(ml_eval_dir, classical_dir):
    ml_eval_dir = Path(ml_eval_dir).resolve()
    classical_dir = Path(classical_dir).resolve()
    ml_summary = load_json(ml_eval_dir / "evaluation_summary.json")
    classical_summary = load_json(classical_dir / "summary.json")

    ml_metrics = summarize_metrics(ml_summary["selected_threshold_metrics"])
    validation_failures = validate_same_split(ml_summary, classical_summary)

    method_reports = {}
    for method_name, method_summary in sorted(classical_summary["methods"].items()):
        classical_metrics = summarize_metrics(method_summary["selected_threshold_metrics"])
        method_reports[method_name] = {
            "selected_threshold": float(method_summary["selected_threshold"]),
            "operating_point": method_summary["operating_point"],
            "metrics": classical_metrics,
            "ml_minus_classical": metric_delta(ml_metrics, classical_metrics),
        }

    report = {
        "status": "pass" if not validation_failures else "fail",
        "ml_eval_dir": str(ml_eval_dir),
        "classical_dir": str(classical_dir),
        "run_dir": str(Path(ml_summary["run_dir"]).resolve()),
        "split": ml_summary["split"],
        "num_samples": int(ml_summary["num_samples"]),
        "operating_point_rule": ml_summary["operating_point"]["rule"],
        "ml_selected_threshold": float(ml_summary["selected_threshold"]),
        "ml_metrics": ml_metrics,
        "classical_methods": method_reports,
        "validation_failures": validation_failures,
    }
    return report


def main():
    args = parse_args()
    report = run_comparison(args.ml_eval_dir, args.classical_dir)
    output_json = Path(args.output_json).resolve() if args.output_json else Path(args.ml_eval_dir).resolve() / "screener_comparison.json"
    output_md = Path(args.output_md).resolve() if args.output_md else Path(args.ml_eval_dir).resolve() / "screener_comparison.md"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_markdown(output_md, report)

    print(json.dumps({key: value for key, value in report.items() if key != "classical_methods"}, indent=2))
    for method_name, method_report in report["classical_methods"].items():
        delta = method_report["ml_minus_classical"]
        print(
            f"{method_name}: "
            f"delta_f1={delta.get('image_f1', np.nan):+.4f} "
            f"delta_recall={delta.get('image_recall', np.nan):+.4f} "
            f"delta_fpr={delta.get('image_false_positive_rate', np.nan):+.4f} "
            f"delta_dice={delta.get('hard_dice_pos', np.nan):+.4f}"
        )
    print(f"Comparison JSON: {output_json}")
    print(f"Comparison markdown: {output_md}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
