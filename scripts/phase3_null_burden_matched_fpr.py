"""
Apply sensitivity-calibrated thresholds to real-map null controls.

This is the matched-FPR null-burden check: every method uses the threshold
tau_{FPR=target} learned from the independent sensitivity negatives, not its
historical validation threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import binomtest

import phase3_train_unet as p3
from phase3_sensitivity_curve import (
    DEFAULT_MODELS,
    parse_model_spec,
    score_classical_methods,
    score_ml_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NULL_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "null_controls_smica_mask090.h5"
DEFAULT_SENSITIVITY_REPORT = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_report.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_null_burden_matched_fpr"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score null controls at thresholds calibrated by phase3_sensitivity_curve.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--null-h5", type=str, default=str(DEFAULT_NULL_H5))
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_SENSITIVITY_REPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", action="append", default=[], help="Model as name:run_dir:checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--classical-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.classical_workers <= 0:
        raise ValueError("--classical-workers must be positive.")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def exact_ci(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def summarize_scores(scores, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    active = scores > float(threshold)
    k = int(active.sum())
    n = int(scores.size)
    return {
        "threshold": float(threshold),
        "num_samples": n,
        "false_positive_count": k,
        "false_positive_rate": float(k / max(n, 1)),
        "false_positive_rate_ci95": exact_ci(k, n),
        "score_mean": float(np.mean(scores)) if n else float("nan"),
        "score_max": float(np.max(scores)) if n else float("nan"),
        "active_score_median": float(np.median(scores[active])) if k else None,
        "active_score_p90": float(np.percentile(scores[active], 90)) if k else None,
    }


def load_null_shape(null_h5):
    with h5py.File(null_h5, "r") as h5:
        n = int(h5["patches"].shape[0])
        labels = np.asarray(h5["labels"][:], dtype=np.uint8) if "labels" in h5 else np.zeros(n, dtype=np.uint8)
    if labels.sum() != 0:
        raise RuntimeError(f"Null HDF5 contains {int(labels.sum())} positive labels; expected all zero.")
    return n


def write_markdown(path, report):
    lines = [
        "# Null Burden At Sensitivity-Matched FPR",
        "",
        f"- Null HDF5: `{report['null_h5']}`",
        f"- Sensitivity report: `{report['sensitivity_report']}`",
        f"- Sensitivity FPR target: `{report['sensitivity_fpr_target']}`",
        f"- Null samples: `{report['num_samples']}`",
        "",
        "| method | threshold | false positives | null FPR | 95% CI | score max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in report["methods"].items():
        ci = row["false_positive_rate_ci95"]
        lines.append(
            f"| {method} | {row['threshold']:.8g} | {row['false_positive_count']} / {row['num_samples']} | "
            f"{row['false_positive_rate']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {row['score_max']:.8g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    null_h5 = Path(args.null_h5).resolve()
    sensitivity_report = Path(args.sensitivity_report).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not null_h5.exists():
        raise FileNotFoundError(f"Null-control HDF5 not found: {null_h5}")
    if not sensitivity_report.exists():
        raise FileNotFoundError(f"Sensitivity report not found: {sensitivity_report}")

    sensitivity = load_json(sensitivity_report)
    thresholds = {name: float(row["threshold"]) for name, row in sensitivity["thresholds"].items()}
    n = load_null_shape(null_h5)
    methods = {}

    if not args.skip_classical:
        theta_grid_deg = tuple(float(x) for x in sensitivity["theta_grid_deg"])
        with h5py.File(sensitivity["data_h5"], "r") as h5:
            beam_fwhm_arcmin = float(h5["summary"].attrs["beam_fwhm_arcmin"])
        classical_scores = score_classical_methods(
            null_h5,
            theta_grid_deg=theta_grid_deg,
            classical_workers=args.classical_workers,
            beam_fwhm_arcmin=beam_fwhm_arcmin,
        )
        for method_name, scores in classical_scores.items():
            methods[method_name] = summarize_scores(scores, thresholds[method_name])

    if not args.skip_ml:
        device = p3.resolve_device(args.device)
        specs = [parse_model_spec(text) for text in (args.model or DEFAULT_MODELS)]
        for spec in specs:
            scores, _ = score_ml_model(spec, null_h5, args, device)
            methods[spec.name] = summarize_scores(scores, thresholds[spec.name])

    report = {
        "null_h5": str(null_h5),
        "sensitivity_report": str(sensitivity_report),
        "sensitivity_fpr_target": float(sensitivity["fpr_target"]),
        "num_samples": int(n),
        "threshold_source": "phase3_sensitivity_curve",
        "score_rule": "scores > sensitivity_threshold",
        "methods": methods,
    }
    json_path = output_dir / "null_burden_matched_fpr.json"
    md_path = output_dir / "null_burden_matched_fpr.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps(report, indent=2))
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
