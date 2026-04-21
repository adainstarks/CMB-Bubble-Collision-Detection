"""
Historical 3D weak-family recall breakdown on the stratified external set.

Rows are amplitude bin x radius bin x edge-smoothing bin, reported for each ML
checkpoint and classical baseline at the matched-FPR thresholds already selected
by the stratified external evaluators.

This script is retained for the pre-remediation stratified validation products.
The remediated-v1 sensitivity grid supersedes it for current claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import binomtest

import phase3_train_unet as p3
from phase3_eval_stratified_external import DEFAULT_MODELS, collect_scores, load_all_labels, parse_model_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_H5 = PROJECT_ROOT / "data" / "validation_stratified_v1" / "validation_data.h5"
DEFAULT_ML_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "stratified_external_eval_v1" / "stratified_eval_report.json"
DEFAULT_CLASSICAL_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "stratified_external_classical_v1" / "classical_stratified_eval_report.json"
DEFAULT_CLASSICAL_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "stratified_external_classical_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "weak_family_breakdown_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-cell recall breakdown over amplitude x radius x edge-smoothing bins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-h5", type=str, default=str(DEFAULT_DATA_H5))
    parser.add_argument("--ml-report", type=str, default=str(DEFAULT_ML_REPORT))
    parser.add_argument("--classical-report", type=str, default=str(DEFAULT_CLASSICAL_REPORT))
    parser.add_argument("--classical-dir", type=str, default=str(DEFAULT_CLASSICAL_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", action="append", default=[], help="Model spec as in phase3_eval_stratified_external.py.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reuse-scores", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def exact_ci(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def model_thresholds(report_path):
    report = load_json(report_path)
    return {row["name"]: float(row["matched_threshold"]) for row in report["models"]}


def load_stratification(data_h5):
    with h5py.File(data_h5, "r") as h5:
        return {
            "labels": np.asarray(h5["labels"][:], dtype=np.uint8),
            "z0_amp_bin": np.asarray(h5["stratification"]["z0_amp_bin"][:], dtype=np.int16),
            "theta_bin": np.asarray(h5["stratification"]["theta_bin"][:], dtype=np.int16),
            "edge_sigma_bin": np.asarray(h5["stratification"]["edge_sigma_bin"][:], dtype=np.int16),
            "bin_schema": json.loads(h5["summary"].attrs["bin_schema"]),
        }


def load_or_score_ml(spec, data_h5, output_dir, args, device):
    score_dir = output_dir / "score_cache"
    score_dir.mkdir(parents=True, exist_ok=True)
    path = score_dir / f"{spec.name}_scores.npz"
    if args.reuse_scores and path.exists():
        with np.load(path) as loaded:
            return np.asarray(loaded["scores"], dtype=np.float32)
    indices, _ = load_all_labels(data_h5)
    scored = collect_scores(spec, data_h5, indices, args, device)
    np.savez_compressed(path, scores=scored["scores"], labels=scored["labels"])
    return scored["scores"]


def load_classical_scores(classical_dir, method):
    path = Path(classical_dir) / f"{method}_scores_masks.npz"
    with np.load(path) as loaded:
        return np.asarray(loaded["scores"], dtype=np.float32)


def format_bin(schema, key, idx):
    bins = schema[key]
    row = bins[int(idx)]
    return f"{row['name']} [{row['low']}, {row['high']}]"


def summarize_cells(method, scores, threshold, strat):
    labels = strat["labels"]
    active = scores >= float(threshold)
    rows = []
    for amp_idx in sorted(int(x) for x in np.unique(strat["z0_amp_bin"][labels == 1]) if int(x) >= 0):
        for theta_idx in sorted(int(x) for x in np.unique(strat["theta_bin"][labels == 1]) if int(x) >= 0):
            for edge_idx in sorted(int(x) for x in np.unique(strat["edge_sigma_bin"][labels == 1]) if int(x) >= 0):
                mask = (
                    (labels == 1)
                    & (strat["z0_amp_bin"] == amp_idx)
                    & (strat["theta_bin"] == theta_idx)
                    & (strat["edge_sigma_bin"] == edge_idx)
                )
                n = int(mask.sum())
                if n == 0:
                    continue
                k = int(active[mask].sum())
                ci = exact_ci(k, n)
                rows.append(
                    {
                        "method": method,
                        "threshold": float(threshold),
                        "z0_amp_bin": amp_idx,
                        "theta_bin": theta_idx,
                        "edge_sigma_bin": edge_idx,
                        "num_positive": n,
                        "detected": k,
                        "recall": float(k / n),
                        "recall_ci95_low": ci[0],
                        "recall_ci95_high": ci[1],
                        "is_weak_family_cell": bool(amp_idx == 0 or theta_idx == 0),
                    }
                )
    return rows


def write_csv(path, rows):
    columns = [
        "method",
        "threshold",
        "z0_amp_bin",
        "theta_bin",
        "edge_sigma_bin",
        "num_positive",
        "detected",
        "recall",
        "recall_ci95_low",
        "recall_ci95_high",
        "is_weak_family_cell",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def write_markdown(path, report):
    lines = [
        "# Weak-Family 3D Recall Breakdown",
        "",
        f"- Data HDF5: `{report['data_h5']}`",
        f"- ML report: `{report['ml_report']}`",
        f"- Classical report: `{report['classical_report']}`",
        "",
        "| method | z0_amp_bin | theta_bin | edge_sigma_bin | detected / n | recall | 95% CI | weak_cell |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['method']} | {row['z0_amp_bin']} | {row['theta_bin']} | {row['edge_sigma_bin']} | "
            f"{row['detected']} / {row['num_positive']} | {row['recall']:.3f} | "
            f"[{row['recall_ci95_low']:.3f}, {row['recall_ci95_high']:.3f}] | {row['is_weak_family_cell']} |"
        )
    lines.extend(["", "## Worst Cells Per Method", ""])
    for method, row in report["worst_cells"].items():
        lines.append(
            f"- `{method}`: bin=({row['z0_amp_bin']}, {row['theta_bin']}, {row['edge_sigma_bin']}), "
            f"recall=`{row['recall']:.3f}` [{row['recall_ci95_low']:.3f}, {row['recall_ci95_high']:.3f}]"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_h5 = Path(args.data_h5).resolve()
    strat = load_stratification(data_h5)
    ml_thresholds = model_thresholds(args.ml_report)
    classical_thresholds = model_thresholds(args.classical_report)
    device = p3.resolve_device(args.device)
    specs = [parse_model_spec(text) for text in (args.model or DEFAULT_MODELS)]

    rows = []
    for method in ("matched_template", "centered_disc"):
        scores = load_classical_scores(args.classical_dir, method)
        rows.extend(summarize_cells(method, scores, classical_thresholds[method], strat))
    for spec in specs:
        scores = load_or_score_ml(spec, data_h5, output_dir, args, device)
        rows.extend(summarize_cells(spec.name, scores, ml_thresholds[spec.name], strat))

    worst_cells = {}
    for method in sorted(set(row["method"] for row in rows)):
        method_rows = [row for row in rows if row["method"] == method]
        worst_cells[method] = min(method_rows, key=lambda row: (row["recall"], row["num_positive"]))

    report = {
        "data_h5": str(data_h5),
        "ml_report": str(Path(args.ml_report).resolve()),
        "classical_report": str(Path(args.classical_report).resolve()),
        "bin_schema": strat["bin_schema"],
        "rows": rows,
        "worst_cells": worst_cells,
    }
    json_path = output_dir / "weak_family_breakdown.json"
    csv_path = output_dir / "weak_family_breakdown.csv"
    md_path = output_dir / "weak_family_breakdown.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, report)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
