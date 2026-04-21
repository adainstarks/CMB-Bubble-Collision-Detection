"""
Evaluate the historical Phase 3 deployment policy from cached scores.

This script is retained for provenance of the pre-remediation composite policy.
It is not the current remediated-v1 deployment path; see PROJECT_HANDOFF.md and
the Batch 6 Nside=32 reports for the active policy interpretation.

Policy definitions:
    normal_candidate:
        v5_consensus passes, and either score_avg ensemble or matched_template passes.
    classical_fallback:
        matched_template passes while v5_consensus does not.
    all_candidates:
        normal_candidate OR classical_fallback.

Thresholds are the frozen sensitivity-grid FPR=0.05 thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "deployment_decision_v1"
DEFAULT_SENS_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "sensitivity_curve_v1" / "sensitivity_report.json"
DEFAULT_SENS_SCORES = PROJECT_ROOT / "runs" / "phase3_unet" / "sensitivity_curve_v1" / "sensitivity_scores.npz"
DEFAULT_ENSEMBLE_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "ensemble_eval_v1" / "ensemble_eval.json"
DEFAULT_STRAT_H5 = PROJECT_ROOT / "data" / "validation_stratified_v1" / "validation_data.h5"
DEFAULT_STRAT_CACHE = PROJECT_ROOT / "runs" / "phase3_unet" / "ensemble_eval_v1" / "score_cache"
DEFAULT_CLASSICAL_STRAT = PROJECT_ROOT / "runs" / "phase3_unet" / "stratified_external_classical_v1" / "matched_template_scores_masks.npz"
DEFAULT_NULL_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "null_burden_matched_fpr_v1" / "null_burden_matched_fpr.json"
DEFAULT_ENSEMBLE_NULL = PROJECT_ROOT / "runs" / "phase3_unet" / "ensemble_eval_v1" / "ensemble_eval.json"
ML_METHODS = ("original_v4", "boundary_v4", "v5_consensus", "v6_aux_only", "v6_hard_w15")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate final Phase 3 composite deployment policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_SENS_REPORT))
    parser.add_argument("--sensitivity-scores", type=str, default=str(DEFAULT_SENS_SCORES))
    parser.add_argument("--ensemble-report", type=str, default=str(DEFAULT_ENSEMBLE_REPORT))
    parser.add_argument("--stratified-h5", type=str, default=str(DEFAULT_STRAT_H5))
    parser.add_argument("--stratified-cache", type=str, default=str(DEFAULT_STRAT_CACHE))
    parser.add_argument("--classical-stratified", type=str, default=str(DEFAULT_CLASSICAL_STRAT))
    parser.add_argument("--null-report", type=str, default=str(DEFAULT_NULL_REPORT))
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def binary_metrics(active, labels):
    labels = np.asarray(labels, dtype=np.uint8)
    active = np.asarray(active, dtype=bool)
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
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    }


def policy_metrics(v5_scores, avg_scores, matched_scores, labels, thresholds):
    v5 = v5_scores > thresholds["v5_consensus"]
    avg = avg_scores > thresholds["score_avg"]
    matched = matched_scores > thresholds["matched_template"]
    policies = {
        "v5_only": v5,
        "score_avg_only": avg,
        "matched_template_only": matched,
        "normal_candidate": v5 & (avg | matched),
        "classical_fallback": matched & ~v5,
        "all_candidates": (v5 & (avg | matched)) | (matched & ~v5),
    }
    return {name: binary_metrics(active, labels) for name, active in policies.items()}


def load_sensitivity_scores(path):
    with np.load(path) as loaded:
        labels = np.asarray(loaded["labels"], dtype=np.uint8)
        scores = {method: np.asarray(loaded[f"score__{method}"], dtype=np.float32) for method in ML_METHODS}
        matched = np.asarray(loaded["score__matched_template"], dtype=np.float32)
    avg = np.vstack([scores[method] for method in ML_METHODS]).mean(axis=0)
    return labels, scores["v5_consensus"], avg.astype(np.float32), matched


def load_stratified_scores(strat_h5, cache_dir, classical_npz):
    with h5py.File(strat_h5, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    cache_dir = Path(cache_dir)
    scores = {}
    for method in ML_METHODS:
        with np.load(cache_dir / f"stratified_{method}_scores.npz") as loaded:
            scores[method] = np.asarray(loaded["scores"], dtype=np.float32)
    avg = np.vstack([scores[method] for method in ML_METHODS]).mean(axis=0)
    with np.load(classical_npz) as loaded:
        matched = np.asarray(loaded["scores"], dtype=np.float32)
    return labels, scores["v5_consensus"], avg.astype(np.float32), matched


def null_policy_metrics(null_report, ensemble_report):
    null = load_json(null_report)
    ensemble = load_json(ensemble_report)
    by_name = null["methods"]
    matched_fp = int(by_name["matched_template"]["false_positive_count"])
    v5_fp = int(by_name["v5_consensus"]["false_positive_count"])
    avg_fp = int(ensemble["null_metrics"]["score_avg"]["fp"])
    n = int(null["num_samples"])
    if matched_fp == 0 and v5_fp == 0 and avg_fp == 0:
        composite_fp = 0
    else:
        composite_fp = None
    return {
        "num_patches": n,
        "matched_template_fp": matched_fp,
        "v5_consensus_fp": v5_fp,
        "score_avg_fp": avg_fp,
        "normal_candidate_fp": composite_fp,
        "classical_fallback_fp": 0 if matched_fp == 0 else None,
        "all_candidates_fp": composite_fp,
        "note": "Composite null FP is exact here because matched_template, v5_consensus, and score_avg each have zero SMICA false positives at the frozen thresholds.",
    }


def write_markdown(path, report):
    lines = ["# Final Composite Policy Evaluation", ""]
    lines.append("Thresholds:")
    lines.append("")
    for name, value in report["thresholds"].items():
        lines.append(f"- `{name}`: `{value:.8f}`")
    lines.append("")
    for section in ("sensitivity", "stratified"):
        lines.append(f"## {section.capitalize()} Metrics")
        lines.append("")
        lines.append("| policy | precision | recall | FPR | F1 | TP | FP | TN | FN |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, metrics in report[section].items():
            lines.append(
                f"| `{name}` | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                f"{metrics['fpr']:.3f} | {metrics['f1']:.3f} | {metrics['tp']} | "
                f"{metrics['fp']} | {metrics['tn']} | {metrics['fn']} |"
            )
        lines.append("")
    lines.append("## SMICA Null")
    lines.append("")
    null = report["smica_null"]
    lines.append(f"- `matched_template`: `{null['matched_template_fp']} / {null['num_patches']}`")
    lines.append(f"- `v5_consensus`: `{null['v5_consensus_fp']} / {null['num_patches']}`")
    lines.append(f"- `score_avg`: `{null['score_avg_fp']} / {null['num_patches']}`")
    lines.append(f"- `normal_candidate`: `{null['normal_candidate_fp']} / {null['num_patches']}`")
    lines.append(f"- `all_candidates`: `{null['all_candidates_fp']} / {null['num_patches']}`")
    lines.append("")
    lines.append(null["note"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sens_report = load_json(args.sensitivity_report)
    ensemble_report = load_json(args.ensemble_report)
    thresholds = {
        "matched_template": float(sens_report["thresholds"]["matched_template"]["threshold"]),
        "v5_consensus": float(sens_report["thresholds"]["v5_consensus"]["threshold"]),
        "score_avg": float(ensemble_report["ensemble_thresholds"]["score_avg"]),
    }

    sens_labels, sens_v5, sens_avg, sens_matched = load_sensitivity_scores(args.sensitivity_scores)
    strat_labels, strat_v5, strat_avg, strat_matched = load_stratified_scores(
        args.stratified_h5,
        args.stratified_cache,
        args.classical_stratified,
    )

    report = {
        "thresholds": thresholds,
        "sensitivity": policy_metrics(sens_v5, sens_avg, sens_matched, sens_labels, thresholds),
        "stratified": policy_metrics(strat_v5, strat_avg, strat_matched, strat_labels, thresholds),
        "smica_null": null_policy_metrics(args.null_report, args.ensemble_report),
    }
    json_path = output_dir / "final_policy_eval.json"
    md_path = output_dir / "final_policy_eval.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
