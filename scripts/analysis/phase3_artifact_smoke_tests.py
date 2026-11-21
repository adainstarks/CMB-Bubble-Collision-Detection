"""
Smoke tests for Phase 3 pushable artifacts.

These are intentionally lightweight: they verify schema, arithmetic, and
cross-artifact consistency without retraining models.
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
import csv
import json
from pathlib import Path


REQUIRED_FUSED_FIELDS = {
    "rank",
    "sample_index",
    "rank_score",
    "proposal_pass",
    "verifier_pass",
    "risk_tag",
    "truth_label",
    "patch_center_glon_deg",
    "patch_center_glat_deg",
    "candidate_glon_deg",
    "candidate_glat_deg",
    "radius_est_deg",
    "proposal_score_max",
    "verifier_score_max",
    "template_delta_chi2_vs_plane_null",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run schema/arithmetic smoke tests on Phase 3 candidate artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-jsonl", type=str, required=True)
    parser.add_argument("--candidate-csv", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()
    failures = []
    warnings = []
    rows = read_jsonl(args.candidate_jsonl)
    summary = json.loads(Path(args.summary_json).read_text())

    with open(args.candidate_csv, "r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    if len(rows) != int(summary["num_emitted_candidates"]):
        failures.append("JSONL row count does not match summary num_emitted_candidates.")
    if len(csv_rows) != len(rows):
        failures.append("CSV row count does not match JSONL row count.")
    if rows:
        missing = sorted(REQUIRED_FUSED_FIELDS - set(rows[0]))
        if missing:
            failures.append(f"Candidate schema missing fields: {missing}")
        ranks = [int(row["rank"]) for row in rows]
        if ranks != list(range(1, len(rows) + 1)):
            failures.append("Candidate ranks are not contiguous starting at 1.")
        scores = [float(row["rank_score"]) for row in rows]
        if any(scores[idx] < scores[idx + 1] for idx in range(len(scores) - 1)):
            failures.append("Candidate rows are not sorted by descending rank_score.")
    else:
        warnings.append("Candidate table is empty.")

    positives = sum(1 for row in rows if int(row["truth_label"]) == 1)
    negatives = sum(1 for row in rows if int(row["truth_label"]) == 0)
    if positives != int(summary["num_truth_positive_emitted"]):
        failures.append("Positive emitted count mismatch.")
    if negatives != int(summary["num_truth_negative_emitted"]):
        failures.append("Negative emitted count mismatch.")
    expected_precision = positives / max(len(rows), 1)
    expected_recall = positives / max(int(summary.get("num_truth_positive", positives)), 1)
    expected_fpr = negatives / max(int(summary.get("num_truth_negative", negatives)), 1)
    if abs(float(summary.get("emitted_precision", expected_precision)) - expected_precision) > 1e-12:
        failures.append("Summary emitted_precision arithmetic mismatch.")
    if abs(float(summary.get("emitted_recall", expected_recall)) - expected_recall) > 1e-12:
        failures.append("Summary emitted_recall arithmetic mismatch.")
    if abs(float(summary.get("emitted_false_positive_rate", expected_fpr)) - expected_fpr) > 1e-12:
        failures.append("Summary emitted_false_positive_rate arithmetic mismatch.")

    report = {
        "status": "pass" if not failures else "fail",
        "num_failures": len(failures),
        "num_warnings": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "num_rows": len(rows),
            "num_csv_rows": len(csv_rows),
            "num_positive_rows": positives,
            "num_negative_rows": negatives,
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
