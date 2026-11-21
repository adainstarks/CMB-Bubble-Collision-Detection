"""
Audit Phase 3 evaluation outputs for candidate-product usability.

This validates that evaluation artifacts are not just plots. The output must be
machine-readable, internally consistent, and tied back to physical sky metadata.
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


REQUIRED_RECORD_FIELDS = {
    "sample_index",
    "threshold",
    "patch_center_glon_deg",
    "patch_center_glat_deg",
    "has_candidate",
    "score_max",
    "score_mean",
    "positive_fraction",
    "candidate_x_pix",
    "candidate_y_pix",
    "candidate_dx_deg",
    "candidate_dy_deg",
    "candidate_glon_deg",
    "candidate_glat_deg",
    "radius_est_deg",
    "area_pixels",
    "truth_label",
    "truth_theta_crit_deg",
    "truth_z0",
    "truth_zcrit",
    "truth_edge_sigma_deg",
    "coord_pool_idx",
    "cmb_realization_idx",
    "background_id",
    "mask_row",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Phase 3 evaluation candidate outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


class Audit:
    def __init__(self):
        self.failures = []
        self.warnings = []
        self.metrics = {}

    def require(self, condition, message):
        if not condition:
            self.failures.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def metric(self, key, value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        self.metrics[key] = value

    def report(self):
        return {
            "status": "pass" if not self.failures else "fail",
            "num_failures": len(self.failures),
            "num_warnings": len(self.warnings),
            "failures": self.failures,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_number(value):
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def audit_record(record, row_id, selected_threshold, audit):
    missing = REQUIRED_RECORD_FIELDS - set(record.keys())
    audit.require(not missing, f"Record {row_id} missing fields: {sorted(missing)}")
    if missing:
        return

    audit.require(int(record["mask_row"]) == row_id, f"Record {row_id} has wrong mask_row")
    audit.require(abs(float(record["threshold"]) - selected_threshold) < 1e-8, f"Record {row_id} threshold mismatch")
    audit.require(0.0 <= float(record["patch_center_glon_deg"]) <= 360.0, f"Record {row_id} patch lon out of range")
    audit.require(-90.0 <= float(record["patch_center_glat_deg"]) <= 90.0, f"Record {row_id} patch lat out of range")
    audit.require(0.0 <= float(record["score_max"]) <= 1.0, f"Record {row_id} score_max out of range")
    audit.require(0.0 <= float(record["score_mean"]) <= 1.0, f"Record {row_id} score_mean out of range")
    audit.require(0.0 <= float(record["positive_fraction"]) <= 1.0, f"Record {row_id} positive_fraction out of range")
    audit.require(int(record["truth_label"]) in {0, 1}, f"Record {row_id} truth_label must be 0/1")

    if record["has_candidate"]:
        for field in (
            "candidate_x_pix",
            "candidate_y_pix",
            "candidate_dx_deg",
            "candidate_dy_deg",
            "candidate_glon_deg",
            "candidate_glat_deg",
            "radius_est_deg",
        ):
            audit.require(is_number(record[field]), f"Record {row_id} candidate field is not finite: {field}")
        audit.require(0.0 <= float(record["candidate_x_pix"]) <= 255.0, f"Record {row_id} candidate x out of range")
        audit.require(0.0 <= float(record["candidate_y_pix"]) <= 255.0, f"Record {row_id} candidate y out of range")
        audit.require(0.0 <= float(record["candidate_glon_deg"]) <= 360.0, f"Record {row_id} candidate lon out of range")
        audit.require(-90.0 <= float(record["candidate_glat_deg"]) <= 90.0, f"Record {row_id} candidate lat out of range")
        audit.require(float(record["radius_est_deg"]) >= 0.0, f"Record {row_id} radius negative")
        audit.require(int(record["area_pixels"]) > 0, f"Record {row_id} candidate area is zero")
    else:
        audit.require(int(record["area_pixels"]) == 0, f"Record {row_id} no-candidate area should be zero")


def run_audit(eval_dir):
    audit = Audit()
    eval_dir = Path(eval_dir).resolve()
    audit.metric("eval_dir", str(eval_dir))

    required_files = {
        "evaluation_summary": eval_dir / "evaluation_summary.json",
        "threshold_metrics": eval_dir / "threshold_metrics.json",
        "stratified_metrics": eval_dir / "stratified_metrics.json",
        "candidate_records": eval_dir / "candidate_records.jsonl",
        "candidate_masks": eval_dir / "candidate_masks.npz",
    }
    for name, path in required_files.items():
        audit.require(path.exists(), f"Missing {name}: {path}")
    if audit.failures:
        return audit.report()

    summary = load_json(required_files["evaluation_summary"])
    threshold_rows = load_json(required_files["threshold_metrics"])
    stratified = load_json(required_files["stratified_metrics"])
    records = load_jsonl(required_files["candidate_records"])
    masks = np.load(required_files["candidate_masks"])

    selected_threshold = float(summary["selected_threshold"])
    num_samples = int(summary["num_samples"])
    audit.metric("num_samples", num_samples)
    audit.metric("num_candidate_records", len(records))
    audit.metric("selected_threshold", selected_threshold)

    audit.require(len(records) == num_samples, "candidate_records row count does not match evaluation summary")
    audit.require(len(threshold_rows) > 1, "threshold_metrics must contain a sweep, not a single row")
    audit.require("sample_indices" in masks, "candidate_masks missing sample_indices")
    audit.require("mask_bits" in masks, "candidate_masks missing mask_bits")
    audit.require("mask_shape" in masks, "candidate_masks missing mask_shape")
    if {"sample_indices", "mask_bits", "mask_shape"}.issubset(set(masks.files)):
        sample_indices = np.asarray(masks["sample_indices"], dtype=np.int64)
        mask_bits = np.asarray(masks["mask_bits"], dtype=np.uint8)
        mask_shape = np.asarray(masks["mask_shape"], dtype=np.int64)
        audit.metric("mask_shape", mask_shape)
        audit.require(len(sample_indices) == num_samples, "candidate mask sample_indices count mismatch")
        audit.require(mask_bits.shape[0] == num_samples, "candidate mask row count mismatch")
        expected_bits = int(np.ceil(int(np.prod(mask_shape)) / 8.0))
        audit.require(mask_bits.shape[1] == expected_bits, "candidate mask packed-bit width mismatch")

    for row_id, record in enumerate(records):
        audit_record(record, row_id, selected_threshold, audit)

    if records:
        sample_indices_from_records = np.asarray([int(row["sample_index"]) for row in records], dtype=np.int64)
        if "sample_indices" in masks:
            audit.require(
                np.array_equal(sample_indices_from_records, np.asarray(masks["sample_indices"], dtype=np.int64)),
                "candidate_records sample_index order differs from candidate_masks sample_indices",
            )
        audit.metric("num_candidates", int(sum(bool(row["has_candidate"]) for row in records)))
        audit.metric("num_truth_positive", int(sum(int(row["truth_label"]) == 1 for row in records)))

    for group_name in ("theta_crit", "amplitude", "edge_strength"):
        audit.require(group_name in stratified, f"stratified_metrics missing {group_name}")
        if group_name in stratified:
            audit.require(bool(stratified[group_name]), f"stratified_metrics/{group_name} is empty")

    selected_rows = [row for row in threshold_rows if abs(float(row["threshold"]) - selected_threshold) < 1e-8]
    audit.require(len(selected_rows) == 1, "selected_threshold not found exactly once in threshold_metrics")
    return audit.report()


def main():
    args = parse_args()
    report = run_audit(args.eval_dir)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
