"""
Compute causal-boundary morphology metrics for classical baseline outputs.
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

import h5py
import numpy as np

from phase3_boundary_analysis import (
    aggregate,
    boundary_core,
    edge_bin_name,
    theta_bin_name,
    tolerant_boundary_metrics,
)
from phase3_audit_outputs import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute contour/interior metrics for classical baseline candidate masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--classical-method-dir", type=str, required=True)
    parser.add_argument("--data-h5", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--ring-half-width-deg", type=float, default=1.0)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def binary_metrics(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    intersection = float(np.logical_and(pred, truth).sum())
    pred_sum = float(pred.sum())
    truth_sum = float(truth.sum())
    union = pred_sum + truth_sum - intersection
    return {
        "dice": float((2.0 * intersection) / max(pred_sum + truth_sum, 1.0)),
        "iou": float(intersection / max(union, 1.0)),
    }


def run(args):
    method_dir = Path(args.classical_method_dir).resolve()
    masks_npz = np.load(method_dir / "candidate_masks.npz")
    sample_indices = np.asarray(masks_npz["sample_indices"], dtype=np.int64)
    pred_masks = np.asarray(masks_npz["pred_masks"], dtype=bool)
    candidate_records = {
        int(row["sample_index"]): row for row in load_jsonl(method_dir / "candidate_records.jsonl")
    }
    method_summary = load_json(args.summary_json)
    threshold = float(method_summary["selected_threshold"])

    rows = []
    by_edge = {}
    by_theta = {}
    tolerance_pix = max(1, int(np.ceil(float(args.ring_half_width_deg) / (13.0 / 60.0))))
    with h5py.File(args.data_h5, "r") as h5:
        truth_masks = h5["masks"]
        truth = h5["truth"]
        labels = np.asarray(h5["labels"][sample_indices], dtype=np.uint8)
        for row_id, sample_idx in enumerate(sample_indices):
            if int(labels[row_id]) != 1:
                continue
            pred_active = float(candidate_records[int(sample_idx)]["score"]) >= threshold
            pred = pred_masks[row_id] if pred_active else np.zeros_like(pred_masks[row_id], dtype=bool)
            truth_disc = np.asarray(truth_masks[sample_idx], dtype=bool)
            theta_crit = float(truth["theta_crit_deg"][sample_idx])
            zcrit_abs = abs(float(truth["zcrit"][sample_idx]))
            contour = tolerant_boundary_metrics(
                pred_boundary=boundary_core(pred),
                truth_boundary=boundary_core(truth_disc),
                tolerance_pix=tolerance_pix,
            )
            disc = binary_metrics(pred, truth_disc)
            row = {
                "sample_index": int(sample_idx),
                "theta_crit_deg": theta_crit,
                "abs_zcrit": zcrit_abs,
                "boundary_tolerance_pix": tolerance_pix,
                "contour_precision": contour["precision"],
                "contour_recall": contour["recall"],
                "contour_f1": contour["f1"],
                "contour_pred_boundary_pixels": contour["pred_boundary_pixels"],
                "contour_truth_boundary_pixels": contour["truth_boundary_pixels"],
                "contour_pred_to_truth_mean_deg": contour["pred_to_truth_mean_deg"],
                "contour_truth_to_pred_mean_deg": contour["truth_to_pred_mean_deg"],
                "disc_dice": disc["dice"],
                "disc_iou": disc["iou"],
            }
            rows.append(row)
            by_edge.setdefault(edge_bin_name(zcrit_abs), []).append(row)
            by_theta.setdefault(theta_bin_name(theta_crit), []).append(row)

    return {
        "classical_method_dir": str(method_dir),
        "data_h5": str(Path(args.data_h5).resolve()),
        "summary_json": str(Path(args.summary_json).resolve()),
        "ring_half_width_deg": float(args.ring_half_width_deg),
        "overall": aggregate(rows),
        "by_edge_strength": {key: aggregate(value) for key, value in sorted(by_edge.items())},
        "by_theta_crit": {key: aggregate(value) for key, value in sorted(by_theta.items())},
        "records": rows,
    }


def main():
    args = parse_args()
    report = run(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(
        json.dumps(
            {
                "overall_contour_f1": report["overall"].get("contour_f1_mean"),
                "strong_edge_contour_f1": report["by_edge_strength"]
                .get("strong_|zcrit|>=3e-5", {})
                .get("contour_f1_mean"),
                "output_json": str(output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
