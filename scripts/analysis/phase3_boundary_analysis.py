"""
Analyze whether predictions recover causal-boundary structure.

This separates recovery of the affected disc interior from recovery of the
causal boundary ring. A detector that only learns broad interior contrast can
score well on mask Dice while failing the boundary signature emphasized in
Feeney et al.
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
from scipy import ndimage as ndi

from phase_dataset_utils import make_angular_distance_grid, patch_offsets_deg_to_pixel
from phase3_audit_outputs import load_jsonl


PATCH_PIX = 256
RESO_ARCMIN = 13.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute interior and causal-boundary recovery metrics from Phase 3 candidate outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--data-h5", type=str, default="")
    parser.add_argument("--ring-half-width-deg", type=float, default=1.0)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def unpack_mask_row(mask_bits_row, mask_shape):
    flat = np.unpackbits(mask_bits_row)[: int(np.prod(mask_shape))]
    return flat.reshape(tuple(mask_shape)).astype(bool)


def binary_metrics(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    intersection = float(np.logical_and(pred, truth).sum())
    pred_sum = float(pred.sum())
    truth_sum = float(truth.sum())
    union = pred_sum + truth_sum - intersection
    precision = intersection / max(pred_sum, 1.0)
    recall = intersection / max(truth_sum, 1.0)
    dice = (2.0 * intersection) / max(pred_sum + truth_sum, 1.0)
    iou = intersection / max(union, 1.0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "truth_pixels": int(truth_sum),
        "pred_pixels": int(pred_sum),
    }


def boundary_core(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask & (~eroded)


def tolerant_boundary_metrics(pred_boundary, truth_boundary, tolerance_pix):
    pred_boundary = np.asarray(pred_boundary, dtype=bool)
    truth_boundary = np.asarray(truth_boundary, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    pred_band = ndi.binary_dilation(pred_boundary, structure=structure, iterations=int(tolerance_pix))
    truth_band = ndi.binary_dilation(truth_boundary, structure=structure, iterations=int(tolerance_pix))

    pred_sum = float(pred_boundary.sum())
    truth_sum = float(truth_boundary.sum())
    precision = float(np.logical_and(pred_boundary, truth_band).sum()) / max(pred_sum, 1.0)
    recall = float(np.logical_and(truth_boundary, pred_band).sum()) / max(truth_sum, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    scale_deg = RESO_ARCMIN / 60.0
    if pred_boundary.any() and truth_boundary.any():
        dist_to_truth = ndi.distance_transform_edt(~truth_boundary) * scale_deg
        dist_to_pred = ndi.distance_transform_edt(~pred_boundary) * scale_deg
        pred_to_truth_mean_deg = float(np.mean(dist_to_truth[pred_boundary]))
        truth_to_pred_mean_deg = float(np.mean(dist_to_pred[truth_boundary]))
    else:
        pred_to_truth_mean_deg = float("nan")
        truth_to_pred_mean_deg = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
        "pred_boundary_pixels": int(pred_sum),
        "truth_boundary_pixels": int(truth_sum),
        "pred_to_truth_mean_deg": pred_to_truth_mean_deg,
        "truth_to_pred_mean_deg": truth_to_pred_mean_deg,
    }


def aggregate(rows):
    if not rows:
        return {"num_samples": 0}
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and np.isfinite(float(value))
    ]
    out = {"num_samples": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows if np.isfinite(float(row[key]))], dtype=np.float64)
        if values.size == 0:
            continue
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_median"] = float(np.median(values))
    return out


def edge_bin_name(zcrit_abs):
    if zcrit_abs <= 5e-6:
        return "smooth_|zcrit|<=5e-6"
    if zcrit_abs < 3e-5:
        return "weak_5e-6_to_3e-5"
    return "strong_|zcrit|>=3e-5"


def theta_bin_name(theta_deg):
    if theta_deg < 10.0:
        return "5-10deg"
    if theta_deg < 15.0:
        return "10-15deg"
    if theta_deg < 20.0:
        return "15-20deg"
    return "20-25deg"


def resolve_data_h5(eval_dir, data_h5_arg):
    if data_h5_arg:
        return Path(data_h5_arg).resolve()
    summary_path = eval_dir / "evaluation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    run_config_path = Path(summary["run_dir"]) / "run_config.json"
    with open(run_config_path, "r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    return Path(run_config["data_h5"]).resolve()


def run_analysis(eval_dir, data_h5, ring_half_width_deg):
    eval_dir = Path(eval_dir).resolve()
    records = load_jsonl(eval_dir / "candidate_records.jsonl")
    mask_npz = np.load(eval_dir / "candidate_masks.npz")
    mask_bits = np.asarray(mask_npz["mask_bits"], dtype=np.uint8)
    mask_shape = tuple(np.asarray(mask_npz["mask_shape"], dtype=np.int64))
    sample_indices = np.asarray(mask_npz["sample_indices"], dtype=np.int64)

    by_edge = {}
    by_theta = {}
    rows = []
    positive_count = 0
    tolerance_pix = max(1, int(np.ceil(float(ring_half_width_deg) / (RESO_ARCMIN / 60.0))))
    with h5py.File(data_h5, "r") as h5:
        truth_masks = h5["masks"]
        for row_id, record in enumerate(records):
            if int(record["truth_label"]) != 1:
                continue
            positive_count += 1
            sample_idx = int(sample_indices[row_id])
            pred = unpack_mask_row(mask_bits[row_id], mask_shape)
            truth_disc = np.asarray(truth_masks[sample_idx], dtype=bool)

            theta_crit = float(record["truth_theta_crit_deg"])
            zcrit_abs = abs(float(record["truth_zcrit"]))
            center_x, center_y = patch_offsets_deg_to_pixel(
                float(record["truth_signal_center_dx_deg"]),
                float(record["truth_signal_center_dy_deg"]),
                npix=PATCH_PIX,
                reso_arcmin=RESO_ARCMIN,
            )
            theta_grid_deg = np.degrees(
                make_angular_distance_grid(
                    PATCH_PIX,
                    RESO_ARCMIN,
                    center_x_pix=center_x,
                    center_y_pix=center_y,
                )
            )
            analytic_boundary_truth = np.abs(theta_grid_deg - theta_crit) <= float(ring_half_width_deg)
            analytic_boundary_truth &= truth_disc
            truth_boundary = boundary_core(truth_disc)
            pred_boundary = boundary_core(pred)
            interior_truth = theta_grid_deg <= max(theta_crit - float(ring_half_width_deg), 0.0)

            filled_mask_vs_analytic_boundary = binary_metrics(pred, analytic_boundary_truth)
            contour = tolerant_boundary_metrics(
                pred_boundary=pred_boundary,
                truth_boundary=truth_boundary,
                tolerance_pix=tolerance_pix,
            )
            interior = binary_metrics(pred, interior_truth)
            disc = binary_metrics(pred, truth_disc)
            row = {
                "sample_index": sample_idx,
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
                "filled_mask_vs_boundary_precision": filled_mask_vs_analytic_boundary["precision"],
                "filled_mask_vs_boundary_recall": filled_mask_vs_analytic_boundary["recall"],
                "filled_mask_vs_boundary_dice": filled_mask_vs_analytic_boundary["dice"],
                "filled_mask_vs_boundary_iou": filled_mask_vs_analytic_boundary["iou"],
                "interior_precision": interior["precision"],
                "interior_recall": interior["recall"],
                "interior_dice": interior["dice"],
                "interior_iou": interior["iou"],
                "disc_dice": disc["dice"],
                "disc_iou": disc["iou"],
            }
            rows.append(row)
            by_edge.setdefault(edge_bin_name(zcrit_abs), []).append(row)
            by_theta.setdefault(theta_bin_name(theta_crit), []).append(row)

    report = {
        "eval_dir": str(eval_dir),
        "data_h5": str(data_h5),
        "ring_half_width_deg": float(ring_half_width_deg),
        "boundary_metric_note": (
            "contour_* compares predicted mask contour to truth contour with tolerance. "
            "filled_mask_vs_boundary_* is diagnostic only and penalizes correct filled-disc masks."
        ),
        "num_positive_samples": int(positive_count),
        "overall": aggregate(rows),
        "by_edge_strength": {key: aggregate(value) for key, value in sorted(by_edge.items())},
        "by_theta_crit": {key: aggregate(value) for key, value in sorted(by_theta.items())},
        "records": rows,
    }
    return report


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    data_h5 = resolve_data_h5(eval_dir, args.data_h5)
    report = run_analysis(eval_dir, data_h5, args.ring_half_width_deg)
    output_path = Path(args.output_json).resolve() if args.output_json else eval_dir / "boundary_analysis.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    print(f"Boundary analysis: {output_path}")


if __name__ == "__main__":
    main()
