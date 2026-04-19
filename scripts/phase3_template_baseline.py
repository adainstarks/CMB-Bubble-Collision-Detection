"""
Classical circular-template baselines for the Phase 3 split.

Two baselines are evaluated on the exact saved split from a Phase 3 run:
    - circular_template_screen: circular template screen with position search over the patch
    - centered_disc: shortcut baseline restricted to the patch center

The goal is not to reproduce Feeney's full needlet+CHT+Bayesian pipeline. The goal is to
provide an immediate non-deep comparator and a trivial shortcut baseline on the same data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve

import phase3_evaluate_run as p3eval
from phase_dataset_utils import make_angular_distance_grid, patch_center_pixel, patch_offsets_deg_to_sky
from phase3_method_registry import CIRCULAR_TEMPLATE_SCREEN, canonical_method_name, method_metadata

METHODS = (CIRCULAR_TEMPLATE_SCREEN, "matched_template", "centered_disc", "both")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate classical circular-template baselines on a saved Phase 3 split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="calibration", choices=["train", "calibration", "val", "test"])
    parser.add_argument("--method", type=str, default="both", choices=METHODS)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument(
        "--fixed-threshold-summary",
        type=str,
        default="",
        help=(
            "Optional summary.json from a calibration split. When supplied, "
            "method thresholds are loaded from that file and applied without "
            "test-split threshold selection."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if not (0.0 <= args.target_fpr < 1.0):
        raise ValueError("--target-fpr must be in [0, 1).")


def make_output_dir(run_dir, split, output_dir):
    if output_dir:
        return Path(output_dir).resolve()
    return run_dir / f"classical_baselines_{split}"


def make_disc_kernel(radius_deg, patch_shape):
    center = patch_center_pixel(patch_shape[-1])
    theta = make_angular_distance_grid(
        patch_shape[-1],
        reso_arcmin=13.0,
        center_x_pix=center,
        center_y_pix=center,
    )
    disc = theta <= np.radians(radius_deg)
    annulus = (theta > np.radians(radius_deg)) & (theta <= np.radians(min(radius_deg * 1.5, 25.0)))
    kernel = disc.astype(np.float32)
    if annulus.any():
        kernel[annulus] = -float(disc.sum()) / float(annulus.sum())
    kernel -= float(kernel.mean())
    norm = float(np.linalg.norm(kernel))
    if norm > 0.0:
        kernel /= norm
    return kernel


def make_ring_kernel(radius_deg, patch_shape, ring_width_deg=1.0):
    center = patch_center_pixel(patch_shape[-1])
    theta = make_angular_distance_grid(
        patch_shape[-1],
        reso_arcmin=13.0,
        center_x_pix=center,
        center_y_pix=center,
    )
    inner = max(radius_deg - ring_width_deg / 2.0, 0.0)
    outer = radius_deg + ring_width_deg / 2.0
    ring = (theta >= np.radians(inner)) & (theta <= np.radians(outer))
    support = theta <= np.radians(min(radius_deg + 2.0 * ring_width_deg, 25.0))
    kernel = np.zeros_like(theta, dtype=np.float32)
    kernel[ring] = 1.0
    background = support & (~ring)
    if background.any():
        kernel[background] = -float(ring.sum()) / float(background.sum())
    kernel -= float(kernel.mean())
    norm = float(np.linalg.norm(kernel))
    if norm > 0.0:
        kernel /= norm
    return kernel


def build_kernel_bank(patch_shape):
    radius_grid = [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]
    kernels = []
    for radius_deg in radius_grid:
        kernels.append(
            {
                "radius_deg": radius_deg,
                "kind": "disc",
                "kernel": make_disc_kernel(radius_deg, patch_shape),
            }
        )
        kernels.append(
            {
                "radius_deg": radius_deg,
                "kind": "ring",
                "kernel": make_ring_kernel(radius_deg, patch_shape),
            }
        )
    return kernels


def make_candidate_mask(center_x_pix, center_y_pix, radius_deg, patch_shape):
    theta = make_angular_distance_grid(
        patch_shape[-1],
        reso_arcmin=13.0,
        center_x_pix=center_x_pix,
        center_y_pix=center_y_pix,
    )
    return (theta <= np.radians(radius_deg)).astype(np.uint8)


def search_best_candidate(patch, kernels, centered_only=False):
    patch = np.asarray(patch, dtype=np.float32)
    patch = patch - float(patch.mean())
    patch_std = float(patch.std())
    if patch_std > 0.0:
        patch = patch / patch_std

    center = patch_center_pixel(patch.shape[-1])
    best = None
    for kernel_info in kernels:
        kernel = kernel_info["kernel"]
        if centered_only:
            response = float(np.sum(patch * kernel))
            score = abs(response)
            candidate = {
                "score": score,
                "center_x_pix": center,
                "center_y_pix": center,
                "radius_deg": kernel_info["radius_deg"],
                "kind": kernel_info["kind"],
            }
        else:
            response_map = fftconvolve(patch, kernel[::-1, ::-1], mode="same")
            peak = np.unravel_index(np.argmax(np.abs(response_map)), response_map.shape)
            score = float(np.abs(response_map[peak]))
            candidate = {
                "score": score,
                "center_y_pix": float(peak[0]),
                "center_x_pix": float(peak[1]),
                "radius_deg": kernel_info["radius_deg"],
                "kind": kernel_info["kind"],
            }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    mask = make_candidate_mask(
        center_x_pix=best["center_x_pix"],
        center_y_pix=best["center_y_pix"],
        radius_deg=best["radius_deg"],
        patch_shape=patch.shape,
    )
    return best, mask


def compute_metrics(scores, pred_masks, truth_masks, labels, threshold):
    active = scores >= threshold
    image_pred = active
    image_true = labels.astype(bool)

    tp = int(np.sum(image_pred & image_true))
    fp = int(np.sum(image_pred & (~image_true)))
    fn = int(np.sum((~image_pred) & image_true))
    tn = int(np.sum((~image_pred) & (~image_true)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)
    false_positive_rate = fp / max(fp + tn, 1)

    positive_rows = np.flatnonzero(labels == 1)
    hard_dice_sum = 0.0
    iou_sum = 0.0
    for row in positive_rows:
        pred = pred_masks[row].astype(bool) if active[row] else np.zeros_like(pred_masks[row], dtype=bool)
        truth = truth_masks[row].astype(bool)
        intersection = float(np.logical_and(pred, truth).sum())
        pred_sum = float(pred.sum())
        truth_sum = float(truth.sum())
        union = pred_sum + truth_sum - intersection
        empty_both = pred_sum == 0.0 and truth_sum == 0.0
        if empty_both:
            hard_dice = 1.0
            iou = 1.0
        else:
            hard_dice = (2.0 * intersection + 1e-8) / (pred_sum + truth_sum + 1e-8)
            iou = (intersection + 1e-8) / (union + 1e-8)
        hard_dice_sum += hard_dice
        iou_sum += iou

    pos_n = max(len(positive_rows), 1)
    return {
        "threshold": float(threshold),
        "image_precision": float(precision),
        "image_recall": float(recall),
        "image_f1": float(f1),
        "image_specificity": float(specificity),
        "image_false_positive_rate": float(false_positive_rate),
        "hard_dice_pos": float(hard_dice_sum / pos_n),
        "iou_pos": float(iou_sum / pos_n),
        "image_tp": tp,
        "image_fp": fp,
        "image_fn": fn,
        "image_tn": tn,
        "num_samples": int(len(labels)),
        "num_positive_samples": int(labels.sum()),
    }


def sweep_thresholds(scores, pred_masks, truth_masks, labels):
    quantiles = np.linspace(0.05, 0.95, 50)
    thresholds = np.unique(np.quantile(scores, quantiles))
    rows = [compute_metrics(scores, pred_masks, truth_masks, labels, thr) for thr in thresholds]
    return rows


def choose_operating_point(rows, target_fpr):
    best_row, operating_point = p3eval.choose_operating_point(
        rows=rows,
        operating_point_rule="fpr_cap",
        selection_metric="image_f1",
        target_fpr=target_fpr,
    )
    return best_row, operating_point


def load_fixed_thresholds(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    thresholds = {}
    for method_name, payload in summary.get("methods", {}).items():
        thresholds[canonical_method_name(method_name)] = float(payload["selected_threshold"])
    return thresholds


def run_method(
    method_name,
    patches,
    truth_masks,
    labels,
    glon,
    glat,
    global_indices,
    kernels,
    target_fpr,
    fixed_threshold=None,
):
    canonical_name = canonical_method_name(method_name)
    scores = np.zeros(len(labels), dtype=np.float32)
    pred_masks = np.zeros_like(truth_masks, dtype=np.uint8)
    candidate_records = []

    centered_only = canonical_name == "centered_disc"
    for row in range(len(labels)):
        best, pred_mask = search_best_candidate(patches[row], kernels, centered_only=centered_only)
        scores[row] = float(best["score"])
        pred_masks[row] = pred_mask

        dx_deg = (best["center_x_pix"] - patch_center_pixel(pred_mask.shape[-1])) * 13.0 / 60.0
        dy_deg = (best["center_y_pix"] - patch_center_pixel(pred_mask.shape[-1])) * 13.0 / 60.0
        cand_glon, cand_glat = patch_offsets_deg_to_sky(glon[row], glat[row], dx_deg, dy_deg)
        candidate_records.append(
            {
                "sample_index": int(global_indices[row]),
                "score": float(best["score"]),
                "kernel_kind": best["kind"],
                "patch_center_glon_deg": float(glon[row]),
                "patch_center_glat_deg": float(glat[row]),
                "radius_est_deg": float(best["radius_deg"]),
                "candidate_dx_deg": float(dx_deg),
                "candidate_dy_deg": float(dy_deg),
                "candidate_glon_deg": float(cand_glon),
                "candidate_glat_deg": float(cand_glat),
                "has_signal": int(labels[row]),
            }
        )

    rows = sweep_thresholds(scores, pred_masks, truth_masks, labels)
    if fixed_threshold is None:
        selected_row, operating_point = choose_operating_point(rows, target_fpr=target_fpr)
    else:
        selected_row = compute_metrics(scores, pred_masks, truth_masks, labels, float(fixed_threshold))
        rows.append(selected_row)
        operating_point = {
            "rule": "fixed_threshold",
            "threshold_source": "fixed_threshold_summary",
            "target_fpr": float(target_fpr),
        }
    return {
        "method_metadata": method_metadata(method_name),
        "threshold_metrics": rows,
        "selected_threshold": float(selected_row["threshold"]),
        "selected_threshold_metrics": selected_row,
        "operating_point": operating_point,
        "scores": scores,
        "pred_masks": pred_masks,
        "candidate_records": candidate_records,
        "sample_indices": np.asarray(global_indices, dtype=np.int64),
    }


def save_method_outputs(output_dir, method_name, result):
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    with open(method_dir / "candidate_records.jsonl", "w", encoding="utf-8") as handle:
        for row_id, record in enumerate(result["candidate_records"]):
            payload = dict(record)
            payload["mask_row"] = row_id
            handle.write(json.dumps(payload) + "\n")
    np.savez_compressed(
        method_dir / "candidate_masks.npz",
        sample_indices=result["sample_indices"].astype(np.int64),
        pred_masks=result["pred_masks"].astype(np.uint8),
    )
    with open(method_dir / "threshold_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result["threshold_metrics"], handle, indent=2)
    with open(method_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method_metadata": result["method_metadata"],
                "selected_threshold": result["selected_threshold"],
                "selected_threshold_metrics": result["selected_threshold_metrics"],
                "operating_point": result["operating_point"],
            },
            handle,
            indent=2,
        )
    return method_dir


def main():
    args = parse_args()
    validate_args(args)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = make_output_dir(run_dir, args.split, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = p3eval.load_json(run_dir / "run_config.json")
    split_indices = p3eval.load_split_indices(run_dir, args.split)
    data_h5 = Path(run_config["data_h5"])

    sorted_order = np.argsort(split_indices)
    sorted_split_indices = np.asarray(split_indices[sorted_order], dtype=np.int64)
    with h5py.File(data_h5, "r") as h5:
        patches = np.asarray(h5["patches"][sorted_split_indices], dtype=np.float32)
        truth_masks = np.asarray(h5["masks"][sorted_split_indices], dtype=np.uint8)
        labels = np.asarray(h5["labels"][sorted_split_indices], dtype=np.uint8)
        glon = np.asarray(h5["metadata"]["glon_deg"][sorted_split_indices], dtype=np.float32)
        glat = np.asarray(h5["metadata"]["glat_deg"][sorted_split_indices], dtype=np.float32)

    kernels = build_kernel_bank(patches[0].shape)
    methods = [CIRCULAR_TEMPLATE_SCREEN, "centered_disc"] if args.method == "both" else [args.method]
    fixed_thresholds = load_fixed_thresholds(args.fixed_threshold_summary)

    combined_summary = {
        "run_dir": str(run_dir),
        "data_h5": str(data_h5),
        "split": args.split,
        "num_samples": int(len(split_indices)),
        "num_positive_samples": int(labels.sum()),
        "target_fpr": float(args.target_fpr),
        "sample_order": "sorted_split_indices_for_hdf5_consistency",
        "methods": {},
    }
    for method_name in methods:
        result = run_method(
            method_name=method_name,
            patches=patches,
            truth_masks=truth_masks,
            labels=labels,
            glon=glon,
            glat=glat,
            global_indices=sorted_split_indices,
            kernels=kernels,
            target_fpr=args.target_fpr,
            fixed_threshold=fixed_thresholds.get(canonical_method_name(method_name)),
        )
        method_dir = save_method_outputs(output_dir, method_name, result)
        combined_summary["methods"][method_name] = {
            "method_metadata": result["method_metadata"],
            "selected_threshold": result["selected_threshold"],
            "selected_threshold_metrics": result["selected_threshold_metrics"],
            "operating_point": result["operating_point"],
            "artifacts_dir": str(method_dir.resolve()),
        }
        print(
            f"{method_name}: threshold={result['selected_threshold']:.4f} "
            f"precision={result['selected_threshold_metrics']['image_precision']:.3f} "
            f"recall={result['selected_threshold_metrics']['image_recall']:.3f} "
            f"fpr={result['selected_threshold_metrics']['image_false_positive_rate']:.3f}"
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(combined_summary, handle, indent=2)

    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
