"""
Phase 3 evaluation with fixed operating-point selection, structured candidate outputs,
and physics-aware breakdowns.

The evaluator reuses the split and normalization saved by `phase3_train_unet.py`,
then performs:
    - threshold sweeps on the requested split
    - a frozen operating-point choice for screening use
    - candidate extraction per patch with machine-readable outputs
    - stratified metrics by angular size, amplitude, and edge strength
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage as ndi

from phase3_audit_outputs import run_audit as run_output_audit
import phase3_train_unet as p3
from phase_config import min_component_area_pixels
from phase_dataset_utils import build_patch_candidate, load_metadata_array, load_optional_metadata_array, load_truth_array

SELECTION_METRICS = (
    "image_f1",
    "image_precision",
    "image_recall",
    "image_specificity",
    "aux_image_f1",
    "aux_image_precision",
    "aux_image_recall",
    "aux_image_specificity",
    "hard_dice_pos",
    "iou_pos",
)
OPERATING_POINT_RULES = ("fpr_cap", "metric_max", "fixed_threshold")
IMAGE_RULES = ("connected_component", "area_floor")
SCORE_MODES = ("component_score", "segmentation_max", "aux_score", "calibrated_composite")
MASK_SHAPE = (256, 256)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Phase 3 checkpoint with threshold sweeps and structured candidate outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint to evaluate: `best`, `last`, or a checkpoint path.",
    )
    parser.add_argument("--split", type=str, default="calibration", choices=["train", "calibration", "val", "test"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override the run batch size. Use 0 to reuse the training run batch size.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-count", type=int, default=50)
    parser.add_argument(
        "--image-min-positive-pixels",
        type=int,
        default=min_component_area_pixels(theta_min_deg=5.0, fraction=0.01),
        help=(
            "Minimum thresholded pixels required before a patch counts as image-positive "
            "for threshold-sweep image metrics."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="image_f1",
        choices=SELECTION_METRICS,
        help="Only used when --operating-point-rule=metric_max.",
    )
    parser.add_argument(
        "--operating-point-rule",
        type=str,
        default="fpr_cap",
        choices=OPERATING_POINT_RULES,
        help="Frozen threshold-selection rule used to choose the screening operating point.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=-1.0,
        help="Required for --split=test unless the checkpoint's configured threshold should be used.",
    )
    parser.add_argument(
        "--use-configured-threshold",
        action="store_true",
        help="Use the training run's configured threshold as the fixed operating point.",
    )
    parser.add_argument(
        "--image-rule",
        type=str,
        default="connected_component",
        choices=IMAGE_RULES,
        help="Image-level detection rule for operating metrics.",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="component_score",
        choices=SCORE_MODES,
        help=(
            "Scalar score family used for image-level threshold sweeps. "
            "`component_score` applies the connected-component pixel rule; "
            "`segmentation_max` thresholds max pixel probability; `aux_score` "
            "thresholds the image auxiliary head; `calibrated_composite` averages "
            "segmentation max with the auxiliary score."
        ),
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.10,
        help="Image-level false-positive-rate cap used by --operating-point-rule=fpr_cap.",
    )
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional evaluation output directory. Defaults to a subdirectory of the run.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be non-negative.")
    if args.threshold_count <= 1:
        raise ValueError("--threshold-count must be greater than 1.")
    if args.image_min_positive_pixels < 1:
        raise ValueError("--image-min-positive-pixels must be at least 1.")
    if not (0.0 < args.threshold_min < 1.0):
        raise ValueError("--threshold-min must be between 0 and 1.")
    if not (0.0 < args.threshold_max < 1.0):
        raise ValueError("--threshold-max must be between 0 and 1.")
    if args.threshold_min >= args.threshold_max:
        raise ValueError("--threshold-min must be smaller than --threshold-max.")
    if args.preview_count <= 0:
        raise ValueError("--preview-count must be positive.")
    if not (0.0 <= args.target_fpr < 1.0):
        raise ValueError("--target-fpr must be in [0, 1).")
    if args.fixed_threshold >= 0.0 and not (0.0 < args.fixed_threshold < 1.0):
        raise ValueError("--fixed-threshold must be in (0, 1), or negative to disable.")
    if args.operating_point_rule == "fixed_threshold" and args.fixed_threshold < 0.0 and not args.use_configured_threshold:
        raise ValueError("--operating-point-rule=fixed_threshold requires --fixed-threshold or --use-configured-threshold.")
    if args.split == "test" and args.fixed_threshold < 0.0 and not args.use_configured_threshold:
        raise ValueError(
            "Refusing to select an operating threshold on the test split. "
            "Pass --fixed-threshold or --use-configured-threshold."
        )


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_checkpoint_path(run_dir, checkpoint_arg):
    if checkpoint_arg == "best":
        path = run_dir / "best_checkpoint.pt"
        label = "best"
    elif checkpoint_arg == "last":
        path = run_dir / "last_checkpoint.pt"
        label = "last"
    else:
        path = Path(checkpoint_arg)
        if not path.is_absolute():
            path = run_dir / path
        label = path.stem
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve(), label


def default_output_dir(run_dir, checkpoint_label, split_name):
    return run_dir / f"evaluation_{checkpoint_label}_{split_name}"


def load_split_indices(run_dir, split_name):
    split_path = run_dir / "split_indices.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split_data = np.load(split_path)
    key = "calibration_idx" if split_name == "calibration" else f"{split_name}_idx"
    if key not in split_data and split_name == "calibration" and "val_idx" in split_data:
        key = "val_idx"
    return np.asarray(split_data[key], dtype=np.int64)


def make_eval_accumulator():
    acc = p3.make_metric_accumulator()
    acc.update(
        {
            "pred_frac_sum": 0.0,
            "pred_frac_pos_sum": 0.0,
            "pred_frac_neg_sum": 0.0,
            "num_negative_samples": 0,
            "aux_image_tp": 0,
            "aux_image_fp": 0,
            "aux_image_fn": 0,
            "aux_image_tn": 0,
            "aux_score_sum": 0.0,
            "aux_score_pos_sum": 0.0,
            "aux_score_neg_sum": 0.0,
            "aux_score_count": 0,
            "operating_score_sum": 0.0,
            "operating_score_pos_sum": 0.0,
            "operating_score_neg_sum": 0.0,
            "operating_score_count": 0,
        }
    )
    return acc


def component_image_predictions(preds, image_min_positive_pixels):
    pred_np = preds.detach().cpu().numpy().astype(bool)
    out = []
    for row in pred_np:
        mask = row[0] if row.ndim == 3 else row
        labels, count = ndi.label(mask)
        if count <= 0:
            out.append(False)
            continue
        areas = ndi.sum(mask, labels, index=np.arange(1, count + 1))
        out.append(bool(np.max(areas) >= int(image_min_positive_pixels)))
    return torch.as_tensor(out, dtype=torch.bool, device=preds.device)


def batch_metrics_from_probs(
    probs,
    targets,
    threshold,
    image_min_positive_pixels=1,
    aux_scores=None,
    image_rule="connected_component",
    score_mode="component_score",
):
    preds = probs >= threshold
    truths = targets >= 0.5

    dims = (1, 2, 3)
    intersection = (preds & truths).sum(dim=dims).float()
    pred_sum = preds.sum(dim=dims).float()
    truth_sum = truths.sum(dim=dims).float()
    union = pred_sum + truth_sum - intersection

    empty_both = (pred_sum == 0) & (truth_sum == 0)
    hard_dice = torch.where(
        empty_both,
        torch.ones_like(intersection),
        (2.0 * intersection + p3.EPS) / (pred_sum + truth_sum + p3.EPS),
    )
    iou = torch.where(
        empty_both,
        torch.ones_like(intersection),
        (intersection + p3.EPS) / (union + p3.EPS),
    )

    image_true = truth_sum > 0
    segmentation_score = probs.flatten(1).max(dim=1).values
    if aux_scores is not None:
        aux_scores = aux_scores.reshape(-1)

    if score_mode == "component_score":
        if image_rule == "connected_component":
            image_pred = component_image_predictions(preds, image_min_positive_pixels)
        else:
            image_pred = pred_sum >= int(image_min_positive_pixels)
        operating_score = segmentation_score
    elif score_mode == "segmentation_max":
        operating_score = segmentation_score
        image_pred = operating_score >= float(threshold)
    elif score_mode == "aux_score":
        if aux_scores is None:
            raise RuntimeError("--score-mode aux_score requires a checkpoint trained with --aux-head-weight > 0.")
        operating_score = aux_scores
        image_pred = operating_score >= float(threshold)
    elif score_mode == "calibrated_composite":
        if aux_scores is None:
            raise RuntimeError("--score-mode calibrated_composite requires a checkpoint trained with --aux-head-weight > 0.")
        operating_score = 0.5 * (segmentation_score + aux_scores)
        image_pred = operating_score >= float(threshold)
    else:
        raise ValueError(f"Unsupported score mode: {score_mode}")

    metrics = {
        "hard_dice": hard_dice.detach(),
        "iou": iou.detach(),
        "positive_mask": image_true.detach(),
        "operating_score": operating_score.detach(),
        "image_tp": int((image_pred & image_true).sum().item()),
        "image_fp": int((image_pred & ~image_true).sum().item()),
        "image_fn": int((~image_pred & image_true).sum().item()),
        "image_tn": int((~image_pred & ~image_true).sum().item()),
    }
    if aux_scores is not None:
        aux_pred = aux_scores >= float(threshold)
        metrics.update(
            {
                "aux_score": aux_scores.detach(),
                "aux_image_tp": int((aux_pred & image_true).sum().item()),
                "aux_image_fp": int((aux_pred & ~image_true).sum().item()),
                "aux_image_fn": int((~aux_pred & image_true).sum().item()),
                "aux_image_tn": int((~aux_pred & ~image_true).sum().item()),
            }
        )
    return metrics, preds


def update_eval_accumulator(acc, probs, metrics, preds, batch_size):
    p3.update_metric_accumulator(
        acc,
        batch_size=batch_size,
        loss_value=0.0,
        bce_value=0.0,
        dice_loss_value=0.0,
        metrics=metrics,
    )

    pred_frac = preds.float().mean(dim=(1, 2, 3))
    positive_mask = metrics["positive_mask"]
    negative_mask = ~positive_mask

    acc["pred_frac_sum"] += float(pred_frac.sum().item())
    if bool(positive_mask.any()):
        acc["pred_frac_pos_sum"] += float(pred_frac[positive_mask].sum().item())
    if bool(negative_mask.any()):
        acc["pred_frac_neg_sum"] += float(pred_frac[negative_mask].sum().item())
        acc["num_negative_samples"] += int(negative_mask.sum().item())
    if "aux_score" in metrics:
        aux_score = metrics["aux_score"]
        acc["aux_score_sum"] += float(aux_score.sum().item())
        acc["aux_score_count"] += int(aux_score.numel())
        if bool(positive_mask.any()):
            acc["aux_score_pos_sum"] += float(aux_score[positive_mask].sum().item())
        if bool(negative_mask.any()):
            acc["aux_score_neg_sum"] += float(aux_score[negative_mask].sum().item())
        acc["aux_image_tp"] += metrics["aux_image_tp"]
        acc["aux_image_fp"] += metrics["aux_image_fp"]
        acc["aux_image_fn"] += metrics["aux_image_fn"]
        acc["aux_image_tn"] += metrics["aux_image_tn"]
    operating_score = metrics["operating_score"]
    acc["operating_score_sum"] += float(operating_score.sum().item())
    acc["operating_score_count"] += int(operating_score.numel())
    if bool(positive_mask.any()):
        acc["operating_score_pos_sum"] += float(operating_score[positive_mask].sum().item())
    if bool(negative_mask.any()):
        acc["operating_score_neg_sum"] += float(operating_score[negative_mask].sum().item())


def finalize_eval_metrics(acc):
    metrics = p3.finalize_metrics(acc)
    n = max(acc["num_samples"], 1)
    pos_n = max(acc["num_positive_samples"], 1)
    neg_n = max(acc["num_negative_samples"], 1)
    metrics.update(
        {
            "mean_predicted_positive_fraction": acc["pred_frac_sum"] / n,
            "mean_predicted_positive_fraction_pos": acc["pred_frac_pos_sum"] / pos_n,
            "mean_predicted_positive_fraction_neg": acc["pred_frac_neg_sum"] / neg_n,
            "num_negative_samples": acc["num_negative_samples"],
            "operating_score_mean": acc["operating_score_sum"] / max(acc["operating_score_count"], 1),
            "operating_score_mean_pos": acc["operating_score_pos_sum"] / pos_n,
            "operating_score_mean_neg": acc["operating_score_neg_sum"] / neg_n,
        }
    )
    if acc["aux_score_count"] > 0:
        aux_precision = acc["aux_image_tp"] / max(acc["aux_image_tp"] + acc["aux_image_fp"], 1)
        aux_recall = acc["aux_image_tp"] / max(acc["aux_image_tp"] + acc["aux_image_fn"], 1)
        aux_f1 = 2.0 * aux_precision * aux_recall / max(aux_precision + aux_recall, p3.EPS)
        aux_specificity = acc["aux_image_tn"] / max(acc["aux_image_tn"] + acc["aux_image_fp"], 1)
        aux_fpr = acc["aux_image_fp"] / max(acc["aux_image_fp"] + acc["aux_image_tn"], 1)
        metrics.update(
            {
                "aux_image_precision": aux_precision,
                "aux_image_recall": aux_recall,
                "aux_image_f1": aux_f1,
                "aux_image_specificity": aux_specificity,
                "aux_image_false_positive_rate": aux_fpr,
                "aux_score_mean": acc["aux_score_sum"] / max(acc["aux_score_count"], 1),
                "aux_score_mean_pos": acc["aux_score_pos_sum"] / pos_n,
                "aux_score_mean_neg": acc["aux_score_neg_sum"] / neg_n,
                "aux_image_tp": acc["aux_image_tp"],
                "aux_image_fp": acc["aux_image_fp"],
                "aux_image_fn": acc["aux_image_fn"],
                "aux_image_tn": acc["aux_image_tn"],
            }
        )
    return metrics


def maybe_collect_preview_samples(store, images, masks, logits, indices, max_examples):
    if len(store["positive"]) >= max_examples and len(store["negative"]) >= max_examples:
        return

    truth_positive = masks.sum(axis=(1, 2, 3)) > 0.5
    for row in range(len(indices)):
        target_key = "positive" if truth_positive[row] else "negative"
        if len(store[target_key]) >= max_examples:
            continue
        store[target_key].append(
            {
                "image": images[row].copy(),
                "mask": masks[row].copy(),
                "logit": logits[row].copy(),
                "index": int(indices[row]),
            }
        )
        if len(store["positive"]) >= max_examples and len(store["negative"]) >= max_examples:
            return


def stack_preview_samples(samples):
    if not samples:
        return None
    return {
        "images": np.stack([sample["image"] for sample in samples], axis=0),
        "masks": np.stack([sample["mask"] for sample in samples], axis=0),
        "logits": np.stack([sample["logit"] for sample in samples], axis=0),
        "indices": np.asarray([sample["index"] for sample in samples], dtype=np.int64),
    }


def save_threshold_sweep_plot(rows, output_path, best_threshold, selection_label, configured_threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = np.asarray([row["threshold"] for row in rows], dtype=np.float64)
    precision = np.asarray([row["image_precision"] for row in rows], dtype=np.float64)
    recall = np.asarray([row["image_recall"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["image_f1"] for row in rows], dtype=np.float64)
    fpr = np.asarray([row["image_false_positive_rate"] for row in rows], dtype=np.float64)
    specificity = np.asarray([row["image_specificity"] for row in rows], dtype=np.float64)
    hard_dice_pos = np.asarray([row["hard_dice_pos"] for row in rows], dtype=np.float64)
    iou_pos = np.asarray([row["iou_pos"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    axes[0].plot(thresholds, precision, label="precision", color="#2563eb")
    axes[0].plot(thresholds, recall, label="recall", color="#dc2626")
    axes[0].plot(thresholds, f1, label="F1", color="#16a34a")
    axes[0].axvline(best_threshold, color="#111827", linestyle="--", linewidth=1.2, label="selected")
    axes[0].axvline(configured_threshold, color="#7c3aed", linestyle=":", linewidth=1.2, label="configured")
    axes[0].set_title("Image-level Metrics vs Threshold")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Metric")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(thresholds, fpr, label="false positive rate", color="#ea580c")
    axes[1].plot(thresholds, specificity, label="specificity", color="#0891b2")
    axes[1].axvline(best_threshold, color="#111827", linestyle="--", linewidth=1.2)
    axes[1].axvline(configured_threshold, color="#7c3aed", linestyle=":", linewidth=1.2)
    axes[1].set_title("Negative-control Metrics vs Threshold")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Metric")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(thresholds, hard_dice_pos, label="hard Dice+", color="#2563eb")
    axes[2].plot(thresholds, iou_pos, label="IoU+", color="#16a34a")
    axes[2].axvline(best_threshold, color="#111827", linestyle="--", linewidth=1.2)
    axes[2].axvline(configured_threshold, color="#7c3aed", linestyle=":", linewidth=1.2)
    axes[2].set_title("Positive-sample Segmentation vs Threshold")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Metric")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    scatter = axes[3].scatter(recall, precision, c=thresholds, cmap="viridis", s=28, edgecolor="none")
    best_idx = int(np.argmin(np.abs(thresholds - best_threshold)))
    axes[3].scatter(
        recall[best_idx],
        precision[best_idx],
        color="#111827",
        s=60,
        marker="x",
        label=selection_label,
    )
    axes[3].set_title("Precision-Recall Curve")
    axes[3].set_xlabel("Recall")
    axes[3].set_ylabel("Precision")
    axes[3].set_xlim(0.0, 1.05)
    axes[3].set_ylim(0.0, 1.05)
    axes[3].grid(alpha=0.25)
    axes[3].legend()
    fig.colorbar(scatter, ax=axes[3], label="Threshold")

    fig.suptitle("Phase 3 Threshold Sweep", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_thresholds(
    model,
    loader,
    thresholds,
    device,
    preview_count,
    split_name,
    image_min_positive_pixels,
    use_image_aux,
    radius_bin_count,
    image_rule,
    score_mode,
):
    accumulators = [make_eval_accumulator() for _ in thresholds]
    previews = {"positive": [], "negative": []}
    progress = p3.ProgressPrinter(len(loader), f"Eval {split_name}")

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits, aux_logits = p3.unpack_model_output(model(images))
            image_logits, _ = p3.split_aux_logits(
                aux_logits,
                use_image_aux=use_image_aux,
                radius_bin_count=radius_bin_count,
            )
            probs = torch.sigmoid(logits)
            aux_scores = torch.sigmoid(image_logits.reshape(-1)) if image_logits is not None else None

            images_cpu = images.detach().cpu().numpy()
            masks_cpu = masks.detach().cpu().numpy()
            logits_cpu = logits.detach().cpu().numpy()
            indices_cpu = np.asarray(batch["index"][:], dtype=np.int64)

            maybe_collect_preview_samples(
                previews,
                images=images_cpu,
                masks=masks_cpu,
                logits=logits_cpu,
                indices=indices_cpu,
                max_examples=preview_count,
            )

            for threshold, acc in zip(thresholds, accumulators):
                metrics, preds = batch_metrics_from_probs(
                    probs,
                    masks,
                    float(threshold),
                    image_min_positive_pixels=image_min_positive_pixels,
                    aux_scores=aux_scores,
                    image_rule=image_rule,
                    score_mode=score_mode,
                )
                update_eval_accumulator(
                    acc,
                    probs=probs,
                    metrics=metrics,
                    preds=preds,
                    batch_size=int(images.shape[0]),
                )

            progress.update(batch_idx)

    return [finalize_eval_metrics(acc) for acc in accumulators], previews


def choose_best_threshold(rows, selection_metric):
    if rows and selection_metric not in rows[0]:
        raise RuntimeError(
            f"Selection metric {selection_metric!r} is unavailable in threshold rows. "
            "Auxiliary-head metrics require a checkpoint trained with --aux-head-weight > 0."
        )
    best_row = None
    for row in rows:
        if best_row is None:
            best_row = row
            continue
        if row[selection_metric] > best_row[selection_metric] + 1e-12:
            best_row = row
            continue
        if abs(row[selection_metric] - best_row[selection_metric]) <= 1e-12 and row["threshold"] > best_row["threshold"]:
            best_row = row
    return best_row


def choose_operating_point(rows, operating_point_rule, selection_metric, target_fpr):
    if operating_point_rule == "metric_max":
        best_row = choose_best_threshold(rows, selection_metric)
        return best_row, {
            "rule": "metric_max",
            "selection_metric": selection_metric,
        }

    feasible = [row for row in rows if row["image_false_positive_rate"] <= target_fpr + 1e-12]
    if feasible:
        best_row = max(
            feasible,
            key=lambda row: (
                row["image_recall"],
                row["image_precision"],
                row["hard_dice_pos"],
                row["threshold"],
            ),
        )
        return best_row, {
            "rule": "max_recall_under_fpr_cap",
            "target_fpr": float(target_fpr),
            "feasible_thresholds": int(len(feasible)),
        }

    best_row = min(
        rows,
        key=lambda row: (
            row["image_false_positive_rate"],
            -row["image_recall"],
            -row["threshold"],
        ),
    )
    return best_row, {
        "rule": "fallback_min_fpr_then_max_recall",
        "target_fpr": float(target_fpr),
        "feasible_thresholds": 0,
    }


def find_closest_threshold_row(rows, target_threshold):
    return min(rows, key=lambda row: abs(row["threshold"] - target_threshold))


def canonical_threshold_grid(values):
    """Return sorted unique thresholds with floating-point duplicates merged."""

    values = np.asarray(values, dtype=np.float64)
    return np.unique(np.round(values, 10))


def collect_prediction_artifacts(
    model,
    loader,
    device,
    threshold,
    preview_count,
    glon_lookup,
    glat_lookup,
    provenance_lookup,
    truth_lookup,
    use_image_aux,
    radius_bin_count,
    score_mode,
):
    previews = {"positive": [], "negative": []}
    candidate_records = []
    mask_bits = []
    sample_indices = []
    progress = p3.ProgressPrinter(len(loader), "Collect predictions")

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits, aux_logits = p3.unpack_model_output(model(images))
            image_logits, _ = p3.split_aux_logits(
                aux_logits,
                use_image_aux=use_image_aux,
                radius_bin_count=radius_bin_count,
            )
            probs = torch.sigmoid(logits)
            aux_scores = torch.sigmoid(image_logits.reshape(-1)) if image_logits is not None else None

            images_cpu = images.detach().cpu().numpy()
            masks_cpu = masks.detach().cpu().numpy()
            logits_cpu = logits.detach().cpu().numpy()
            probs_cpu = probs.detach().cpu().numpy()
            aux_scores_cpu = aux_scores.detach().cpu().numpy() if aux_scores is not None else None
            indices_cpu = np.asarray(batch["index"][:], dtype=np.int64)

            maybe_collect_preview_samples(
                previews,
                images=images_cpu,
                masks=masks_cpu,
                logits=logits_cpu,
                indices=indices_cpu,
                max_examples=preview_count,
            )

            for row, sample_idx in enumerate(indices_cpu):
                candidate, candidate_mask = build_patch_candidate(
                    prob_map=probs_cpu[row, 0],
                    threshold=threshold,
                    patch_center_glon_deg=float(glon_lookup[sample_idx]),
                    patch_center_glat_deg=float(glat_lookup[sample_idx]),
                    sample_index=int(sample_idx),
                )
                candidate.update(
                    {
                        "score_mode": score_mode,
                        "segmentation_max_score": float(np.max(probs_cpu[row, 0])),
                        "aux_score": float(aux_scores_cpu[row]) if aux_scores_cpu is not None else None,
                        "calibrated_composite_score": (
                            float(0.5 * (np.max(probs_cpu[row, 0]) + aux_scores_cpu[row]))
                            if aux_scores_cpu is not None
                            else None
                        ),
                        "truth_label": int(truth_lookup["has_signal"][sample_idx]),
                        "truth_theta_crit_deg": float(truth_lookup["theta_crit_deg"][sample_idx]),
                        "truth_z0": float(truth_lookup["z0"][sample_idx]),
                        "truth_zcrit": float(truth_lookup["zcrit"][sample_idx]),
                        "truth_edge_sigma_deg": float(truth_lookup["edge_sigma_deg"][sample_idx]),
                        "truth_signal_center_dx_deg": float(truth_lookup["signal_center_dx_deg"][sample_idx]),
                        "truth_signal_center_dy_deg": float(truth_lookup["signal_center_dy_deg"][sample_idx]),
                        "coord_pool_idx": int(provenance_lookup["coord_pool_idx"][sample_idx]),
                        "coord_mask_fraction": float(provenance_lookup["coord_mask_fraction"][sample_idx]),
                        "cmb_realization_idx": int(provenance_lookup["cmb_realization_idx"][sample_idx]),
                        "background_id": int(provenance_lookup["background_id"][sample_idx]),
                    }
                )
                candidate_records.append(candidate)
                sample_indices.append(int(sample_idx))
                mask_bits.append(np.packbits(candidate_mask.reshape(-1)).astype(np.uint8))

            progress.update(batch_idx)

    return {
        "records": candidate_records,
        "sample_indices": np.asarray(sample_indices, dtype=np.int64),
        "mask_bits": np.stack(mask_bits, axis=0),
        "mask_shape": np.asarray(MASK_SHAPE, dtype=np.int32),
        "previews": previews,
    }


def unpack_mask_row(mask_bits_row, mask_shape):
    flat = np.unpackbits(mask_bits_row)[: int(np.prod(mask_shape))]
    return flat.reshape(mask_shape).astype(bool)


def positive_subset_metrics(mask_bits, sample_indices, truth_masks_h5, include_mask):
    selected_positions = np.flatnonzero(include_mask)
    if len(selected_positions) == 0:
        return {"num_samples": 0}

    hard_dice_sum = 0.0
    iou_sum = 0.0
    detected = 0
    for position in selected_positions:
        sample_idx = int(sample_indices[position])
        pred = unpack_mask_row(mask_bits[position], MASK_SHAPE)
        truth = np.asarray(truth_masks_h5[sample_idx], dtype=bool)
        pred_any = bool(pred.any())
        if pred_any:
            detected += 1

        intersection = float(np.logical_and(pred, truth).sum())
        pred_sum = float(pred.sum())
        truth_sum = float(truth.sum())
        union = pred_sum + truth_sum - intersection
        empty_both = pred_sum == 0.0 and truth_sum == 0.0
        if empty_both:
            hard_dice = 1.0
            iou = 1.0
        else:
            hard_dice = (2.0 * intersection + p3.EPS) / (pred_sum + truth_sum + p3.EPS)
            iou = (intersection + p3.EPS) / (union + p3.EPS)

        hard_dice_sum += hard_dice
        iou_sum += iou

    n = max(len(selected_positions), 1)
    return {
        "num_samples": int(len(selected_positions)),
        "detected": int(detected),
        "image_recall": float(detected / n),
        "hard_dice_pos": float(hard_dice_sum / n),
        "iou_pos": float(iou_sum / n),
    }


def make_stratified_metrics(data_h5, artifact_bundle):
    sample_indices = artifact_bundle["sample_indices"]
    records = artifact_bundle["records"]
    mask_bits = artifact_bundle["mask_bits"]

    theta = np.asarray([record["truth_theta_crit_deg"] for record in records], dtype=np.float64)
    signal_strength = np.asarray(
        [max(abs(record["truth_z0"]), abs(record["truth_zcrit"])) for record in records],
        dtype=np.float64,
    )
    edge_strength = np.asarray([abs(record["truth_zcrit"]) for record in records], dtype=np.float64)
    has_signal = np.asarray([record["truth_label"] == 1 for record in records], dtype=bool)

    theta_bins = [
        ("5-10deg", (theta >= 5.0) & (theta < 10.0)),
        ("10-15deg", (theta >= 10.0) & (theta < 15.0)),
        ("15-20deg", (theta >= 15.0) & (theta < 20.0)),
        ("20-25deg", (theta >= 20.0) & (theta <= 25.0)),
    ]
    amplitude_bins = [
        ("1e-6_to_1e-5", (signal_strength >= 1e-6) & (signal_strength < 1e-5)),
        ("1e-5_to_3e-5", (signal_strength >= 1e-5) & (signal_strength < 3e-5)),
        ("3e-5_to_5e-5", (signal_strength >= 3e-5) & (signal_strength < 5e-5)),
        ("5e-5_to_1e-4", (signal_strength >= 5e-5) & (signal_strength <= 1e-4)),
    ]
    edge_bins = [
        ("smooth_|zcrit|<=5e-6", edge_strength <= 5e-6),
        ("weak_5e-6_to_3e-5", (edge_strength > 5e-6) & (edge_strength < 3e-5)),
        ("strong_|zcrit|>=3e-5", edge_strength >= 3e-5),
    ]

    stratified = {"theta_crit": {}, "amplitude": {}, "edge_strength": {}}
    with h5py.File(data_h5, "r") as h5:
        truth_masks_h5 = h5["masks"]
        for name, mask in theta_bins:
            stratified["theta_crit"][name] = positive_subset_metrics(
                mask_bits=mask_bits,
                sample_indices=sample_indices,
                truth_masks_h5=truth_masks_h5,
                include_mask=has_signal & mask,
            )
        for name, mask in amplitude_bins:
            stratified["amplitude"][name] = positive_subset_metrics(
                mask_bits=mask_bits,
                sample_indices=sample_indices,
                truth_masks_h5=truth_masks_h5,
                include_mask=has_signal & mask,
            )
        for name, mask in edge_bins:
            stratified["edge_strength"][name] = positive_subset_metrics(
                mask_bits=mask_bits,
                sample_indices=sample_indices,
                truth_masks_h5=truth_masks_h5,
                include_mask=has_signal & mask,
            )
    return stratified


def save_candidate_outputs(output_dir, artifact_bundle):
    candidate_records_path = output_dir / "candidate_records.jsonl"
    candidate_masks_path = output_dir / "candidate_masks.npz"

    with open(candidate_records_path, "w", encoding="utf-8") as handle:
        for row_id, record in enumerate(artifact_bundle["records"]):
            payload = dict(record)
            payload["mask_row"] = row_id
            handle.write(json.dumps(payload) + "\n")

    np.savez_compressed(
        candidate_masks_path,
        sample_indices=artifact_bundle["sample_indices"],
        mask_bits=artifact_bundle["mask_bits"],
        mask_shape=artifact_bundle["mask_shape"],
    )
    return candidate_records_path, candidate_masks_path


def main():
    args = parse_args()
    validate_args(args)
    p3.require_ml_packages()
    p3.seed_everything(42)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, args.checkpoint)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(run_dir, checkpoint_label, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_json(run_dir / "run_config.json")
    split_indices = load_split_indices(run_dir, args.split)
    train_args = run_config["args"]
    batch_size = args.batch_size or int(train_args["batch_size"])
    configured_threshold = float(train_args["threshold"])
    data_h5 = Path(run_config["data_h5"])

    mean = float(run_config["normalization"]["train_mean"])
    std = float(run_config["normalization"]["train_std"])
    dataset = p3.H5BubbleDataset(
        h5_path=data_h5,
        indices=split_indices,
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(train_args["seed"]) + (0 if args.split == "train" else 1),
        max_translate_pixels=0,
    )
    device = p3.resolve_device(args.device)
    loader = p3.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=p3.seed_worker if args.num_workers > 0 else None,
        generator=torch.Generator().manual_seed(0),
    )

    model_args = p3.model_args_from_run_config(run_config)
    radius_bin_edges = p3.parse_radius_bin_edges(getattr(model_args, "radius_bin_edges_deg", "5,10,15,20,25"))
    radius_bin_count = (
        p3.radius_bin_count_from_edges(radius_bin_edges)
        if getattr(model_args, "radius_head_weight", 0.0) > 0.0
        else 0
    )
    use_image_aux = getattr(model_args, "aux_head_weight", 0.0) > 0.0
    if args.score_mode in {"aux_score", "calibrated_composite"} and not use_image_aux:
        raise RuntimeError(f"--score-mode {args.score_mode} requires a checkpoint trained with --aux-head-weight > 0.")
    model = p3.build_model(model_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])

    thresholds = canonical_threshold_grid(np.linspace(args.threshold_min, args.threshold_max, args.threshold_count, dtype=np.float64))
    fixed_threshold_source = None
    if args.use_configured_threshold:
        args.fixed_threshold = configured_threshold
        fixed_threshold_source = "run_config"
    if args.fixed_threshold > 0.0:
        if fixed_threshold_source is None:
            fixed_threshold_source = "cli"
        thresholds = canonical_threshold_grid(np.concatenate([thresholds, np.asarray([args.fixed_threshold], dtype=np.float64)]))
    metrics_per_threshold, previews = evaluate_thresholds(
        model=model,
        loader=loader,
        thresholds=thresholds,
        device=device,
        preview_count=args.preview_count,
        split_name=args.split,
        image_min_positive_pixels=args.image_min_positive_pixels,
        use_image_aux=use_image_aux,
        radius_bin_count=radius_bin_count,
        image_rule=args.image_rule,
        score_mode=args.score_mode,
    )

    rows = []
    for threshold, metrics in zip(thresholds, metrics_per_threshold):
        row = {"threshold": float(threshold)}
        row.update(metrics)
        rows.append(row)

    if args.fixed_threshold > 0.0:
        operating_row = find_closest_threshold_row(rows, args.fixed_threshold)
        operating_point = {
            "rule": "fixed_threshold",
            "threshold_source": fixed_threshold_source,
            "threshold": float(args.fixed_threshold),
        }
    else:
        operating_row, operating_point = choose_operating_point(
            rows=rows,
            operating_point_rule=args.operating_point_rule,
            selection_metric=args.selection_metric,
            target_fpr=args.target_fpr,
        )
    configured_row = find_closest_threshold_row(rows, configured_threshold)

    selection_label = operating_point["rule"]
    p3.save_json(output_dir / "threshold_metrics.json", rows)
    save_threshold_sweep_plot(
        rows=rows,
        output_path=output_dir / "threshold_sweep.png",
        best_threshold=float(operating_row["threshold"]),
        selection_label=selection_label,
        configured_threshold=configured_threshold,
    )

    positive_preview = stack_preview_samples(previews["positive"])
    negative_preview = stack_preview_samples(previews["negative"])
    if positive_preview is not None:
        p3.save_prediction_preview(
            images=positive_preview["images"],
            masks=positive_preview["masks"],
            logits=positive_preview["logits"],
            output_path=output_dir / "selected_threshold_positive_preview.png",
            mean=mean,
            std=std,
            threshold=float(operating_row["threshold"]),
            indices=positive_preview["indices"],
        )
    if negative_preview is not None:
        p3.save_prediction_preview(
            images=negative_preview["images"],
            masks=negative_preview["masks"],
            logits=negative_preview["logits"],
            output_path=output_dir / "selected_threshold_negative_preview.png",
            mean=mean,
            std=std,
            threshold=float(operating_row["threshold"]),
            indices=negative_preview["indices"],
        )

    glon_lookup = load_metadata_array(data_h5, "glon_deg", dtype=np.float32)
    glat_lookup = load_metadata_array(data_h5, "glat_deg", dtype=np.float32)
    provenance_lookup = {
        "coord_pool_idx": load_optional_metadata_array(data_h5, "coord_pool_idx", dtype=np.int32, default_value=-1),
        "coord_mask_fraction": load_optional_metadata_array(
            data_h5,
            "coord_mask_fraction",
            dtype=np.float32,
            default_value=np.nan,
        ),
        "cmb_realization_idx": load_metadata_array(data_h5, "cmb_realization_idx", dtype=np.int32),
        "background_id": load_optional_metadata_array(data_h5, "background_id", dtype=np.uint64, default_value=0),
    }
    truth_lookup = {
        "has_signal": load_truth_array(data_h5, "has_signal", dtype=np.uint8),
        "theta_crit_deg": load_truth_array(data_h5, "theta_crit_deg", dtype=np.float32),
        "z0": load_truth_array(data_h5, "z0", dtype=np.float32),
        "zcrit": load_truth_array(data_h5, "zcrit", dtype=np.float32),
        "edge_sigma_deg": load_truth_array(data_h5, "edge_sigma_deg", dtype=np.float32),
        "signal_center_dx_deg": load_truth_array(data_h5, "signal_center_dx_deg", dtype=np.float32),
        "signal_center_dy_deg": load_truth_array(data_h5, "signal_center_dy_deg", dtype=np.float32),
    }

    artifact_loader = p3.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=p3.seed_worker if args.num_workers > 0 else None,
        generator=torch.Generator().manual_seed(0),
    )
    artifact_bundle = collect_prediction_artifacts(
        model=model,
        loader=artifact_loader,
        device=device,
        threshold=float(operating_row["threshold"]),
        preview_count=args.preview_count,
        glon_lookup=glon_lookup,
        glat_lookup=glat_lookup,
        provenance_lookup=provenance_lookup,
        truth_lookup=truth_lookup,
        use_image_aux=use_image_aux,
        radius_bin_count=radius_bin_count,
        score_mode=args.score_mode,
    )
    stratified_metrics = make_stratified_metrics(data_h5=data_h5, artifact_bundle=artifact_bundle)
    p3.save_json(output_dir / "stratified_metrics.json", stratified_metrics)
    candidate_records_path, candidate_masks_path = save_candidate_outputs(output_dir, artifact_bundle)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": checkpoint_label,
        "split": args.split,
        "num_samples": int(len(split_indices)),
        "configured_threshold": configured_threshold,
        "image_min_positive_pixels": int(args.image_min_positive_pixels),
        "image_rule": args.image_rule,
        "score_mode": args.score_mode,
        "aux_head_evaluated": bool(use_image_aux),
        "configured_threshold_metrics": configured_row,
        "operating_point": operating_point,
        "selected_threshold": float(operating_row["threshold"]),
        "selected_threshold_metrics": operating_row,
        "artifacts": {
            "threshold_metrics_json": str((output_dir / "threshold_metrics.json").resolve()),
            "threshold_sweep_plot": str((output_dir / "threshold_sweep.png").resolve()),
            "positive_preview": str((output_dir / "selected_threshold_positive_preview.png").resolve()),
            "negative_preview": str((output_dir / "selected_threshold_negative_preview.png").resolve()),
            "stratified_metrics_json": str((output_dir / "stratified_metrics.json").resolve()),
            "candidate_records_jsonl": str(candidate_records_path.resolve()),
            "candidate_masks_npz": str(candidate_masks_path.resolve()),
        },
    }
    p3.save_json(output_dir / "evaluation_summary.json", summary)

    output_audit = run_output_audit(output_dir)
    p3.save_json(output_dir / "output_audit.json", output_audit)
    if output_audit["status"] != "pass":
        raise RuntimeError(f"Evaluation outputs failed audit. See {output_dir / 'output_audit.json'}")

    print("\n=== Phase 3 evaluation summary ===")
    print(f"  Run dir:                {run_dir}")
    print(f"  Checkpoint:             {checkpoint_path}")
    print(f"  Split:                  {args.split} ({len(split_indices)} samples)")
    print(f"  Configured threshold:   {configured_threshold:.3f}")
    print(f"  Operating-point rule:   {operating_point['rule']}")
    if "target_fpr" in operating_point:
        print(f"  Target image FPR:       {operating_point['target_fpr']:.3f}")
    print(f"  Selected threshold:     {operating_row['threshold']:.3f}")
    print(
        f"  Selected image F1:      {operating_row['image_f1']:.3f} "
        f"(precision={operating_row['image_precision']:.3f}, "
        f"recall={operating_row['image_recall']:.3f}, "
        f"fpr={operating_row['image_false_positive_rate']:.3f})"
    )
    print(f"  Threshold sweep plot:   {output_dir / 'threshold_sweep.png'}")
    print(f"  Stratified metrics:     {output_dir / 'stratified_metrics.json'}")
    print(f"  Candidate records:      {candidate_records_path}")
    print(f"  Output audit:           {output_dir / 'output_audit.json'}")
    print(f"  Evaluation summary:     {output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
