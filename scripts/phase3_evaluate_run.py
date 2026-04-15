"""
Phase 3: Evaluate a trained U-Net checkpoint with threshold sweeps and previews.

This script reuses the split, normalization, and model configuration saved by
`phase3_train_unet.py` so a finished run can be scored consistently after
training. The main outputs are:
    - a threshold sweep JSON file
    - a threshold sweep plot
    - summary JSON with the best threshold on the chosen split
    - separate positive/negative preview panels at the best threshold
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import phase3_train_unet as p3

SELECTION_METRICS = (
    "image_f1",
    "image_precision",
    "image_recall",
    "image_specificity",
    "hard_dice_pos",
    "iou_pos",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Phase 3 checkpoint with a threshold sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint to evaluate: `best`, `last`, or a checkpoint path.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
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
        "--selection-metric",
        type=str,
        default="image_f1",
        choices=SELECTION_METRICS,
        help="Metric used to choose the best threshold from the sweep.",
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
    if not (0.0 < args.threshold_min < 1.0):
        raise ValueError("--threshold-min must be between 0 and 1.")
    if not (0.0 < args.threshold_max < 1.0):
        raise ValueError("--threshold-max must be between 0 and 1.")
    if args.threshold_min >= args.threshold_max:
        raise ValueError("--threshold-min must be smaller than --threshold-max.")
    if args.preview_count <= 0:
        raise ValueError("--preview-count must be positive.")


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
    return np.asarray(split_data[f"{split_name}_idx"], dtype=np.int64)


def make_eval_accumulator():
    acc = p3.make_metric_accumulator()
    acc.update(
        {
            "pred_frac_sum": 0.0,
            "pred_frac_pos_sum": 0.0,
            "pred_frac_neg_sum": 0.0,
            "num_negative_samples": 0,
        }
    )
    return acc


def batch_metrics_from_probs(probs, targets, threshold):
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

    image_pred = pred_sum > 0
    image_true = truth_sum > 0

    return {
        "hard_dice": hard_dice.detach(),
        "iou": iou.detach(),
        "positive_mask": image_true.detach(),
        "image_tp": int((image_pred & image_true).sum().item()),
        "image_fp": int((image_pred & ~image_true).sum().item()),
        "image_fn": int((~image_pred & image_true).sum().item()),
        "image_tn": int((~image_pred & ~image_true).sum().item()),
    }, preds


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
    images = np.stack([sample["image"] for sample in samples], axis=0)
    masks = np.stack([sample["mask"] for sample in samples], axis=0)
    logits = np.stack([sample["logit"] for sample in samples], axis=0)
    indices = np.asarray([sample["index"] for sample in samples], dtype=np.int64)
    return {
        "images": images,
        "masks": masks,
        "logits": logits,
        "indices": indices,
    }


def save_threshold_sweep_plot(rows, output_path, best_threshold, selection_metric, configured_threshold):
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
    axes[0].axvline(best_threshold, color="#111827", linestyle="--", linewidth=1.2, label="best")
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

    scatter = axes[3].scatter(
        recall,
        precision,
        c=thresholds,
        cmap="viridis",
        s=28,
        edgecolor="none",
    )
    best_idx = int(np.argmax([row[selection_metric] for row in rows]))
    axes[3].scatter(
        recall[best_idx],
        precision[best_idx],
        color="#111827",
        s=60,
        marker="x",
        label=f"best {selection_metric}",
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


def evaluate_thresholds(model, loader, thresholds, device, preview_count, split_name):
    accumulators = [make_eval_accumulator() for _ in thresholds]
    previews = {"positive": [], "negative": []}
    progress = p3.ProgressPrinter(len(loader), f"Eval {split_name}")

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)

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
                metrics, preds = batch_metrics_from_probs(probs, masks, float(threshold))
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


def find_closest_threshold_row(rows, target_threshold):
    return min(rows, key=lambda row: abs(row["threshold"] - target_threshold))


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
        mean=mean,
        std=std,
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
    )

    model_args = argparse.Namespace(
        encoder_name=train_args["encoder_name"],
        encoder_weights=train_args["encoder_weights"],
    )
    model = p3.build_model(model_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_count, dtype=np.float64)
    metrics_per_threshold, previews = evaluate_thresholds(
        model=model,
        loader=loader,
        thresholds=thresholds,
        device=device,
        preview_count=args.preview_count,
        split_name=args.split,
    )

    rows = []
    for threshold, metrics in zip(thresholds, metrics_per_threshold):
        row = {"threshold": float(threshold)}
        row.update(metrics)
        rows.append(row)

    best_row = choose_best_threshold(rows, args.selection_metric)
    configured_row = find_closest_threshold_row(rows, configured_threshold)

    p3.save_json(output_dir / "threshold_metrics.json", rows)
    save_threshold_sweep_plot(
        rows=rows,
        output_path=output_dir / "threshold_sweep.png",
        best_threshold=float(best_row["threshold"]),
        selection_metric=args.selection_metric,
        configured_threshold=configured_threshold,
    )

    positive_preview = stack_preview_samples(previews["positive"])
    negative_preview = stack_preview_samples(previews["negative"])
    if positive_preview is not None:
        p3.save_prediction_preview(
            images=positive_preview["images"],
            masks=positive_preview["masks"],
            logits=positive_preview["logits"],
            output_path=output_dir / "best_threshold_positive_preview.png",
            mean=mean,
            std=std,
            threshold=float(best_row["threshold"]),
            indices=positive_preview["indices"],
        )
    if negative_preview is not None:
        p3.save_prediction_preview(
            images=negative_preview["images"],
            masks=negative_preview["masks"],
            logits=negative_preview["logits"],
            output_path=output_dir / "best_threshold_negative_preview.png",
            mean=mean,
            std=std,
            threshold=float(best_row["threshold"]),
            indices=negative_preview["indices"],
        )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": checkpoint_label,
        "split": args.split,
        "num_samples": int(len(split_indices)),
        "selection_metric": args.selection_metric,
        "configured_threshold": configured_threshold,
        "best_threshold": float(best_row["threshold"]),
        "configured_threshold_metrics": configured_row,
        "best_threshold_metrics": best_row,
        "artifacts": {
            "threshold_metrics_json": str((output_dir / "threshold_metrics.json").resolve()),
            "threshold_sweep_plot": str((output_dir / "threshold_sweep.png").resolve()),
            "positive_preview": str((output_dir / "best_threshold_positive_preview.png").resolve()),
            "negative_preview": str((output_dir / "best_threshold_negative_preview.png").resolve()),
        },
    }
    p3.save_json(output_dir / "evaluation_summary.json", summary)

    print("\n=== Phase 3 evaluation summary ===")
    print(f"  Run dir:                {run_dir}")
    print(f"  Checkpoint:             {checkpoint_path}")
    print(f"  Split:                  {args.split} ({len(split_indices)} samples)")
    print(f"  Configured threshold:   {configured_threshold:.3f}")
    print(
        f"  Best threshold ({args.selection_metric}): "
        f"{best_row['threshold']:.3f}"
    )
    print(
        f"  Best image F1:          {best_row['image_f1']:.3f} "
        f"(precision={best_row['image_precision']:.3f}, "
        f"recall={best_row['image_recall']:.3f}, "
        f"fpr={best_row['image_false_positive_rate']:.3f})"
    )
    print(f"  Threshold sweep plot:   {output_dir / 'threshold_sweep.png'}")
    print(f"  Evaluation summary:     {output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
