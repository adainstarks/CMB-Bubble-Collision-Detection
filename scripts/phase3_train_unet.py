"""
Phase 3: Train a U-Net segmentation model on the Phase 2 HDF5 dataset.

This script expects the Phase 2 generator to have already produced an HDF5 file
with the datasets:
    - patches : [N, 256, 256] float32
    - labels  : [N] uint8
    - masks   : [N, 256, 256] uint8
    - metadata/*

The default model is a U-Net with an EfficientNet encoder using
segmentation-models-pytorch, which keeps the implementation compact while still
matching the project plan.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
from pathlib import Path

import h5py

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as ndi_shift

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover - handled at runtime
    smp = None
    SMP_IMPORT_ERROR = exc
else:
    SMP_IMPORT_ERROR = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_H5 = PROJECT_ROOT / "data" / "training_v2_fixed_10000" / "training_data.h5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "phase3_unet"
EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet-backed U-Net on the synthetic bubble-collision dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-h5", type=str, default=str(DEFAULT_DATA_H5))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-positive-amplitude", type=float, default=0.0)
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b0")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated CUDA device ids to use with DataParallel, e.g. `0,1`.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--max-translate-pixels", type=int, default=48)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-augment", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the dataset, compute normalization statistics, and exit before importing the model.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1.")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")
    if args.preview_count <= 0:
        raise ValueError("--preview-count must be positive.")
    if args.max_train_samples < 0 or args.max_val_samples < 0:
        raise ValueError("--max-train-samples and --max-val-samples must be non-negative.")
    if args.max_translate_pixels < 0:
        raise ValueError("--max-translate-pixels must be non-negative.")
    if args.min_positive_amplitude < 0.0:
        raise ValueError("--min-positive-amplitude must be non-negative.")
    if args.gpu_ids.strip():
        try:
            parse_gpu_ids(args.gpu_ids)
        except ValueError as exc:
            raise ValueError("--gpu-ids must be a comma-separated list of non-negative integers.") from exc


def require_ml_packages():
    missing = []
    if TORCH_IMPORT_ERROR is not None:
        missing.append(f"torch ({TORCH_IMPORT_ERROR})")
    if SMP_IMPORT_ERROR is not None:
        missing.append(f"segmentation_models_pytorch/timm ({SMP_IMPORT_ERROR})")
    if missing:
        raise RuntimeError(
            "Phase 3 training needs additional ML packages.\n"
            "Missing imports:\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\nInstall a CUDA-enabled PyTorch build plus torchvision, "
            "segmentation-models-pytorch, and timm before training."
        )


def make_run_dir(output_root, run_name):
    root = Path(output_root)
    if run_name:
        run_dir = root / run_name
    else:
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_dataset_summary(h5_path):
    summary = {}
    with h5py.File(h5_path, "r") as h5:
        if "summary" in h5:
            for key, value in h5["summary"].attrs.items():
                if isinstance(value, np.generic):
                    summary[key] = value.item()
                else:
                    summary[key] = value
    return summary


def load_labels(h5_path):
    with h5py.File(h5_path, "r") as h5:
        return np.asarray(h5["labels"][:], dtype=np.uint8)


def load_positive_signal_strength(h5_path):
    with h5py.File(h5_path, "r") as h5:
        z0 = np.asarray(h5["metadata"]["z0"][:], dtype=np.float32)
        zcrit = np.asarray(h5["metadata"]["zcrit"][:], dtype=np.float32)
    return np.maximum(np.abs(z0), np.abs(zcrit))


def format_seconds(seconds):
    seconds = max(int(round(seconds)), 0)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class ProgressPrinter:
    def __init__(self, total, label):
        self.total = max(int(total), 1)
        self.label = label
        self.start_time = dt.datetime.utcnow()
        self.last_step = 0

    def update(self, current):
        current = min(max(int(current), 0), self.total)
        if current == self.last_step and current != self.total:
            return
        self.last_step = current

        elapsed = (dt.datetime.utcnow() - self.start_time).total_seconds()
        rate = current / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - current) / rate if rate > 0 else 0.0
        percent = 100.0 * current / self.total
        print(
            f"  {self.label}: {current:4d}/{self.total} "
            f"({percent:5.1f}%) | elapsed {format_seconds(elapsed)} "
            f"| eta {format_seconds(remaining)}",
            flush=True,
        )


def limit_indices(indices, labels, limit, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if limit <= 0 or limit >= len(indices):
        return indices

    selected_labels = labels[indices]
    pos = indices[selected_labels == 1]
    neg = indices[selected_labels == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_target = int(round(limit * len(pos) / max(len(indices), 1)))
    pos_target = min(len(pos), pos_target)
    neg_target = min(len(neg), limit - pos_target)

    if pos_target + neg_target < limit:
        remaining = limit - (pos_target + neg_target)
        extra_pos = min(len(pos) - pos_target, remaining)
        pos_target += extra_pos
        remaining -= extra_pos
        neg_target += min(len(neg) - neg_target, remaining)

    limited = np.concatenate([pos[:pos_target], neg[:neg_target]])
    rng.shuffle(limited)
    return limited


def select_candidate_indices(labels, signal_strength, seed, min_positive_amplitude):
    all_indices = np.arange(len(labels), dtype=np.int64)
    if min_positive_amplitude <= 0.0:
        return all_indices, {
            "min_positive_amplitude": 0.0,
            "retained_positive": int((labels == 1).sum()),
            "retained_negative": int((labels == 0).sum()),
            "candidate_samples": int(len(labels)),
        }

    rng = np.random.default_rng(seed)
    positive_idx = all_indices[(labels == 1) & (signal_strength >= min_positive_amplitude)]
    negative_idx = all_indices[labels == 0]

    if len(positive_idx) == 0:
        raise RuntimeError(
            f"No positive samples remain after applying --min-positive-amplitude={min_positive_amplitude:.2e}."
        )

    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)

    kept_negatives = negative_idx[: len(positive_idx)]
    candidate_indices = np.concatenate([positive_idx, kept_negatives])
    rng.shuffle(candidate_indices)

    return candidate_indices, {
        "min_positive_amplitude": float(min_positive_amplitude),
        "retained_positive": int(len(positive_idx)),
        "retained_negative": int(len(kept_negatives)),
        "candidate_samples": int(len(candidate_indices)),
    }


def stratified_split(labels, train_fraction, seed, max_train_samples=0, max_val_samples=0, candidate_indices=None):
    rng = np.random.default_rng(seed)
    if candidate_indices is None:
        all_indices = np.arange(len(labels), dtype=np.int64)
    else:
        all_indices = np.asarray(candidate_indices, dtype=np.int64)
    selected_labels = labels[all_indices]
    pos = all_indices[selected_labels == 1]
    neg = all_indices[selected_labels == 0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_train = int(round(len(pos) * train_fraction))
    neg_train = int(round(len(neg) * train_fraction))

    train_idx = np.concatenate([pos[:pos_train], neg[:neg_train]])
    val_idx = np.concatenate([pos[pos_train:], neg[neg_train:]])

    train_idx = limit_indices(train_idx, labels, max_train_samples, rng)
    val_idx = limit_indices(val_idx, labels, max_val_samples, rng)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def iter_index_chunks(indices, chunk_size):
    for start in range(0, len(indices), chunk_size):
        chunk = np.asarray(indices[start:start + chunk_size], dtype=np.int64)
        yield np.sort(chunk)


def compute_patch_normalization(h5_path, indices, chunk_size=64):
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0
    progress = ProgressPrinter(len(indices), "Normalization")
    processed = 0

    with h5py.File(h5_path, "r") as h5:
        patches = h5["patches"]
        for chunk in iter_index_chunks(indices, chunk_size):
            batch = np.asarray(patches[chunk], dtype=np.float64)
            total_sum += float(batch.sum())
            total_sq += float(np.square(batch).sum())
            total_count += int(batch.size)
            processed += len(chunk)
            progress.update(processed)

    mean = total_sum / max(total_count, 1)
    variance = max(total_sq / max(total_count, 1) - mean**2, 1e-12)
    std = math.sqrt(variance)
    return float(mean), float(std)


def compute_positive_pixel_fraction(h5_path, indices, chunk_size=128):
    total_positive = 0.0
    total_pixels = 0
    progress = ProgressPrinter(len(indices), "Mask stats")
    processed = 0

    with h5py.File(h5_path, "r") as h5:
        masks = h5["masks"]
        for chunk in iter_index_chunks(indices, chunk_size):
            batch = np.asarray(masks[chunk], dtype=np.float64)
            total_positive += float(batch.sum())
            total_pixels += int(batch.size)
            processed += len(chunk)
            progress.update(processed)

    fraction = total_positive / max(total_pixels, 1)
    return float(fraction)


def compute_pos_weight(positive_fraction):
    if positive_fraction <= 0.0:
        return 1.0
    return float((1.0 - positive_fraction) / positive_fraction)


def count_class_balance(labels, indices):
    selected = labels[indices]
    positives = int(selected.sum())
    negatives = int(len(indices) - positives)
    return positives, negatives


def random_dihedral(patch, mask, rng):
    k = int(rng.integers(0, 4))
    if k:
        patch = np.rot90(patch, k=k)
        mask = np.rot90(mask, k=k)
    if rng.random() < 0.5:
        patch = np.flip(patch, axis=0)
        mask = np.flip(mask, axis=0)
    if rng.random() < 0.5:
        patch = np.flip(patch, axis=1)
        mask = np.flip(mask, axis=1)
    return patch.copy(), mask.copy()


def translate_patch_and_mask(patch, mask, shift_y, shift_x):
    """
    Translate a training example so the network cannot solve the task by
    memorizing that positive masks are always centered in the patch.

    We reflect-pad the CMB patch to avoid introducing artificial blank borders,
    while the binary target mask is shifted with zero fill.
    """
    if shift_x == 0 and shift_y == 0:
        return patch, mask

    patch_shifted = ndi_shift(
        patch,
        shift=(shift_y, shift_x),
        order=1,
        mode="reflect",
        prefilter=False,
    )
    mask_shifted = ndi_shift(
        mask,
        shift=(shift_y, shift_x),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return patch_shifted.astype(np.float32), (mask_shifted > 0.5).astype(np.float32)


def random_translate(patch, mask, rng, max_translate_pixels):
    if max_translate_pixels <= 0:
        return patch, mask

    shift_y = int(rng.integers(-max_translate_pixels, max_translate_pixels + 1))
    shift_x = int(rng.integers(-max_translate_pixels, max_translate_pixels + 1))
    return translate_patch_and_mask(patch, mask, shift_y=shift_y, shift_x=shift_x)


class H5BubbleDataset(Dataset):
    def __init__(self, h5_path, indices, mean, std, augment=False, seed=42, max_translate_pixels=0):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = float(mean)
        self.std = float(max(std, 1e-8))
        self.augment = bool(augment)
        self.seed = int(seed)
        self.max_translate_pixels = int(max_translate_pixels)
        self._h5 = None
        self._rng = None

    def __len__(self):
        return len(self.indices)

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _get_rng(self):
        if self._rng is None:
            if torch is not None:
                worker_info = torch.utils.data.get_worker_info()
                worker_seed = worker_info.seed if worker_info is not None else self.seed
            else:
                worker_seed = self.seed
            self._rng = np.random.default_rng(worker_seed % (2**32))
        return self._rng

    def __getitem__(self, item):
        h5 = self._get_h5()
        index = int(self.indices[item])

        patch = np.asarray(h5["patches"][index], dtype=np.float32)
        mask = np.asarray(h5["masks"][index], dtype=np.float32)

        if self.augment:
            patch, mask = random_translate(
                patch,
                mask,
                self._get_rng(),
                max_translate_pixels=self.max_translate_pixels,
            )
            patch, mask = random_dihedral(patch, mask, self._get_rng())

        patch = (patch - self.mean) / self.std
        patch = patch[None, :, :]
        mask = mask[None, :, :]

        return {
            "image": torch.from_numpy(patch),
            "mask": torch.from_numpy(mask),
            "index": index,
        }

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_gpu_ids(gpu_ids_arg):
    gpu_ids = []
    for token in gpu_ids_arg.split(","):
        token = token.strip()
        if not token:
            continue
        gpu_id = int(token)
        if gpu_id < 0:
            raise ValueError("GPU ids must be non-negative.")
        gpu_ids.append(gpu_id)
    return gpu_ids


def select_data_parallel_device_ids(device, gpu_ids_arg):
    if device.type != "cuda":
        return []
    if not gpu_ids_arg.strip():
        return []

    requested_ids = parse_gpu_ids(gpu_ids_arg)
    visible_count = torch.cuda.device_count()
    if visible_count <= 0:
        raise RuntimeError("CUDA requested, but torch.cuda.device_count() returned zero.")
    invalid_ids = [gpu_id for gpu_id in requested_ids if gpu_id >= visible_count]
    if invalid_ids:
        raise RuntimeError(
            f"Requested GPU ids {invalid_ids}, but only {visible_count} CUDA device(s) are visible."
        )
    return requested_ids


def wrap_model_for_data_parallel(model, device_ids):
    if len(device_ids) <= 1:
        return model
    return nn.DataParallel(model, device_ids=device_ids)


def model_state_dict_for_checkpoint(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    model.load_state_dict(stripped)


def build_model(args):
    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    return smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
        activation=None,
    )


def dice_loss_from_logits(logits, targets):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + EPS) / (denominator + EPS)
    return 1.0 - dice.mean()


def batch_metrics_from_logits(logits, targets, threshold):
    probs = torch.sigmoid(logits)
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
        (2.0 * intersection + EPS) / (pred_sum + truth_sum + EPS),
    )
    iou = torch.where(
        empty_both,
        torch.ones_like(intersection),
        (intersection + EPS) / (union + EPS),
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
    }


def make_metric_accumulator():
    return {
        "loss_sum": 0.0,
        "bce_sum": 0.0,
        "dice_loss_sum": 0.0,
        "soft_dice_sum": 0.0,
        "hard_dice_sum": 0.0,
        "hard_dice_pos_sum": 0.0,
        "iou_sum": 0.0,
        "iou_pos_sum": 0.0,
        "num_samples": 0,
        "num_positive_samples": 0,
        "image_tp": 0,
        "image_fp": 0,
        "image_fn": 0,
        "image_tn": 0,
    }


def update_metric_accumulator(acc, batch_size, loss_value, bce_value, dice_loss_value, metrics):
    acc["loss_sum"] += loss_value * batch_size
    acc["bce_sum"] += bce_value * batch_size
    acc["dice_loss_sum"] += dice_loss_value * batch_size
    acc["soft_dice_sum"] += (1.0 - dice_loss_value) * batch_size
    acc["hard_dice_sum"] += float(metrics["hard_dice"].sum().item())
    acc["iou_sum"] += float(metrics["iou"].sum().item())
    acc["num_samples"] += batch_size

    positive_mask = metrics["positive_mask"].cpu().numpy().astype(bool)
    positive_count = int(positive_mask.sum())
    if positive_count > 0:
        acc["hard_dice_pos_sum"] += float(metrics["hard_dice"][metrics["positive_mask"]].sum().item())
        acc["iou_pos_sum"] += float(metrics["iou"][metrics["positive_mask"]].sum().item())
        acc["num_positive_samples"] += positive_count

    acc["image_tp"] += metrics["image_tp"]
    acc["image_fp"] += metrics["image_fp"]
    acc["image_fn"] += metrics["image_fn"]
    acc["image_tn"] += metrics["image_tn"]


def finalize_metrics(acc):
    n = max(acc["num_samples"], 1)
    pos_n = max(acc["num_positive_samples"], 1)
    precision = acc["image_tp"] / max(acc["image_tp"] + acc["image_fp"], 1)
    recall = acc["image_tp"] / max(acc["image_tp"] + acc["image_fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    specificity = acc["image_tn"] / max(acc["image_tn"] + acc["image_fp"], 1)
    false_positive_rate = acc["image_fp"] / max(acc["image_fp"] + acc["image_tn"], 1)

    return {
        "loss": acc["loss_sum"] / n,
        "bce": acc["bce_sum"] / n,
        "dice_loss": acc["dice_loss_sum"] / n,
        "soft_dice": acc["soft_dice_sum"] / n,
        "hard_dice": acc["hard_dice_sum"] / n,
        "hard_dice_pos": acc["hard_dice_pos_sum"] / pos_n,
        "iou": acc["iou_sum"] / n,
        "iou_pos": acc["iou_pos_sum"] / pos_n,
        "image_precision": precision,
        "image_recall": recall,
        "image_f1": f1,
        "image_specificity": specificity,
        "image_false_positive_rate": false_positive_rate,
        "num_samples": acc["num_samples"],
        "num_positive_samples": acc["num_positive_samples"],
        "image_tp": acc["image_tp"],
        "image_fp": acc["image_fp"],
        "image_fn": acc["image_fn"],
        "image_tn": acc["image_tn"],
    }


def save_prediction_preview(images, masks, logits, output_path, mean, std, threshold, indices):
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = probs >= threshold
    nrows = min(len(images), 6)
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 4 * nrows))
    axes = np.atleast_2d(axes)

    for row in range(nrows):
        raw_patch = images[row, 0] * std + mean
        target_mask = masks[row, 0]
        prob_map = probs[row, 0]
        pred_mask = preds[row, 0]
        vmin, vmax = np.percentile(raw_patch, [1, 99])

        axes[row, 0].imshow(raw_patch, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Patch idx={indices[row]}", fontsize=10)
        axes[row, 1].imshow(target_mask, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 1].set_title("Target mask", fontsize=10)
        axes[row, 2].imshow(prob_map, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axes[row, 2].set_title("Predicted probability", fontsize=10)
        axes[row, 3].imshow(pred_mask, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 3].set_title(f"Thresholded @ {threshold:.2f}", fontsize=10)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("Validation prediction preview", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_history_plot(history, output_path):
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train"]["loss"] for entry in history]
    val_loss = [entry["val"]["loss"] for entry in history]
    train_dice = [entry["train"]["hard_dice_pos"] for entry in history]
    val_dice = [entry["val"]["hard_dice_pos"] for entry in history]
    val_f1 = [entry["val"]["image_f1"] for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(epochs, train_loss, label="train", color="#2563eb")
    axes[0].plot(epochs, val_loss, label="val", color="#dc2626")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_dice, label="train", color="#2563eb")
    axes[1].plot(epochs, val_dice, label="val", color="#dc2626")
    axes[1].set_title("Positive-sample hard Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, val_f1, color="#16a34a")
    axes[2].set_title("Val image-level F1")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_fn,
    device,
    threshold,
    bce_weight,
    dice_weight,
    use_amp,
    grad_clip,
    train_mode,
):
    accumulator = make_metric_accumulator()
    preview = None
    phase_name = "Train" if train_mode else "Val"
    progress = ProgressPrinter(len(loader), f"{phase_name} epoch")
    batch_counter = 0

    if train_mode:
        model.train()
    else:
        model.eval()

    autocast_enabled = bool(use_amp and device.type == "cuda")

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(images)
                bce = loss_fn(logits, masks)
                dice = dice_loss_from_logits(logits, masks)
                loss = bce_weight * bce + dice_weight * dice

            if train_mode:
                scaler.scale(loss).backward()
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        metrics = batch_metrics_from_logits(logits.detach(), masks.detach(), threshold)
        batch_size = int(images.shape[0])
        update_metric_accumulator(
            accumulator,
            batch_size=batch_size,
            loss_value=float(loss.detach().item()),
            bce_value=float(bce.detach().item()),
            dice_loss_value=float(dice.detach().item()),
            metrics=metrics,
        )

        if not train_mode and preview is None:
            preview = {
                "images": images.detach().cpu().numpy(),
                "masks": masks.detach().cpu().numpy(),
                "logits": logits.detach().cpu().numpy(),
                "indices": np.asarray(batch["index"][:], dtype=np.int64),
            }

        batch_counter += 1
        progress.update(batch_counter)

    return finalize_metrics(accumulator), preview


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    h5_path = Path(args.data_h5)
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found: {h5_path}")

    run_dir = make_run_dir(args.output_root, args.run_name)
    summary = load_dataset_summary(h5_path)
    labels = load_labels(h5_path)
    signal_strength = load_positive_signal_strength(h5_path)
    candidate_indices, candidate_summary = select_candidate_indices(
        labels,
        signal_strength=signal_strength,
        seed=args.seed,
        min_positive_amplitude=args.min_positive_amplitude,
    )
    train_idx, val_idx = stratified_split(
        labels,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        candidate_indices=candidate_indices,
    )

    train_pos, train_neg = count_class_balance(labels, train_idx)
    val_pos, val_neg = count_class_balance(labels, val_idx)
    train_mean, train_std = compute_patch_normalization(h5_path, train_idx)
    positive_fraction = compute_positive_pixel_fraction(h5_path, train_idx)
    pos_weight = compute_pos_weight(positive_fraction)

    run_config = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "data_h5": str(h5_path.resolve()),
        "summary": summary,
        "args": vars(args),
        "split": {
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "train_positive": train_pos,
            "train_negative": train_neg,
            "val_positive": val_pos,
            "val_negative": val_neg,
        },
        "normalization": {
            "train_mean": train_mean,
            "train_std": train_std,
        },
        "loss_reweighting": {
            "positive_pixel_fraction": positive_fraction,
            "bce_pos_weight": pos_weight,
        },
        "candidate_selection": candidate_summary,
    }

    save_json(run_dir / "run_config.json", run_config)
    np.savez_compressed(run_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx)

    print("\n=== Phase 3 dataset summary ===")
    print(f"  HDF5:            {h5_path}")
    print(f"  Train samples:   {len(train_idx)} ({train_pos} pos / {train_neg} neg)")
    print(f"  Val samples:     {len(val_idx)} ({val_pos} pos / {val_neg} neg)")
    if args.min_positive_amplitude > 0.0:
        print(
            f"  Candidate set:   {candidate_summary['candidate_samples']} "
            f"({candidate_summary['retained_positive']} pos / {candidate_summary['retained_negative']} neg)"
        )
        print(f"  Min amplitude:   {args.min_positive_amplitude:.2e}")
    print(f"  Train mean:      {train_mean:.6e}")
    print(f"  Train std:       {train_std:.6e}")
    print(f"  Positive pixels: {positive_fraction:.3%}")
    print(f"  BCE pos_weight:  {pos_weight:.3f}")

    if args.dry_run:
        print(f"  Run dir:         {run_dir}")
        print("\nDry run complete. Skipping model construction and training.")
        return

    require_ml_packages()
    device = resolve_device(args.device)
    data_parallel_device_ids = select_data_parallel_device_ids(device, args.gpu_ids)
    use_amp = bool(not args.disable_amp and device.type == "cuda")

    if data_parallel_device_ids:
        print(f"  DataParallel:    enabled on GPUs {data_parallel_device_ids}")
    elif device.type == "cuda":
        print(f"  DataParallel:    disabled (visible CUDA devices: {torch.cuda.device_count()})")
    print(f"  Run dir:         {run_dir}")

    train_dataset = H5BubbleDataset(
        h5_path=h5_path,
        indices=train_idx,
        mean=train_mean,
        std=train_std,
        augment=not args.disable_augment,
        seed=args.seed,
        max_translate_pixels=args.max_translate_pixels,
    )
    val_dataset = H5BubbleDataset(
        h5_path=h5_path,
        indices=val_idx,
        mean=train_mean,
        std=train_std,
        augment=False,
        seed=args.seed + 1,
        max_translate_pixels=0,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(args).to(device)
    model = wrap_model_for_data_parallel(model, data_parallel_device_ids)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    history = []
    best_val_score = -float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            threshold=args.threshold,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            train_mode=True,
        )

        with torch.no_grad():
            val_metrics, preview = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                scaler=scaler,
                loss_fn=loss_fn,
                device=device,
                threshold=args.threshold,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                use_amp=use_amp,
                grad_clip=0.0,
                train_mode=False,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_metrics["hard_dice_pos"])

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        save_json(run_dir / "history.json", history)
        save_history_plot(history, run_dir / "training_curves.png")

        preview_count = min(args.preview_count, len(preview["indices"]))
        save_prediction_preview(
            images=preview["images"][:preview_count],
            masks=preview["masks"][:preview_count],
            logits=preview["logits"][:preview_count],
            indices=preview["indices"][:preview_count],
            output_path=run_dir / "latest_val_preview.png",
            mean=train_mean,
            std=train_std,
            threshold=args.threshold,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict_for_checkpoint(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "run_config": run_config,
            "history": history,
            "best_val_hard_dice_pos": best_val_score,
        }
        torch.save(checkpoint, run_dir / "last_checkpoint.pt")

        val_score = float(val_metrics["hard_dice_pos"])
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            checkpoint["best_val_hard_dice_pos"] = best_val_score
            torch.save(checkpoint, run_dir / "best_checkpoint.pt")
            save_prediction_preview(
                images=preview["images"][:preview_count],
                masks=preview["masks"][:preview_count],
                logits=preview["logits"][:preview_count],
                indices=preview["indices"][:preview_count],
                output_path=run_dir / "best_val_preview.png",
                mean=train_mean,
                std=train_std,
                threshold=args.threshold,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val dice+ {val_metrics['hard_dice_pos']:.4f} | "
            f"val img F1 {val_metrics['image_f1']:.4f} | "
            f"lr {current_lr:.2e}"
        )

    print("\n=== Training complete ===")
    print(f"  Best epoch:                 {best_epoch}")
    print(f"  Best val positive Dice:     {best_val_score:.4f}")
    print(f"  Outputs saved under:        {run_dir}")


if __name__ == "__main__":
    main()
