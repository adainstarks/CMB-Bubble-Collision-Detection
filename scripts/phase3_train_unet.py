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
import hashlib
from pathlib import Path

import h5py

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from phase2_audit_dataset import run_audit
from phase_config import min_component_area_pixels
from phase_dataset_utils import load_predefined_split_indices, load_signal_strength

try:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except Exception as exc:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    WeightedRandomSampler = None
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
DEFAULT_DATA_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "training_data.h5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "phase3_unet"
EPS = 1e-6
DICE_SMOOTH = 1.0


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
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine"],
        help="Learning-rate scheduler. Use cosine for short fixed-length fine-tunes.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument(
        "--split-source",
        type=str,
        default="auto",
        choices=["auto", "predefined", "random"],
        help="Use dataset-provided split indices when available, or force a random split for older datasets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-positive-amplitude", type=float, default=0.0)
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b0")
    parser.add_argument("--encoder-weights", type=str, default="none")
    parser.add_argument(
        "--extra-channel-dataset",
        action="append",
        default=[],
        help=(
            "Optional HDF5 dataset path to append as an input channel, e.g. "
            "`features/circular_template_response`. Legacy files may still contain "
            "`features/matched_filter_response`. Can be repeated."
        ),
    )
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument(
        "--aux-head-weight",
        type=float,
        default=0.0,
        help="Joint image-level presence loss weight. Uses segmentation_models_pytorch aux_params when > 0.",
    )
    parser.add_argument("--aux-head-dropout", type=float, default=0.2)
    parser.add_argument(
        "--radius-head-weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary radius-bin classification loss for positive samples. "
            "This supervises theta_crit as a physical scale cue; negatives use ignore_index."
        ),
    )
    parser.add_argument(
        "--radius-bin-edges-deg",
        type=str,
        default="5,10,15,20,25",
        help=(
            "Comma-separated theta_crit bin edges used when --radius-head-weight > 0. "
            "The default creates bins [5,10), [10,15), [15,20), [20,25]."
        ),
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.0,
        help=(
            "Extra BCE pixel weight applied inside a narrow band around the target causal boundary. "
            "The segmentation target remains the full causal disc; this only emphasizes boundary pixels."
        ),
    )
    parser.add_argument(
        "--boundary-width-pixels",
        type=int,
        default=5,
        help="Half-width, in patch pixels, of the target-boundary emphasis band when --boundary-weight > 0.",
    )
    parser.add_argument(
        "--hard-positive-mining-json",
        type=str,
        default="",
        help="Optional phase3_error_mining.py JSON used to upweight documented hard positive bins.",
    )
    parser.add_argument(
        "--hard-positive-weight",
        type=float,
        default=1.0,
        help="Sampling weight multiplier for hard-positive bins when --hard-positive-mining-json is set.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Monitoring threshold used for training-time metrics and preview panels. Deployment thresholds are chosen later.",
    )
    parser.add_argument(
        "--image-min-positive-pixels",
        type=int,
        default=min_component_area_pixels(theta_min_deg=5.0, fraction=0.01),
        help=(
            "Minimum thresholded pixels required before a patch counts as image-positive "
            "for training-time image metrics. This prevents single-pixel activations from "
            "driving checkpoint selection."
        ),
    )
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
    parser.add_argument(
        "--normalization-config",
        type=str,
        default="",
        help=(
            "Optional previous run_config.json to reuse train normalization and "
            "positive-sample-only positive-pixel statistics."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--max-translate-pixels", type=int, default=48)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="image_f1",
        choices=["image_f1", "hard_dice_pos", "iou_pos", "image_recall"],
        help="Validation metric used for LR scheduling and best-checkpoint selection.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to resume model, optimizer, scheduler, scaler, and history.",
    )
    parser.add_argument(
        "--model-only-resume",
        action="store_true",
        help=(
            "When --resume-checkpoint is provided, load only compatible model weights and reset "
            "optimizer/scheduler/history. Use this for controlled fine-tune ablations."
        ),
    )
    parser.add_argument(
        "--cache-data",
        action="store_true",
        help="Load selected patches/masks into RAM once to avoid repeated HDF5 reads during training.",
    )
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-augment", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the dataset, compute normalization statistics, and exit before importing the model.",
    )
    parser.add_argument(
        "--skip-data-audit",
        action="store_true",
        help="Skip the strict Phase 2 dataset audit before training. Intended only for debugging.",
    )
    parser.add_argument(
        "--allow-legacy-data",
        action="store_true",
        help="Convert dataset-audit failures to warnings. Do not use for production results.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1.")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")
    if args.image_min_positive_pixels < 1:
        raise ValueError("--image-min-positive-pixels must be at least 1.")
    if args.preview_count <= 0:
        raise ValueError("--preview-count must be positive.")
    if args.max_train_samples < 0 or args.max_val_samples < 0:
        raise ValueError("--max-train-samples and --max-val-samples must be non-negative.")
    if args.max_translate_pixels < 0:
        raise ValueError("--max-translate-pixels must be non-negative.")
    if args.min_positive_amplitude < 0.0:
        raise ValueError("--min-positive-amplitude must be non-negative.")
    if args.boundary_weight < 0.0:
        raise ValueError("--boundary-weight must be non-negative.")
    if args.boundary_width_pixels < 0:
        raise ValueError("--boundary-width-pixels must be non-negative.")
    if args.aux_head_weight < 0.0:
        raise ValueError("--aux-head-weight must be non-negative.")
    if args.radius_head_weight < 0.0:
        raise ValueError("--radius-head-weight must be non-negative.")
    radius_edges = parse_radius_bin_edges(args.radius_bin_edges_deg)
    if args.radius_head_weight > 0.0 and len(radius_edges) < 3:
        raise ValueError("--radius-bin-edges-deg must define at least two radius bins.")
    if not (0.0 <= args.aux_head_dropout < 1.0):
        raise ValueError("--aux-head-dropout must be in [0, 1).")
    if args.hard_positive_weight < 1.0:
        raise ValueError("--hard-positive-weight must be >= 1.")
    if len(parse_extra_channel_datasets(args.extra_channel_dataset)) != len(set(parse_extra_channel_datasets(args.extra_channel_dataset))):
        raise ValueError("--extra-channel-dataset contains duplicate dataset paths.")
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


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_positive_signal_strength(h5_path):
    return load_signal_strength(h5_path)


def parse_extra_channel_datasets(value):
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    out = []
    for item in values:
        for token in str(item).split(","):
            token = token.strip().strip("/")
            if token:
                out.append(token)
    return out


def parse_radius_bin_edges(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        edges = [float(x) for x in value]
    else:
        edges = [float(token.strip()) for token in str(value).split(",") if token.strip()]
    if len(edges) < 2:
        raise ValueError("--radius-bin-edges-deg must contain at least two values.")
    if any(not np.isfinite(edge) for edge in edges):
        raise ValueError("--radius-bin-edges-deg contains a non-finite edge.")
    if any(edges[idx] >= edges[idx + 1] for idx in range(len(edges) - 1)):
        raise ValueError("--radius-bin-edges-deg must be strictly increasing.")
    return tuple(edges)


def radius_bin_count_from_edges(edges):
    return max(len(edges) - 1, 0)


def theta_to_radius_bin(theta_crit_deg, radius_bin_edges):
    edges = np.asarray(radius_bin_edges, dtype=np.float32)
    inner_edges = edges[1:-1]
    bin_idx = int(np.searchsorted(inner_edges, float(theta_crit_deg), side="right"))
    return int(np.clip(bin_idx, 0, radius_bin_count_from_edges(edges) - 1))


def input_channel_count_from_args(args):
    explicit = getattr(args, "input_channels", None)
    if explicit is not None:
        return int(explicit)
    return 1 + len(parse_extra_channel_datasets(getattr(args, "extra_channel_dataset", [])))


def input_config_from_run_config(run_config):
    train_args = run_config.get("args", {})
    normalization = run_config.get("normalization", {})
    extra_channel_datasets = parse_extra_channel_datasets(train_args.get("extra_channel_dataset", []))
    input_channels = int(train_args.get("input_channels", 1 + len(extra_channel_datasets)))
    channel_means = normalization.get("channel_means")
    channel_stds = normalization.get("channel_stds")
    if channel_means is None:
        channel_means = [float(normalization["train_mean"])] + [0.0] * len(extra_channel_datasets)
    if channel_stds is None:
        channel_stds = [float(normalization["train_std"])] + [1.0] * len(extra_channel_datasets)
    return {
        "extra_channel_datasets": extra_channel_datasets,
        "input_channels": input_channels,
        "channel_means": [float(x) for x in channel_means],
        "channel_stds": [float(x) for x in channel_stds],
    }


def model_args_from_run_config(run_config):
    train_args = run_config["args"]
    input_config = input_config_from_run_config(run_config)
    return argparse.Namespace(
        encoder_name=train_args["encoder_name"],
        encoder_weights=train_args["encoder_weights"],
        aux_head_weight=float(train_args.get("aux_head_weight", 0.0) or 0.0),
        aux_head_dropout=float(train_args.get("aux_head_dropout", 0.0) or 0.0),
        radius_head_weight=float(train_args.get("radius_head_weight", 0.0) or 0.0),
        radius_bin_edges_deg=train_args.get("radius_bin_edges_deg", "5,10,15,20,25"),
        input_channels=input_config["input_channels"],
        extra_channel_dataset=input_config["extra_channel_datasets"],
    )


def dataset_kwargs_from_run_config(run_config):
    normalization = run_config["normalization"]
    input_config = input_config_from_run_config(run_config)
    return {
        "mean": float(normalization["train_mean"]),
        "std": float(normalization["train_std"]),
        "extra_channel_datasets": input_config["extra_channel_datasets"],
        "channel_means": input_config["channel_means"],
        "channel_stds": input_config["channel_stds"],
    }


def h5_dataset_exists(h5, dataset_path):
    try:
        obj = h5[dataset_path]
    except KeyError:
        return False
    return isinstance(obj, h5py.Dataset)


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


def select_candidate_indices(labels, signal_strength, seed, min_positive_amplitude, base_indices=None):
    if base_indices is None:
        all_indices = np.arange(len(labels), dtype=np.int64)
    else:
        all_indices = np.asarray(base_indices, dtype=np.int64)
    if min_positive_amplitude <= 0.0:
        return all_indices, {
            "min_positive_amplitude": 0.0,
            "retained_positive": int((labels[all_indices] == 1).sum()),
            "retained_negative": int((labels[all_indices] == 0).sum()),
            "candidate_samples": int(len(all_indices)),
        }

    rng = np.random.default_rng(seed)
    selected_labels = labels[all_indices]
    selected_signal = signal_strength[all_indices]
    positive_idx = all_indices[(selected_labels == 1) & (selected_signal >= min_positive_amplitude)]
    negative_idx = all_indices[selected_labels == 0]

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


def resolve_split_indices(
    h5_path,
    labels,
    signal_strength,
    train_fraction,
    seed,
    min_positive_amplitude,
    max_train_samples=0,
    max_val_samples=0,
    split_source="auto",
):
    predefined = None
    if split_source in {"auto", "predefined"}:
        predefined = load_predefined_split_indices(h5_path)
        if split_source == "predefined" and predefined is None:
            raise RuntimeError("Requested --split-source predefined but the dataset does not provide split indices.")

    if predefined is not None:
        train_idx, train_summary = select_candidate_indices(
            labels,
            signal_strength=signal_strength,
            seed=seed,
            min_positive_amplitude=min_positive_amplitude,
            base_indices=predefined["train_idx"],
        )
        val_idx, val_summary = select_candidate_indices(
            labels,
            signal_strength=signal_strength,
            seed=seed + 1,
            min_positive_amplitude=min_positive_amplitude,
            base_indices=predefined["val_idx"],
        )
        rng = np.random.default_rng(seed)
        train_idx = limit_indices(train_idx, labels, max_train_samples, rng)
        val_idx = limit_indices(val_idx, labels, max_val_samples, rng)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        return train_idx, val_idx, {
            "split_source": "predefined",
            "train": train_summary,
            "val": val_summary,
        }

    candidate_indices, candidate_summary = select_candidate_indices(
        labels,
        signal_strength=signal_strength,
        seed=seed,
        min_positive_amplitude=min_positive_amplitude,
    )
    train_idx, val_idx = stratified_split(
        labels,
        train_fraction=train_fraction,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        candidate_indices=candidate_indices,
    )
    return train_idx, val_idx, {
        "split_source": "random",
        "all": candidate_summary,
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


def compute_patch_normalization(h5_path, indices, chunk_size=256):
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


def compute_dataset_normalization(h5_path, indices, dataset_path, label=None, chunk_size=256):
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0
    progress = ProgressPrinter(len(indices), label or f"Normalization {dataset_path}")
    processed = 0

    with h5py.File(h5_path, "r") as h5:
        if not h5_dataset_exists(h5, dataset_path):
            raise KeyError(f"Normalization dataset not found in {h5_path}: {dataset_path}")
        dataset = h5[dataset_path]
        if dataset.shape[0] != h5["labels"].shape[0]:
            raise RuntimeError(f"Dataset {dataset_path} has incompatible leading dimension: {dataset.shape}")
        for chunk in iter_index_chunks(indices, chunk_size):
            batch = np.asarray(dataset[chunk], dtype=np.float64)
            total_sum += float(batch.sum())
            total_sq += float(np.square(batch).sum())
            total_count += int(batch.size)
            processed += len(chunk)
            progress.update(processed)

    mean = total_sum / max(total_count, 1)
    variance = max(total_sq / max(total_count, 1) - mean**2, 1e-12)
    std = math.sqrt(variance)
    return float(mean), float(std)


def compute_extra_channel_normalization(h5_path, indices, extra_channel_datasets, chunk_size=256):
    means = []
    stds = []
    for dataset_path in extra_channel_datasets:
        mean, std = compute_dataset_normalization(
            h5_path,
            indices,
            dataset_path=dataset_path,
            label=f"Normalization {dataset_path}",
            chunk_size=chunk_size,
        )
        means.append(mean)
        stds.append(std)
    return means, stds


def compute_positive_pixel_fraction(h5_path, indices, chunk_size=256):
    total_positive = 0.0
    total_pixels = 0
    progress = ProgressPrinter(len(indices), "Mask stats")
    processed = 0

    with h5py.File(h5_path, "r") as h5:
        masks = h5["masks"]
        labels = h5["labels"]
        for chunk in iter_index_chunks(indices, chunk_size):
            chunk_labels = np.asarray(labels[chunk], dtype=np.uint8)
            positive_rows = chunk[chunk_labels == 1]
            if len(positive_rows) == 0:
                processed += len(chunk)
                progress.update(processed)
                continue
            batch = np.asarray(masks[positive_rows], dtype=np.float64)
            total_positive += float(batch.sum())
            total_pixels += int(batch.size)
            processed += len(chunk)
            progress.update(processed)

    if total_pixels == 0:
        raise ValueError("Cannot compute positive-pixel fraction: training split has no positive samples.")
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
        patch = np.rot90(patch, k=k, axes=(-2, -1))
        mask = np.rot90(mask, k=k)
    if rng.random() < 0.5:
        patch = np.flip(patch, axis=-2)
        mask = np.flip(mask, axis=0)
    if rng.random() < 0.5:
        patch = np.flip(patch, axis=-1)
        mask = np.flip(mask, axis=1)
    return patch.copy(), mask.copy()


def translate_patch_and_mask(patch, mask, shift_y, shift_x):
    """
    Translate a training example so the network cannot solve the task by
    memorizing that positive masks are always centered in the patch.

    We reflect-pad the CMB patch to avoid introducing artificial blank borders,
    while the binary target mask is shifted with zero fill. Shifts are integer
    pixels, so direct slicing is much faster than scipy.ndimage.shift and avoids
    interpolation artifacts.
    """
    if shift_x == 0 and shift_y == 0:
        return patch, mask

    h, w = patch.shape[-2:]
    pad_y = abs(int(shift_y))
    pad_x = abs(int(shift_x))

    if patch.ndim == 2:
        pad_width = ((pad_y, pad_y), (pad_x, pad_x))
        patch_y_slice = slice(pad_y - int(shift_y), pad_y - int(shift_y) + h)
        patch_x_slice = slice(pad_x - int(shift_x), pad_x - int(shift_x) + w)
        patch_padded = np.pad(patch, pad_width, mode="reflect")
        patch_shifted = patch_padded[patch_y_slice, patch_x_slice]
    elif patch.ndim == 3:
        pad_width = ((0, 0), (pad_y, pad_y), (pad_x, pad_x))
        patch_y_slice = slice(pad_y - int(shift_y), pad_y - int(shift_y) + h)
        patch_x_slice = slice(pad_x - int(shift_x), pad_x - int(shift_x) + w)
        patch_padded = np.pad(patch, pad_width, mode="reflect")
        patch_shifted = patch_padded[:, patch_y_slice, patch_x_slice]
    else:
        raise ValueError(f"Expected 2D or 3D patch array, got shape {patch.shape}.")

    mask_shifted = np.zeros_like(mask, dtype=np.float32)
    src_y0 = max(0, -int(shift_y))
    src_y1 = min(h, h - int(shift_y))
    dst_y0 = max(0, int(shift_y))
    dst_y1 = min(h, h + int(shift_y))
    src_x0 = max(0, -int(shift_x))
    src_x1 = min(w, w - int(shift_x))
    dst_x0 = max(0, int(shift_x))
    dst_x1 = min(w, w + int(shift_x))
    if src_y1 > src_y0 and src_x1 > src_x0:
        mask_shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]

    return patch_shifted.astype(np.float32), (mask_shifted > 0.5).astype(np.float32)


def random_translate(patch, mask, rng, max_translate_pixels):
    if max_translate_pixels <= 0:
        return patch, mask

    positive = mask > 0.5
    if np.any(positive):
        ys, xs = np.nonzero(positive)
        max_up = min(int(max_translate_pixels), int(ys.min()))
        max_down = min(int(max_translate_pixels), int(mask.shape[0] - 1 - ys.max()))
        max_left = min(int(max_translate_pixels), int(xs.min()))
        max_right = min(int(max_translate_pixels), int(mask.shape[1] - 1 - xs.max()))
        shift_y = int(rng.integers(-max_up, max_down + 1))
        shift_x = int(rng.integers(-max_left, max_right + 1))
    else:
        shift_y = int(rng.integers(-max_translate_pixels, max_translate_pixels + 1))
        shift_x = int(rng.integers(-max_translate_pixels, max_translate_pixels + 1))
    return translate_patch_and_mask(patch, mask, shift_y=shift_y, shift_x=shift_x)


def read_h5_rows(dataset, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    values_sorted = np.asarray(dataset[sorted_indices])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return values_sorted[inverse]


class H5BubbleDataset(Dataset):
    def __init__(
        self,
        h5_path,
        indices,
        mean,
        std,
        augment=False,
        seed=42,
        max_translate_pixels=0,
        cache_data=False,
        extra_channel_datasets=None,
        channel_means=None,
        channel_stds=None,
        radius_bin_edges=None,
    ):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.extra_channel_datasets = parse_extra_channel_datasets(extra_channel_datasets)
        self.radius_bin_edges = None if radius_bin_edges is None else parse_radius_bin_edges(radius_bin_edges)
        self.num_channels = 1 + len(self.extra_channel_datasets)
        if channel_means is None:
            channel_means = [float(mean)] + [0.0] * len(self.extra_channel_datasets)
        if channel_stds is None:
            channel_stds = [float(max(std, 1e-8))] + [1.0] * len(self.extra_channel_datasets)
        self.channel_means = np.asarray(channel_means, dtype=np.float32)
        self.channel_stds = np.maximum(np.asarray(channel_stds, dtype=np.float32), 1e-8)
        if self.channel_means.size != self.num_channels or self.channel_stds.size != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channel normalization values, got "
                f"{self.channel_means.size} means and {self.channel_stds.size} stds."
            )
        self.mean = float(self.channel_means[0])
        self.std = float(self.channel_stds[0])
        self.augment = bool(augment)
        self.seed = int(seed)
        self.max_translate_pixels = int(max_translate_pixels)
        self.cache_data = bool(cache_data)
        self._h5 = None
        self._rng = None
        self._patches = None
        self._extra_channels = None
        self._masks = None
        self._labels = None
        self._theta_crit = None
        if self.cache_data:
            with h5py.File(self.h5_path, "r") as h5:
                for dataset_path in self.extra_channel_datasets:
                    if not h5_dataset_exists(h5, dataset_path):
                        raise KeyError(f"Extra input channel dataset not found in {self.h5_path}: {dataset_path}")
                if self.radius_bin_edges is not None and "truth/theta_crit_deg" not in h5:
                    raise KeyError(f"Radius-bin supervision requires truth/theta_crit_deg in {self.h5_path}.")
                self._patches = read_h5_rows(h5["patches"], self.indices).astype(np.float32, copy=False)
                self._extra_channels = [
                    read_h5_rows(h5[dataset_path], self.indices).astype(np.float32, copy=False)
                    for dataset_path in self.extra_channel_datasets
                ]
                self._masks = read_h5_rows(h5["masks"], self.indices).astype(np.float32, copy=False)
                self._labels = read_h5_rows(h5["labels"], self.indices).astype(np.float32, copy=False)
                if self.radius_bin_edges is not None:
                    self._theta_crit = read_h5_rows(h5["truth/theta_crit_deg"], self.indices).astype(np.float32, copy=False)

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
        index = int(self.indices[item])
        if self.cache_data:
            patch = np.asarray(self._patches[item], dtype=np.float32)
            extra_channels = [
                np.asarray(channel[item], dtype=np.float32)
                for channel in (self._extra_channels or [])
            ]
            mask = np.asarray(self._masks[item], dtype=np.float32)
            label = float(self._labels[item])
            theta_crit = None if self._theta_crit is None else float(self._theta_crit[item])
        else:
            h5 = self._get_h5()
            for dataset_path in self.extra_channel_datasets:
                if not h5_dataset_exists(h5, dataset_path):
                    raise KeyError(f"Extra input channel dataset not found in {self.h5_path}: {dataset_path}")
            if self.radius_bin_edges is not None and "truth/theta_crit_deg" not in h5:
                raise KeyError(f"Radius-bin supervision requires truth/theta_crit_deg in {self.h5_path}.")
            patch = np.asarray(h5["patches"][index], dtype=np.float32)
            extra_channels = [
                np.asarray(h5[dataset_path][index], dtype=np.float32)
                for dataset_path in self.extra_channel_datasets
            ]
            mask = np.asarray(h5["masks"][index], dtype=np.float32)
            label = float(h5["labels"][index])
            theta_crit = None if self.radius_bin_edges is None else float(h5["truth/theta_crit_deg"][index])

        if not np.all(np.isfinite(patch)):
            raise ValueError(f"Patch index {index} contains non-finite values.")
        if float(np.max(np.abs(patch))) > 1.0:
            raise ValueError(
                f"Patch index {index} is outside Kelvin anisotropy scale; "
                "check microkelvin/Kelvin conversion."
            )
        if not np.all(np.isfinite(mask)):
            raise ValueError(f"Mask index {index} contains non-finite values.")

        if extra_channels:
            patch = np.stack([patch, *extra_channels], axis=0).astype(np.float32, copy=False)

        if self.augment:
            patch, mask = random_translate(
                patch,
                mask,
                self._get_rng(),
                max_translate_pixels=self.max_translate_pixels,
            )
            patch, mask = random_dihedral(patch, mask, self._get_rng())

        if patch.ndim == 2:
            patch = patch[None, :, :]
        patch = (patch - self.channel_means[:, None, None]) / self.channel_stds[:, None, None]
        mask = mask[None, :, :]

        radius_bin = -100
        if self.radius_bin_edges is not None and label >= 0.5:
            radius_bin = theta_to_radius_bin(theta_crit, self.radius_bin_edges)

        return {
            "image": torch.from_numpy(patch),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.float32),
            "radius_bin": torch.tensor(radius_bin, dtype=torch.long),
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


def load_model_state_dict(model, state_dict, strict=True):
    def filtered(target_model, candidate_state):
        target_state = target_model.state_dict()
        return {
            key: value
            for key, value in candidate_state.items()
            if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
        }

    def candidates(state):
        yield state
        if isinstance(model, nn.DataParallel):
            yield {
                (key if key.startswith("module.") else f"module.{key}"): value
                for key, value in state.items()
            }
        yield {
            (key[len("module."):] if key.startswith("module.") else key): value
            for key, value in state.items()
        }

    if strict:
        last_error = None
        for variant in candidates(state_dict):
            try:
                model.load_state_dict(variant, strict=True)
                return
            except RuntimeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return

    target_state = model.state_dict()
    best_variant = None
    best_count = -1
    for variant in candidates(state_dict):
        compatible = filtered(model, variant)
        count = len(compatible)
        if count > best_count:
            best_count = count
            best_variant = compatible
    target_total = len(target_state)
    if best_count == 0:
        raise RuntimeError(
            "load_model_state_dict (strict=False): no checkpoint keys matched any target "
            f"parameters ({target_total} target keys). Refusing to silently load random weights."
        )
    print(
        f"  Resume: matched {best_count}/{target_total} target parameters from checkpoint "
        f"(strict=False, partial resume).",
        flush=True,
    )
    model.load_state_dict(best_variant, strict=False)


def build_model(args):
    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    input_channels = input_channel_count_from_args(args)
    radius_bin_edges = parse_radius_bin_edges(getattr(args, "radius_bin_edges_deg", "5,10,15,20,25"))
    radius_bin_count = radius_bin_count_from_edges(radius_bin_edges) if getattr(args, "radius_head_weight", 0.0) > 0.0 else 0
    image_aux_count = 1 if getattr(args, "aux_head_weight", 0.0) > 0.0 else 0
    aux_params = None
    if image_aux_count + radius_bin_count > 0:
        aux_params = {
            "pooling": "avg",
            "dropout": float(args.aux_head_dropout),
            "activation": None,
            "classes": image_aux_count + radius_bin_count,
        }
    return smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1,
        activation=None,
        aux_params=aux_params,
    )


def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        mask_logits = output[0]
        aux_logits = output[1] if len(output) > 1 else None
        return mask_logits, aux_logits
    return output, None


def split_aux_logits(aux_logits, use_image_aux, radius_bin_count):
    image_logits = None
    radius_logits = None
    if aux_logits is None:
        return image_logits, radius_logits
    if aux_logits.ndim == 1:
        aux_logits = aux_logits.reshape(-1, 1)
    offset = 0
    if use_image_aux:
        image_logits = aux_logits[:, 0]
        offset = 1
    if radius_bin_count > 0:
        radius_logits = aux_logits[:, offset : offset + radius_bin_count]
    return image_logits, radius_logits


def dice_loss_from_logits(logits, targets):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + DICE_SMOOTH) / (denominator + DICE_SMOOTH)
    return 1.0 - dice.mean()


def target_boundary_band(targets, width_pixels):
    if width_pixels <= 0:
        return torch.zeros_like(targets)

    kernel_size = 2 * int(width_pixels) + 1
    padding = int(width_pixels)
    dilated = F.max_pool2d(targets, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool2d(1.0 - targets, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp_(0.0, 1.0)


def weighted_bce_with_logits(logits, targets, base_pos_weight, boundary_weight, boundary_width_pixels):
    if boundary_weight <= 0.0 or boundary_width_pixels <= 0:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=base_pos_weight)

    boundary = target_boundary_band(targets, boundary_width_pixels)
    pixel_weights = 1.0 + float(boundary_weight) * boundary
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=pixel_weights,
        pos_weight=base_pos_weight,
    )


def batch_metrics_from_logits(logits, targets, threshold, image_min_positive_pixels=1):
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

    image_pred = pred_sum >= int(image_min_positive_pixels)
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
        "radius_loss_sum": 0.0,
        "radius_correct": 0,
        "radius_count": 0,
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
    if "radius_loss" in metrics:
        acc["radius_loss_sum"] += float(metrics["radius_loss"]) * batch_size
        acc["radius_correct"] += int(metrics.get("radius_correct", 0))
        acc["radius_count"] += int(metrics.get("radius_count", 0))


def build_hard_positive_sample_weights(h5_path, train_idx, error_mining_json, hard_positive_weight):
    if not error_mining_json:
        return None, {}

    report = load_json(error_mining_json)
    hard_samples = set()
    selected_groups = ("amplitude", "edge_strength", "theta_crit", "offcenter_distance", "local_background_std")
    for group_name in selected_groups:
        for _, row in report.get("groups", {}).get(group_name, {}).items():
            if float(row.get("recall", 1.0)) <= 0.40:
                hard_samples.update(int(x) for x in row.get("sample_misses", []))

    with h5py.File(h5_path, "r") as h5:
        labels = read_h5_rows(h5["labels"], train_idx).astype(np.uint8, copy=False)
        truth = h5["truth"]
        theta = read_h5_rows(truth["theta_crit_deg"], train_idx).astype(np.float32, copy=False)
        z0 = read_h5_rows(truth["z0"], train_idx).astype(np.float32, copy=False)
        zcrit = read_h5_rows(truth["zcrit"], train_idx).astype(np.float32, copy=False)
        dx = read_h5_rows(truth["signal_center_dx_deg"], train_idx).astype(np.float32, copy=False)
        dy = read_h5_rows(truth["signal_center_dy_deg"], train_idx).astype(np.float32, copy=False)

    amplitude = np.maximum(np.abs(z0), np.abs(zcrit))
    offset = np.hypot(dx, dy)
    hard_mask = (labels == 1) & (
        (amplitude < 3e-5)
        | (np.abs(zcrit) < 3e-5)
        | (theta < 15.0)
        | (offset >= 10.0)
    )
    weights = np.ones(len(train_idx), dtype=np.float64)
    weights[hard_mask] = float(hard_positive_weight)

    return weights, {
        "source": str(Path(error_mining_json).resolve()),
        "hard_positive_weight": float(hard_positive_weight),
        "candidate_hard_samples": int(len(hard_samples)),
        "train_hard_samples": int(hard_mask.sum()),
        "selection_rule": (
            "upweight train positives in documented weak families: amplitude < 3e-5, "
            "|zcrit| < 3e-5, theta_crit < 15 deg, or off-center distance >= 10 deg"
        ),
    }


def finalize_metrics(acc):
    n = max(acc["num_samples"], 1)
    pos_n = max(acc["num_positive_samples"], 1)
    precision = acc["image_tp"] / max(acc["image_tp"] + acc["image_fp"], 1)
    recall = acc["image_tp"] / max(acc["image_tp"] + acc["image_fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    specificity = acc["image_tn"] / max(acc["image_tn"] + acc["image_fp"], 1)
    false_positive_rate = acc["image_fp"] / max(acc["image_fp"] + acc["image_tn"], 1)

    radius_count = max(acc["radius_count"], 1)
    return {
        "loss": acc["loss_sum"] / n,
        "bce": acc["bce_sum"] / n,
        "dice_loss": acc["dice_loss_sum"] / n,
        "radius_loss": acc["radius_loss_sum"] / n,
        "radius_bin_accuracy": acc["radius_correct"] / radius_count,
        "radius_bin_count": acc["radius_count"],
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
    logits = np.asarray(logits, dtype=np.float32)
    probs = np.empty_like(logits, dtype=np.float32)
    positive = logits >= 0.0
    probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probs[~positive] = exp_logits / (1.0 + exp_logits)
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_normalization_config(path):
    payload = load_json(path)
    normalization = payload.get("normalization", {})
    loss_reweighting = payload.get("loss_reweighting", {})
    required_norm = ("train_mean", "train_std")
    required_loss = (
        "positive_pixel_fraction",
        "positive_pixel_fraction_denominator",
        "bce_pos_weight",
    )
    missing = [key for key in required_norm if key not in normalization]
    missing += [key for key in required_loss if key not in loss_reweighting]
    if missing:
        raise RuntimeError(f"Normalization config is missing fields: {missing}")
    denominator = str(loss_reweighting["positive_pixel_fraction_denominator"])
    if denominator != "positive_samples_only":
        raise RuntimeError(
            "Normalization config uses an obsolete positive-pixel denominator "
            f"({denominator!r}); recompute normalization/loss statistics."
        )
    result = {
        "train_mean": float(normalization["train_mean"]),
        "train_std": float(normalization["train_std"]),
        "positive_pixel_fraction": float(loss_reweighting["positive_pixel_fraction"]),
        "bce_pos_weight": float(loss_reweighting["bce_pos_weight"]),
        "source": str(Path(path).resolve()),
    }
    if "channel_means" in normalization and "channel_stds" in normalization:
        result["channel_means"] = [float(x) for x in normalization["channel_means"]]
        result["channel_stds"] = [float(x) for x in normalization["channel_stds"]]
    return result


def run_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_fn,
    device,
    threshold,
    image_min_positive_pixels,
    bce_weight,
    dice_weight,
    aux_head_weight,
    radius_head_weight,
    radius_bin_count,
    use_amp,
    grad_clip,
    boundary_weight,
    boundary_width_pixels,
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
        labels = batch["label"].to(device, non_blocking=True)
        radius_bins = batch["radius_bin"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                model_output = model(images)
                logits, aux_logits = unpack_model_output(model_output)
                image_logits, radius_logits = split_aux_logits(
                    aux_logits,
                    use_image_aux=aux_head_weight > 0.0,
                    radius_bin_count=radius_bin_count,
                )
                if boundary_weight > 0.0 and boundary_width_pixels > 0:
                    bce = weighted_bce_with_logits(
                        logits=logits,
                        targets=masks,
                        base_pos_weight=loss_fn.pos_weight,
                        boundary_weight=boundary_weight,
                        boundary_width_pixels=boundary_width_pixels,
                    )
                else:
                    bce = loss_fn(logits, masks)
                dice = dice_loss_from_logits(logits, masks)
                loss = bce_weight * bce + dice_weight * dice
                if aux_head_weight > 0.0 and image_logits is not None:
                    image_logits = image_logits.reshape(-1)
                    image_loss = F.binary_cross_entropy_with_logits(image_logits, labels.float())
                    loss = loss + float(aux_head_weight) * image_loss
                radius_loss = None
                if radius_head_weight > 0.0 and radius_logits is not None:
                    valid_radius_for_loss = radius_bins >= 0
                    if bool(valid_radius_for_loss.any()):
                        radius_loss = F.cross_entropy(radius_logits, radius_bins, ignore_index=-100)
                    else:
                        radius_loss = radius_logits.sum() * 0.0
                    loss = loss + float(radius_head_weight) * radius_loss

            if train_mode:
                scaler.scale(loss).backward()
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        metrics = batch_metrics_from_logits(
            logits.detach(),
            masks.detach(),
            threshold,
            image_min_positive_pixels=image_min_positive_pixels,
        )
        if radius_head_weight > 0.0 and radius_logits is not None and radius_loss is not None:
            valid_radius = radius_bins >= 0
            if bool(valid_radius.any()):
                radius_pred = torch.argmax(radius_logits.detach(), dim=1)
                metrics["radius_correct"] = int((radius_pred[valid_radius] == radius_bins[valid_radius]).sum().item())
                metrics["radius_count"] = int(valid_radius.sum().item())
            else:
                metrics["radius_correct"] = 0
                metrics["radius_count"] = 0
            metrics["radius_loss"] = float(radius_loss.detach().item())
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
    input_audit_report = None
    if not args.skip_data_audit:
        print("\n=== Auditing Phase 2 dataset before training ===")
        input_audit_report = run_audit(
            h5_path,
            allow_legacy=args.allow_legacy_data,
            sample_patch_count=256,
        )
        audit_path = run_dir / "input_dataset_audit.json"
        save_json(audit_path, input_audit_report)
        print(f"  Audit status:    {input_audit_report['status']}")
        print(f"  Audit failures:  {input_audit_report['num_failures']}")
        print(f"  Audit warnings:  {input_audit_report['num_warnings']}")
        print(f"  Audit report:    {audit_path}")
        if input_audit_report["status"] != "pass":
            raise RuntimeError(f"Input dataset failed audit. See {audit_path}")

    summary = load_dataset_summary(h5_path)
    labels = load_labels(h5_path)
    signal_strength = load_positive_signal_strength(h5_path)
    extra_channel_datasets = parse_extra_channel_datasets(args.extra_channel_dataset)
    radius_bin_edges = parse_radius_bin_edges(args.radius_bin_edges_deg)
    radius_bin_count = radius_bin_count_from_edges(radius_bin_edges) if args.radius_head_weight > 0.0 else 0
    args.extra_channel_dataset = extra_channel_datasets
    args.input_channels = 1 + len(extra_channel_datasets)
    args.radius_bin_edges_deg = ",".join(f"{edge:g}" for edge in radius_bin_edges)
    if extra_channel_datasets:
        with h5py.File(h5_path, "r") as h5:
            for dataset_path in extra_channel_datasets:
                if not h5_dataset_exists(h5, dataset_path):
                    raise KeyError(f"Extra input channel dataset not found in {h5_path}: {dataset_path}")
                if h5[dataset_path].shape != h5["patches"].shape:
                    raise RuntimeError(
                        f"Extra input channel {dataset_path} shape {h5[dataset_path].shape} "
                        f"does not match patches shape {h5['patches'].shape}."
                    )
    train_idx, val_idx, candidate_summary = resolve_split_indices(
        h5_path=h5_path,
        labels=labels,
        signal_strength=signal_strength,
        train_fraction=args.train_fraction,
        seed=args.seed,
        min_positive_amplitude=args.min_positive_amplitude,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        split_source=args.split_source,
    )
    predefined_splits = load_predefined_split_indices(h5_path)
    test_idx = None
    if predefined_splits is not None and "test_idx" in predefined_splits:
        test_idx = np.asarray(predefined_splits["test_idx"], dtype=np.int64)

    train_pos, train_neg = count_class_balance(labels, train_idx)
    val_pos, val_neg = count_class_balance(labels, val_idx)
    test_pos = test_neg = 0
    if test_idx is not None:
        test_pos, test_neg = count_class_balance(labels, test_idx)
    normalization_source = None
    channel_means = None
    channel_stds = None
    if args.normalization_config:
        cached_norm = load_normalization_config(args.normalization_config)
        train_mean = cached_norm["train_mean"]
        train_std = cached_norm["train_std"]
        positive_fraction = cached_norm["positive_pixel_fraction"]
        pos_weight = cached_norm["bce_pos_weight"]
        normalization_source = cached_norm["source"]
        if (
            "channel_means" in cached_norm
            and len(cached_norm["channel_means"]) == args.input_channels
            and "channel_stds" in cached_norm
            and len(cached_norm["channel_stds"]) == args.input_channels
        ):
            channel_means = cached_norm["channel_means"]
            channel_stds = cached_norm["channel_stds"]
        elif extra_channel_datasets:
            extra_means, extra_stds = compute_extra_channel_normalization(h5_path, train_idx, extra_channel_datasets)
            channel_means = [train_mean, *extra_means]
            channel_stds = [train_std, *extra_stds]
    else:
        train_mean, train_std = compute_patch_normalization(h5_path, train_idx)
        extra_means, extra_stds = compute_extra_channel_normalization(h5_path, train_idx, extra_channel_datasets)
        channel_means = [train_mean, *extra_means]
        channel_stds = [train_std, *extra_stds]
        positive_fraction = compute_positive_pixel_fraction(h5_path, train_idx)
        pos_weight = compute_pos_weight(positive_fraction)
    if channel_means is None:
        channel_means = [train_mean]
        channel_stds = [train_std]

    run_config = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "data_h5": str(h5_path.resolve()),
        "data_h5_sha256": file_sha256(h5_path),
        "summary": summary,
        "args": vars(args),
        "split": {
            "train_samples": int(len(train_idx)),
            "calibration_samples": int(len(val_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)) if test_idx is not None else 0,
            "train_positive": train_pos,
            "train_negative": train_neg,
            "calibration_positive": val_pos,
            "calibration_negative": val_neg,
            "val_positive": val_pos,
            "val_negative": val_neg,
            "test_positive": test_pos,
            "test_negative": test_neg,
            "calibration_alias": "val",
        },
        "normalization": {
            "train_mean": train_mean,
            "train_std": train_std,
            "channel_means": [float(x) for x in channel_means],
            "channel_stds": [float(x) for x in channel_stds],
            "source": normalization_source or "computed_from_training_split",
        },
        "loss_reweighting": {
            "positive_pixel_fraction": positive_fraction,
            "positive_pixel_fraction_denominator": "positive_samples_only",
            "bce_pos_weight": pos_weight,
        },
        "candidate_selection": candidate_summary,
        "input_dataset_audit": input_audit_report,
    }
    if args.radius_head_weight > 0.0:
        run_config["radius_head"] = {
            "enabled": True,
            "weight": float(args.radius_head_weight),
            "bin_edges_deg": [float(edge) for edge in radius_bin_edges],
            "num_bins": int(radius_bin_count),
            "target": "truth/theta_crit_deg for positive samples; negatives ignored",
            "scientific_note": (
                "Auxiliary scale supervision only. The Feeney-template generator, masks, "
                "and deployment thresholds are unchanged."
            ),
        }
    sample_weights = None
    hard_positive_summary = None
    if args.hard_positive_mining_json:
        sample_weights, hard_positive_summary = build_hard_positive_sample_weights(
            h5_path=h5_path,
            train_idx=train_idx,
            error_mining_json=args.hard_positive_mining_json,
            hard_positive_weight=args.hard_positive_weight,
        )
        run_config["hard_positive_mining"] = hard_positive_summary

    save_json(run_dir / "run_config.json", run_config)
    split_payload = {"train_idx": train_idx, "val_idx": val_idx, "calibration_idx": val_idx}
    if test_idx is not None:
        split_payload["test_idx"] = test_idx
    np.savez_compressed(run_dir / "split_indices.npz", **split_payload)

    print("\n=== Phase 3 dataset summary ===")
    print(f"  HDF5:            {h5_path}")
    print(f"  Split source:    {candidate_summary['split_source']}")
    print(f"  Train samples:   {len(train_idx)} ({train_pos} pos / {train_neg} neg)")
    print(f"  Calibration:     {len(val_idx)} ({val_pos} pos / {val_neg} neg)")
    if test_idx is not None:
        print(f"  Test samples:    {len(test_idx)} ({test_pos} pos / {test_neg} neg)")
    if args.min_positive_amplitude > 0.0:
        if candidate_summary["split_source"] == "predefined":
            train_summary = candidate_summary["train"]
            val_summary = candidate_summary["val"]
            print(
                f"  Train candidate: {train_summary['candidate_samples']} "
                f"({train_summary['retained_positive']} pos / {train_summary['retained_negative']} neg)"
            )
            print(
                f"  Val candidate:   {val_summary['candidate_samples']} "
                f"({val_summary['retained_positive']} pos / {val_summary['retained_negative']} neg)"
            )
        else:
            all_summary = candidate_summary["all"]
            print(
                f"  Candidate set:   {all_summary['candidate_samples']} "
                f"({all_summary['retained_positive']} pos / {all_summary['retained_negative']} neg)"
            )
        print(f"  Min amplitude:   {args.min_positive_amplitude:.2e}")
    print(f"  Train mean:      {train_mean:.6e}")
    print(f"  Train std:       {train_std:.6e}")
    if extra_channel_datasets:
        print(f"  Input channels:  {args.input_channels} ({', '.join(['patches', *extra_channel_datasets])})")
        for idx, (mean_value, std_value) in enumerate(zip(channel_means, channel_stds)):
            print(f"    Channel {idx}: mean {mean_value:.6e}, std {std_value:.6e}")
    if normalization_source:
        print(f"  Norm source:     {normalization_source}")
    print(f"  Positive pixels: {positive_fraction:.3%}")
    print(f"  BCE pos_weight:  {pos_weight:.3f}")
    if args.boundary_weight > 0.0:
        print(
            f"  Boundary loss:   +{args.boundary_weight:.2f}x BCE weight "
            f"within {args.boundary_width_pixels} px of target edge"
        )
    if args.radius_head_weight > 0.0:
        print(
            f"  Radius head:     {radius_bin_count} bins {list(radius_bin_edges)} "
            f"weighted x{args.radius_head_weight:.2f}"
        )
    if hard_positive_summary:
        print(
            f"  Hard positives:  {hard_positive_summary['train_hard_samples']} train samples "
            f"weighted x{hard_positive_summary['hard_positive_weight']:.2f}"
        )

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
        cache_data=args.cache_data,
        extra_channel_datasets=extra_channel_datasets,
        channel_means=channel_means,
        channel_stds=channel_stds,
        radius_bin_edges=radius_bin_edges if args.radius_head_weight > 0.0 else None,
    )
    val_dataset = H5BubbleDataset(
        h5_path=h5_path,
        indices=val_idx,
        mean=train_mean,
        std=train_std,
        augment=False,
        seed=args.seed + 1,
        max_translate_pixels=0,
        cache_data=args.cache_data,
        extra_channel_datasets=extra_channel_datasets,
        channel_means=channel_means,
        channel_stds=channel_stds,
        radius_bin_edges=radius_bin_edges if args.radius_head_weight > 0.0 else None,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker if args.num_workers > 0 else None,
        "generator": torch.Generator().manual_seed(int(args.seed)),
    }
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
    train_loader = DataLoader(train_dataset, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(args).to(device)
    model = wrap_model_for_data_parallel(model, data_parallel_device_ids)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(args.epochs), 1),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    history = []
    best_val_score = -float("inf")
    best_epoch = -1
    start_epoch = 1

    resume_checkpoint_path = None
    if args.resume_checkpoint:
        resume_checkpoint_path = Path(args.resume_checkpoint)
        if not resume_checkpoint_path.is_absolute():
            resume_checkpoint_path = run_dir / resume_checkpoint_path
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        partial_resume = bool(args.model_only_resume or args.aux_head_weight > 0.0 or args.radius_head_weight > 0.0)
        load_model_state_dict(model, resume_checkpoint["model_state_dict"], strict=not partial_resume)
        if not partial_resume:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        history = [] if partial_resume else list(resume_checkpoint.get("history", []))
        start_epoch = 1 if partial_resume else int(resume_checkpoint["epoch"]) + 1
        for entry in history:
            score = float(entry["val"].get(args.checkpoint_metric, -float("inf")))
            if score > best_val_score:
                best_val_score = score
                best_epoch = int(entry["epoch"])
        if not partial_resume and history and int(history[-1]["epoch"]) == int(resume_checkpoint["epoch"]):
            last_score = float(history[-1]["val"].get(args.checkpoint_metric, -float("inf")))
            if abs(last_score - best_val_score) <= 1e-12:
                resume_checkpoint["checkpoint_metric"] = args.checkpoint_metric
                resume_checkpoint["best_val_score"] = best_val_score
                resume_checkpoint["best_epoch"] = best_epoch
                torch.save(resume_checkpoint, run_dir / "best_checkpoint.pt")
        print(
            f"  Resumed from:    {resume_checkpoint_path} "
            f"(next epoch {start_epoch}, best {args.checkpoint_metric}={best_val_score:.4f}"
            f"{', partial model-only resume' if partial_resume else ''})"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            threshold=args.threshold,
            image_min_positive_pixels=args.image_min_positive_pixels,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            aux_head_weight=args.aux_head_weight,
            radius_head_weight=args.radius_head_weight,
            radius_bin_count=radius_bin_count,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            boundary_weight=args.boundary_weight,
            boundary_width_pixels=args.boundary_width_pixels,
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
                image_min_positive_pixels=args.image_min_positive_pixels,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                aux_head_weight=args.aux_head_weight,
                radius_head_weight=args.radius_head_weight,
                radius_bin_count=radius_bin_count,
                use_amp=use_amp,
                grad_clip=0.0,
                boundary_weight=args.boundary_weight,
                boundary_width_pixels=args.boundary_width_pixels,
                train_mode=False,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        checkpoint_score = float(val_metrics[args.checkpoint_metric])
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(checkpoint_score)

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
            "checkpoint_metric": args.checkpoint_metric,
            "best_val_score": best_val_score,
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, run_dir / "last_checkpoint.pt")

        val_score = checkpoint_score
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            checkpoint["best_val_score"] = best_val_score
            checkpoint["best_epoch"] = best_epoch
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

        radius_text = (
            f" | val radius acc {val_metrics['radius_bin_accuracy']:.4f}"
            if args.radius_head_weight > 0.0
            else ""
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val dice+ {val_metrics['hard_dice_pos']:.4f} | "
            f"val img F1 {val_metrics['image_f1']:.4f}"
            f"{radius_text} | "
            f"lr {current_lr:.2e}"
        )

    print("\n=== Training complete ===")
    print(f"  Best epoch:                 {best_epoch}")
    print(f"  Best val {args.checkpoint_metric}:        {best_val_score:.4f}")
    print(f"  Outputs saved under:        {run_dir}")


if __name__ == "__main__":
    main()
