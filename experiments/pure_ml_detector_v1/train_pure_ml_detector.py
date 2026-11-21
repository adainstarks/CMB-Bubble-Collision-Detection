"""Train an isolated pure-ML CMB bubble-collision detector.

Assumptions
-----------
* This is a candidate-screening experiment, not a Bayesian evidence calculation.
* The deployed score is produced only by a neural network from raw CMB patches.
  No Wiener, SMHW, circular-template, or other classical score is used as an
  input feature.
* Positive examples use remediated-v1 simulated bubble injections. Negative
  examples can combine synthetic nulls with real-map null-control patches.
* Patches are CMB anisotropies in Kelvin. Non-finite values and implausibly
  large Kelvin-scale excursions are rejected before training.
* Network tensors use float32. Threshold selection and summary metrics use
  float64 to reduce avoidable numerical drift in fixed-FPR calculations.

References
----------
The physical signal model follows the Feeney et al. bubble-collision template
family used elsewhere in this repository. The detector objective is a
Neyman-Pearson-style screening proxy: maximize recall at fixed false-positive
rate before downstream Bayesian/template-likelihood follow-up.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models as tv_models
except Exception as exc:  # pragma: no cover - handled at runtime.
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    tv_models = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import timm
except Exception as exc:  # pragma: no cover - optional architecture backend.
    timm = None
    TIMM_IMPORT_ERROR = exc
else:
    TIMM_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_SYNTHETIC_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "training_data.h5"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "runs"
DEFAULT_NULL_TEMPLATE = PROJECT_ROOT / "data" / "remediated_v1" / "null_controls_{map}_mask090.h5"
DEFAULT_MAPS = ("smica", "nilc", "sevem", "commander")
FPR_POINTS = (0.05, 0.08)
EPS = 1e-8
ARCHITECTURES = (
    "small_cnn",
    "resnet18",
    "resnet34",
    "resnet50",
    "wide_resnet50_2",
    "resunet",
    "convnext_tiny",
    "convnextv2_small",
    "convnextv2_base",
    "efficientnet_b0",
    "efficientnet_v2_s",
)


@dataclass(frozen=True)
class Record:
    """One HDF5 row used by the detector dataset."""

    h5_path: str
    row_index: int
    label: int
    source: str
    map_name: str
    mask_tag: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Train a pure neural detector for CMB bubble-collision screening.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synthetic-h5", type=str, default=str(DEFAULT_SYNTHETIC_H5))
    parser.add_argument("--null-template", type=str, default=str(DEFAULT_NULL_TEMPLATE))
    parser.add_argument("--maps", type=str, default=",".join(DEFAULT_MAPS))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--experiment-name", type=str, default="smoke")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--architecture",
        type=str,
        default="small_cnn",
        choices=ARCHITECTURES,
        help=(
            "Pure neural detector architecture. All choices consume only raw CMB "
            "patch pixels; no classical response maps are allowed as inputs."
        ),
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--synthetic-negatives-per-positive", type=float, default=1.0)
    parser.add_argument("--real-null-negatives-per-map", type=int, default=0)
    parser.add_argument("--max-train-positives", type=int, default=0)
    parser.add_argument("--max-val-positives", type=int, default=0)
    parser.add_argument("--normalization-samples", type=int, default=2048)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--augment-dihedral", action="store_true")
    parser.add_argument(
        "--cache-patches",
        action="store_true",
        help="Load selected train/validation patches into RAM before training. Useful for small pilots on compressed HDF5 files.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Write live step progress every N batches during train/validation/scoring phases.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated CUDA ids for DataParallel, e.g. 0,1. Leave empty for single-GPU.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Optional pure-ML detector checkpoint to load model weights from before training.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision for faster full runs.")
    parser.add_argument("--save-scores", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate user-controlled settings."""

    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(f"PyTorch import failed: {TORCH_IMPORT_ERROR}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not (0.0 < args.learning_rate < 1.0):
        raise ValueError("--learning-rate must be in (0, 1).")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be in [0, 1).")
    if args.base_channels <= 0:
        raise ValueError("--base-channels must be positive.")
    if args.architecture not in ARCHITECTURES:
        raise ValueError(f"--architecture must be one of {ARCHITECTURES}.")
    if not (0.0 < args.target_fpr < 1.0):
        raise ValueError("--target-fpr must be in (0, 1).")
    if args.synthetic_negatives_per_positive < 0.0:
        raise ValueError("--synthetic-negatives-per-positive must be non-negative.")
    if args.real_null_negatives_per_map < 0:
        raise ValueError("--real-null-negatives-per-map must be non-negative.")
    if args.max_train_positives < 0 or args.max_val_positives < 0:
        raise ValueError("--max-train-positives and --max-val-positives must be non-negative.")
    if args.normalization_samples <= 0:
        raise ValueError("--normalization-samples must be positive.")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive.")
    if args.ranking_loss_weight < 0.0:
        raise ValueError("--ranking-loss-weight must be non-negative.")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative.")
    if args.gpu_ids.strip():
        parse_gpu_ids(args.gpu_ids)
    if args.resume_checkpoint and not Path(args.resume_checkpoint).expanduser().exists():
        raise FileNotFoundError(f"Missing --resume-checkpoint: {args.resume_checkpoint}")
    if not Path(args.synthetic_h5).expanduser().exists():
        raise FileNotFoundError(f"Missing synthetic HDF5: {args.synthetic_h5}")
    for map_name in parse_maps(args.maps):
        null_path = Path(str(args.null_template).format(map=map_name)).expanduser()
        if args.real_null_negatives_per_map > 0 and not null_path.exists():
            raise FileNotFoundError(f"Missing null-control HDF5 for {map_name}: {null_path}")


def parse_maps(text: str) -> tuple[str, ...]:
    """Return cleaned-map names from a comma-separated string."""

    maps = tuple(token.strip().lower() for token in str(text).split(",") if token.strip())
    if not maps:
        raise ValueError("--maps must name at least one cleaned map.")
    return maps


def seed_everything(seed: int) -> None:
    """Seed numpy, Python, and torch RNGs."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = True


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested torch device."""

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_gpu_ids(gpu_ids: str) -> list[int]:
    """Parse comma-separated CUDA ids."""

    parsed: list[int] = []
    for token in str(gpu_ids).split(","):
        token = token.strip()
        if not token:
            continue
        gpu_id = int(token)
        if gpu_id < 0:
            raise ValueError("GPU ids must be non-negative.")
        parsed.append(gpu_id)
    return parsed


def wrap_data_parallel(model: nn.Module, device: torch.device, gpu_ids: str) -> nn.Module:
    """Wrap a model in DataParallel when requested."""

    if device.type != "cuda" or not str(gpu_ids).strip():
        return model
    ids = parse_gpu_ids(gpu_ids)
    if len(ids) <= 1:
        return model
    visible = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in ids if gpu_id >= visible]
    if invalid:
        raise RuntimeError(f"Requested GPU ids {invalid}, but only {visible} CUDA devices are visible.")
    return nn.DataParallel(model, device_ids=ids)


def model_state_for_save(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a checkpoint state dict without DataParallel prefixes."""

    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_weights(model: nn.Module, checkpoint_path: Path) -> int:
    """Load compatible model weights and return source epoch."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError(f"Checkpoint lacks model_state_dict: {checkpoint_path}")
    target = model.module if isinstance(model, nn.DataParallel) else model
    cleaned = {
        (key[len("module."):] if str(key).startswith("module.") else key): value
        for key, value in state.items()
    }
    target.load_state_dict(cleaned, strict=True)
    return int(checkpoint.get("epoch", 0))


def read_indices(h5: h5py.File, split_name: str) -> np.ndarray:
    """Read split indices from an HDF5 file."""

    path = f"splits/{split_name}_idx"
    if path not in h5:
        raise KeyError(f"Missing split dataset: {path}")
    return np.asarray(h5[path][:], dtype=np.int64)


def choose_rows(
    candidates: np.ndarray,
    *,
    limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle and optionally cap a row-index array."""

    values = np.asarray(candidates, dtype=np.int64).copy()
    rng.shuffle(values)
    if int(limit) > 0:
        values = values[: int(limit)]
    return values


def build_records(args: argparse.Namespace) -> tuple[list[Record], list[Record], dict[str, Any]]:
    """Build train/validation records from synthetic positives and null controls."""

    rng = np.random.default_rng(int(args.seed))
    synthetic_h5 = str(Path(args.synthetic_h5).expanduser().resolve())
    map_names = parse_maps(args.maps)
    train_records: list[Record] = []
    val_records: list[Record] = []

    with h5py.File(synthetic_h5, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        train_idx = read_indices(h5, "train")
        val_idx = read_indices(h5, "val")
        train_pos = choose_rows(train_idx[labels[train_idx] == 1], limit=args.max_train_positives, rng=rng)
        val_pos = choose_rows(val_idx[labels[val_idx] == 1], limit=args.max_val_positives, rng=rng)
        train_neg_pool = train_idx[labels[train_idx] == 0]
        val_neg_pool = val_idx[labels[val_idx] == 0]
        train_synth_neg_n = int(round(len(train_pos) * float(args.synthetic_negatives_per_positive)))
        val_synth_neg_n = int(round(len(val_pos) * float(args.synthetic_negatives_per_positive)))
        train_neg = choose_rows(train_neg_pool, limit=train_synth_neg_n, rng=rng)
        val_neg = choose_rows(val_neg_pool, limit=val_synth_neg_n, rng=rng)

    for row in train_pos:
        train_records.append(Record(synthetic_h5, int(row), 1, "synthetic_positive", "synthetic", "none"))
    for row in val_pos:
        val_records.append(Record(synthetic_h5, int(row), 1, "synthetic_positive", "synthetic", "none"))
    for row in train_neg:
        train_records.append(Record(synthetic_h5, int(row), 0, "synthetic_negative", "synthetic", "none"))
    for row in val_neg:
        val_records.append(Record(synthetic_h5, int(row), 0, "synthetic_negative", "synthetic", "none"))

    null_summary = {}
    if int(args.real_null_negatives_per_map) > 0:
        for map_name in map_names:
            null_path = str(Path(str(args.null_template).format(map=map_name)).expanduser().resolve())
            with h5py.File(null_path, "r") as h5:
                labels = np.asarray(h5["labels"][:], dtype=np.uint8)
                if int(labels.sum()) != 0:
                    raise ValueError(f"Null-control file contains positive labels: {null_path}")
                train_idx = read_indices(h5, "train")
                val_idx = read_indices(h5, "val")
                train_rows = choose_rows(train_idx, limit=args.real_null_negatives_per_map, rng=rng)
                val_rows = choose_rows(val_idx, limit=max(1, args.real_null_negatives_per_map // 4), rng=rng)
            for row in train_rows:
                train_records.append(Record(null_path, int(row), 0, "real_null", map_name, "mask090"))
            for row in val_rows:
                val_records.append(Record(null_path, int(row), 0, "real_null", map_name, "mask090"))
            null_summary[map_name] = {
                "train_real_null": int(len(train_rows)),
                "val_real_null": int(len(val_rows)),
                "path": null_path,
            }

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    summary = {
        "train_records": int(len(train_records)),
        "val_records": int(len(val_records)),
        "train_positive": int(sum(row.label for row in train_records)),
        "val_positive": int(sum(row.label for row in val_records)),
        "train_negative": int(sum(1 - row.label for row in train_records)),
        "val_negative": int(sum(1 - row.label for row in val_records)),
        "null_summary": null_summary,
    }
    if summary["train_positive"] == 0 or summary["val_positive"] == 0:
        raise ValueError("Training and validation require at least one positive sample.")
    if summary["train_negative"] == 0 or summary["val_negative"] == 0:
        raise ValueError("Training and validation require at least one negative sample.")
    return train_records, val_records, summary


def record_to_json(row: Record) -> dict[str, Any]:
    """Return a JSON-serializable record."""

    return {
        "h5_path": row.h5_path,
        "row_index": int(row.row_index),
        "label": int(row.label),
        "source": row.source,
        "map_name": row.map_name,
        "mask_tag": row.mask_tag,
    }


def read_patch(h5_path: str, row_index: int) -> np.ndarray:
    """Read one patch and validate physical sanity."""

    with h5py.File(h5_path, "r") as h5:
        patch = np.asarray(h5["patches"][int(row_index)], dtype=np.float32)
    if patch.shape != (256, 256):
        raise ValueError(f"Expected a 256x256 patch, got {patch.shape} from {h5_path}:{row_index}")
    if not np.all(np.isfinite(patch)):
        raise ValueError(f"Patch contains non-finite values: {h5_path}:{row_index}")
    if float(np.max(np.abs(patch))) > 1.0:
        raise ValueError(
            f"Patch has implausible Kelvin anisotropy scale: max abs={float(np.max(np.abs(patch))):.3g}"
        )
    return patch


def load_patch_block(records: list[Record]) -> np.ndarray:
    """Load patches for records, grouping HDF5 reads by source file."""

    patches = np.empty((len(records), 256, 256), dtype=np.float32)
    positions_by_path: dict[str, list[int]] = {}
    for pos, row in enumerate(records):
        positions_by_path.setdefault(row.h5_path, []).append(pos)

    for h5_path, positions in positions_by_path.items():
        rows = np.asarray([records[pos].row_index for pos in positions], dtype=np.int64)
        order = np.argsort(rows)
        sorted_rows = rows[order]
        if np.unique(sorted_rows).size != sorted_rows.size:
            # h5py fancy indexing does not accept duplicate indices reliably.
            with h5py.File(h5_path, "r") as h5:
                for pos in positions:
                    patches[pos] = np.asarray(h5["patches"][records[pos].row_index], dtype=np.float32)
        else:
            with h5py.File(h5_path, "r") as h5:
                block = np.asarray(h5["patches"][sorted_rows], dtype=np.float32)
            inverse = np.empty_like(order)
            inverse[order] = np.arange(order.size)
            for local_i, pos in enumerate(positions):
                patches[pos] = block[inverse[local_i]]
    if not np.all(np.isfinite(patches)):
        raise ValueError("Cached patch block contains non-finite values.")
    if float(np.max(np.abs(patches))) > 1.0:
        raise ValueError(
            f"Cached patch block has implausible Kelvin scale: max abs={float(np.max(np.abs(patches))):.3g}"
        )
    return patches


def compute_normalization(records: list[Record], max_samples: int, seed: int) -> dict[str, float]:
    """Compute global raw-patch normalization from train records."""

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(records), dtype=np.int64)
    rng.shuffle(idx)
    idx = idx[: min(int(max_samples), len(idx))]
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0
    sample_records = [records[int(pos)] for pos in idx]
    for start in range(0, len(sample_records), 256):
        block = load_patch_block(sample_records[start:start + 256]).astype(np.float64, copy=False)
        total_sum += float(np.sum(block))
        total_sq += float(np.sum(np.square(block)))
        total_count += int(block.size)
    mean = total_sum / max(total_count, 1)
    var = max(total_sq / max(total_count, 1) - mean * mean, 1e-20)
    std = math.sqrt(var)
    return {
        "mean": float(mean),
        "std": float(max(std, 1e-12)),
        "num_samples": int(len(idx)),
    }


class DetectorDataset(Dataset):
    """Torch dataset for multi-HDF5 pure detector records."""

    def __init__(
        self,
        records: list[Record],
        normalization: dict[str, float],
        *,
        augment: bool,
        seed: int,
        cache_patches: bool,
    ):
        self.records = list(records)
        self.mean = float(normalization["mean"])
        self.std = float(normalization["std"])
        self.augment = bool(augment)
        self.seed = int(seed)
        self._handles: dict[str, h5py.File] = {}
        self._rng = None
        self._patch_cache = load_patch_block(self.records) if bool(cache_patches) else None

    def __len__(self) -> int:
        return len(self.records)

    def _handle(self, path: str) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_seed = worker_info.seed if worker_info is not None else self.seed
            self._rng = np.random.default_rng(int(worker_seed) % (2**32))
        return self._rng

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.records[int(item)]
        if self._patch_cache is None:
            h5 = self._handle(row.h5_path)
            patch = np.asarray(h5["patches"][row.row_index], dtype=np.float32)
        else:
            patch = np.asarray(self._patch_cache[int(item)], dtype=np.float32)
        if patch.shape != (256, 256):
            raise ValueError(f"Expected 256x256 patch, got {patch.shape}")
        if not np.all(np.isfinite(patch)):
            raise ValueError(f"Patch contains non-finite values: {row.h5_path}:{row.row_index}")
        if float(np.max(np.abs(patch))) > 1.0:
            raise ValueError(f"Patch outside Kelvin sanity range: {row.h5_path}:{row.row_index}")
        if self.augment:
            rng = self._get_rng()
            k = int(rng.integers(0, 4))
            if k:
                patch = np.rot90(patch, k=k).copy()
            if rng.random() < 0.5:
                patch = np.flip(patch, axis=0).copy()
            if rng.random() < 0.5:
                patch = np.flip(patch, axis=1).copy()
        image = ((patch - self.mean) / self.std).astype(np.float32, copy=False)[None, :, :]
        return {
            "image": torch.from_numpy(image),
            "label": torch.tensor(float(row.label), dtype=torch.float32),
            "row": torch.tensor(int(row.row_index), dtype=torch.long),
        }

    def __del__(self) -> None:  # pragma: no cover - cleanup only.
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


class ResidualBlock(nn.Module):
    """Small residual block for 256x256 CMB patches."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DownBlock(nn.Module):
    """Downsampling convolution followed by residual refinement."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PureMLDetector(nn.Module):
    """A pure CNN detector whose output is an image-level bubble score."""

    def __init__(self, base_channels: int, dropout: float):
        super().__init__()
        c = int(base_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            ResidualBlock(c),
            DownBlock(c, c * 2),
            DownBlock(c * 2, c * 4),
            DownBlock(c * 4, c * 8),
            DownBlock(c * 8, c * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(c * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).reshape(-1)


def group_norm(channels: int) -> nn.GroupNorm:
    """Return GroupNorm with a divisor-safe group count for small GPU batches."""

    groups = min(8, int(channels))
    while int(channels) % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


class ResUNetBlock(nn.Module):
    """Residual convolution block used by the raw-patch ResUNet detector."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False)
        )
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1, bias=False),
            group_norm(int(out_channels)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1, bias=False),
            group_norm(int(out_channels)),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x) + self.net(x))


class ResUNetDown(nn.Module):
    """Downsample by two and refine with a residual block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, stride=2, padding=1, bias=False),
            group_norm(int(out_channels)),
            nn.SiLU(inplace=True),
            ResUNetBlock(int(out_channels), int(out_channels)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResUNetUp(nn.Module):
    """Upsample, concatenate an encoder skip map, and refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = ResUNetBlock(int(in_channels) + int(skip_channels), int(out_channels))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResUNetGlobalDetector(nn.Module):
    """Localization-aware ResUNet that still emits one screening logit.

    The model keeps a high-resolution decoder path so the scalar score can be
    driven by localized bubble-like evidence rather than only by global average
    pooling. This is still an image-level detector because the current
    benchmark provides scalar labels; a later mask/center-radius loss can reuse
    the same decoder without changing the training data contract.
    """

    def __init__(self, base_channels: int, dropout: float):
        super().__init__()
        c = int(base_channels)
        self.stem = ResUNetBlock(1, c)
        self.down1 = ResUNetDown(c, c * 2)
        self.down2 = ResUNetDown(c * 2, c * 4)
        self.down3 = ResUNetDown(c * 4, c * 8)
        self.down4 = ResUNetDown(c * 8, c * 8)
        self.up3 = ResUNetUp(c * 8, c * 8, c * 4)
        self.up2 = ResUNetUp(c * 4, c * 4, c * 2)
        self.up1 = ResUNetUp(c * 2, c * 2, c)
        self.up0 = ResUNetUp(c, c, c)
        self.heatmap_head = nn.Conv2d(c, 1, kernel_size=1)
        self.bottleneck_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(c * 8 + 3, c * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(c * 2, 1),
        )

    @staticmethod
    def _top_fraction_mean(heatmap: torch.Tensor, fraction: float = 0.01) -> torch.Tensor:
        """Return the mean of the strongest localized heatmap responses."""

        flat = heatmap.flatten(start_dim=1)
        k = max(1, int(round(flat.shape[1] * float(fraction))))
        return torch.topk(flat, k=k, dim=1).values.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        y = self.up3(x4, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        heatmap = self.heatmap_head(y)
        pooled = self.bottleneck_pool(x4).flatten(start_dim=1)
        heat_stats = torch.cat(
            [
                heatmap.amax(dim=(2, 3)),
                heatmap.mean(dim=(2, 3)),
                self._top_fraction_mean(heatmap),
            ],
            dim=1,
        )
        return self.head(torch.cat([pooled, heat_stats], dim=1)).reshape(-1)


def replace_first_conv(module: nn.Module, in_channels: int = 1) -> None:
    """Replace a torchvision model's first convolution with a raw-patch stem."""

    if hasattr(module, "conv1") and isinstance(module.conv1, nn.Conv2d):
        old = module.conv1
        module.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        return
    if hasattr(module, "features") and isinstance(module.features, nn.Sequential):
        first = module.features[0]
        if isinstance(first, nn.Sequential) and isinstance(first[0], nn.Conv2d):
            old = first[0]
            first[0] = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=old.bias is not None,
            )
            return
        if isinstance(first, nn.Conv2d):
            old = first
            module.features[0] = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=old.bias is not None,
            )
            return
    raise TypeError(f"Could not locate first convolution in {module.__class__.__name__}.")


def replace_classifier(module: nn.Module, dropout: float) -> None:
    """Replace a torchvision classifier with a one-logit detector head."""

    if hasattr(module, "fc") and isinstance(module.fc, nn.Linear):
        in_features = int(module.fc.in_features)
        module.fc = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(in_features, 1))
        return
    if hasattr(module, "classifier") and isinstance(module.classifier, nn.Sequential):
        for idx in range(len(module.classifier) - 1, -1, -1):
            layer = module.classifier[idx]
            if isinstance(layer, nn.Linear):
                in_features = int(layer.in_features)
                replacement: nn.Module
                if float(dropout) > 0.0:
                    replacement = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(in_features, 1))
                else:
                    replacement = nn.Linear(in_features, 1)
                module.classifier[idx] = replacement
                return
    raise TypeError(f"Could not locate classifier in {module.__class__.__name__}.")


class TorchvisionDetector(nn.Module):
    """Thin wrapper around a torchvision image-classification backbone."""

    def __init__(self, architecture: str, dropout: float):
        super().__init__()
        if tv_models is None:
            raise RuntimeError("torchvision.models is unavailable.")
        if architecture == "resnet18":
            model = tv_models.resnet18(weights=None)
        elif architecture == "resnet34":
            model = tv_models.resnet34(weights=None)
        elif architecture == "resnet50":
            model = tv_models.resnet50(weights=None)
        elif architecture == "wide_resnet50_2":
            model = tv_models.wide_resnet50_2(weights=None)
        elif architecture == "convnext_tiny":
            model = tv_models.convnext_tiny(weights=None)
        elif architecture == "efficientnet_b0":
            model = tv_models.efficientnet_b0(weights=None)
        elif architecture == "efficientnet_v2_s":
            model = tv_models.efficientnet_v2_s(weights=None)
        else:
            raise ValueError(f"Unsupported torchvision detector architecture: {architecture!r}")
        replace_first_conv(model, in_channels=1)
        replace_classifier(model, dropout=float(dropout))
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).reshape(-1)


class TimmDetector(nn.Module):
    """Thin wrapper around timm backbones that support one-channel inputs."""

    def __init__(self, architecture: str, dropout: float):
        super().__init__()
        if timm is None:
            raise RuntimeError(f"timm import failed, cannot build {architecture}: {TIMM_IMPORT_ERROR}")
        model_name_by_arch = {
            "convnextv2_small": "convnextv2_small",
            "convnextv2_base": "convnextv2_base",
        }
        model_name = model_name_by_arch.get(str(architecture))
        if model_name is None:
            raise ValueError(f"Unsupported timm detector architecture: {architecture!r}")
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=1,
            num_classes=1,
            drop_rate=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).reshape(-1)


def build_detector_model(args: Any) -> nn.Module:
    """Build a pure-ML detector from CLI args or checkpoint args."""

    if isinstance(args, dict):
        architecture = str(args.get("architecture", "small_cnn"))
        base_channels = int(args.get("base_channels", 32))
        dropout = float(args.get("dropout", 0.2))
    else:
        architecture = str(args.architecture)
        base_channels = int(args.base_channels)
        dropout = float(args.dropout)
    if architecture == "small_cnn":
        return PureMLDetector(base_channels=base_channels, dropout=dropout)
    if architecture == "resunet":
        return ResUNetGlobalDetector(base_channels=base_channels, dropout=dropout)
    if architecture.startswith("convnextv2_"):
        return TimmDetector(architecture=architecture, dropout=dropout)
    return TorchvisionDetector(architecture=architecture, dropout=dropout)


def binary_focal_with_logits(logits: torch.Tensor, labels: torch.Tensor, gamma: float) -> torch.Tensor:
    """Binary focal loss for image-level detection."""

    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if gamma <= 0.0:
        return bce.mean()
    probs = torch.sigmoid(logits)
    pt = probs * labels + (1.0 - probs) * (1.0 - labels)
    return (bce * (1.0 - pt).pow(float(gamma))).mean()


def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Encourage positive scores to exceed negative scores inside each batch."""

    pos = logits[labels >= 0.5]
    neg = logits[labels < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.sum() * 0.0
    diff = pos[:, None] - neg[None, :]
    return F.softplus(-diff).mean()


def make_loader(
    records: list[Record],
    normalization: dict[str, float],
    *,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int,
    num_workers: int,
    cache_patches: bool,
) -> DataLoader:
    """Build a DataLoader."""

    dataset = DetectorDataset(records, normalization, augment=augment, seed=seed, cache_patches=cache_patches)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def threshold_for_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> tuple[float, float]:
    """Return split-conformal score threshold and realized FPR.

    The rule matches the repository's classical same-grid reports: choose the
    ``ceil((n_null + 1) * (1 - alpha))`` null-score order statistic and declare
    detections with ``score > threshold``. This is slightly conservative at
    finite null count and avoids the off-by-one leniency from ``>=``.
    """

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    neg = scores[labels == 0]
    if neg.size == 0:
        raise ValueError("Cannot calibrate FPR without negative scores.")
    sorted_neg = np.sort(neg)
    rank_1indexed = int(math.ceil((int(neg.size) + 1) * (1.0 - float(fpr))))
    rank_1indexed = min(max(rank_1indexed, 1), int(neg.size))
    threshold = float(sorted_neg[rank_1indexed - 1])
    realized = float(np.mean(neg > threshold))
    return threshold, realized


def summarize_scores(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Summarize detector scores at fixed FPR points."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite detector scores encountered.")
    out: dict[str, Any] = {
        "num_samples": int(labels.size),
        "num_positive": int(np.count_nonzero(labels == 1)),
        "num_negative": int(np.count_nonzero(labels == 0)),
        "score_mean_positive": float(np.mean(scores[labels == 1])) if np.any(labels == 1) else float("nan"),
        "score_mean_negative": float(np.mean(scores[labels == 0])) if np.any(labels == 0) else float("nan"),
    }
    if out["num_positive"] > 0 and out["num_negative"] > 0:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    fixed = {}
    for fpr in FPR_POINTS:
        threshold, realized = threshold_for_fpr(scores, labels, fpr)
        active = scores > threshold
        tp = int(np.count_nonzero(active & (labels == 1)))
        fp = int(np.count_nonzero(active & (labels == 0)))
        fn = int(np.count_nonzero((~active) & (labels == 1)))
        tn = int(np.count_nonzero((~active) & (labels == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, EPS)
        fixed[f"fpr_{fpr:.2f}"] = {
            "threshold": float(threshold),
            "realized_fpr": float(realized),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    out["fixed_fpr"] = fixed
    return out


def autocast_context(device: torch.device, use_amp: bool) -> Any:
    """Return a version-compatible AMP autocast context."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=str(device.type), enabled=bool(use_amp))
    return torch.cuda.amp.autocast(enabled=bool(use_amp))


def make_grad_scaler(device: torch.device, use_amp: bool) -> Any:
    """Return a version-compatible gradient scaler."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(str(device.type), enabled=bool(use_amp))
    return torch.cuda.amp.GradScaler(enabled=bool(use_amp))


def optimizer_lr(optimizer: torch.optim.Optimizer | None) -> float | None:
    """Return the first optimizer learning rate for progress telemetry."""

    if optimizer is None or not optimizer.param_groups:
        return None
    return float(optimizer.param_groups[0].get("lr", 0.0))


def write_progress(
    progress_path: Path | None,
    progress_jsonl_path: Path | None,
    payload: dict[str, Any],
    *,
    append: bool = True,
) -> None:
    """Write live progress as current JSON plus optional JSONL history."""

    if progress_path is None:
        return
    row = dict(payload)
    row["updated_utc"] = utc_now()
    write_json(progress_path, row)
    if append and progress_jsonl_path is not None:
        append_jsonl(progress_jsonl_path, row)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    use_amp: bool,
    focal_gamma: float,
    ranking_loss_weight: float,
    epoch: int | None = None,
    phase: str = "train",
    progress_path: Path | None = None,
    progress_jsonl_path: Path | None = None,
    progress_interval: int = 25,
) -> dict[str, float]:
    """Run one train or evaluation epoch."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_bce = 0.0
    total_rank = 0.0
    total_n = 0
    total_steps = len(loader)
    start_time = time.monotonic()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bool(use_amp)):
            logits = model(images)
            bce = binary_focal_with_logits(logits, labels, gamma=float(focal_gamma))
            rank = pairwise_ranking_loss(logits, labels)
            loss = bce + float(ranking_loss_weight) * rank
        if training:
            if scaler is not None and bool(use_amp):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch_n = int(labels.numel())
        total_loss += float(loss.detach().item()) * batch_n
        total_bce += float(bce.detach().item()) * batch_n
        total_rank += float(rank.detach().item()) * batch_n
        total_n += batch_n
        if step == 1 or step == total_steps or step % int(progress_interval) == 0:
            elapsed = max(time.monotonic() - start_time, 1e-6)
            steps_per_sec = float(step / elapsed)
            samples_per_sec = float(total_n / elapsed)
            eta = float((total_steps - step) / max(steps_per_sec, 1e-12))
            write_progress(
                progress_path,
                progress_jsonl_path,
                {
                    "event": "step",
                    "phase": str(phase),
                    "epoch": None if epoch is None else int(epoch),
                    "step": int(step),
                    "total_steps": int(total_steps),
                    "samples": int(total_n),
                    "total_samples": int(len(loader.dataset)),
                    "percent": float(100.0 * step / max(total_steps, 1)),
                    "avg_loss": float(total_loss / max(total_n, 1)),
                    "avg_bce_or_focal": float(total_bce / max(total_n, 1)),
                    "avg_ranking_loss": float(total_rank / max(total_n, 1)),
                    "last_loss": float(loss.detach().item()),
                    "last_bce_or_focal": float(bce.detach().item()),
                    "last_ranking_loss": float(rank.detach().item()),
                    "learning_rate": optimizer_lr(optimizer),
                    "elapsed_sec": float(elapsed),
                    "eta_sec": eta,
                    "samples_per_sec": samples_per_sec,
                    "steps_per_sec": steps_per_sec,
                },
            )
    return {
        "loss": total_loss / max(total_n, 1),
        "bce_or_focal": total_bce / max(total_n, 1),
        "ranking_loss": total_rank / max(total_n, 1),
    }


@torch.no_grad()
def collect_scores(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool = False,
    epoch: int | None = None,
    progress_path: Path | None = None,
    progress_jsonl_path: Path | None = None,
    progress_interval: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect sigmoid scores and labels from a loader."""

    model.eval()
    score_parts = []
    label_parts = []
    total_steps = len(loader)
    total_n = 0
    start_time = time.monotonic()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        with autocast_context(device, bool(use_amp)):
            logits = model(images)
        scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        labels = batch["label"].detach().cpu().numpy().astype(np.uint8)
        score_parts.append(scores)
        label_parts.append(labels)
        total_n += int(labels.size)
        if step == 1 or step == total_steps or step % int(progress_interval) == 0:
            elapsed = max(time.monotonic() - start_time, 1e-6)
            steps_per_sec = float(step / elapsed)
            write_progress(
                progress_path,
                progress_jsonl_path,
                {
                    "event": "step",
                    "phase": "score_validation",
                    "epoch": None if epoch is None else int(epoch),
                    "step": int(step),
                    "total_steps": int(total_steps),
                    "samples": int(total_n),
                    "total_samples": int(len(loader.dataset)),
                    "percent": float(100.0 * step / max(total_steps, 1)),
                    "elapsed_sec": float(elapsed),
                    "eta_sec": float((total_steps - step) / max(steps_per_sec, 1e-12)),
                    "samples_per_sec": float(total_n / elapsed),
                    "steps_per_sec": steps_per_sec,
                },
            )
    return np.concatenate(score_parts), np.concatenate(label_parts)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def utc_now() -> str:
    """Return compact UTC timestamp."""

    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    """Write epoch history as CSV."""

    if not history:
        return
    keys = sorted({key for row in history for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    """Train and evaluate the isolated pure-ML detector."""

    args = parse_args()
    validate_args(args)
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    run_dir = Path(args.output_root).expanduser().resolve() / str(args.experiment_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    live_metrics_path = run_dir / "live_metrics.jsonl"
    history_csv_path = run_dir / "history.csv"
    progress_path = run_dir / "progress.json"
    progress_jsonl_path = run_dir / "progress.jsonl"

    def mark_interrupted(signum: int, _frame: Any) -> None:
        """Record clean interrupted state before exiting on SIGINT/SIGTERM."""

        write_json(
            status_path,
            {
                "state": "interrupted",
                "pid": int(os.getpid()),
                "signal": int(signum),
                "updated_utc": utc_now(),
                "experiment_name": str(args.experiment_name),
                "architecture": str(args.architecture),
                "device": str(device),
                "gpu_ids": str(args.gpu_ids),
                "best_checkpoint": str(run_dir / "best_checkpoint.pt")
                if (run_dir / "best_checkpoint.pt").exists()
                else None,
                "progress_json": str(progress_path),
                "progress_jsonl": str(progress_jsonl_path),
                "live_metrics_jsonl": str(live_metrics_path),
                "history_csv": str(history_csv_path),
            },
        )
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, mark_interrupted)
    signal.signal(signal.SIGTERM, mark_interrupted)

    write_json(
        status_path,
        {
            "state": "starting",
            "pid": int(os.getpid()),
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "experiment_name": str(args.experiment_name),
            "architecture": str(args.architecture),
            "device": str(device),
            "gpu_ids": str(args.gpu_ids),
            "current_epoch": None,
            "total_requested_epochs": int(args.epochs),
            "best_recall": None,
            "best_checkpoint": None,
            "progress_json": str(progress_path),
            "progress_jsonl": str(progress_jsonl_path),
        },
    )

    train_records, val_records, data_summary = build_records(args)
    normalization = compute_normalization(train_records, int(args.normalization_samples), int(args.seed))
    write_json(
        run_dir / "records_summary.json",
        {
            "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "data_summary": data_summary,
            "normalization": normalization,
            "train_records_preview": [record_to_json(row) for row in train_records[:20]],
            "val_records_preview": [record_to_json(row) for row in val_records[:20]],
        },
    )

    train_loader = make_loader(
        train_records,
        normalization,
        batch_size=int(args.batch_size),
        shuffle=True,
        augment=bool(args.augment_dihedral),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        cache_patches=bool(args.cache_patches),
    )
    val_loader = make_loader(
        val_records,
        normalization,
        batch_size=int(args.batch_size),
        shuffle=False,
        augment=False,
        seed=int(args.seed) + 10_000,
        num_workers=int(args.num_workers),
        cache_patches=bool(args.cache_patches),
    )

    model = build_detector_model(args).to(device)
    model = wrap_data_parallel(model, device, str(args.gpu_ids))
    resumed_from_epoch = 0
    if str(args.resume_checkpoint).strip():
        resumed_from_epoch = load_model_weights(model, Path(args.resume_checkpoint).expanduser().resolve())
        print(f"Loaded model weights from epoch {resumed_from_epoch}: {args.resume_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(device, use_amp)
    write_json(
        status_path,
        {
            "state": "running",
            "pid": int(os.getpid()),
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "experiment_name": str(args.experiment_name),
            "architecture": str(args.architecture),
            "device": str(device),
            "gpu_ids": str(args.gpu_ids),
            "data_parallel": bool(isinstance(model, nn.DataParallel)),
            "amp": bool(use_amp),
            "resumed_from_epoch": int(resumed_from_epoch),
            "current_epoch": int(resumed_from_epoch),
            "final_epoch": int(resumed_from_epoch + int(args.epochs)),
            "best_recall": None,
            "best_checkpoint": None,
            "progress_json": str(progress_path),
            "progress_jsonl": str(progress_jsonl_path),
            "live_metrics_jsonl": str(live_metrics_path),
            "history_csv": str(history_csv_path),
        },
    )

    history = []
    best_recall = -1.0
    best_report: dict[str, Any] | None = None
    best_path = run_dir / "best_checkpoint.pt"

    for epoch in range(resumed_from_epoch + 1, resumed_from_epoch + int(args.epochs) + 1):
        write_json(
            status_path,
            {
                "state": "running",
                "pid": int(os.getpid()),
                "created_utc": None,
                "updated_utc": utc_now(),
                "experiment_name": str(args.experiment_name),
                "architecture": str(args.architecture),
                "device": str(device),
                "gpu_ids": str(args.gpu_ids),
                "data_parallel": bool(isinstance(model, nn.DataParallel)),
                "amp": bool(use_amp),
                "resumed_from_epoch": int(resumed_from_epoch),
                "completed_epoch": int(epoch - 1),
                "active_epoch": int(epoch),
                "current_epoch": int(epoch - 1),
                "final_epoch": int(resumed_from_epoch + int(args.epochs)),
                "best_recall": None if best_recall < 0.0 else float(best_recall),
                "best_checkpoint": None if best_recall < 0.0 else str(best_path),
                "progress_json": str(progress_path),
                "progress_jsonl": str(progress_jsonl_path),
                "live_metrics_jsonl": str(live_metrics_path),
                "history_csv": str(history_csv_path),
            },
        )
        train_loss = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            focal_gamma=float(args.focal_gamma),
            ranking_loss_weight=float(args.ranking_loss_weight),
            epoch=int(epoch),
            phase="train",
            progress_path=progress_path,
            progress_jsonl_path=progress_jsonl_path,
            progress_interval=int(args.progress_interval),
        )
        val_loss = run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            focal_gamma=float(args.focal_gamma),
            ranking_loss_weight=float(args.ranking_loss_weight),
            epoch=int(epoch),
            phase="validate_loss",
            progress_path=progress_path,
            progress_jsonl_path=progress_jsonl_path,
            progress_interval=int(args.progress_interval),
        )
        val_scores, val_labels = collect_scores(
            model,
            val_loader,
            device=device,
            use_amp=use_amp,
            epoch=int(epoch),
            progress_path=progress_path,
            progress_jsonl_path=progress_jsonl_path,
            progress_interval=int(args.progress_interval),
        )
        score_summary = summarize_scores(val_scores, val_labels)
        target_key = f"fpr_{float(args.target_fpr):.2f}"
        target_row = score_summary["fixed_fpr"][target_key]
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss["loss"]),
            "train_bce_or_focal": float(train_loss["bce_or_focal"]),
            "train_ranking_loss": float(train_loss["ranking_loss"]),
            "val_loss": float(val_loss["loss"]),
            "val_bce_or_focal": float(val_loss["bce_or_focal"]),
            "val_ranking_loss": float(val_loss["ranking_loss"]),
            "val_auroc": float(score_summary["auroc"]),
            "val_auprc": float(score_summary["auprc"]),
            "target_fpr": float(args.target_fpr),
            "target_threshold": float(target_row["threshold"]),
            "target_recall": float(target_row["recall"]),
            "target_precision": float(target_row["precision"]),
            "target_realized_fpr": float(target_row["realized_fpr"]),
            "target_tp": int(target_row["tp"]),
            "target_fp": int(target_row["fp"]),
            "target_fn": int(target_row["fn"]),
            "target_tn": int(target_row["tn"]),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train loss {row['train_loss']:.4f} | "
            f"val auroc {row['val_auroc']:.4f} | recall@{args.target_fpr:.2f} {row['target_recall']:.4f}",
            flush=True,
        )
        if row["target_recall"] > best_recall:
            best_recall = float(row["target_recall"])
            best_report = score_summary
            torch.save(
                {
                    "model_state_dict": model_state_for_save(model),
                    "args": vars(args),
                    "architecture": str(args.architecture),
                    "normalization": normalization,
                    "data_summary": data_summary,
                    "epoch": int(epoch),
                    "score_summary": score_summary,
                },
                best_path,
            )
            if args.save_scores:
                np.savez_compressed(
                    run_dir / "best_val_scores.npz",
                    scores=val_scores.astype(np.float32),
                    labels=val_labels.astype(np.uint8),
                )
        row["best_recall_so_far"] = float(best_recall)
        row["updated_utc"] = utc_now()
        append_jsonl(live_metrics_path, row)
        write_history_csv(history_csv_path, history)
        write_json(
            status_path,
            {
                "state": "running",
                "pid": int(os.getpid()),
                "created_utc": None,
                "updated_utc": utc_now(),
                "experiment_name": str(args.experiment_name),
                "architecture": str(args.architecture),
                "device": str(device),
                "gpu_ids": str(args.gpu_ids),
                "data_parallel": bool(isinstance(model, nn.DataParallel)),
                "amp": bool(use_amp),
                "resumed_from_epoch": int(resumed_from_epoch),
                "completed_epoch": int(epoch),
                "active_epoch": None,
                "current_epoch": int(epoch),
                "final_epoch": int(resumed_from_epoch + int(args.epochs)),
                "last_epoch_metrics": row,
                "best_recall": float(best_recall),
                "best_checkpoint": str(best_path),
                "progress_json": str(progress_path),
                "progress_jsonl": str(progress_jsonl_path),
                "live_metrics_jsonl": str(live_metrics_path),
                "history_csv": str(history_csv_path),
            },
        )

    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "experiment_name": str(args.experiment_name),
        "run_dir": str(run_dir),
        "args": vars(args),
        "device": str(device),
        "data_summary": data_summary,
        "normalization": normalization,
        "architecture": str(args.architecture),
        "best_checkpoint": str(best_path),
        "best_score_summary": best_report,
        "history": history,
        "assumption_notes": [
            "The detector score is neural-only and uses raw patch pixels as input.",
            "No classical score or response map is used as a deployed input.",
            "This report is a screening metric report, not a Bayesian evidence calculation.",
        ],
    }
    write_json(run_dir / "report.json", report)
    write_history_csv(history_csv_path, history)
    write_json(
        status_path,
        {
            "state": "completed",
            "pid": int(os.getpid()),
            "updated_utc": utc_now(),
            "experiment_name": str(args.experiment_name),
            "architecture": str(args.architecture),
            "device": str(device),
            "gpu_ids": str(args.gpu_ids),
            "data_parallel": bool(isinstance(model, nn.DataParallel)),
            "amp": bool(use_amp),
            "resumed_from_epoch": int(resumed_from_epoch),
            "completed_epoch": int(resumed_from_epoch + int(args.epochs)),
            "active_epoch": None,
            "current_epoch": int(resumed_from_epoch + int(args.epochs)),
            "final_epoch": int(resumed_from_epoch + int(args.epochs)),
            "last_epoch_metrics": history[-1] if history else None,
            "best_recall": float(best_recall),
            "best_checkpoint": str(best_path),
            "report": str(run_dir / "report.json"),
            "progress_json": str(progress_path),
            "progress_jsonl": str(progress_jsonl_path),
            "live_metrics_jsonl": str(live_metrics_path),
            "history_csv": str(history_csv_path),
        },
    )
    print(json.dumps({"report": str(run_dir / "report.json"), "best_checkpoint": str(best_path)}, indent=2))


if __name__ == "__main__":
    main()
