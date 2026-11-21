"""Score an HDF5 patch set with an isolated pure-ML detector checkpoint.

Assumptions
-----------
* The checkpoint was trained by ``train_pure_ml_detector.py`` and uses raw CMB
  patch pixels as its only deployed input.
* This script calibrates a score threshold on the negative rows in the supplied
  HDF5 file. For publication-grade claims, the calibration and test negatives
  must be split externally; this script is a benchmark/scoring utility.
* When amplitude/radius truth is present, mean recall is the arithmetic mean over
  amplitude-radius cells, matching the same-grid screening summaries.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    torch = None
    DataLoader = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_pure_ml_detector import build_detector_model, summarize_scores, threshold_for_fpr, wrap_data_parallel  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATCH_H5 = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_same_grid_fullsky_manifest"
    / "same_grid_fullsky_patches.h5"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Score a patch HDF5 with a pure-ML detector checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--patch-h5", type=str, default=str(DEFAULT_PATCH_H5))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--cache-patches",
        action="store_true",
        help="Load all patches into RAM before scoring. Recommended for the 12k-row same-grid benchmark.",
    )
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated CUDA ids for DataParallel scoring, e.g. 0,1.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate inputs."""

    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(f"PyTorch import failed: {TORCH_IMPORT_ERROR}")
    if not Path(args.checkpoint).expanduser().exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not Path(args.patch_h5).expanduser().exists():
        raise FileNotFoundError(f"Missing patch HDF5: {args.patch_h5}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if not (0.0 < args.target_fpr < 1.0):
        raise ValueError("--target-fpr must be in (0, 1).")


def resolve_device(device_arg: str) -> torch.device:
    """Resolve torch device."""

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class H5PatchDataset(Dataset):
    """Read raw patches from one HDF5 file."""

    def __init__(self, h5_path: Path, normalization: dict[str, float], *, cache_patches: bool):
        self.h5_path = str(h5_path)
        self.mean = float(normalization["mean"])
        self.std = max(float(normalization["std"]), 1e-12)
        self._h5 = None
        self._patches = None
        with h5py.File(self.h5_path, "r") as h5:
            self.n = int(h5["patches"].shape[0])
            if bool(cache_patches):
                self._patches = np.asarray(h5["patches"][:], dtype=np.float32)
        if self._patches is not None:
            if not np.all(np.isfinite(self._patches)):
                raise ValueError("Cached patches contain non-finite values.")
            if float(np.max(np.abs(self._patches))) > 1.0:
                raise ValueError("Cached patches are outside Kelvin sanity range.")

    def __len__(self) -> int:
        return self.n

    def _handle(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, item: int) -> dict[str, Any]:
        if self._patches is None:
            patch = np.asarray(self._handle()["patches"][int(item)], dtype=np.float32)
        else:
            patch = np.asarray(self._patches[int(item)], dtype=np.float32)
        if patch.shape != (256, 256):
            raise ValueError(f"Expected 256x256 patch, got {patch.shape}.")
        if not np.all(np.isfinite(patch)):
            raise ValueError(f"Patch {item} contains non-finite values.")
        if float(np.max(np.abs(patch))) > 1.0:
            raise ValueError(f"Patch {item} is outside Kelvin sanity range.")
        image = ((patch - self.mean) / self.std).astype(np.float32, copy=False)[None, :, :]
        return {"image": torch.from_numpy(image), "index": int(item)}

    def __del__(self) -> None:  # pragma: no cover
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


@torch.no_grad()
def score_h5(model: torch.nn.Module, h5_path: Path, normalization: dict[str, float], args: argparse.Namespace, device: torch.device) -> np.ndarray:
    """Return neural scores for every row in an HDF5 file."""

    dataset = H5PatchDataset(h5_path, normalization, cache_patches=bool(args.cache_patches))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    scores = np.empty(len(dataset), dtype=np.float32)
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        idx = batch["index"].detach().cpu().numpy().astype(np.int64)
        logits = model(images)
        scores[idx] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model produced non-finite scores.")
    return scores


def load_labels(path: Path) -> np.ndarray:
    """Load binary labels from HDF5."""

    with h5py.File(path, "r") as h5:
        return np.asarray(h5["labels"][:], dtype=np.uint8)


def cell_recall_rows(path: Path, scores: np.ndarray, labels: np.ndarray, threshold: float, realized_fpr: float) -> list[dict[str, Any]]:
    """Build amplitude-radius recall rows when truth metadata is available."""

    with h5py.File(path, "r") as h5:
        if "truth/amplitude" not in h5 or "truth/theta_crit_deg" not in h5:
            return []
        amplitude = np.asarray(h5["truth/amplitude"][:], dtype=np.float64)
        theta = np.asarray(h5["truth/theta_crit_deg"][:], dtype=np.float64)

    positive = labels == 1
    rows = []
    for amp in sorted(float(value) for value in np.unique(amplitude[positive])):
        for theta_deg in sorted(float(value) for value in np.unique(theta[positive])):
            mask = positive & np.isclose(amplitude, amp) & np.isclose(theta, theta_deg)
            n = int(np.count_nonzero(mask))
            if n == 0:
                continue
            detected = int(np.count_nonzero(scores[mask] > float(threshold)))
            rows.append(
                {
                    "method": "pure_ml_detector",
                    "amplitude": float(amp),
                    "theta_crit_deg": float(theta_deg),
                    "num_positive": n,
                    "detected": detected,
                    "p_det": float(detected / max(n, 1)),
                    "threshold": float(threshold),
                    "negative_fpr": float(realized_fpr),
                }
            )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write recall rows."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a local detector checkpoint without relying on PyTorch defaults."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Older PyTorch versions do not expose weights_only.
        return torch.load(path, map_location=device)


def main() -> None:
    """Load checkpoint, score patches, and write benchmark report."""

    args = parse_args()
    validate_args(args)
    device = resolve_device(str(args.device))
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    patch_h5 = Path(args.patch_h5).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else checkpoint_path.parent / f"score_{patch_h5.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path, device)
    ckpt_args = checkpoint["args"]
    normalization = checkpoint["normalization"]
    model = build_detector_model(ckpt_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = wrap_data_parallel(model, device, str(args.gpu_ids))

    scores = score_h5(model, patch_h5, normalization, args, device)
    labels = load_labels(patch_h5)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("Score and label arrays have different lengths.")

    score_summary = summarize_scores(scores, labels)
    threshold, realized_fpr = threshold_for_fpr(scores, labels, float(args.target_fpr))
    rows = cell_recall_rows(patch_h5, scores, labels, threshold, realized_fpr)
    mean_cell_recall = float(np.mean([row["p_det"] for row in rows])) if rows else float("nan")

    np.savez_compressed(output_dir / "scores.npz", scores=scores.astype(np.float32), labels=labels.astype(np.uint8))
    write_rows_csv(output_dir / "cell_recall_rows.csv", rows)
    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "checkpoint": str(checkpoint_path),
        "patch_h5": str(patch_h5),
        "num_rows": int(scores.shape[0]),
        "target_fpr": float(args.target_fpr),
        "target_threshold": float(threshold),
        "target_realized_fpr": float(realized_fpr),
        "mean_cell_recall": mean_cell_recall,
        "score_summary": score_summary,
        "cell_recall_rows": rows,
        "assumption_notes": [
            "Scores are neural-only raw-patch detector outputs.",
            "Threshold uses the same split-conformal null quantile and score > threshold rule as the classical same-grid reports.",
            "This is a screening benchmark, not a Bayesian evidence calculation.",
        ],
    }
    write_json(output_dir / "score_report.json", report)
    print(json.dumps({"report": str(output_dir / "score_report.json"), "mean_cell_recall": mean_cell_recall}, indent=2))


if __name__ == "__main__":
    main()
