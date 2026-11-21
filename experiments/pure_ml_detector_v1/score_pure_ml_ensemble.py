"""Score a neural-only ensemble for the pure-ML detector track.

Assumptions
-----------
* Every checkpoint is produced by ``train_pure_ml_detector.py`` and consumes raw
  CMB patches only.
* Ensemble aggregation is over neural model outputs only. No Wiener, SMHW,
  circular-template, or other mathematical score is used.
* The default aggregation is mean logit, which avoids probability saturation
  while keeping a monotonic neural score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_pure_ml_detector import DEFAULT_PATCH_H5, load_checkpoint, load_labels, resolve_device, score_h5  # noqa: E402
from train_pure_ml_detector import build_detector_model, summarize_scores, threshold_for_fpr  # noqa: E402
from score_pure_ml_detector import cell_recall_rows, write_json, write_rows_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Score a pure-neural checkpoint ensemble on an HDF5 patch set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint path. Repeat for ensemble.")
    parser.add_argument("--patch-h5", type=str, default=str(DEFAULT_PATCH_H5))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--aggregation", type=str, default="mean_logit", choices=("mean_logit", "mean_prob", "max_prob"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--gpu-ids", type=str, default="", help="Comma-separated CUDA ids for DataParallel scoring.")
    parser.add_argument("--cache-patches", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate inputs."""

    if not args.checkpoint:
        raise ValueError("At least one --checkpoint is required.")
    for ckpt in args.checkpoint:
        if not Path(ckpt).expanduser().exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    if not Path(args.patch_h5).expanduser().exists():
        raise FileNotFoundError(f"Missing patch HDF5: {args.patch_h5}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if not (0.0 < args.target_fpr < 1.0):
        raise ValueError("--target-fpr must be in (0, 1).")


def probability_to_logit(prob: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits with clipping."""

    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def load_checkpoint_model(
    path: Path,
    device: torch.device,
    gpu_ids: str,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Load one pure detector checkpoint."""

    checkpoint = load_checkpoint(path, device)
    args = checkpoint["args"]
    model = build_detector_model(args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    from train_pure_ml_detector import wrap_data_parallel  # noqa: WPS433

    model = wrap_data_parallel(model, device, str(gpu_ids))
    return model, args, checkpoint["normalization"]


def aggregate_scores(score_matrix: np.ndarray, mode: str) -> np.ndarray:
    """Aggregate per-checkpoint probability scores."""

    if score_matrix.ndim != 2:
        raise ValueError(f"Expected score matrix with shape (models, rows), got {score_matrix.shape}.")
    if mode == "mean_prob":
        return np.mean(score_matrix, axis=0)
    if mode == "max_prob":
        return np.max(score_matrix, axis=0)
    if mode == "mean_logit":
        return 1.0 / (1.0 + np.exp(-np.mean(probability_to_logit(score_matrix), axis=0)))
    raise ValueError(f"Unknown aggregation: {mode}")


def main() -> None:
    """Score the neural ensemble."""

    args = parse_args()
    validate_args(args)
    device = resolve_device(str(args.device))
    patch_h5 = Path(args.patch_h5).expanduser().resolve()
    checkpoint_paths = [Path(path).expanduser().resolve() for path in args.checkpoint]
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else checkpoint_paths[0].parent / f"ensemble_{len(checkpoint_paths)}_{patch_h5.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = []
    scores = []
    for path in checkpoint_paths:
        model, ckpt_args, normalization = load_checkpoint_model(path, device, str(args.gpu_ids))
        ckpt_scores = score_h5(model, patch_h5, normalization, args, device)
        scores.append(ckpt_scores.astype(np.float64))
        checkpoint_rows.append(
            {
                "checkpoint": str(path),
                "architecture": str(ckpt_args.get("architecture", "small_cnn")),
                "seed": int(ckpt_args.get("seed", -1)),
                "experiment_name": str(ckpt_args.get("experiment_name", "")),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    score_matrix = np.stack(scores, axis=0)
    labels = load_labels(patch_h5)
    ensemble_scores = aggregate_scores(score_matrix, str(args.aggregation))
    summary = summarize_scores(ensemble_scores, labels)
    threshold, realized_fpr = threshold_for_fpr(ensemble_scores, labels, float(args.target_fpr))
    rows = cell_recall_rows(patch_h5, ensemble_scores, labels, threshold, realized_fpr)
    mean_cell_recall = float(np.mean([row["p_det"] for row in rows])) if rows else float("nan")

    np.savez_compressed(
        output_dir / "ensemble_scores.npz",
        scores=ensemble_scores.astype(np.float32),
        labels=labels.astype(np.uint8),
        score_matrix=score_matrix.astype(np.float32),
    )
    write_rows_csv(output_dir / "cell_recall_rows.csv", rows)
    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "patch_h5": str(patch_h5),
        "aggregation": str(args.aggregation),
        "checkpoints": checkpoint_rows,
        "num_models": int(score_matrix.shape[0]),
        "num_rows": int(score_matrix.shape[1]),
        "target_fpr": float(args.target_fpr),
        "target_threshold": float(threshold),
        "target_realized_fpr": float(realized_fpr),
        "mean_cell_recall": mean_cell_recall,
        "score_summary": summary,
        "cell_recall_rows": rows,
        "assumption_notes": [
            "Every component score is neural-only from raw patch pixels.",
            "Aggregation uses only neural outputs, not classical scores.",
            "Threshold uses the same split-conformal null quantile and score > threshold rule as the classical same-grid reports.",
            "This is a screening benchmark, not Bayesian evidence.",
        ],
    }
    write_json(output_dir / "ensemble_report.json", report)
    print(json.dumps({"report": str(output_dir / "ensemble_report.json"), "mean_cell_recall": mean_cell_recall}, indent=2))


if __name__ == "__main__":
    main()
