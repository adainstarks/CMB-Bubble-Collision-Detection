"""
Lightweight Phase 3 evaluation for architecture pilot studies.

This utility reuses the saved run configuration, split indices, and checkpoint
loading logic from the main evaluation pipeline, but skips artifact generation
and output audits. The intent is to obtain directly comparable threshold-sweep
metrics for controlled model ablations without paying the full evaluation cost.

The operating-point logic matches ``phase3_evaluate_run.py`` so recall and FPR
remain comparable to the production screening policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import phase3_evaluate_run as ev
import phase3_train_unet as p3
from phase_config import min_component_area_pixels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a lightweight threshold sweep for a saved Phase 3 run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best", help="`best`, `last`, or a checkpoint path.")
    parser.add_argument("--split", type=str, default="calibration", choices=["train", "calibration", "val", "test"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-count", type=int, default=25)
    parser.add_argument(
        "--image-min-positive-pixels",
        type=int,
        default=min_component_area_pixels(theta_min_deg=5.0, fraction=0.01),
    )
    parser.add_argument(
        "--operating-point-rule",
        type=str,
        default="fpr_cap",
        choices=ev.OPERATING_POINT_RULES,
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="image_f1",
        choices=ev.SELECTION_METRICS,
        help="Used only when --operating-point-rule=metric_max.",
    )
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument(
        "--image-rule",
        type=str,
        default="connected_component",
        choices=ev.IMAGE_RULES,
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="component_score",
        choices=ev.SCORE_MODES,
    )
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save the summary JSON.")
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.threshold_count <= 1:
        raise ValueError("--threshold-count must be greater than 1.")
    if not (0.0 < args.threshold_min < args.threshold_max < 1.0):
        raise ValueError("Threshold range must satisfy 0 < min < max < 1.")
    if args.image_min_positive_pixels < 1:
        raise ValueError("--image-min-positive-pixels must be at least 1.")
    if not (0.0 <= args.target_fpr < 1.0):
        raise ValueError("--target-fpr must be in [0, 1).")


def build_loader(run_config, split_indices, batch_size, num_workers, device):
    train_args = run_config["args"]
    dataset = p3.H5BubbleDataset(
        h5_path=Path(run_config["data_h5"]),
        indices=split_indices,
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(train_args["seed"]) + 1,
        max_translate_pixels=0,
    )
    pin_memory = device.type == "cuda"
    loader = p3.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=p3.seed_worker if num_workers > 0 else None,
        generator=torch.Generator().manual_seed(0),
    )
    return dataset, loader


def main():
    args = parse_args()
    validate_args(args)
    p3.require_ml_packages()
    p3.seed_everything(42)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_config = ev.load_json(run_dir / "run_config.json")
    split_indices = ev.load_split_indices(run_dir, args.split)
    checkpoint_path, checkpoint_label = ev.resolve_checkpoint_path(run_dir, args.checkpoint)

    device = p3.resolve_device(args.device)
    dataset, loader = build_loader(
        run_config=run_config,
        split_indices=split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
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
        raise RuntimeError(f"--score-mode {args.score_mode} requires --aux-head-weight > 0.")

    model = p3.build_model(model_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])

    thresholds = ev.canonical_threshold_grid(
        np.linspace(args.threshold_min, args.threshold_max, args.threshold_count, dtype=np.float64)
    )
    metrics_per_threshold, _ = ev.evaluate_thresholds(
        model=model,
        loader=loader,
        thresholds=thresholds,
        device=device,
        preview_count=1,
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

    operating_row, operating_point = ev.choose_operating_point(
        rows=rows,
        operating_point_rule=args.operating_point_rule,
        selection_metric=args.selection_metric,
        target_fpr=args.target_fpr,
    )
    best_f1_row = ev.choose_best_threshold(rows, "image_f1")
    summary = {
        "run_dir": str(run_dir),
        "checkpoint_label": checkpoint_label,
        "split": args.split,
        "num_samples": int(len(dataset)),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "score_mode": args.score_mode,
        "image_rule": args.image_rule,
        "operating_point": operating_point,
        "selected_threshold": float(operating_row["threshold"]),
        "selected_threshold_metrics": operating_row,
        "best_image_f1_threshold": float(best_f1_row["threshold"]),
        "best_image_f1_metrics": best_f1_row,
        "min_image_fpr": float(min(row["image_false_positive_rate"] for row in rows)),
        "max_image_recall": float(max(row["image_recall"] for row in rows)),
        "max_image_f1": float(max(row["image_f1"] for row in rows)),
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
