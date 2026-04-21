"""
Score classical circular-template baselines on real null-control patches.

This applies the synthetic-split thresholds saved by phase3_template_baseline.py
to real cleaned-map null controls, producing a direct null-burden comparator for
the ML screeners.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from phase3_template_baseline import build_kernel_bank, search_best_candidate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply saved classical baseline thresholds to real-map null-control patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--classical-dir", type=str, required=True)
    parser.add_argument("--null-h5", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "calibration", "test"],
    )
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_null_indices(null_h5, split):
    with h5py.File(null_h5, "r") as h5:
        if split == "all":
            return np.arange(h5["patches"].shape[0], dtype=np.int64)
        split_keys = [f"{split}_idx"]
        if split == "calibration":
            split_keys.append("val_idx")
        if split == "val":
            split_keys.append("calibration_idx")
        for key in split_keys:
            if key in h5["splits"]:
                return np.asarray(h5["splits"][key][:], dtype=np.int64)
        raise KeyError(f"Null-control HDF5 missing any split key in {split_keys}.")


def score_method(patches, indices, kernels, threshold, centered_only):
    false_positive_count = 0
    score_sum = 0.0
    score_max = 0.0
    active_scores = []
    for offset, sample_idx in enumerate(indices):
        best, _ = search_best_candidate(patches[int(sample_idx)], kernels, centered_only=centered_only)
        score = float(best["score"])
        score_sum += score
        score_max = max(score_max, score)
        if score >= threshold:
            false_positive_count += 1
            active_scores.append(score)
        if (offset + 1) % 250 == 0 or offset + 1 == len(indices):
            print(f"  Scored {offset + 1:5d} / {len(indices)}", flush=True)
    return {
        "threshold": float(threshold),
        "num_samples": int(len(indices)),
        "false_positive_count": int(false_positive_count),
        "false_positive_rate": float(false_positive_count / max(len(indices), 1)),
        "score_mean": float(score_sum / max(len(indices), 1)),
        "score_max": float(score_max),
        "active_score_median": float(np.median(active_scores)) if active_scores else None,
        "active_score_p90": float(np.percentile(active_scores, 90)) if active_scores else None,
    }


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    classical_dir = Path(args.classical_dir).resolve()
    summary = load_json(classical_dir / "summary.json")
    indices = load_null_indices(args.null_h5, args.split)

    results = {
        "classical_dir": str(classical_dir),
        "null_h5": str(Path(args.null_h5).resolve()),
        "split": args.split,
        "num_samples": int(len(indices)),
        "methods": {},
    }
    with h5py.File(args.null_h5, "r") as h5:
        patches = h5["patches"]
        kernels = build_kernel_bank(patches[0].shape)
        for method_name, payload in sorted(summary["methods"].items()):
            print(f"Scoring {method_name}...", flush=True)
            result = score_method(
                patches=patches,
                indices=indices,
                kernels=kernels,
                threshold=float(payload["selected_threshold"]),
                centered_only=method_name == "centered_disc",
            )
            results["methods"][method_name] = result
            print(
                f"{method_name}: false positives {result['false_positive_count']} / {result['num_samples']} "
                f"({result['false_positive_rate']:.4f})"
            )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Classical null summary: {output_json}")


if __name__ == "__main__":
    main()
