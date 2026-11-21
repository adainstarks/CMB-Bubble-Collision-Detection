"""Audit complementarity between two pure-ML detector score files.

Assumptions
-----------
* Both score files evaluate the same HDF5 rows in the same order.
* Scores are neural-only raw-patch detector outputs.
* Thresholds use the same split-conformal null quantile and ``score >
  threshold`` rule as the classical same-grid reports.
* This is a diagnostic for model diversity, not a Bayesian evidence
  calculation or deployment tile-burden audit.
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_pure_ml_detector import threshold_for_fpr  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATCH_H5 = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_data.h5"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit overlap and unique detections for two pure-ML score files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--score-a", required=True, type=str)
    parser.add_argument("--name-a", default="model_a", type=str)
    parser.add_argument("--score-b", required=True, type=str)
    parser.add_argument("--name-b", default="model_b", type=str)
    parser.add_argument("--patch-h5", default=str(DEFAULT_PATCH_H5), type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--target-fpr", default=0.05, type=float)
    return parser.parse_args()


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load scores and labels from an ``scores.npz``-style file."""

    if not path.exists():
        raise FileNotFoundError(f"Missing score file: {path}")
    with np.load(path) as data:
        if "scores" not in data or "labels" not in data:
            raise KeyError(f"Score file must contain scores and labels: {path}")
        scores = np.asarray(data["scores"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.uint8)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError(f"Invalid score/label shape in {path}: {scores.shape}, {labels.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite scores in {path}")
    return scores, labels


def load_truth(path: Path, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Load amplitude and radius metadata."""

    with h5py.File(path, "r") as h5:
        if "truth/amplitude" not in h5 or "truth/theta_crit_deg" not in h5:
            raise KeyError(f"Missing truth/amplitude or truth/theta_crit_deg in {path}")
        amplitude = np.asarray(h5["truth/amplitude"][:], dtype=np.float64)
        theta = np.asarray(h5["truth/theta_crit_deg"][:], dtype=np.float64)
    if amplitude.shape[0] != n_rows or theta.shape[0] != n_rows:
        raise ValueError("Truth arrays do not match score length.")
    return amplitude, theta


def count(mask: np.ndarray) -> int:
    """Return the number of true elements."""

    return int(np.count_nonzero(mask))


def summarize_partition(active_a: np.ndarray, active_b: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Summarize overlap/unique partitions for positives and negatives."""

    pos = labels == 1
    neg = labels == 0
    both = active_a & active_b
    a_only = active_a & ~active_b
    b_only = active_b & ~active_a
    union = active_a | active_b
    return {
        "positive": {
            "total": count(pos),
            "a_detected": count(active_a & pos),
            "b_detected": count(active_b & pos),
            "both_detected": count(both & pos),
            "a_only_detected": count(a_only & pos),
            "b_only_detected": count(b_only & pos),
            "union_detected": count(union & pos),
            "a_recall": float(np.mean(active_a[pos])),
            "b_recall": float(np.mean(active_b[pos])),
            "both_recall": float(np.mean(both[pos])),
            "a_only_recall": float(np.mean(a_only[pos])),
            "b_only_recall": float(np.mean(b_only[pos])),
            "union_recall": float(np.mean(union[pos])),
            "jaccard_detected": float(count(both & pos) / max(count(union & pos), 1)),
        },
        "negative": {
            "total": count(neg),
            "a_false_positive": count(active_a & neg),
            "b_false_positive": count(active_b & neg),
            "both_false_positive": count(both & neg),
            "a_only_false_positive": count(a_only & neg),
            "b_only_false_positive": count(b_only & neg),
            "union_false_positive": count(union & neg),
            "a_fpr": float(np.mean(active_a[neg])),
            "b_fpr": float(np.mean(active_b[neg])),
            "union_fpr": float(np.mean(union[neg])),
            "jaccard_false_positive": float(count(both & neg) / max(count(union & neg), 1)),
        },
    }


def cell_rows(
    active_a: np.ndarray,
    active_b: np.ndarray,
    labels: np.ndarray,
    amplitude: np.ndarray,
    theta: np.ndarray,
    name_a: str,
    name_b: str,
) -> list[dict[str, Any]]:
    """Return cell-level overlap rows for positive injections."""

    pos = labels == 1
    rows: list[dict[str, Any]] = []
    for amp in sorted(float(value) for value in np.unique(amplitude[pos])):
        for theta_deg in sorted(float(value) for value in np.unique(theta[pos])):
            mask = pos & np.isclose(amplitude, amp) & np.isclose(theta, theta_deg)
            n = count(mask)
            if n == 0:
                continue
            both = active_a & active_b & mask
            a_only = active_a & ~active_b & mask
            b_only = active_b & ~active_a & mask
            union = (active_a | active_b) & mask
            rows.append(
                {
                    "amplitude": float(amp),
                    "theta_crit_deg": float(theta_deg),
                    "num_positive": n,
                    f"{name_a}_detected": count(active_a & mask),
                    f"{name_b}_detected": count(active_b & mask),
                    "both_detected": count(both),
                    f"{name_a}_only_detected": count(a_only),
                    f"{name_b}_only_detected": count(b_only),
                    "union_detected": count(union),
                    f"{name_a}_recall": float(np.mean(active_a[mask])),
                    f"{name_b}_recall": float(np.mean(active_b[mask])),
                    "both_recall": float(np.mean((active_a & active_b)[mask])),
                    f"{name_a}_only_recall": float(np.mean((active_a & ~active_b)[mask])),
                    f"{name_b}_only_recall": float(np.mean((active_b & ~active_a)[mask])),
                    "union_recall": float(np.mean((active_a | active_b)[mask])),
                }
            )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV file."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the complementarity audit."""

    args = parse_args()
    if not (0.0 < float(args.target_fpr) < 1.0):
        raise ValueError("--target-fpr must be in (0, 1).")
    score_a, labels_a = load_scores(Path(args.score_a).expanduser().resolve())
    score_b, labels_b = load_scores(Path(args.score_b).expanduser().resolve())
    if not np.array_equal(labels_a, labels_b):
        raise ValueError("Score files do not contain identical labels in the same order.")
    amplitude, theta = load_truth(Path(args.patch_h5).expanduser().resolve(), int(labels_a.shape[0]))

    threshold_a, fpr_a = threshold_for_fpr(score_a, labels_a, float(args.target_fpr))
    threshold_b, fpr_b = threshold_for_fpr(score_b, labels_b, float(args.target_fpr))
    active_a = score_a > threshold_a
    active_b = score_b > threshold_b

    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = cell_rows(active_a, active_b, labels_a, amplitude, theta, str(args.name_a), str(args.name_b))
    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "score_a": str(Path(args.score_a).expanduser().resolve()),
        "name_a": str(args.name_a),
        "threshold_a": float(threshold_a),
        "realized_fpr_a": float(fpr_a),
        "score_b": str(Path(args.score_b).expanduser().resolve()),
        "name_b": str(args.name_b),
        "threshold_b": float(threshold_b),
        "realized_fpr_b": float(fpr_b),
        "patch_h5": str(Path(args.patch_h5).expanduser().resolve()),
        "target_fpr": float(args.target_fpr),
        "overall": summarize_partition(active_a, active_b, labels_a),
        "cell_rows": rows,
        "assumption_notes": [
            "Both model thresholds use score > threshold.",
            "Union recall is diagnostic only because union FPR can exceed the target.",
            "This audit measures score complementarity, not deployment tile burden.",
        ],
    }
    write_json(output_dir / "complementarity_report.json", report)
    write_csv(output_dir / "complementarity_cells.csv", rows)
    print(
        json.dumps(
            {
                "report": str(output_dir / "complementarity_report.json"),
                "a_recall": report["overall"]["positive"]["a_recall"],
                "b_recall": report["overall"]["positive"]["b_recall"],
                "b_only_recall": report["overall"]["positive"]["b_only_recall"],
                "union_recall": report["overall"]["positive"]["union_recall"],
                "union_fpr": report["overall"]["negative"]["union_fpr"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
