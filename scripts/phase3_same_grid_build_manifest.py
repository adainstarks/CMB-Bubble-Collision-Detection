"""Build a reproducible row manifest for same-grid full-sky benchmarking.

Assumptions
-----------
* The source HDF5 is the remediated sensitivity grid or another HDF5 with the
  same ``labels``, ``truth``, and ``stratification`` layout.
* A benchmark manifest defines a fixed set of source rows shared by every
  method. This is the object that should be cited when claiming "same-grid"
  comparison, not an ad-hoc terminal command.
* Positive sampling is stratified by `(amplitude, theta_crit)` because that is
  the current paper-facing cell grid. Negatives are drawn from the shared null
  pool without cell structure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_H5 = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_data.h5"
)
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_same_grid_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stratified same-grid row manifest and shard plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-h5", type=str, default=str(DEFAULT_SOURCE_H5))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument(
        "--positive-per-cell",
        type=int,
        default=200,
        help="0 keeps all positives from each amplitude/theta cell.",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=5000,
        help="0 keeps all negatives.",
    )
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260421)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.positive_per_cell < 0:
        raise ValueError("--positive-per-cell must be non-negative.")
    if args.num_negatives < 0:
        raise ValueError("--num-negatives must be non-negative.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")


def read_source(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        truth = h5["truth"]
        amplitude = np.asarray(truth["amplitude"][:], dtype=np.float64)
        theta = np.asarray(truth["theta_crit_deg"][:], dtype=np.float64)
        summary = dict(h5["summary"].attrs)
    return {
        "labels": labels,
        "amplitude": amplitude,
        "theta_crit_deg": theta,
        "summary": summary,
    }


def sample_manifest_rows(payload: dict, *, positive_per_cell: int, num_negatives: int, seed: int) -> tuple[np.ndarray, dict]:
    labels = payload["labels"]
    amplitude = payload["amplitude"]
    theta = payload["theta_crit_deg"]
    rng = np.random.default_rng(int(seed))

    positive_rows = []
    cell_counts = []
    pos_mask = labels == 1
    amplitude_values = sorted(float(value) for value in np.unique(amplitude[pos_mask]))
    theta_values = sorted(float(value) for value in np.unique(theta[pos_mask]))
    for amp in amplitude_values:
        for theta_deg in theta_values:
            cell_rows = np.flatnonzero(
                pos_mask & np.isclose(amplitude, amp, rtol=0.0, atol=0.0) & np.isclose(theta, theta_deg, rtol=0.0, atol=0.0)
            ).astype(np.int64)
            if cell_rows.size == 0:
                continue
            if positive_per_cell and cell_rows.size > positive_per_cell:
                chosen = np.sort(rng.choice(cell_rows, size=int(positive_per_cell), replace=False))
            else:
                chosen = np.sort(cell_rows)
            positive_rows.append(chosen)
            cell_counts.append(
                {
                    "amplitude": float(amp),
                    "theta_crit_deg": float(theta_deg),
                    "available_positive": int(cell_rows.size),
                    "selected_positive": int(chosen.size),
                }
            )

    positive_rows = np.concatenate(positive_rows) if positive_rows else np.zeros(0, dtype=np.int64)

    negative_rows = np.flatnonzero(labels == 0).astype(np.int64)
    if num_negatives and negative_rows.size > num_negatives:
        negative_rows = np.sort(rng.choice(negative_rows, size=int(num_negatives), replace=False))
    else:
        negative_rows = np.sort(negative_rows)

    selected_rows = np.concatenate([positive_rows, negative_rows]).astype(np.int64, copy=False)
    selected_rows = np.sort(np.unique(selected_rows))
    return selected_rows, {
        "cell_counts": cell_counts,
        "selected_positive": int(positive_rows.size),
        "selected_negative": int(negative_rows.size),
    }


def shard_rows(rows: np.ndarray, num_shards: int) -> list[dict]:
    shards = np.array_split(np.asarray(rows, dtype=np.int64), int(num_shards))
    payload = []
    for shard_id, shard_rows_arr in enumerate(shards):
        payload.append(
            {
                "shard_id": int(shard_id),
                "rows": [int(value) for value in shard_rows_arr.tolist()],
                "num_rows": int(shard_rows_arr.size),
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    validate_args(args)
    source_h5 = Path(args.source_h5).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = read_source(source_h5)
    rows, stats = sample_manifest_rows(
        payload,
        positive_per_cell=int(args.positive_per_cell),
        num_negatives=int(args.num_negatives),
        seed=int(args.seed),
    )
    manifest = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source_h5": str(source_h5),
        "seed": int(args.seed),
        "positive_per_cell": int(args.positive_per_cell),
        "num_negatives": int(args.num_negatives),
        "num_rows": int(rows.size),
        "num_positive": int(stats["selected_positive"]),
        "num_negative": int(stats["selected_negative"]),
        "num_shards": int(args.num_shards),
        "rows": [int(value) for value in rows.tolist()],
        "shards": shard_rows(rows, int(args.num_shards)),
        "cell_counts": stats["cell_counts"],
        "source_summary": {
            "fpr_target": float(payload["summary"].get("fpr_target", 0.05)),
            "beam_fwhm_arcmin": float(payload["summary"].get("beam_fwhm_arcmin", 0.0)),
            "noise_sigma_uk_arcmin": float(payload["summary"].get("noise_sigma_uk_arcmin", 0.0)),
            "nside": int(payload["summary"].get("nside", 0)),
        },
    }
    output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output_json), "num_rows": int(rows.size)}, indent=2))


if __name__ == "__main__":
    main()
