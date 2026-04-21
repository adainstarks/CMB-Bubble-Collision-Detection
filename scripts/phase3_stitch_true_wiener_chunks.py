"""Stitch standalone true-Wiener chunk files back into a source HDF5.

Assumptions
-----------
* Each chunk file was produced by `phase3_cache_true_wiener_chunk.py` and
  contains explicit `rows` and `patches` datasets.
* Stitching is a single-writer operation performed after chunk generation
  completes; this script owns the target HDF5 lock while it runs.
* The target dataset is an auxiliary response channel, not a temperature map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "training_data.h5"
DEFAULT_DATASET = "features/wiener_feeney_response"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch true-Wiener chunk HDF5s back into a source HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-h5", type=str, default=str(DEFAULT_SOURCE_H5))
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--chunk-h5", action="append", default=[], help="Chunk HDF5. Can be repeated.")
    parser.add_argument("--chunk-dir", type=str, default="", help="Directory of chunk HDF5 files.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_chunk_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in args.chunk_h5]
    if args.chunk_dir:
        paths.extend(sorted(Path(args.chunk_dir).expanduser().resolve().glob("*.h5")))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise ValueError("Provide at least one --chunk-h5 or a --chunk-dir.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing chunk files: {missing}")
    return paths


def ensure_parent_group(h5: h5py.File, dataset_path: str) -> None:
    parent = str(Path(dataset_path).parent).strip(".")
    if parent and parent != "/":
        h5.require_group(parent)


def main() -> None:
    args = parse_args()
    source_h5 = Path(args.source_h5).expanduser().resolve()
    chunk_paths = collect_chunk_paths(args)
    dataset_path = str(args.dataset).strip("/")

    with h5py.File(source_h5, "r+") as h5:
        num_rows = int(h5["labels"].shape[0])
        shape = (num_rows, *h5["patches"].shape[1:])
        if dataset_path in h5 and args.overwrite:
            del h5[dataset_path]
        if dataset_path not in h5:
            ensure_parent_group(h5, dataset_path)
            out = h5.create_dataset(
                dataset_path,
                shape=shape,
                dtype=np.float32,
                fillvalue=np.nan,
                chunks=(1, shape[1], shape[2]),
                compression="lzf",
                shuffle=True,
            )
        else:
            out = h5[dataset_path]
            if tuple(out.shape) != tuple(shape):
                raise RuntimeError(f"Existing dataset {dataset_path} has shape {out.shape}; expected {shape}.")

        total_rows = 0
        for chunk_path in chunk_paths:
            with h5py.File(chunk_path, "r") as chunk:
                rows = np.asarray(chunk["rows"][:], dtype=np.int64)
                patches = np.asarray(chunk["patches"][:], dtype=np.float32)
            if rows.ndim != 1 or patches.ndim != 3 or patches.shape[0] != rows.size:
                raise RuntimeError(f"Invalid chunk layout in {chunk_path}.")
            out[rows] = patches
            total_rows += int(rows.size)
            print(f"  stitched {rows.size} rows from {chunk_path.name}", flush=True)
        h5.flush()

    print(
        json.dumps(
            {
                "source_h5": str(source_h5),
                "dataset": dataset_path,
                "chunk_files": [str(path) for path in chunk_paths],
                "stitched_rows": total_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
