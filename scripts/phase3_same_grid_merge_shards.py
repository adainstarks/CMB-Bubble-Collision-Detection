"""Merge same-grid shard outputs into one benchmark artifact.

Assumptions
-----------
* Every shard comes from the same manifest and the same source HDF5. Row order
  is defined by the manifest, not by filesystem globbing.
* Shard patch HDF5 files already contain the observed patches projected from
  the full-sky benchmark maps. This script only merges them; it does not
  regenerate spherical filters.
* If ML scores are requested here, they are evaluated on the merged patch HDF5
  against the exact same manifest rows used for the classical shard outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import h5py
import numpy as np

import phase3_train_unet as p3
from phase3_method_registry import method_metadata
from phase3_same_grid_fullsky_benchmark import (
    DEFAULT_SOURCE_H5,
    read_selected_rows,
    score_ml_models,
    summarize_scores,
    write_markdown,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_JSON = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_same_grid_manifest.json"
DEFAULT_SHARD_ROOT = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_same_grid_shards"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_same_grid_fullsky_manifest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge same-grid benchmark shards into one reportable artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest-json", type=str, default=str(DEFAULT_MANIFEST_JSON))
    parser.add_argument("--source-h5", type=str, default=str(DEFAULT_SOURCE_H5))
    parser.add_argument("--shard-root", type=str, default=str(DEFAULT_SHARD_ROOT))
    parser.add_argument("--shard-dir-template", type=str, default="shard_{shard_id:02d}")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--model", action="append", default=[], help="ML model as name:run_dir:checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-ml-data", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be non-negative.")


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in manifest or "shards" not in manifest:
        raise ValueError(f"Manifest {path} is missing `rows` or `shards`.")
    return manifest


def is_stratified_subset(manifest: dict) -> bool:
    for row in manifest.get("cell_counts", []):
        if int(row.get("selected_positive", 0)) < int(row.get("available_positive", 0)):
            return True
    return False


def shard_dir(shard_root: Path, template: str, shard_id: int) -> Path:
    return shard_root / template.format(shard_id=int(shard_id))


def copy_group_dataset(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    name: str,
    *,
    total_rows: int,
    row_offset: int,
    num_rows: int,
) -> None:
    src = src_group[name]
    if name not in dst_group:
        dst_group.create_dataset(
            name,
            shape=(int(total_rows), *src.shape[1:]),
            dtype=src.dtype,
            compression="gzip",
            shuffle=True,
        )
    dst_group[name][row_offset : row_offset + num_rows] = src[:]


def merge_patch_h5(manifest: dict, shard_root: Path, shard_template: str, output_h5: Path) -> None:
    rows = [int(value) for value in manifest["rows"]]
    total_rows = len(rows)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "w") as out_h5:
        row_offset = 0
        for shard in manifest["shards"]:
            shard_id = int(shard["shard_id"])
            expected_rows = np.asarray(shard["rows"], dtype=np.int64)
            shard_path = shard_dir(shard_root, shard_template, shard_id) / "same_grid_fullsky_patches.h5"
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard patch HDF5: {shard_path}")
            with h5py.File(shard_path, "r") as in_h5:
                num_rows = int(in_h5["patches"].shape[0])
                if num_rows != expected_rows.size:
                    raise RuntimeError(
                        f"Shard {shard_id} has {num_rows} rows but manifest expects {expected_rows.size}."
                    )
                source_rows = np.asarray(in_h5["metadata"]["source_row_index"][:], dtype=np.int64)
                if not np.array_equal(source_rows, expected_rows):
                    raise RuntimeError(f"Shard {shard_id} row order does not match manifest.")
                for name in ("patches", "labels", "masks"):
                    copy_group_dataset(
                        in_h5,
                        out_h5,
                        name,
                        total_rows=total_rows,
                        row_offset=row_offset,
                        num_rows=num_rows,
                    )
                for group_name in ("metadata", "truth", "stratification"):
                    src_group = in_h5[group_name]
                    dst_group = out_h5.require_group(group_name)
                    for name in src_group.keys():
                        copy_group_dataset(
                            src_group,
                            dst_group,
                            name,
                            total_rows=total_rows,
                            row_offset=row_offset,
                            num_rows=num_rows,
                        )
                row_offset += num_rows

        if row_offset != total_rows:
            raise RuntimeError(f"Merged {row_offset} rows but manifest defines {total_rows}.")

        summary_group = out_h5.require_group("summary")
        summary_group.attrs["created_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        summary_group.attrs["manifest_num_rows"] = int(total_rows)
        summary_group.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)


def merge_classical_scores(manifest: dict, shard_root: Path, shard_template: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    labels = np.zeros(len(manifest["rows"]), dtype=np.uint8)
    source_row_index = np.zeros(len(manifest["rows"]), dtype=np.int64)
    merged_scores: dict[str, np.ndarray] = {}
    row_offset = 0
    for shard in manifest["shards"]:
        shard_id = int(shard["shard_id"])
        expected_rows = np.asarray(shard["rows"], dtype=np.int64)
        shard_path = shard_dir(shard_root, shard_template, shard_id) / "same_grid_fullsky_scores.npz"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard score NPZ: {shard_path}")
        with np.load(shard_path) as data:
            shard_labels = np.asarray(data["labels"], dtype=np.uint8)
            shard_rows = np.asarray(data["source_row_index"], dtype=np.int64)
            if not np.array_equal(shard_rows, expected_rows):
                raise RuntimeError(f"Shard {shard_id} score row order does not match manifest.")
            n = shard_rows.size
            labels[row_offset : row_offset + n] = shard_labels
            source_row_index[row_offset : row_offset + n] = shard_rows
            score_keys = [key for key in data.files if key.startswith("score__")]
            for key in score_keys:
                if key not in merged_scores:
                    merged_scores[key] = np.zeros(len(manifest["rows"]), dtype=np.float32)
                merged_scores[key][row_offset : row_offset + n] = np.asarray(data[key], dtype=np.float32)
            row_offset += n
    if row_offset != len(manifest["rows"]):
        raise RuntimeError(f"Merged {row_offset} score rows but manifest defines {len(manifest['rows'])}.")
    return labels, source_row_index, merged_scores


def main() -> None:
    args = parse_args()
    validate_args(args)
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    source_h5 = Path(args.source_h5).expanduser().resolve()
    shard_root = Path(args.shard_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    merged_h5 = output_dir / "same_grid_fullsky_patches.h5"
    merge_patch_h5(manifest, shard_root, str(args.shard_dir_template), merged_h5)
    labels, source_row_index, merged_scores = merge_classical_scores(manifest, shard_root, str(args.shard_dir_template))

    rows = read_selected_rows(source_h5, np.asarray(manifest["rows"], dtype=np.int64))
    if not np.array_equal(rows.row_index, source_row_index):
        raise RuntimeError("Merged source_row_index does not match manifest row order.")
    ml_metadata = {"skipped": True}
    if not args.skip_ml:
        score_args = argparse.Namespace(
            skip_ml=False,
            model=args.model,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            cache_ml_data=bool(args.cache_ml_data),
            device=str(args.device),
        )
        ml_scores, ml_metadata = score_ml_models(h5_path=merged_h5, rows=rows, args=score_args)
        for name, values in ml_scores.items():
            merged_scores[f"score__{name}"] = np.asarray(values, dtype=np.float32)

    scores_npz = output_dir / "same_grid_fullsky_scores.npz"
    np.savez_compressed(scores_npz, labels=labels, source_row_index=source_row_index, **merged_scores)

    scores_by_method = {
        (key[7:] if key.startswith("score__") else key): value
        for key, value in merged_scores.items()
    }
    fpr_target = float(manifest.get("source_summary", {}).get("fpr_target", 0.05))
    recall_rows, thresholds = summarize_scores(scores_by_method, rows, fpr_target)

    benchmark_status = "stratified_same_grid_complete" if is_stratified_subset(manifest) else "full_same_grid_complete"
    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "benchmark_status": benchmark_status,
        "manifest_json": str(manifest_path),
        "source_h5": str(source_h5),
        "patch_h5": str(merged_h5),
        "scores_npz": str(scores_npz),
        "num_rows": int(len(manifest["rows"])),
        "num_positive": int(np.count_nonzero(labels == 1)),
        "num_negative": int(np.count_nonzero(labels == 0)),
        "num_shards": int(len(manifest["shards"])),
        "positive_per_cell": int(manifest.get("positive_per_cell", 0)),
        "requested_negatives": int(manifest.get("num_negatives", 0)),
        "fpr_target": float(fpr_target),
        "rows": recall_rows,
        "thresholds": thresholds,
        "method_metadata": {name: method_metadata(name) for name in scores_by_method},
        "ml_metadata": ml_metadata,
        "assumption_warnings": [
            "This artifact closes the same-grid Wiener/SMHW comparison on the fixed manifest cited in `manifest_json`.",
            "If the manifest is stratified, the result is a same-grid benchmark closure on that stratified manifest, not the full 33000-row sensitivity table.",
            "The classical score is still an unmasked full-sky filter with local candidate-region maximization, not masked-sky Bayesian evidence.",
        ],
    }
    report_path = output_dir / "same_grid_fullsky_report.json"
    markdown_path = output_dir / "same_grid_fullsky_report.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(markdown_path, report)
    print(json.dumps({"report": str(report_path), "markdown": str(markdown_path), "scores": str(scores_npz)}, indent=2))


if __name__ == "__main__":
    main()
