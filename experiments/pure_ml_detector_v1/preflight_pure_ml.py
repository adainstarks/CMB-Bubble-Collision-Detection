"""Preflight audit for the isolated pure-ML detector track.

Assumptions
-----------
* This audit checks dataset integrity and experiment hygiene. It does not
  certify that a trained model beats any classical screener.
* The pure-ML contract allows raw CMB patches and labels only. HDF5 feature
  groups containing Wiener, SMHW, or circular-template responses are reported as
  forbidden deployed inputs.
* Split overlap is checked in the row namespace of each HDF5 file. Same-grid
  ``source_row_index`` values are only comparable when their source HDF5 matches
  the synthetic training HDF5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYNTHETIC_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "training_data.h5"
DEFAULT_SAME_GRID_H5 = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_same_grid_fullsky_manifest"
    / "same_grid_fullsky_patches.h5"
)
DEFAULT_SAME_GRID_REPORT = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_same_grid_fullsky_manifest"
    / "same_grid_fullsky_report.json"
)
DEFAULT_NULL_TEMPLATE = PROJECT_ROOT / "data" / "remediated_v1" / "null_controls_{map}_mask090.h5"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "pure_ml_detector_v1" / "preflight_report.json"
DEFAULT_MAPS = ("smica", "nilc", "sevem", "commander")
FORBIDDEN_FEATURE_TOKENS = ("wiener", "smhw", "matched", "template", "circular")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Audit pure-ML experiment data and leakage risks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synthetic-h5", type=str, default=str(DEFAULT_SYNTHETIC_H5))
    parser.add_argument("--same-grid-h5", type=str, default=str(DEFAULT_SAME_GRID_H5))
    parser.add_argument("--same-grid-report", type=str, default=str(DEFAULT_SAME_GRID_REPORT))
    parser.add_argument("--null-template", type=str, default=str(DEFAULT_NULL_TEMPLATE))
    parser.add_argument("--maps", type=str, default=",".join(DEFAULT_MAPS))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_maps(text: str) -> tuple[str, ...]:
    """Return cleaned-map names."""

    maps = tuple(token.strip().lower() for token in str(text).split(",") if token.strip())
    if not maps:
        raise ValueError("--maps must not be empty.")
    return maps


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON if it exists."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_list(path: Path) -> list[str]:
    """List HDF5 datasets."""

    names: list[str] = []
    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                names.append(str(name))

        h5.visititems(visitor)
    return sorted(names)


def split_indices(h5: h5py.File, split_name: str) -> set[int]:
    """Read split row indices as a set."""

    key = f"splits/{split_name}_idx"
    if key not in h5:
        return set()
    return set(int(value) for value in np.asarray(h5[key][:], dtype=np.int64))


def overlap_count(a: set[int], b: set[int]) -> int:
    """Return set-overlap size."""

    return int(len(a & b))


def split_report(path: Path) -> dict[str, Any]:
    """Audit split disjointness and label balance."""

    with h5py.File(path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        splits = {name: split_indices(h5, name) for name in ("train", "val", "calibration", "test")}
        split_rows = {}
        for name, idx in splits.items():
            if not idx:
                continue
            arr = np.asarray(sorted(idx), dtype=np.int64)
            split_rows[name] = {
                "count": int(arr.size),
                "positive": int(np.count_nonzero(labels[arr] == 1)),
                "negative": int(np.count_nonzero(labels[arr] == 0)),
            }
        overlaps = {}
        blocking_overlap_pairs = []
        names = sorted(splits)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                if splits[left] and splits[right]:
                    pair = f"{left}__{right}"
                    count = overlap_count(splits[left], splits[right])
                    overlaps[pair] = count
                    if count and pair != "calibration__val":
                        blocking_overlap_pairs.append(pair)
        realization_by_split = {}
        if "metadata/cmb_realization_idx" in h5:
            realization = np.asarray(h5["metadata/cmb_realization_idx"][:], dtype=np.int64)
            for name, idx in splits.items():
                if idx:
                    arr = np.asarray(sorted(idx), dtype=np.int64)
                    realization_by_split[name] = sorted(int(value) for value in np.unique(realization[arr]))
        return {
            "path": str(path),
            "num_rows": int(labels.size),
            "positive": int(np.count_nonzero(labels == 1)),
            "negative": int(np.count_nonzero(labels == 0)),
            "split_rows": split_rows,
            "split_overlaps": overlaps,
            "allowed_split_overlaps": {
                "calibration__val": (
                    "Existing remediated_v1 training_data.h5 uses the validation "
                    "rows as calibration rows. Pure-ML training must not report "
                    "validation and calibration as independent evidence."
                )
            },
            "blocking_split_overlap_pairs": blocking_overlap_pairs,
            "realization_by_split_count": {name: len(values) for name, values in realization_by_split.items()},
            "blocking_split_overlap_ok": len(blocking_overlap_pairs) == 0,
        }


def forbidden_feature_report(path: Path) -> dict[str, Any]:
    """Report classical feature datasets that must not become inputs."""

    datasets = dataset_list(path)
    forbidden = [
        name
        for name in datasets
        if name.startswith("features/")
        and any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    return {
        "path": str(path),
        "feature_datasets": [name for name in datasets if name.startswith("features/")],
        "forbidden_deployed_inputs": forbidden,
        "pure_ml_input_contract": "Allowed deployed input is patches only for this experiment track.",
    }


def same_grid_report(path: Path, same_grid_report_json: Path, synthetic_h5: Path) -> dict[str, Any]:
    """Audit same-grid benchmark data."""

    report_json = load_json(same_grid_report_json)
    with h5py.File(path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        amplitude = np.asarray(h5["truth/amplitude"][:], dtype=np.float64) if "truth/amplitude" in h5 else np.array([])
        theta = np.asarray(h5["truth/theta_crit_deg"][:], dtype=np.float64) if "truth/theta_crit_deg" in h5 else np.array([])
        realization = (
            np.asarray(h5["metadata/cmb_realization_idx"][:], dtype=np.int64)
            if "metadata/cmb_realization_idx" in h5
            else np.array([], dtype=np.int64)
        )
        source_rows = (
            np.asarray(h5["metadata/source_row_index"][:], dtype=np.int64)
            if "metadata/source_row_index" in h5
            else np.array([], dtype=np.int64)
        )
    source_h5 = Path(str(report_json.get("source_h5", ""))).resolve() if report_json.get("source_h5") else None
    comparable_source_rows = bool(source_h5 and source_h5 == synthetic_h5.resolve())
    return {
        "path": str(path),
        "source_h5": str(source_h5) if source_h5 else None,
        "source_rows_comparable_to_synthetic": comparable_source_rows,
        "num_rows": int(labels.size),
        "positive": int(np.count_nonzero(labels == 1)),
        "negative": int(np.count_nonzero(labels == 0)),
        "amplitude_values": [float(value) for value in sorted(np.unique(amplitude[labels == 1]))] if amplitude.size else [],
        "theta_values_deg": [float(value) for value in sorted(np.unique(theta[labels == 1]))] if theta.size else [],
        "realization_count": int(np.unique(realization).size) if realization.size else 0,
        "source_row_count": int(np.unique(source_rows).size) if source_rows.size else 0,
    }


def null_control_report(template: str, maps: tuple[str, ...]) -> dict[str, Any]:
    """Audit real-map null controls."""

    rows = {}
    for map_name in maps:
        path = Path(str(template).format(map=map_name)).expanduser().resolve()
        if not path.exists():
            rows[map_name] = {"path": str(path), "exists": False}
            continue
        with h5py.File(path, "r") as h5:
            labels = np.asarray(h5["labels"][:], dtype=np.uint8)
            row = {
                "path": str(path),
                "exists": True,
                "num_rows": int(labels.size),
                "positive_labels": int(np.count_nonzero(labels == 1)),
                "negative_labels": int(np.count_nonzero(labels == 0)),
            }
            for split in ("train", "val", "calibration", "test"):
                idx = split_indices(h5, split)
                if idx:
                    row[f"{split}_count"] = int(len(idx))
            rows[map_name] = row
    return rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a compact preflight markdown report."""

    lines = [
        "# Pure ML Detector Preflight",
        "",
        "## Assumptions",
        "",
        "- Deployed pure-ML inputs are raw `patches` only.",
        "- Classical response maps may be external comparators, not model inputs.",
        "- Split overlap checks are row-namespace checks inside each source HDF5.",
        "",
        "## Status",
        "",
        f"- `blocking_split_overlap_ok`: `{report['synthetic_split_report']['blocking_split_overlap_ok']}`",
        f"- `forbidden_deployed_inputs_count`: `{len(report['synthetic_forbidden_features']['forbidden_deployed_inputs'])}`",
        f"- `same_grid_source_rows_comparable_to_synthetic`: `{report['same_grid_report']['source_rows_comparable_to_synthetic']}`",
        "",
        "## Forbidden Feature Datasets",
        "",
    ]
    forbidden = report["synthetic_forbidden_features"]["forbidden_deployed_inputs"]
    if forbidden:
        lines.extend(f"- `{name}`" for name in forbidden)
    else:
        lines.append("- None present.")
    lines.extend(
        [
            "",
            "These datasets are allowed to exist in the HDF5 file for other experiments, but this track must not use them as deployed inputs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the preflight audit."""

    args = parse_args()
    synthetic_h5 = Path(args.synthetic_h5).expanduser().resolve()
    same_grid_h5 = Path(args.same_grid_h5).expanduser().resolve()
    same_grid_report_json = Path(args.same_grid_report).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    maps = parse_maps(args.maps)

    if not synthetic_h5.exists():
        raise FileNotFoundError(f"Missing synthetic HDF5: {synthetic_h5}")
    if not same_grid_h5.exists():
        raise FileNotFoundError(f"Missing same-grid HDF5: {same_grid_h5}")

    report = {
        "synthetic_split_report": split_report(synthetic_h5),
        "synthetic_forbidden_features": forbidden_feature_report(synthetic_h5),
        "same_grid_report": same_grid_report(same_grid_h5, same_grid_report_json, synthetic_h5),
        "null_controls": null_control_report(args.null_template, maps),
        "contract": {
            "allowed_deployed_inputs": ["patches"],
            "forbidden_deployed_inputs": [
                "features/wiener_feeney_response",
                "features/*matched*",
                "features/*template*",
                "features/*smhw*",
                "any external classical score used as a model feature",
            ],
            "benchmark_to_beat": {
                "method": "wiener_feeney_matched_filter",
                "mean_same_grid_recall_at_fpr_0p05": 0.3841,
            },
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(output_json.with_suffix(".md"), report)
    print(json.dumps({"json": str(output_json), "markdown": str(output_json.with_suffix('.md'))}, indent=2))


if __name__ == "__main__":
    main()
