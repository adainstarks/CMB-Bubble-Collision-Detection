"""Score one or more trained models on an existing sensitivity HDF5 grid.

Assumptions
-----------
* The sensitivity HDF5 already exists and defines the frozen amplitude/radius
  benchmark grid.
* Existing score arrays loaded from the source NPZ are treated as authoritative
  for unchanged methods; this script only adds or refreshes requested ML model
  scores.
* Model comparisons remain fixed-FPR screening comparisons on the sensitivity
  grid, not Bayesian evidence statements.
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
from phase3_sensitivity_curve import (
    parse_model_spec,
    plot_sensitivity,
    score_ml_model,
    summarize_sensitivity,
    write_csv,
    write_markdown,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "phase3_unet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score trained models on an existing sensitivity benchmark grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", type=str, default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", action="append", required=True, help="Model as name:run_dir:checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-ml-data", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = source_dir / "sensitivity_data.h5"
    scores_path = source_dir / "sensitivity_scores.npz"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing sensitivity HDF5: {h5_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing sensitivity score NPZ: {scores_path}")

    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        summary = dict(h5["summary"].attrs)
        theta_grid_deg = tuple(float(x) for x in json.loads(summary["theta_grid_deg"]))
        fpr_target = float(summary["fpr_target"])

    with np.load(scores_path) as loaded:
        scores_by_method = {
            (key[7:] if key.startswith("score__") else key): np.asarray(loaded[key], dtype=np.float32)
            for key in loaded.files
            if key.startswith("score__")
        }

    device = p3.resolve_device(args.device)
    model_metadata = {}
    for spec_text in args.model:
        spec = parse_model_spec(spec_text)
        if spec.name in scores_by_method and not args.overwrite:
            model_metadata[spec.name] = {
                "reused": True,
                "score_source": str(scores_path),
            }
            continue
        score_args = argparse.Namespace(
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            cache_ml_data=bool(args.cache_ml_data),
        )
        scores, metadata = score_ml_model(spec, h5_path, score_args, device)
        scores_by_method[spec.name] = np.asarray(scores, dtype=np.float32)
        model_metadata[spec.name] = metadata

    output_scores = output_dir / "sensitivity_scores.npz"
    np.savez_compressed(
        output_scores,
        labels=labels,
        **{f"score__{name}": values.astype(np.float32) for name, values in scores_by_method.items()},
    )

    rows, thresholds = summarize_sensitivity(scores_by_method, h5_path, fpr_target)
    csv_path = output_dir / "sensitivity_curve.csv"
    write_csv(csv_path, rows)
    plot_path = output_dir / "sensitivity_curve.png"
    plot_sensitivity(plot_path, rows, list(scores_by_method.keys()), theta_grid_deg)

    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source_dir": str(source_dir),
        "data_h5": str(h5_path),
        "scores_npz": str(output_scores),
        "csv": str(csv_path),
        "plot_png": str(plot_path),
        "fpr_target": float(fpr_target),
        "num_positive": int(summary["num_positive"]),
        "num_negative": int(summary["num_negative"]),
        "num_per_cell": int(summary["num_per_cell"]),
        "amplitude_grid": json.loads(summary["amplitude_grid"]),
        "theta_grid_deg": json.loads(summary["theta_grid_deg"]),
        "zcrit_ratio_grid": json.loads(summary.get("zcrit_ratio_grid", "[1.0]")),
        "amplitude_definition": summary["amplitude_definition"],
        "signal_strength_definition": summary.get("signal_strength_definition", ""),
        "edge_sigma_deg": float(summary["edge_sigma_deg"]),
        "thresholds": thresholds,
        "method_metadata": {name: method_metadata(name) for name in scores_by_method},
        "model_metadata": model_metadata,
        "rows": rows,
    }
    report_path = output_dir / "sensitivity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_dir / "sensitivity_report.md"
    write_markdown(markdown_path, report)
    print(json.dumps({"report": str(report_path), "scores": str(output_scores), "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
