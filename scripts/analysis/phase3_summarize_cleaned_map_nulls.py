"""
Summarize real-null screener burden across Planck cleaned-map products.
"""

from __future__ import annotations
# Bootstrap organized script folders for direct CLI execution.
import sys as _cmb_sys
from pathlib import Path as _cmb_Path

_cmb_scripts_root = _cmb_Path(__file__).resolve().parents[1]
_cmb_paths = [_cmb_scripts_root]
_cmb_paths.extend(
    sorted(
        _cmb_path
        for _cmb_path in _cmb_scripts_root.iterdir()
        if _cmb_path.is_dir() and _cmb_path.name != "__pycache__"
    )
)
for _cmb_path in reversed(_cmb_paths):
    _cmb_path_text = str(_cmb_path)
    if _cmb_path_text not in _cmb_sys.path:
        _cmb_sys.path.insert(0, _cmb_path_text)
del _cmb_Path, _cmb_sys, _cmb_scripts_root, _cmb_paths, _cmb_path, _cmb_path_text

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a compact cross-map null robustness table for multiple screeners.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Entry formatted as model_label,map_name,path/to/null_summary.json",
    )
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_entry(value):
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise ValueError("--entry must be formatted as model_label,map_name,path")
    return parts[0], parts[1], Path(parts[2])


def write_md(path, rows):
    lines = [
        "# Cleaned-Map Null Robustness",
        "",
        "| model | map | samples | frozen threshold | false candidates | FPR | mean positive fraction | null-calibrated threshold |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["map"],
                    str(row["num_samples"]),
                    f"{row['frozen_threshold']:.3f}",
                    str(row["false_positive_count"]),
                    f"{row['false_positive_rate']:.4f}",
                    f"{row['mean_positive_fraction']:.3e}",
                    f"{row['null_calibrated_threshold']:.3f}",
                ]
            )
            + " |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = []
    for entry in args.entry:
        model, map_name, path = parse_entry(entry)
        summary = load_json(path)
        metrics = summary["frozen_threshold_nearest_grid_metrics"]
        rows.append(
            {
                "model": model,
                "map": map_name,
                "summary_path": str(path.resolve()),
                "num_samples": int(metrics["num_samples"]),
                "frozen_threshold": float(summary["frozen_threshold"]),
                "false_positive_count": int(metrics["false_positive_count"]),
                "false_positive_rate": float(metrics["false_positive_rate"]),
                "mean_positive_fraction": float(metrics["mean_positive_fraction"]),
                "null_calibrated_threshold": float(summary["null_calibrated_threshold"]),
                "null_calibrated_false_positive_rate": float(
                    summary["null_calibrated_threshold_metrics"]["false_positive_rate"]
                ),
            }
        )
    rows.sort(key=lambda row: (row["map"], row["model"]))
    report = {"rows": rows}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_md(args.output_md, rows)
    print(json.dumps(report, indent=2))
    print(f"Cross-map null JSON: {args.output_json}")
    print(f"Cross-map null MD:   {args.output_md}")


if __name__ == "__main__":
    main()
