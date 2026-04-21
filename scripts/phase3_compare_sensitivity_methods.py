"""Compare sensitivity-report methods against a fixed baseline method.

Assumptions
-----------
* All compared reports are candidate-screening sensitivity artifacts generated
  on the same amplitude/theta grid and fixed-FPR calibration protocol.
* This comparison summarizes cell-level detection-efficiency deltas. It is not
  a Bayesian evidence comparison and it does not replace the full same-grid
  classical benchmark.
* The primary scientific use is ablation ranking: identify which changes
  improve recall in hard cells without silently changing the evaluation grid.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_method_compare"
CELL_KEYS = ("amplitude", "theta_crit_deg", "zcrit_ratio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one or more sensitivity-report methods to a baseline method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-report", type=str, required=True)
    parser.add_argument("--baseline-method", type=str, required=True)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Comparison spec label:path:method. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hard-max-amplitude", type=float, default=5.0e-6)
    parser.add_argument("--moderate-min-amplitude", type=float, default=1.0e-5)
    parser.add_argument("--large-radius-min-deg", type=float, default=15.0)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def parse_compare_spec(text: str) -> tuple[str, Path, str]:
    parts = str(text).split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --compare spec `{text}`; expected label:path:method.")
    label, path_text, method = parts
    return label.strip(), Path(path_text).expanduser().resolve(), method.strip()


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_key(row: dict[str, Any]) -> tuple[float, ...]:
    values: list[float] = []
    for key in CELL_KEYS:
        if key in row:
            values.append(float(row[key]))
    return tuple(values)


def rows_for_method(report: dict[str, Any], method: str) -> dict[tuple[float, ...], dict[str, Any]]:
    rows = {}
    for row in report.get("rows", []):
        if str(row.get("method")) != str(method):
            continue
        rows[cell_key(row)] = row
    if not rows:
        raise KeyError(f"Method `{method}` not found in report.")
    return rows


def threshold_for_method(report: dict[str, Any], method: str) -> dict[str, Any]:
    thresholds = report.get("thresholds", {})
    if method not in thresholds:
        raise KeyError(f"Threshold block for method `{method}` not found in report.")
    return thresholds[method]


def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_cells": 0,
            "baseline_mean_p_det": None,
            "compare_mean_p_det": None,
            "mean_delta_p_det": None,
            "median_delta_p_det": None,
            "improved_cells": 0,
            "worsened_cells": 0,
            "tied_cells": 0,
        }
    delta = np.asarray([float(row["delta_p_det"]) for row in rows], dtype=np.float64)
    baseline = np.asarray([float(row["baseline_p_det"]) for row in rows], dtype=np.float64)
    compare = np.asarray([float(row["compare_p_det"]) for row in rows], dtype=np.float64)
    improved = int(np.count_nonzero(delta > 0.0))
    worsened = int(np.count_nonzero(delta < 0.0))
    return {
        "num_cells": int(len(rows)),
        "baseline_mean_p_det": float(np.mean(baseline)),
        "compare_mean_p_det": float(np.mean(compare)),
        "mean_delta_p_det": float(np.mean(delta)),
        "median_delta_p_det": float(np.median(delta)),
        "improved_cells": improved,
        "worsened_cells": worsened,
        "tied_cells": int(len(rows) - improved - worsened),
    }


def compare_methods(
    *,
    baseline_report: dict[str, Any],
    baseline_method: str,
    compare_report: dict[str, Any],
    compare_method: str,
    label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline_rows = rows_for_method(baseline_report, baseline_method)
    compare_rows = rows_for_method(compare_report, compare_method)
    common_keys = sorted(set(baseline_rows) & set(compare_rows))
    if not common_keys:
        raise RuntimeError(f"No overlapping sensitivity cells for `{label}`.")

    cell_rows = []
    for key in common_keys:
        base_row = baseline_rows[key]
        cmp_row = compare_rows[key]
        row = {
            "cell_key": key,
            "amplitude": float(base_row["amplitude"]),
            "theta_crit_deg": float(base_row["theta_crit_deg"]),
            "baseline_p_det": float(base_row["p_det"]),
            "compare_p_det": float(cmp_row["p_det"]),
            "delta_p_det": float(cmp_row["p_det"]) - float(base_row["p_det"]),
            "baseline_detected": int(base_row["detected"]),
            "compare_detected": int(cmp_row["detected"]),
            "num_positive": int(base_row["num_positive"]),
        }
        if "zcrit_ratio" in base_row:
            row["zcrit_ratio"] = float(base_row["zcrit_ratio"])
        cell_rows.append(row)

    cell_rows.sort(key=lambda row: (row["amplitude"], row["theta_crit_deg"]))
    hard_rows = [row for row in cell_rows if float(row["amplitude"]) <= float(args.hard_max_amplitude)]
    moderate_rows = [row for row in cell_rows if float(row["amplitude"]) >= float(args.moderate_min_amplitude)]
    large_radius_rows = [row for row in cell_rows if float(row["theta_crit_deg"]) >= float(args.large_radius_min_deg)]
    sorted_gain = sorted(cell_rows, key=lambda row: row["delta_p_det"], reverse=True)
    sorted_loss = sorted(cell_rows, key=lambda row: row["delta_p_det"])

    return {
        "label": label,
        "baseline_method": baseline_method,
        "compare_method": compare_method,
        "baseline_report": str(Path(args.baseline_report).expanduser().resolve()),
        "compare_report": str(compare_report.get("source_dir", "")) or "",
        "baseline_threshold": threshold_for_method(baseline_report, baseline_method),
        "compare_threshold": threshold_for_method(compare_report, compare_method),
        "overall": subset_summary(cell_rows),
        "hard_subset": {
            "rule": f"amplitude <= {float(args.hard_max_amplitude):.3g}",
            **subset_summary(hard_rows),
        },
        "moderate_subset": {
            "rule": f"amplitude >= {float(args.moderate_min_amplitude):.3g}",
            **subset_summary(moderate_rows),
        },
        "large_radius_subset": {
            "rule": f"theta_crit_deg >= {float(args.large_radius_min_deg):.1f}",
            **subset_summary(large_radius_rows),
        },
        "top_gains": sorted_gain[: int(args.top_k)],
        "top_losses": sorted_loss[: int(args.top_k)],
        "cell_rows": cell_rows,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Sensitivity Method Comparison", ""]
    lines.append(f"Created: `{report['created_utc']}`")
    lines.append(f"Baseline: `{report['baseline_method']}` from `{report['baseline_report']}`")
    lines.append("")
    for item in report["comparisons"]:
        lines.append(f"## {item['label']}: `{item['compare_method']}`")
        lines.append("")
        lines.append("| subset | cells | baseline mean | compare mean | mean delta | improved | worsened |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for key, title in (
            ("overall", "overall"),
            ("hard_subset", "hard"),
            ("moderate_subset", "moderate"),
            ("large_radius_subset", "large-radius"),
        ):
            row = item[key]
            lines.append(
                f"| {title} | {row['num_cells']} | "
                f"{row['baseline_mean_p_det'] if row['baseline_mean_p_det'] is not None else 'n/a'} | "
                f"{row['compare_mean_p_det'] if row['compare_mean_p_det'] is not None else 'n/a'} | "
                f"{row['mean_delta_p_det'] if row['mean_delta_p_det'] is not None else 'n/a'} | "
                f"{row['improved_cells']} | {row['worsened_cells']} |"
            )
        lines.append("")
        lines.append("Top gains:")
        for row in item["top_gains"]:
            lines.append(
                f"- A={row['amplitude']:.3g}, theta={row['theta_crit_deg']:.1f} deg: "
                f"{row['baseline_p_det']:.3f} -> {row['compare_p_det']:.3f} "
                f"(delta {row['delta_p_det']:+.3f})"
            )
        lines.append("")
        lines.append("Top losses:")
        for row in item["top_losses"]:
            lines.append(
                f"- A={row['amplitude']:.3g}, theta={row['theta_crit_deg']:.1f} deg: "
                f"{row['baseline_p_det']:.3f} -> {row['compare_p_det']:.3f} "
                f"(delta {row['delta_p_det']:+.3f})"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.compare:
        raise ValueError("Provide at least one --compare label:path:method spec.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_report_path = Path(args.baseline_report).expanduser().resolve()
    baseline_report = load_report(baseline_report_path)

    comparisons = []
    for spec in args.compare:
        label, report_path, method = parse_compare_spec(spec)
        compare_report = load_report(report_path)
        comparisons.append(
            compare_methods(
                baseline_report=baseline_report,
                baseline_method=str(args.baseline_method),
                compare_report=compare_report,
                compare_method=method,
                label=label,
                args=args,
            )
        )

    payload = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "baseline_report": str(baseline_report_path),
        "baseline_method": str(args.baseline_method),
        "comparisons": comparisons,
    }
    json_path = output_dir / "sensitivity_method_compare.json"
    md_path = output_dir / "sensitivity_method_compare.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
