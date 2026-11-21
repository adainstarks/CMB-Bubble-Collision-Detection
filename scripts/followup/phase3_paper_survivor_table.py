"""Build a paper-facing ranked survivor table from follow-up artifacts.

Assumptions
-----------
* Input rows are screened follow-up targets, not detections or posterior
  samples.
* The ranked table is intended for the results/follow-up section of the paper:
  it records which adaptive systems survive the current real-sky gates and
  whether a projection-robust re-extraction yields a usable template seed.
* Equal-area fit parameters are treated as deterministic initialization
  metadata, not final physical parameter constraints.
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
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECTION_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_projection_robust_survivor_followup"
    / "projection_robust_survivor_followup.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_paper_survivor_table"
)
DEFAULT_GEOMETRY_RESOLUTION_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_candidate_geometry_resolution"
    / "nilc_9628_geometry_resolution.json"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build a ranked paper-facing survivor table from the current follow-up reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--projection-json", type=str, default=str(DEFAULT_PROJECTION_JSON))
    parser.add_argument(
        "--geometry-resolution-json",
        type=str,
        default=str(DEFAULT_GEOMETRY_RESOLUTION_JSON),
        help="Optional candidate-specific geometry-resolution result used to refine paper-facing notes.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_float(value: Any) -> float | None:
    """Return a finite float or ``None``."""

    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def load_optional_json(path: Path) -> dict[str, Any] | None:
    """Load one JSON object when present, otherwise return ``None``."""

    if not path.exists():
        return None
    return load_json(path)


def build_note(
    member_row: dict[str, Any],
    system_row: dict[str, Any],
    *,
    geometry_decision: dict[str, Any] | None,
) -> str:
    """Return a compact paper-facing note for one representative row."""

    compare = member_row.get("projection_comparison", {})
    if str(system_row["projection_robust_status"]) == "projection_robust_ready":
        return "survives HM+frequency follow-up and remains stable on equal-area re-extraction"
    if geometry_decision is not None and str(geometry_decision.get("status")) == "center_unstable_screening_only":
        return (
            "candidate-centered follow-up collapses under the geometry study; "
            "keep as screening-only until native-sphere center-marginalized follow-up exists"
        )
    failures = [str(item) for item in compare.get("failures", [])]
    if failures:
        return "projection-robust follow-up is blocked by candidate-centered mask support"
    return "survives HM+frequency follow-up but still shifts under projection-robust re-extraction"


def promotion_status(
    system_row: dict[str, Any],
    geometry_decision: dict[str, Any] | None,
) -> str:
    """Return the paper-facing promotion state."""

    if str(system_row["projection_robust_status"]) == "projection_robust_ready":
        return "downstream_handoff_ready"
    if geometry_decision is not None and str(geometry_decision.get("status")) == "center_unstable_screening_only":
        return "screening_only"
    return "geometry_blocked"


def build_rows(
    report: dict[str, Any],
    *,
    geometry_resolution: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build flat ranked table rows."""

    member_index = {row["candidate_id"]: row for row in report.get("members", [])}
    rows: list[dict[str, Any]] = []
    for system_row in report.get("systems", []):
        candidate_id = str(system_row["representative_candidate_id"])
        member_row = member_index[candidate_id]
        geometry_decision = None
        if geometry_resolution is not None and str(geometry_resolution.get("candidate_id")) == candidate_id:
            geometry_decision = geometry_resolution.get("decision")
        equal_area = member_row.get("candidate_centered_equal_area_fit", {})
        original = member_row.get("original_template_seed", {})
        projection_ready = str(system_row["projection_robust_status"]) == "projection_robust_ready"
        parameter_source = "equal_area_reextract" if projection_ready else "original_screening_seed"
        rows.append(
            {
                "system_rank": int(system_row["system_rank"]),
                "candidate_id": candidate_id,
                "map": str(member_row["map"]),
                "patch_index": int(member_row["patch_index"]),
                "glon_deg": float(member_row["candidate_glon_deg"]),
                "glat_deg": float(member_row["candidate_glat_deg"]),
                "screening_priority_tier": str(member_row["screening_priority_tier"]),
                "screening_pooled_bh_q": finite_float(member_row["screening_pooled_bh_q"]),
                "hm_signflip_p_value": finite_float(member_row["hm_signflip_p_value"]),
                "frequency_jackknife_stable": bool(member_row["frequency_jackknife_stable"]),
                "adaptive_system_status": str(system_row["system_status"]),
                "projection_robust_status": str(system_row["projection_robust_status"]),
                "promotion_status": promotion_status(system_row, geometry_decision),
                "geometry_resolution_status": (
                    None if geometry_decision is None else str(geometry_decision.get("status"))
                ),
                "parameter_source": parameter_source,
                "theta_crit_deg": finite_float(
                    equal_area.get("theta_crit_fit_deg") if projection_ready else original.get("theta_crit_deg")
                ),
                "z0": finite_float(
                    equal_area.get("z0_fit") if projection_ready else original.get("z0")
                ),
                "zcrit": finite_float(
                    equal_area.get("zcrit_fit") if projection_ready else original.get("zcrit")
                ),
                "theta_shift_deg": finite_float(member_row.get("projection_comparison", {}).get("theta_shift_deg")),
                "z0_retention": finite_float(member_row.get("projection_comparison", {}).get("z0_retention")),
                "note": build_note(member_row, system_row, geometry_decision=geometry_decision),
            }
        )
    rows.sort(key=lambda row: int(row["system_rank"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat CSV rows."""

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the paper-facing markdown table."""

    lines = ["# Paper Survivor Table", ""]
    lines.append("## Assumptions")
    lines.append("")
    lines.append("- This table ranks the current surviving adaptive systems after HM sign-flip and frequency jackknife follow-up.")
    lines.append("- Equal-area fit parameters are deterministic follow-up seeds, not posterior constraints.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- `num_rows`: `{summary['num_rows']}`")
    lines.append(f"- `projection_robust_status_counts`: `{summary['projection_robust_status_counts']}`")
    lines.append(f"- `promotion_status_counts`: `{summary['promotion_status_counts']}`")
    lines.append("")
    lines.append("## Ranked Table")
    lines.append("")
    lines.append("| rank | candidate | map | q | HM p | theta | z0 | zcrit | parameter source | projection status | promotion | note |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|---|---|---|")
    for row in rows:
        q_value = 0.0 if row["screening_pooled_bh_q"] is None else row["screening_pooled_bh_q"]
        hm_p = 0.0 if row["hm_signflip_p_value"] is None else row["hm_signflip_p_value"]
        theta = 0.0 if row["theta_crit_deg"] is None else row["theta_crit_deg"]
        z0 = 0.0 if row["z0"] is None else row["z0"]
        zcrit = 0.0 if row["zcrit"] is None else row["zcrit"]
        lines.append(
            f"| {row['system_rank']} | {row['candidate_id']} | {row['map']} | {q_value:.6f} | {hm_p:.4f} | "
            f"{theta:.2f} | {z0:.3e} | {zcrit:.3e} | {row['parameter_source']} | "
            f"{row['projection_robust_status']} | {row['promotion_status']} | {row['note']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the paper-facing survivor table builder."""

    args = parse_args()
    projection_path = Path(args.projection_json).expanduser().resolve()
    if not projection_path.exists():
        raise FileNotFoundError(f"Missing projection follow-up JSON: {projection_path}")
    geometry_path = Path(args.geometry_resolution_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = load_json(projection_path)
    geometry_resolution = load_optional_json(geometry_path)
    rows = build_rows(report, geometry_resolution=geometry_resolution)
    summary = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "projection_json": str(projection_path),
        "geometry_resolution_json": str(geometry_path) if geometry_resolution is not None else None,
        "num_rows": int(len(rows)),
        "projection_robust_status_counts": {
            key: int(sum(str(row["projection_robust_status"]) == key for row in rows))
            for key in sorted({str(row["projection_robust_status"]) for row in rows})
        },
        "promotion_status_counts": {
            key: int(sum(str(row["promotion_status"]) == key for row in rows))
            for key in sorted({str(row["promotion_status"]) for row in rows})
        },
    }

    json_path = output_dir / "paper_survivor_table.json"
    csv_path = output_dir / "paper_survivor_table.csv"
    md_path = output_dir / "paper_survivor_table.md"
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, summary, rows)
    print(
        json.dumps(
            {
                "output_json": str(json_path),
                "output_csv": str(csv_path),
                "output_markdown": str(md_path),
                "num_rows": int(len(rows)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
