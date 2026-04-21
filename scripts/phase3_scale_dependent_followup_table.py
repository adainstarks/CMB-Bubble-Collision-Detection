"""Build a scale-dependent follow-up table for frozen screening candidates.

Assumptions
-----------
* Input rows are screening-derived candidates with deterministic local
  Feeney-template seeds. They are not detections or Bayesian posteriors.
* A scale-dependent association radius ``r_assoc ~= theta_crit / 2`` is used
  only for follow-up bookkeeping, motivated by the scale-aware blob handling in
  McEwen et al. (2012, Phys. Rev. D 85, 103502). It is not a likelihood-based
  source matcher.
* Large fitted radii or large offsets from the patch center are treated as
  geometry cautions. Those systems should be re-extracted on the sphere or with
  a less distorted equal-area geometry before any parameter statement.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_bayesian_template_handoff"
    / "bayesian_template_handoff.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_scale_dependent_followup_table"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Group frozen follow-up candidates with a scale-dependent sky-association rule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-jsonl", type=str, default=str(DEFAULT_INPUT_JSONL))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=0.5,
        help="Association radius factor applied to fitted theta_crit.",
    )
    parser.add_argument(
        "--min-radius-deg",
        type=float,
        default=5.0,
        help="Lower bound for the scale-dependent association radius.",
    )
    parser.add_argument(
        "--max-radius-deg",
        type=float,
        default=15.0,
        help="Upper bound for the scale-dependent association radius.",
    )
    parser.add_argument(
        "--large-radius-caution-deg",
        type=float,
        default=20.0,
        help="Template-fit radius above which parameter interpretation remains geometry-limited.",
    )
    parser.add_argument(
        "--large-offset-caution-deg",
        type=float,
        default=10.0,
        help="Candidate offset above which parameter interpretation remains geometry-limited.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate physical ranges and required inputs."""

    input_path = Path(args.input_jsonl).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {input_path}")
    for label, value in (
        ("--radius-scale", args.radius_scale),
        ("--min-radius-deg", args.min_radius_deg),
        ("--max-radius-deg", args.max_radius_deg),
        ("--large-radius-caution-deg", args.large_radius_caution_deg),
        ("--large-offset-caution-deg", args.large_offset_caution_deg),
    ):
        if not np.isfinite(float(value)):
            raise ValueError(f"{label} must be finite.")
    if float(args.radius_scale) <= 0.0:
        raise ValueError("--radius-scale must be positive.")
    if float(args.min_radius_deg) <= 0.0:
        raise ValueError("--min-radius-deg must be positive.")
    if float(args.max_radius_deg) < float(args.min_radius_deg):
        raise ValueError("--max-radius-deg must be >= --min-radius-deg.")
    if float(args.large_radius_caution_deg) <= 0.0:
        raise ValueError("--large-radius-caution-deg must be positive.")
    if float(args.large_offset_caution_deg) < 0.0:
        raise ValueError("--large-offset-caution-deg must be non-negative.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def finite_float(value: Any) -> float | None:
    """Return a finite float or ``None``."""

    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Reject malformed or non-physical follow-up rows."""

    for idx, row in enumerate(rows, start=1):
        for key in ("candidate_glon_deg", "candidate_glat_deg", "global_cluster_rank", "map", "patch_index"):
            if key not in row:
                raise KeyError(f"Row {idx} is missing required field {key!r}.")
        lat = float(row["candidate_glat_deg"])
        if lat < -90.0 or lat > 90.0:
            raise ValueError(f"Row {idx} has non-physical latitude {lat:.6f} deg.")
        theta = finite_float(row.get("template_seed", {}).get("theta_crit_deg"))
        if theta is not None and theta <= 0.0:
            raise ValueError(f"Row {idx} has non-physical theta_crit {theta:.6f} deg.")


def association_radius_deg(row: dict[str, Any], args: argparse.Namespace) -> float:
    """Return the adaptive sky-association radius for one row."""

    theta = finite_float(row.get("template_seed", {}).get("theta_crit_deg"))
    if theta is None:
        theta = float(args.min_radius_deg)
    radius = float(theta) * float(args.radius_scale)
    radius = min(max(radius, float(args.min_radius_deg)), float(args.max_radius_deg))
    return float((radius * u.deg).to_value(u.deg))


def screening_sort_key(row: dict[str, Any]) -> tuple[float, float, int]:
    """Sort candidates by strongest screening evidence first."""

    q_value = finite_float(row.get("screening_pooled_bh_q"))
    pooled_p = finite_float(row.get("screening_pooled_survival_p"))
    global_rank = int(row.get("global_cluster_rank", 10**9))
    return (
        float(q_value if q_value is not None else np.inf),
        float(pooled_p if pooled_p is not None else np.inf),
        global_rank,
    )


def build_adjacency(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[list[int]]:
    """Connect rows whose sky separation is within either adaptive radius."""

    coords = SkyCoord(
        l=np.asarray([float(row["candidate_glon_deg"]) for row in rows], dtype=np.float64) * u.deg,
        b=np.asarray([float(row["candidate_glat_deg"]) for row in rows], dtype=np.float64) * u.deg,
        frame="galactic",
    )
    radii = np.asarray([association_radius_deg(row, args) for row in rows], dtype=np.float64)
    adjacency = [[] for _ in rows]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            separation_deg = float(coords[i].separation(coords[j]).deg)
            allowed_deg = max(float(radii[i]), float(radii[j]))
            if separation_deg <= allowed_deg:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    """Return connected components of the adaptive association graph."""

    visited = [False] * len(adjacency)
    components: list[list[int]] = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def summarize_system(
    system_id: int,
    member_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build one paper-facing follow-up system summary."""

    ranked_members = sorted(member_rows, key=screening_sort_key)
    representative = ranked_members[0]
    theta_values = np.asarray(
        [
            finite_float(row.get("template_seed", {}).get("theta_crit_deg"))
            for row in ranked_members
            if finite_float(row.get("template_seed", {}).get("theta_crit_deg")) is not None
        ],
        dtype=np.float64,
    )
    offset_values = np.asarray(
        [
            finite_float(row.get("template_seed", {}).get("candidate_offset_deg"))
            for row in ranked_members
            if finite_float(row.get("template_seed", {}).get("candidate_offset_deg")) is not None
        ],
        dtype=np.float64,
    )
    association_values = np.asarray(
        [association_radius_deg(row, args) for row in ranked_members],
        dtype=np.float64,
    )
    survivors = [
        f"{row['map']}:{int(row['patch_index'])}"
        for row in ranked_members
        if str(row.get("real_sky_followup_status")) == "survives_hm_and_frequency_followup"
    ]
    pending = [
        f"{row['map']}:{int(row['patch_index'])}"
        for row in ranked_members
        if str(row.get("real_sky_followup_status")) == "pending_real_sky_followup"
    ]
    geometry_limited = bool(
        any(bool(row.get("projection_systematics_caution", False)) for row in ranked_members)
        or (theta_values.size and float(np.max(theta_values)) >= float(args.large_radius_caution_deg))
        or (offset_values.size and float(np.max(offset_values)) >= float(args.large_offset_caution_deg))
    )
    if survivors:
        system_status = "survives_real_sky_followup"
    elif pending:
        system_status = "pending_real_sky_followup"
    else:
        system_status = "fails_real_sky_followup"
    if geometry_limited:
        interpretation = "screening_only_until_projection_robust_followup"
    else:
        interpretation = "template_seed_usable_for_non_geometry_limited_followup"

    return {
        "system_id": int(system_id),
        "system_rank": int(system_id),
        "system_status": system_status,
        "geometry_interpretation": interpretation,
        "geometry_limited": geometry_limited,
        "representative_global_cluster_rank": int(representative["global_cluster_rank"]),
        "representative_map": str(representative["map"]),
        "representative_patch_index": int(representative["patch_index"]),
        "representative_glon_deg": float(representative["candidate_glon_deg"]),
        "representative_glat_deg": float(representative["candidate_glat_deg"]),
        "representative_theta_crit_deg": finite_float(representative.get("template_seed", {}).get("theta_crit_deg")),
        "representative_z0": finite_float(representative.get("template_seed", {}).get("z0")),
        "representative_zcrit": finite_float(representative.get("template_seed", {}).get("zcrit")),
        "representative_screening_tier": str(representative.get("screening_priority_tier")),
        "representative_screening_q": finite_float(representative.get("screening_pooled_bh_q")),
        "adaptive_radius_deg": association_radius_deg(representative, args),
        "max_member_adaptive_radius_deg": float(np.max(association_values)),
        "num_members": int(len(ranked_members)),
        "maps": [str(row["map"]) for row in ranked_members],
        "member_ids": [f"{row['map']}:{int(row['patch_index'])}" for row in ranked_members],
        "surviving_member_ids": survivors,
        "pending_member_ids": pending,
        "projection_caution_member_ids": [
            f"{row['map']}:{int(row['patch_index'])}"
            for row in ranked_members
            if bool(row.get("projection_systematics_caution", False))
        ],
        "theta_crit_deg_median": (
            float(np.median(theta_values))
            if theta_values.size
            else None
        ),
        "theta_crit_deg_max": (
            float(np.max(theta_values))
            if theta_values.size
            else None
        ),
        "candidate_offset_deg_median": (
            float(np.median(offset_values))
            if offset_values.size
            else None
        ),
        "candidate_offset_deg_max": (
            float(np.max(offset_values))
            if offset_values.size
            else None
        ),
        "members": ranked_members,
    }


def build_system_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Group follow-up rows into adaptive sky systems."""

    ordered_rows = sorted(rows, key=screening_sort_key)
    adjacency = build_adjacency(ordered_rows, args)
    components = connected_components(adjacency)
    systems = [
        summarize_system(system_id=index + 1, member_rows=[ordered_rows[i] for i in component], args=args)
        for index, component in enumerate(sorted(components, key=lambda comp: screening_sort_key(ordered_rows[min(comp, key=lambda idx: screening_sort_key(ordered_rows[idx]))])))
    ]
    systems.sort(key=lambda row: screening_sort_key(row["members"][0]))
    for rank, row in enumerate(systems, start=1):
        row["system_id"] = int(rank)
        row["system_rank"] = int(rank)
    return systems


def system_summary(systems: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Build a compact JSON summary for the adaptive follow-up table."""

    status_counts: dict[str, int] = {}
    interpretation_counts: dict[str, int] = {}
    for row in systems:
        status = str(row["system_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        interpretation = str(row["geometry_interpretation"])
        interpretation_counts[interpretation] = interpretation_counts.get(interpretation, 0) + 1
    return {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "association_rule": {
            "formula": "clamp(radius_scale * theta_crit_fit_deg, min_radius_deg, max_radius_deg)",
            "radius_scale": float(args.radius_scale),
            "min_radius_deg": float(args.min_radius_deg),
            "max_radius_deg": float(args.max_radius_deg),
        },
        "geometry_guardrails": {
            "large_radius_caution_deg": float(args.large_radius_caution_deg),
            "large_offset_caution_deg": float(args.large_offset_caution_deg),
        },
        "num_input_candidates": int(sum(int(row["num_members"]) for row in systems)),
        "num_systems": int(len(systems)),
        "system_status_counts": status_counts,
        "geometry_interpretation_counts": interpretation_counts,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a flat CSV summary table."""

    flat_rows = [
        {
            "system_rank": row["system_rank"],
            "system_status": row["system_status"],
            "geometry_interpretation": row["geometry_interpretation"],
            "representative_global_cluster_rank": row["representative_global_cluster_rank"],
            "representative_map": row["representative_map"],
            "representative_patch_index": row["representative_patch_index"],
            "representative_glon_deg": row["representative_glon_deg"],
            "representative_glat_deg": row["representative_glat_deg"],
            "representative_theta_crit_deg": row["representative_theta_crit_deg"],
            "representative_z0": row["representative_z0"],
            "representative_zcrit": row["representative_zcrit"],
            "adaptive_radius_deg": row["adaptive_radius_deg"],
            "max_member_adaptive_radius_deg": row["max_member_adaptive_radius_deg"],
            "num_members": row["num_members"],
            "maps": ",".join(row["maps"]),
            "surviving_member_ids": ",".join(row["surviving_member_ids"]),
            "projection_caution_member_ids": ",".join(row["projection_caution_member_ids"]),
        }
        for row in rows
    ]
    if not flat_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_markdown(path: Path, summary: dict[str, Any], systems: list[dict[str, Any]]) -> None:
    """Write a paper-facing markdown table."""

    lines = ["# Scale-Dependent Follow-Up Table", ""]
    lines.append("## Assumptions")
    lines.append("")
    lines.append("- This is a follow-up grouping table for screened candidates, not a cosmological detection catalog.")
    lines.append("- Systems use an adaptive association radius derived from the fitted template scale, bounded to avoid non-physical over-merging.")
    lines.append("- Geometry-limited systems require native-sphere or equal-area re-extraction before any parameter statement.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- `num_input_candidates`: `{summary['num_input_candidates']}`")
    lines.append(f"- `num_systems`: `{summary['num_systems']}`")
    lines.append(f"- `system_status_counts`: `{summary['system_status_counts']}`")
    lines.append(f"- `geometry_interpretation_counts`: `{summary['geometry_interpretation_counts']}`")
    lines.append(
        "- `association_rule`: "
        f"`clamp({summary['association_rule']['radius_scale']:.2f} * theta_crit_fit_deg, "
        f"{summary['association_rule']['min_radius_deg']:.1f}, {summary['association_rule']['max_radius_deg']:.1f}) deg`"
    )
    lines.append("")
    lines.append("## Ranked Systems")
    lines.append("")
    lines.append("| system | status | rep rank | rep id | glon | glat | theta_fit | adaptive radius | members | surviving members | geometry |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|")
    for row in systems:
        rep_id = f"{row['representative_map']}:{row['representative_patch_index']}"
        surviving = ", ".join(row["surviving_member_ids"]) if row["surviving_member_ids"] else "none"
        lines.append(
            f"| {row['system_rank']} | {row['system_status']} | {row['representative_global_cluster_rank']} | "
            f"{rep_id} | {row['representative_glon_deg']:.2f} | {row['representative_glat_deg']:.2f} | "
            f"{0.0 if row['representative_theta_crit_deg'] is None else row['representative_theta_crit_deg']:.2f} | "
            f"{row['adaptive_radius_deg']:.2f} | {row['num_members']} | {surviving} | {row['geometry_interpretation']} |"
        )
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- Treat the fitted amplitude/radius/location as initialization metadata, not posterior constraints.")
    lines.append("- If `geometry_interpretation` contains `screening_only`, do not make a paper claim beyond candidate-screening persistence.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the scale-dependent follow-up table builder."""

    args = parse_args()
    validate_args(args)
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    validate_rows(rows)
    systems = build_system_rows(rows, args)
    summary = system_summary(systems, args)

    json_path = output_dir / "scale_dependent_followup_table.json"
    jsonl_path = output_dir / "scale_dependent_followup_systems.jsonl"
    csv_path = output_dir / "scale_dependent_followup_systems.csv"
    md_path = output_dir / "scale_dependent_followup_table.md"

    json_path.write_text(json.dumps({"summary": summary, "systems": systems}, indent=2), encoding="utf-8")
    write_jsonl(jsonl_path, systems)
    write_csv(csv_path, systems)
    write_markdown(md_path, summary, systems)
    print(
        json.dumps(
            {
                "output_json": str(json_path),
                "output_jsonl": str(jsonl_path),
                "output_csv": str(csv_path),
                "output_markdown": str(md_path),
                "num_systems": summary["num_systems"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
