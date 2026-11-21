"""Recommend an operational cluster radius for frozen screening candidates.

Assumptions
-----------
* This is candidate-volume accounting for screened Planck cleaned-map tiles.
  It is not a detection statistic or an independent-trials correction.
* The input candidate JSONL is assumed to be the frozen science-footprint
  emission from the current deployment-safe policy.
* Greedy great-circle clustering follows the same operational convention as
  ``phase3_fullsky_tile.py``. The recommended radius is therefore a follow-up
  workload convention, not a physical source-association law.
* McEwen et al. (2012, Phys. Rev. D 85, 103502) use scale-aware elimination in
  their blob-detection pipeline. This script does not infer a physical match
  scale; it records a reproducible fixed-radius compromise for the current
  emitted candidate set.
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from phase3_fullsky_tile import greedy_cluster


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_tile_constrained_candidates"
    / "candidate_records.jsonl"
)
DEFAULT_BAYESIAN_HANDOFF_JSONL = (
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
    / "remediated_v1_cluster_radius_recommendation"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Recommend a paper-facing cluster radius for frozen candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-jsonl", type=str, default=str(DEFAULT_CANDIDATE_JSONL))
    parser.add_argument(
        "--bayesian-handoff-jsonl",
        type=str,
        default=str(DEFAULT_BAYESIAN_HANDOFF_JSONL),
        help="Optional candidate-level follow-up packet used only for theta/offset context.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cluster-radii-deg", type=str, default="5,10,15,20,25")
    parser.add_argument(
        "--min-global-reduction-factor",
        type=float,
        default=4.0,
        help="Require at least raw_candidates / clusters >= this value.",
    )
    parser.add_argument(
        "--max-global-cluster-members",
        type=int,
        default=15,
        help="Avoid radii that over-merge the frozen candidate set beyond this member cap.",
    )
    return parser.parse_args()


def parse_radii(text: str) -> np.ndarray:
    """Parse a comma-separated radius list into finite float64 degrees."""

    radii = np.asarray(
        [float(part.strip()) for part in str(text).split(",") if part.strip()],
        dtype=np.float64,
    )
    if radii.size == 0:
        raise ValueError("--cluster-radii-deg must contain at least one radius.")
    if not np.all(np.isfinite(radii)):
        raise ValueError("--cluster-radii-deg contains non-finite values.")
    if np.any(radii <= 0.0) or np.any(radii >= 180.0):
        raise ValueError("Cluster radii must lie in the open interval (0, 180) deg.")
    if np.unique(radii).size != radii.size:
        raise ValueError("--cluster-radii-deg must not contain duplicates.")
    return np.sort(radii)


def validate_args(args: argparse.Namespace) -> None:
    """Validate required inputs and operational thresholds."""

    args.cluster_radii_deg = parse_radii(args.cluster_radii_deg)
    candidate_path = Path(args.candidate_jsonl).expanduser()
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing candidate JSONL: {candidate_path}")
    if not np.isfinite(float(args.min_global_reduction_factor)):
        raise ValueError("--min-global-reduction-factor must be finite.")
    if float(args.min_global_reduction_factor) <= 0.0:
        raise ValueError("--min-global-reduction-factor must be positive.")
    if int(args.max_global_cluster_members) <= 0:
        raise ValueError("--max-global-cluster-members must be positive.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"No candidate rows found in {path}.")
    return rows


def validate_candidate_rows(rows: list[dict[str, Any]]) -> None:
    """Raise if candidate rows are malformed or non-finite."""

    required = (
        "map",
        "patch_index",
        "peak_glon_deg",
        "peak_glat_deg",
        "policy_margin",
    )
    for idx, row in enumerate(rows, start=1):
        for key in required:
            if key not in row:
                raise KeyError(f"Candidate row {idx} is missing required field {key!r}.")
        for key in ("peak_glon_deg", "peak_glat_deg", "policy_margin"):
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"Candidate row {idx} has non-finite {key}.")
        glat = float(row["peak_glat_deg"])
        if glat < -90.0 or glat > 90.0:
            raise ValueError(f"Candidate row {idx} has non-physical latitude {glat:.6f} deg.")


def cluster_records(records: list[dict[str, Any]], radius_deg: float) -> list[dict[str, Any]]:
    """Cluster one map's records with the shared greedy sky-linkage rule."""

    cluster_inputs = [
        {
            "patch_index": int(record["patch_index"]),
            "peak_glon_deg": float(record["peak_glon_deg"]),
            "peak_glat_deg": float(record["peak_glat_deg"]),
            "gbt_score": float(record["policy_margin"]),
        }
        for record in records
    ]
    clusters, _assignment = greedy_cluster(cluster_inputs, float(radius_deg))
    return clusters


def load_followup_context(path: Path | None) -> dict[str, Any]:
    """Load optional theta/offset context from the Bayesian handoff packet."""

    if path is None or not path.exists():
        return {"available": False}
    rows = load_jsonl(path)
    theta = np.asarray(
        [
            float(row["template_seed"]["theta_crit_deg"])
            for row in rows
            if row.get("template_seed", {}).get("theta_crit_deg") is not None
        ],
        dtype=np.float64,
    )
    offset = np.asarray(
        [
            float(row["template_seed"]["candidate_offset_deg"])
            for row in rows
            if row.get("template_seed", {}).get("candidate_offset_deg") is not None
        ],
        dtype=np.float64,
    )
    caution_flags = np.asarray(
        [bool(row.get("projection_systematics_caution", False)) for row in rows],
        dtype=bool,
    )
    out: dict[str, Any] = {
        "available": True,
        "num_rows": int(len(rows)),
        "num_projection_cautions": int(np.count_nonzero(caution_flags)),
    }
    if theta.size:
        out.update(
            {
                "theta_crit_deg_median": float(np.median(theta)),
                "theta_crit_deg_p90": float(np.quantile(theta, 0.9)),
                "theta_crit_deg_max": float(np.max(theta)),
            }
        )
    if offset.size:
        out.update(
            {
                "candidate_offset_deg_median": float(np.median(offset)),
                "candidate_offset_deg_p90": float(np.quantile(offset, 0.9)),
                "candidate_offset_deg_max": float(np.max(offset)),
            }
        )
    return out


def summarize_radius_tradeoffs(
    candidate_rows: list[dict[str, Any]],
    radii_deg: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-map and global tradeoff rows for each cluster radius."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["map"]).lower()].append(row)

    per_map_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    total_candidates = int(len(candidate_rows))
    for radius in radii_deg:
        radius_quantity = float((float(radius) * u.deg).to_value(u.deg))
        total_clusters = 0
        max_cluster_size = 0
        mean_cluster_sizes: list[float] = []
        for map_name in sorted(grouped):
            clusters = cluster_records(grouped[map_name], radius_quantity)
            n_clusters = int(len(clusters))
            cluster_sizes = np.asarray(
                [int(cluster["n_members"]) for cluster in clusters],
                dtype=np.int32,
            )
            total_clusters += n_clusters
            max_cluster_size = max(
                max_cluster_size,
                int(np.max(cluster_sizes)) if cluster_sizes.size else 0,
            )
            mean_size = float(np.mean(cluster_sizes)) if cluster_sizes.size else 0.0
            mean_cluster_sizes.append(mean_size)
            per_map_rows.append(
                {
                    "cluster_radius_deg": radius_quantity,
                    "map": map_name,
                    "num_candidates": int(len(grouped[map_name])),
                    "num_clusters": n_clusters,
                    "reduction_factor": float(len(grouped[map_name]) / max(n_clusters, 1)),
                    "max_cluster_size": int(np.max(cluster_sizes)) if cluster_sizes.size else 0,
                    "mean_cluster_size": mean_size,
                }
            )
        global_rows.append(
            {
                "cluster_radius_deg": radius_quantity,
                "num_candidates": total_candidates,
                "num_clusters": int(total_clusters),
                "reduction_factor": float(total_candidates / max(total_clusters, 1)),
                "max_cluster_size": int(max_cluster_size),
                "mean_map_cluster_size": float(np.mean(mean_cluster_sizes)) if mean_cluster_sizes else 0.0,
            }
        )
    return per_map_rows, global_rows


def recommend_radius(
    global_rows: list[dict[str, Any]],
    *,
    min_reduction_factor: float,
    max_cluster_members: int,
) -> tuple[float, dict[str, Any]]:
    """Select the smallest operationally acceptable radius and record why."""

    if not global_rows:
        raise ValueError("No global rows available for cluster-radius recommendation.")

    qualified = [
        row
        for row in global_rows
        if float(row["reduction_factor"]) >= float(min_reduction_factor)
        and int(row["max_cluster_size"]) <= int(max_cluster_members)
    ]
    if qualified:
        chosen = qualified[0]
        return float(chosen["cluster_radius_deg"]), {
            "status": "qualified",
            "rule": (
                "Choose the smallest radius that reaches the requested global "
                "candidate-reduction factor without exceeding the over-merge cap."
            ),
        }

    chosen = sorted(
        global_rows,
        key=lambda row: (
            abs(float(row["reduction_factor"]) - float(min_reduction_factor)),
            int(row["max_cluster_size"]),
            float(row["cluster_radius_deg"]),
        ),
    )[0]
    return float(chosen["cluster_radius_deg"]), {
        "status": "fallback",
        "rule": (
            "No radius met both operational thresholds, so choose the closest "
            "reduction target with the smallest over-merge burden."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a flat CSV table."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plot(path: Path, global_rows: list[dict[str, Any]], recommended_radius_deg: float) -> None:
    """Write a compact publication-style tradeoff plot."""

    radii = np.asarray([float(row["cluster_radius_deg"]) for row in global_rows], dtype=np.float64)
    clusters = np.asarray([int(row["num_clusters"]) for row in global_rows], dtype=np.float64)
    max_members = np.asarray([int(row["max_cluster_size"]) for row in global_rows], dtype=np.float64)

    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "savefig.dpi": 180,
        }
    )
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot(radii, clusters, marker="o", color="#1f3b5c", linewidth=2.0, label="Cluster representatives")
    ax_right.plot(radii, max_members, marker="s", color="#c15a2b", linewidth=2.0, label="Max cluster size")
    ax_left.axvline(float(recommended_radius_deg), color="#2d7d46", linestyle="--", linewidth=1.5)
    ax_left.set_xlabel("Cluster radius [deg]")
    ax_left.set_ylabel("Total cluster representatives")
    ax_right.set_ylabel("Largest merged cluster [members]")
    ax_left.set_title("Frozen Candidate Burden vs Cluster Radius")

    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(left_handles + right_handles, left_labels + right_labels, loc="upper right", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a human-readable recommendation summary."""

    lines = ["# Cluster Radius Recommendation", ""]
    lines.append("## Assumptions")
    lines.append("")
    for note in report["assumption_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"- Recommended fixed cluster radius: `{report['recommended_cluster_radius_deg']:.1f} deg`"
    )
    lines.append(f"- Recommendation status: `{report['recommendation_metadata']['status']}`")
    lines.append(f"- Decision rule: {report['recommendation_metadata']['rule']}")
    lines.append(
        f"- Operational thresholds: reduction factor `>= {report['selection_rule']['min_global_reduction_factor']:.2f}`, "
        f"max cluster members `<= {report['selection_rule']['max_global_cluster_members']}`"
    )
    if report["followup_context"].get("available"):
        lines.append(
            f"- Follow-up context: median fitted `theta_crit = {report['followup_context'].get('theta_crit_deg_median', 0.0):.1f} deg`, "
            f"median candidate offset `{report['followup_context'].get('candidate_offset_deg_median', 0.0):.1f} deg`, "
            f"`{report['followup_context'].get('num_projection_cautions', 0)}` projection-caution representatives"
        )
    lines.append("")
    lines.append("## Global Tradeoff")
    lines.append("")
    lines.append("| radius deg | raw candidates | cluster reps | reduction x | max cluster size |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in report["global_rows"]:
        lines.append(
            f"| {row['cluster_radius_deg']:.1f} | {row['num_candidates']} | {row['num_clusters']} | "
            f"{row['reduction_factor']:.2f} | {row['max_cluster_size']} |"
        )
    lines.append("")
    lines.append("## Per-Map Tradeoff")
    lines.append("")
    lines.append("| radius deg | map | raw candidates | cluster reps | reduction x | max cluster size |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for row in report["per_map_rows"]:
        lines.append(
            f"| {row['cluster_radius_deg']:.1f} | {row['map']} | {row['num_candidates']} | "
            f"{row['num_clusters']} | {row['reduction_factor']:.2f} | {row['max_cluster_size']} |"
        )
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- This is an operational follow-up convention for screened candidates, not a physical source-association scale.")
    lines.append("- Keep the fixed radius aligned across candidate-burden tables and follow-up manifests unless a later scale-dependent study replaces it.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the frozen-candidate cluster-radius recommendation."""

    args = parse_args()
    validate_args(args)

    candidate_path = Path(args.candidate_jsonl).expanduser().resolve()
    bayesian_path = Path(args.bayesian_handoff_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = load_jsonl(candidate_path)
    validate_candidate_rows(candidate_rows)
    followup_context = load_followup_context(bayesian_path if bayesian_path.exists() else None)
    per_map_rows, global_rows = summarize_radius_tradeoffs(candidate_rows, args.cluster_radii_deg)
    recommended_radius_deg, recommendation_metadata = recommend_radius(
        global_rows,
        min_reduction_factor=float(args.min_global_reduction_factor),
        max_cluster_members=int(args.max_global_cluster_members),
    )

    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "candidate_jsonl": str(candidate_path),
        "bayesian_handoff_jsonl": str(bayesian_path) if bayesian_path.exists() else "",
        "selection_rule": {
            "min_global_reduction_factor": float(args.min_global_reduction_factor),
            "max_global_cluster_members": int(args.max_global_cluster_members),
        },
        "recommended_cluster_radius_deg": float(recommended_radius_deg),
        "recommendation_metadata": recommendation_metadata,
        "followup_context": followup_context,
        "global_rows": global_rows,
        "per_map_rows": per_map_rows,
        "assumption_notes": [
            "The input candidate list is the frozen science-footprint emission from the deployment-safe composite policy.",
            "Cluster representatives are workload/accounting units for downstream follow-up, not statistical trials.",
            "The recommendation is intentionally conservative: it picks the smallest radius that reaches the requested burden reduction without large over-merging.",
        ],
    }

    json_path = output_dir / "cluster_radius_recommendation.json"
    md_path = output_dir / "cluster_radius_recommendation.md"
    global_csv = output_dir / "cluster_radius_global_tradeoff.csv"
    map_csv = output_dir / "cluster_radius_per_map_tradeoff.csv"
    plot_path = output_dir / "cluster_radius_tradeoff.png"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    write_csv(global_csv, global_rows)
    write_csv(map_csv, per_map_rows)
    make_plot(plot_path, global_rows, float(recommended_radius_deg))

    print(
        json.dumps(
            {
                "output_json": str(json_path),
                "output_markdown": str(md_path),
                "recommended_cluster_radius_deg": float(recommended_radius_deg),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
