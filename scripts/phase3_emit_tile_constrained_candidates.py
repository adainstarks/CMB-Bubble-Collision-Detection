"""Emit full-sky candidates for the tile-constrained remediated policy.

Assumptions
-----------
* The input tile score caches were produced from signal-free Planck cleaned-map
  tiles with the same scorer definitions used by the policy search.
* The selected policy row is a candidate-screening rule, not a cosmological
  detection claim or a Bayesian evidence threshold.
* Greedy great-circle clustering is candidate-volume accounting on overlapping
  tiles. It is not an independent-trials p-value.
* Candidate emission defaults to the canonical ``mask_fraction >= 0.9`` science
  footprint even though the Nside=32 tile caches also contain ``>=0.5``
  stress-test tiles.
* Candidate coordinates are peak locations from the ML segmentation peak when
  an ML score participates in the policy margin; otherwise they fall back to
  the tile center for classical-only policies.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from phase3_fullsky_tile import greedy_cluster, peak_sky_coord
from phase3_policy_pareto_search import METHOD_KEYS
from phase3_remediated_policy_tile_audit import (
    ML_METHODS,
    apply_policy,
    best_ml_peak,
    policy_margin,
    slugify,
)
from phase3_tile_constrained_policy_search import (
    DEFAULT_MAPS,
    DEFAULT_TILE_SCORE_TEMPLATE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_tile_constrained_policy_search"
    / "tile_constrained_policy_search.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_tile_constrained_candidates"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Emit candidate and cluster JSONL files for the constrained policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy-json", type=str, default=str(DEFAULT_POLICY_JSON))
    parser.add_argument("--tile-score-template", type=str, default=str(DEFAULT_TILE_SCORE_TEMPLATE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--maps", type=str, default=DEFAULT_MAPS)
    parser.add_argument("--policy-rank", type=int, default=1)
    parser.add_argument("--cluster-radius-deg", type=float, default=15.0)
    parser.add_argument(
        "--min-mask-fraction",
        type=float,
        default=0.9,
        help="Minimum projected common-mask fraction for emitted science candidates.",
    )
    parser.add_argument(
        "--max-tile-candidates-per-map",
        type=int,
        default=0,
        help="Optional rank-score cap for exploratory small outputs. Use 0 to emit all triggers.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths and non-physical arguments."""

    args.maps = tuple(item.strip().lower() for item in str(args.maps).split(",") if item.strip())
    if not args.maps:
        raise ValueError("--maps must contain at least one map.")
    if args.policy_rank <= 0:
        raise ValueError("--policy-rank must be positive.")
    if args.cluster_radius_deg <= 0.0:
        raise ValueError("--cluster-radius-deg must be positive.")
    if not (0.0 <= args.min_mask_fraction <= 1.0):
        raise ValueError("--min-mask-fraction must lie in [0, 1].")
    if args.max_tile_candidates_per_map < 0:
        raise ValueError("--max-tile-candidates-per-map must be non-negative.")
    policy_path = Path(args.policy_json).expanduser()
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing policy JSON: {policy_path}")


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_policy_row(path: Path, rank: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Load one ranked row from the tile-constrained policy-search report."""

    report = load_json(path)
    rows = report.get("top_rows", [])
    if int(rank) > len(rows):
        raise ValueError(f"Requested policy rank {rank}, but {path} contains {len(rows)} rows.")
    row = dict(rows[int(rank) - 1])
    thresholds = row.get("thresholds", {})
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("Selected policy row has no thresholds.")
    unknown = sorted(set(thresholds) - set(METHOD_KEYS))
    if unknown:
        raise ValueError(f"Selected policy row has unsupported methods: {unknown}")
    slug = slugify(f"tile_constrained_rank{rank}_{row.get('family', 'policy')}")
    return slug, row, report


def load_tile_scores(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load cached tile scores and peak-pixel arrays."""

    if not path.exists():
        raise FileNotFoundError(f"Missing tile score cache: {path}")
    with np.load(path) as data:
        scores: dict[str, np.ndarray] = {
            "glon_deg": np.asarray(data["glon_deg"], dtype=np.float64),
            "glat_deg": np.asarray(data["glat_deg"], dtype=np.float64),
            "mask_fraction": np.asarray(data["mask_fraction"], dtype=np.float64),
        }
        peaks: dict[str, np.ndarray] = {}
        for method, key in METHOD_KEYS.items():
            if key not in data.files:
                raise KeyError(f"{path} missing {key}.")
            scores[method] = np.asarray(data[key], dtype=np.float64)
            if method in ML_METHODS:
                peaks[f"peak_i__{method}"] = np.asarray(data[f"peak_i__{method}"], dtype=np.int32)
                peaks[f"peak_j__{method}"] = np.asarray(data[f"peak_j__{method}"], dtype=np.int32)
    n = int(scores["glon_deg"].shape[0])
    for key, values in scores.items():
        if int(values.shape[0]) != n:
            raise ValueError(f"{path} score length mismatch for {key}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains non-finite values for {key}.")
    return scores, peaks


def candidate_records_for_map(
    map_name: str,
    policy_slug: str,
    row: dict[str, Any],
    scores: dict[str, np.ndarray],
    peaks: dict[str, np.ndarray],
    *,
    min_mask_fraction: float,
    max_tile_candidates: int,
) -> list[dict[str, Any]]:
    """Build tile-trigger candidate records for one cleaned map."""

    trigger = apply_policy(row, scores)
    if float(min_mask_fraction) > 0.0:
        trigger &= np.asarray(scores["mask_fraction"], dtype=np.float64) >= float(min_mask_fraction)
    margins = policy_margin(row, scores)
    trigger_indices = np.flatnonzero(trigger)
    if max_tile_candidates:
        order = np.argsort(-margins[trigger_indices], kind="mergesort")[: int(max_tile_candidates)]
        trigger_indices = trigger_indices[order]
    else:
        trigger_indices = trigger_indices[np.argsort(-margins[trigger_indices], kind="mergesort")]

    records: list[dict[str, Any]] = []
    for local_rank, idx in enumerate(trigger_indices, start=1):
        peak_model, peak_i, peak_j = best_ml_peak(int(idx), row, scores, peaks)
        if peak_model is None:
            peak_glon = float(scores["glon_deg"][idx])
            peak_glat = float(scores["glat_deg"][idx])
        else:
            peak_glon, peak_glat = peak_sky_coord(
                float(scores["glon_deg"][idx]),
                float(scores["glat_deg"][idx]),
                int(peak_i),
                int(peak_j),
            )
        record = {
            "map": map_name,
            "source_kind": "planck_cleaned_map_tile",
            "policy_slug": policy_slug,
            "policy": row["policy"],
            "policy_family": row["family"],
            "policy_thresholds": {
                method: float(threshold)
                for method, threshold in row["thresholds"].items()
            },
            "has_candidate": True,
            "local_candidate_rank": int(local_rank),
            "patch_index": int(idx),
            "patch_center_glon_deg": float(scores["glon_deg"][idx]),
            "patch_center_glat_deg": float(scores["glat_deg"][idx]),
            "patch_glon_deg": float(scores["glon_deg"][idx]),
            "patch_glat_deg": float(scores["glat_deg"][idx]),
            "candidate_glon_deg": float(peak_glon),
            "candidate_glat_deg": float(peak_glat),
            "peak_glon_deg": float(peak_glon),
            "peak_glat_deg": float(peak_glat),
            "peak_source_model": peak_model,
            "peak_pixel_i": peak_i,
            "peak_pixel_j": peak_j,
            "rank_score": float(margins[idx]),
            "gbt_score": float(margins[idx]),
            "policy_margin": float(margins[idx]),
            "mask_fraction": float(scores["mask_fraction"][idx]),
            "diagnostic_real_recall": float(row["real_recall"]),
            "diagnostic_real_fpr": float(row["real_fpr"]),
            "pooled_null_fpr_ci_high": float(row["pooled_null_fpr_ci_high"]),
            **{f"score__{method}": float(scores[method][idx]) for method in METHOD_KEYS},
        }
        if row.get("real_recall_gain_vs_best_feasible_single") is not None:
            record["real_recall_gain_vs_best_feasible_single"] = float(
                row["real_recall_gain_vs_best_feasible_single"]
            )
        records.append(record)
    return records


def assign_clusters(
    records: list[dict[str, Any]],
    radius_deg: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign greedy clusters and return cluster records plus representatives."""

    cluster_inputs = [
        {
            "patch_index": int(record["patch_index"]),
            "peak_glon_deg": float(record["peak_glon_deg"]),
            "peak_glat_deg": float(record["peak_glat_deg"]),
            "gbt_score": float(record["policy_margin"]),
        }
        for record in records
    ]
    clusters, assignment = greedy_cluster(cluster_inputs, float(radius_deg))
    by_patch = {int(record["patch_index"]): record for record in records}
    representatives: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_id = int(cluster["cluster_id"])
        for idx, assigned_id in enumerate(assignment):
            if int(assigned_id) == cluster_id:
                records[idx]["cluster_id"] = cluster_id
                records[idx]["cluster_radius_deg"] = float(radius_deg)
                records[idx][f"cluster_id_{float(radius_deg):g}deg"] = cluster_id
        seed = by_patch[int(cluster["seed_patch_index"])]
        rep = {
            **seed,
            "cluster_id": cluster_id,
            "cluster_radius_deg": float(radius_deg),
            "cluster_n_members": int(cluster["n_members"]),
            "cluster_max_policy_margin": float(cluster["max_gbt_score"]),
            "cluster_member_patches": [int(value) for value in cluster["member_patches"]],
        }
        representatives.append(rep)
    return clusters, representatives


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact candidate CSV."""

    columns = [
        "global_candidate_rank",
        "map",
        "patch_index",
        "patch_glon_deg",
        "patch_glat_deg",
        "candidate_glon_deg",
        "candidate_glat_deg",
        "policy_margin",
        "mask_fraction",
        "cluster_id",
        "cluster_n_members",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a human-readable emission report."""

    lines = ["# Tile-Constrained Candidate Emission", ""]
    lines.append("This freezes the current constrained policy into downstream candidate files.")
    lines.append("It is a screening output, not a detection claim.")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append(f"- `policy_slug`: `{report['policy_slug']}`")
    lines.append(f"- `policy`: `{report['policy']['policy']}`")
    lines.append(f"- `family`: `{report['policy']['family']}`")
    lines.append(f"- `thresholds`: `{report['policy']['thresholds']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| map | eligible tiles | tile candidates | eligible trigger frac | cluster reps | max cluster size |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report["map_rows"]:
        lines.append(
            f"| {row['map']} | {row['num_tiles_passing_mask_fraction']} | "
            f"{row['num_tile_candidates']} | {row['eligible_trigger_fraction']:.4f} | "
            f"{row['num_cluster_representatives']} | {row['max_cluster_size']} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in report["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_json = Path(args.policy_json).expanduser().resolve()
    policy_slug, policy_row, policy_report = load_policy_row(policy_json, int(args.policy_rank))

    all_candidates: list[dict[str, Any]] = []
    all_representatives: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    for map_name in args.maps:
        score_path = Path(str(args.tile_score_template).format(map=map_name)).expanduser().resolve()
        scores, peaks = load_tile_scores(score_path)
        records = candidate_records_for_map(
            map_name,
            policy_slug,
            policy_row,
            scores,
            peaks,
            max_tile_candidates=int(args.max_tile_candidates_per_map),
            min_mask_fraction=float(args.min_mask_fraction),
        )
        clusters, representatives = assign_clusters(records, float(args.cluster_radius_deg))
        map_dir = output_dir / map_name
        candidate_path = map_dir / "candidate_records.jsonl"
        cluster_path = map_dir / f"clusters_{int(args.cluster_radius_deg)}deg.jsonl"
        representative_path = map_dir / f"cluster_representatives_{int(args.cluster_radius_deg)}deg.jsonl"
        write_jsonl(candidate_path, records)
        write_jsonl(cluster_path, clusters)
        write_jsonl(representative_path, representatives)
        max_cluster_size = max((int(cluster["n_members"]) for cluster in clusters), default=0)
        map_rows.append(
            {
                "map": map_name,
                "tile_score_cache": str(score_path),
                "candidate_jsonl": str(candidate_path),
                "cluster_jsonl": str(cluster_path),
                "cluster_representative_jsonl": str(representative_path),
                "num_tiles": int(scores["glon_deg"].shape[0]),
                "num_tiles_passing_mask_fraction": int(
                    np.count_nonzero(np.asarray(scores["mask_fraction"]) >= float(args.min_mask_fraction))
                ),
                "num_tile_candidates": int(len(records)),
                "trigger_fraction": float(len(records) / max(int(scores["glon_deg"].shape[0]), 1)),
                "eligible_trigger_fraction": float(
                    len(records)
                    / max(
                        int(
                            np.count_nonzero(
                                np.asarray(scores["mask_fraction"]) >= float(args.min_mask_fraction)
                            )
                        ),
                        1,
                    )
                ),
                "num_cluster_representatives": int(len(representatives)),
                "max_cluster_size": int(max_cluster_size),
            }
        )
        all_candidates.extend(records)
        all_representatives.extend(representatives)

    all_candidates.sort(key=lambda record: float(record["policy_margin"]), reverse=True)
    all_representatives.sort(key=lambda record: float(record["cluster_max_policy_margin"]), reverse=True)
    for rank, record in enumerate(all_candidates, start=1):
        record["global_candidate_rank"] = int(rank)
    for rank, record in enumerate(all_representatives, start=1):
        record["global_cluster_rank"] = int(rank)

    global_candidates_path = output_dir / "candidate_records.jsonl"
    global_representatives_path = (
        output_dir / f"cluster_representatives_{int(args.cluster_radius_deg)}deg.jsonl"
    )
    global_csv_path = output_dir / "candidate_records.csv"
    representative_csv_path = output_dir / f"cluster_representatives_{int(args.cluster_radius_deg)}deg.csv"
    write_jsonl(global_candidates_path, all_candidates)
    write_jsonl(global_representatives_path, all_representatives)
    write_candidate_csv(global_csv_path, all_candidates)
    write_candidate_csv(representative_csv_path, all_representatives)

    report = {
        "policy_json": str(policy_json),
        "policy_rank": int(args.policy_rank),
        "policy_slug": policy_slug,
        "policy": {
            "policy": policy_row["policy"],
            "family": policy_row["family"],
            "thresholds": policy_row["thresholds"],
            "real_recall": policy_row.get("real_recall"),
            "real_fpr": policy_row.get("real_fpr"),
            "pooled_null_fpr_ci_high": policy_row.get("pooled_null_fpr_ci_high"),
            "real_recall_gain_vs_best_feasible_single": policy_row.get(
                "real_recall_gain_vs_best_feasible_single"
            ),
        },
        "constraints": policy_report.get("constraints", {}),
        "cluster_radius_deg": float(args.cluster_radius_deg),
        "min_mask_fraction": float(args.min_mask_fraction),
        "max_tile_candidates_per_map": int(args.max_tile_candidates_per_map),
        "map_rows": map_rows,
        "totals": {
            "num_tile_candidates": int(len(all_candidates)),
            "num_cluster_representatives": int(len(all_representatives)),
            "maps": list(args.maps),
        },
        "artifacts": {
            "candidate_records_jsonl": str(global_candidates_path),
            "candidate_records_csv": str(global_csv_path),
            "cluster_representatives_jsonl": str(global_representatives_path),
            "cluster_representatives_csv": str(representative_csv_path),
        },
        "assumption_notes": [
            "Candidate rows are emitted from cached full-sky tile scores.",
            "Cluster representatives are highest-policy-margin seeds from greedy clustering.",
            "Use cluster representatives, not all overlapping tile rows, for first-pass follow-up volume.",
        ],
    }
    json_path = output_dir / "candidate_emission_summary.json"
    md_path = output_dir / "candidate_emission_summary.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
