"""Calibrate emitted candidate margins against real-map null controls.

Assumptions
-----------
* Calibration uses the real-map null-control calibration split only; the
  held-out test split remains reserved for reporting false-positive burden.
* The calibrated statistic is the frozen policy margin from the current
  candidate-screening rule. It is not a posterior probability, Bayesian
  evidence ratio, or global LambdaCDM p-value.
* Empirical survival probabilities are computed with a plus-one correction:
  ``p = (1 + #{null margin >= candidate margin}) / (1 + N_null)``.
* Benjamini-Hochberg q-values are descriptive multiple-candidate screening
  metadata. Correlations from overlapping tiles and CMB structure mean they
  should be treated conservatively.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from phase3_policy_pareto_search import METHOD_KEYS
from phase3_remediated_policy_tile_audit import apply_policy, policy_margin
from phase5_half_mission_signflip_null import load_policy_rows, policy_slug


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_tile_constrained_policy_search"
    / "tile_constrained_policy_search.json"
)
DEFAULT_CANDIDATE_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_tile_constrained_candidates"
    / "cluster_representatives_15deg.jsonl"
)
DEFAULT_NULL_SCORE_TEMPLATE = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_null_policy_calibration"
    / "null_policy_scores_{map}_mask090_calibration.npz"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_candidate_score_calibration"
)
DEFAULT_MAPS = "smica,nilc,sevem,commander"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Assign calibration-split empirical null-survival scores to emitted candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy-json", type=str, default=str(DEFAULT_POLICY_JSON))
    parser.add_argument("--candidate-jsonl", type=str, default=str(DEFAULT_CANDIDATE_JSONL))
    parser.add_argument("--null-score-template", type=str, default=str(DEFAULT_NULL_SCORE_TEMPLATE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--maps", type=str, default=DEFAULT_MAPS)
    parser.add_argument("--policy-slug", type=str, default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths and map list."""

    args.maps = tuple(item.strip().lower() for item in str(args.maps).split(",") if item.strip())
    if not args.maps:
        raise ValueError("--maps must contain at least one map.")
    for path_text, label in ((args.policy_json, "policy JSON"), (args.candidate_jsonl, "candidate JSONL")):
        path = Path(path_text).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL candidate records."""

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row.setdefault("candidate_jsonl", str(path))
            row.setdefault("candidate_jsonl_line", int(line_number))
            rows.append(row)
    if not rows:
        raise ValueError(f"No candidates found in {path}.")
    return rows


def select_policy(path: Path, slug_override: str, candidates: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Select the policy row referenced by the candidates."""

    rows = load_policy_rows(path)
    by_slug = {policy_slug(row): row for row in rows}
    slug = slug_override or str(candidates[0].get("policy_slug", ""))
    if not slug and len(by_slug) == 1:
        slug = next(iter(by_slug))
    if slug not in by_slug:
        raise KeyError(f"Could not resolve policy slug {slug!r}; available={sorted(by_slug)}")
    for row in candidates:
        candidate_slug = str(row.get("policy_slug", slug))
        if candidate_slug != slug:
            raise ValueError(
                f"Candidate file mixes policy slugs: expected {slug!r}, found {candidate_slug!r}."
            )
    return slug, by_slug[slug]


def load_null_scores(path: Path) -> dict[str, np.ndarray]:
    """Load one calibration split score cache."""

    if not path.exists():
        raise FileNotFoundError(f"Missing calibration score cache: {path}")
    with np.load(path) as data:
        scores = {}
        for method, key in METHOD_KEYS.items():
            if key not in data.files:
                raise KeyError(f"{path} missing {key}.")
            scores[method] = np.asarray(data[key], dtype=np.float64)
    n = len(next(iter(scores.values())))
    for method, values in scores.items():
        if len(values) != n:
            raise ValueError(f"{path} length mismatch for {method}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains non-finite {method} scores.")
    return scores


def empirical_survival(value: float, null_values: np.ndarray) -> tuple[float, int, int]:
    """Return plus-one empirical survival probability."""

    values = np.asarray(null_values, dtype=np.float64)
    n = int(values.shape[0])
    exceed = int(np.count_nonzero(values >= float(value)))
    return float((1 + exceed) / (1 + n)), exceed, n


def bh_q_values(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values."""

    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    n = p.size
    adjusted_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    out = np.empty_like(adjusted_sorted)
    out[order] = adjusted_sorted
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact calibrated candidate CSV."""

    columns = [
        "global_cluster_rank",
        "global_candidate_rank",
        "map",
        "cluster_id",
        "patch_index",
        "candidate_glon_deg",
        "candidate_glat_deg",
        "policy_margin",
        "calibration_map_survival_p",
        "calibration_pooled_survival_p",
        "calibration_pooled_bh_q",
        "calibration_map_exceedance_count",
        "calibration_map_null_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a human-readable calibration report."""

    lines = ["# Candidate Score Calibration", ""]
    lines.append("Empirical null-survival scores are calibrated on real-map null-control calibration splits.")
    lines.append("The held-out test split is not used for candidate score calibration.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- `policy_slug`: `{report['policy_slug']}`")
    lines.append(f"- `num_candidates`: `{report['num_candidates']}`")
    lines.append(f"- `pooled_null_count`: `{report['pooled_null_count']}`")
    lines.append("")
    lines.append("| map | null count | policy pass frac | candidates | min pooled p | min q |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report["map_rows"]:
        lines.append(
            f"| {row['map']} | {row['null_count']} | {row['policy_pass_fraction']:.4f} | "
            f"{row['candidate_count']} | {row['min_pooled_survival_p']:.4f} | "
            f"{row['min_pooled_bh_q']:.4f} |"
        )
    lines.append("")
    lines.append("## Top Calibrated Candidates")
    lines.append("")
    lines.append("| rank | map | cluster | margin | pooled p | q |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for idx, row in enumerate(report["top_candidates"], start=1):
        lines.append(
            f"| {idx} | {row.get('map')} | {row.get('cluster_id')} | "
            f"{row.get('policy_margin'):.4f} | {row.get('calibration_pooled_survival_p'):.4f} | "
            f"{row.get('calibration_pooled_bh_q'):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(Path(args.candidate_jsonl).expanduser().resolve())
    selected_slug, policy_row = select_policy(
        Path(args.policy_json).expanduser().resolve(),
        str(args.policy_slug),
        candidates,
    )

    null_by_map = {}
    margin_by_map = {}
    pass_by_map = {}
    for map_name in args.maps:
        path = Path(str(args.null_score_template).format(map=map_name)).expanduser().resolve()
        scores = load_null_scores(path)
        null_by_map[map_name] = scores
        margin_by_map[map_name] = policy_margin(policy_row, scores)
        pass_by_map[map_name] = apply_policy(policy_row, scores)
    pooled_margin = np.concatenate([margin_by_map[map_name] for map_name in args.maps])

    calibrated = []
    for row in candidates:
        map_name = str(row.get("map", "")).lower()
        if map_name not in margin_by_map:
            raise KeyError(f"Candidate references map {map_name!r}; expected one of {args.maps}.")
        out = dict(row)
        margin = float(out["policy_margin"])
        map_p, map_exceed, map_n = empirical_survival(margin, margin_by_map[map_name])
        pooled_p, pooled_exceed, pooled_n = empirical_survival(margin, pooled_margin)
        out.update(
            {
                "calibration_policy_slug": selected_slug,
                "calibration_map_survival_p": map_p,
                "calibration_map_exceedance_count": int(map_exceed),
                "calibration_map_null_count": int(map_n),
                "calibration_pooled_survival_p": pooled_p,
                "calibration_pooled_exceedance_count": int(pooled_exceed),
                "calibration_pooled_null_count": int(pooled_n),
            }
        )
        calibrated.append(out)

    q_values = bh_q_values(np.asarray([row["calibration_pooled_survival_p"] for row in calibrated]))
    for row, q_value in zip(calibrated, q_values):
        row["calibration_pooled_bh_q"] = float(q_value)
    calibrated.sort(
        key=lambda row: (
            row["calibration_pooled_survival_p"],
            row["calibration_pooled_bh_q"],
            -float(row["policy_margin"]),
        )
    )
    for rank, row in enumerate(calibrated, start=1):
        row["calibrated_rank"] = int(rank)

    map_rows = []
    for map_name in args.maps:
        subset = [row for row in calibrated if str(row.get("map", "")).lower() == map_name]
        map_rows.append(
            {
                "map": map_name,
                "null_count": int(margin_by_map[map_name].shape[0]),
                "policy_pass_count": int(np.count_nonzero(pass_by_map[map_name])),
                "policy_pass_fraction": float(np.mean(pass_by_map[map_name])),
                "candidate_count": int(len(subset)),
                "min_pooled_survival_p": float(
                    min((row["calibration_pooled_survival_p"] for row in subset), default=np.nan)
                ),
                "min_pooled_bh_q": float(
                    min((row["calibration_pooled_bh_q"] for row in subset), default=np.nan)
                ),
            }
        )

    jsonl_path = output_dir / "calibrated_candidates.jsonl"
    csv_path = output_dir / "calibrated_candidates.csv"
    json_path = output_dir / "candidate_score_calibration.json"
    md_path = output_dir / "candidate_score_calibration.md"
    write_jsonl(jsonl_path, calibrated)
    write_csv(csv_path, calibrated)
    report = {
        "policy_json": str(Path(args.policy_json).expanduser().resolve()),
        "candidate_jsonl": str(Path(args.candidate_jsonl).expanduser().resolve()),
        "null_score_template": str(args.null_score_template),
        "policy_slug": selected_slug,
        "policy": {
            "policy": policy_row.get("policy"),
            "family": policy_row.get("family"),
            "thresholds": policy_row.get("thresholds"),
        },
        "num_candidates": int(len(calibrated)),
        "pooled_null_count": int(pooled_margin.shape[0]),
        "map_rows": map_rows,
        "top_candidates": calibrated[:10],
        "artifacts": {
            "calibrated_candidates_jsonl": str(jsonl_path),
            "calibrated_candidates_csv": str(csv_path),
        },
        "assumption_notes": [
            "Calibration uses null-control calibration split scores, not held-out test scores.",
            "Survival scores are empirical null-tail probabilities for the frozen policy margin.",
            "BH q-values are screening metadata for the emitted candidate set.",
        ],
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
