"""Search policies under synthetic, real-null, and full-sky tile constraints.

Assumptions
-----------
* This is a candidate-screening policy search. It does not train models and
  does not create a cosmological detection claim.
* CAMB and real-injection recalls are diagnostic estimates from existing
  score caches. The full-sky tile score caches are deployment-burden stress
  tests on overlapping gnomonic patches.
* A policy is deployable only if it obeys both patch-level false-positive
  limits and cross-map tile/cluster burden limits.
* Cluster counts use the same greedy great-circle linkage used by
  ``phase3_remediated_policy_tile_audit.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from phase2_signal_model import PATCH_PIX, RESO_ARCMIN
from phase3_policy_pareto_search import (
    METHOD_KEYS,
    binomial_ci,
    build_policy_rows,
    load_scores,
    metric_row,
)
from phase3_remediated_policy_tile_audit import (
    ML_METHODS,
    apply_policy,
    policy_margin,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENSITIVITY_SCORES = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_sensitivity_curve"
    / "sensitivity_scores.npz"
)
DEFAULT_REAL_SKY_SCORES = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_real_sky_injection_smica_mask090"
    / "real_sky_scores.npz"
)
DEFAULT_TILE_SCORE_TEMPLATE = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_policy_tile_audit"
    / "tile_scores_{map}_nside32.npz"
)
DEFAULT_NULL_SCORE_TEMPLATE = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_null_policy_audit"
    / "null_policy_scores_{map}_mask090_test.npz"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_tile_constrained_policy_search"
)
DEFAULT_MAPS = "smica,nilc,sevem,commander"


PATCH_CENTER = (PATCH_PIX - 1) / 2.0
PATCH_RESO_RAD = np.radians(RESO_ARCMIN / 60.0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Search score policies that satisfy full-sky tile burden constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sensitivity-scores", type=str, default=str(DEFAULT_SENSITIVITY_SCORES))
    parser.add_argument("--real-sky-scores", type=str, default=str(DEFAULT_REAL_SKY_SCORES))
    parser.add_argument("--tile-score-template", type=str, default=str(DEFAULT_TILE_SCORE_TEMPLATE))
    parser.add_argument("--null-score-template", type=str, default=str(DEFAULT_NULL_SCORE_TEMPLATE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--maps", type=str, default=DEFAULT_MAPS)
    parser.add_argument("--max-camb-fpr", type=float, default=0.08)
    parser.add_argument("--max-real-fpr", type=float, default=0.08)
    parser.add_argument("--max-pooled-null-fpr-ci-high", type=float, default=0.04)
    parser.add_argument("--max-trigger-fraction-any-map", type=float, default=0.15)
    parser.add_argument("--max-clusters-any-map", type=int, default=70)
    parser.add_argument("--cluster-radius-deg", type=float, default=15.0)
    parser.add_argument("--num-quantiles", type=int, default=52)
    parser.add_argument("--kof3-step", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--max-cluster-evals",
        type=int,
        default=0,
        help=(
            "Maximum number of recall-ranked policies to evaluate with full greedy clustering. "
            "Use 0 for an exhaustive run; positive values are for exploratory smoke runs."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""

    args.maps = tuple(item.strip().lower() for item in str(args.maps).split(",") if item.strip())
    if not args.maps:
        raise ValueError("--maps must contain at least one map.")
    for name in ("max_camb_fpr", "max_real_fpr", "max_pooled_null_fpr_ci_high", "max_trigger_fraction_any_map"):
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0, 1].")
    if args.max_clusters_any_map < 0:
        raise ValueError("--max-clusters-any-map must be non-negative.")
    if args.cluster_radius_deg <= 0.0:
        raise ValueError("--cluster-radius-deg must be positive.")
    if args.num_quantiles < 8:
        raise ValueError("--num-quantiles must be at least 8.")
    if args.kof3_step <= 0:
        raise ValueError("--kof3-step must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.max_cluster_evals < 0:
        raise ValueError("--max-cluster-evals must be non-negative.")
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must lie in (0, 1).")
    for path_text in (args.sensitivity_scores, args.real_sky_scores):
        if not Path(path_text).expanduser().exists():
            raise FileNotFoundError(f"Missing score file: {path_text}")


def load_npz_scores(path: Path) -> dict[str, np.ndarray]:
    """Load method-keyed score arrays from a cache file."""

    with np.load(path) as data:
        out = {}
        for method in METHOD_KEYS:
            key = METHOD_KEYS[method]
            if key not in data.files:
                raise KeyError(f"{path} missing {key}.")
            out[method] = np.asarray(data[key], dtype=np.float64)
    return out


def load_tile_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load one map's full-sky tile scores and peak arrays."""

    if not path.exists():
        raise FileNotFoundError(f"Missing tile score cache: {path}")
    with np.load(path) as data:
        scores: dict[str, np.ndarray] = {
            "glon_deg": np.asarray(data["glon_deg"], dtype=np.float64),
            "glat_deg": np.asarray(data["glat_deg"], dtype=np.float64),
            "mask_fraction": np.asarray(data["mask_fraction"], dtype=np.float64),
        }
        peaks = {}
        for method in METHOD_KEYS:
            score_key = METHOD_KEYS[method]
            if score_key not in data.files:
                raise KeyError(f"{path} missing {score_key}.")
            scores[method] = np.asarray(data[score_key], dtype=np.float64)
            if method in ML_METHODS:
                peaks[f"peak_i__{method}"] = np.asarray(data[f"peak_i__{method}"], dtype=np.int32)
                peaks[f"peak_j__{method}"] = np.asarray(data[f"peak_j__{method}"], dtype=np.int32)
    return scores, peaks


def load_all_tiles(args: argparse.Namespace) -> dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Load tile score caches for every requested map."""

    out = {}
    for map_name in args.maps:
        path = Path(str(args.tile_score_template).format(map=map_name)).expanduser().resolve()
        out[map_name] = load_tile_cache(path)
    return out


def load_pooled_null_scores(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Load and concatenate held-out real-null policy score caches."""

    chunks = {method: [] for method in METHOD_KEYS}
    for map_name in args.maps:
        path = Path(str(args.null_score_template).format(map=map_name)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing null score cache: {path}")
        scores = load_npz_scores(path)
        for method, values in scores.items():
            chunks[method].append(values)
    return {method: np.concatenate(values) for method, values in chunks.items()}


def threshold_grid_from_sources(
    method: str,
    score_sources: list[np.ndarray],
    num_quantiles: int,
) -> np.ndarray:
    """Build a high-tail threshold grid from simulation, real, null, and tile scores."""

    values = np.concatenate([np.asarray(source, dtype=np.float64) for source in score_sources])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No finite threshold source values for {method}.")
    quantiles = np.unique(
        np.r_[
            np.linspace(0.50, 0.95, max(int(num_quantiles) // 2, 4)),
            np.linspace(0.955, 0.999, max(int(num_quantiles) // 2, 4)),
            [0.9995, 0.9999, 0.99995],
        ]
    )
    grid = np.unique(np.quantile(values, quantiles))
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError(f"Non-finite threshold grid for {method}.")
    return grid


def build_threshold_grids(
    sensitivity_labels: np.ndarray,
    sensitivity_scores: dict[str, np.ndarray],
    real_labels: np.ndarray,
    real_scores: dict[str, np.ndarray],
    null_scores: dict[str, np.ndarray],
    tiles: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    num_quantiles: int,
) -> dict[str, np.ndarray]:
    """Build threshold grids including deployment score tails."""

    grids = {}
    for method in METHOD_KEYS:
        sources = [
            sensitivity_scores[method][~sensitivity_labels],
            sensitivity_scores[method][sensitivity_labels],
            real_scores[method][~real_labels],
            real_scores[method][real_labels],
            null_scores[method],
        ]
        sources.extend(scores[method] for scores, _peaks in tiles.values())
        grids[method] = threshold_grid_from_sources(method, sources, int(num_quantiles))
    return grids


def add_exact_single_rows(
    rows: list[dict[str, Any]],
    sensitivity_labels: np.ndarray,
    sensitivity_scores: dict[str, np.ndarray],
    real_labels: np.ndarray,
    real_scores: dict[str, np.ndarray],
    grids: dict[str, np.ndarray],
    confidence: float,
) -> None:
    """Add single-score rows at every deployment-aware grid threshold."""

    seen = {
        (
            row["family"],
            tuple(sorted((key, round(float(value), 12)) for key, value in row["thresholds"].items())),
        )
        for row in rows
    }
    for method in METHOD_KEYS:
        for threshold in grids[method]:
            key = ("single_exact", ((method, round(float(threshold), 12)),))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                metric_row(
                    f"{method} >= {threshold:.6g}",
                    "single_exact",
                    sensitivity_scores[method] >= threshold,
                    real_scores[method] >= threshold,
                    sensitivity_labels,
                    real_labels,
                    confidence=confidence,
                    thresholds={method: float(threshold)},
                )
            )


def pooled_null_summary(
    row: dict[str, Any],
    null_scores: dict[str, np.ndarray],
    confidence: float,
) -> dict[str, Any]:
    """Return pooled null FPR and confidence interval for a policy."""

    trigger = apply_policy(row, null_scores)
    n = int(trigger.shape[0])
    fp = int(np.count_nonzero(trigger))
    lo, hi = binomial_ci(fp, n, confidence)
    return {
        "pooled_null_count": n,
        "pooled_null_fp": fp,
        "pooled_null_fpr": float(fp / max(n, 1)),
        "pooled_null_fpr_ci_low": lo,
        "pooled_null_fpr_ci_high": hi,
    }


def gnomonic_inverse_vectorized(
    glon0_deg: np.ndarray,
    glat0_deg: np.ndarray,
    dx_rad: np.ndarray,
    dy_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized inverse gnomonic projection for patch-peak coordinates."""

    glat0 = np.radians(np.asarray(glat0_deg, dtype=np.float64))
    glon0 = np.radians(np.asarray(glon0_deg, dtype=np.float64))
    dx = np.asarray(dx_rad, dtype=np.float64)
    dy = np.asarray(dy_rad, dtype=np.float64)
    rho = np.hypot(dx, dy)
    safe_rho = np.where(rho > 0.0, rho, 1.0)
    c = np.arctan(rho)
    sin_c = np.sin(c)
    cos_c = np.cos(c)
    sin_lat = cos_c * np.sin(glat0) + (dy * sin_c * np.cos(glat0)) / safe_rho
    glat = np.arcsin(np.clip(sin_lat, -1.0, 1.0))
    glon = glon0 + np.arctan2(
        dx * sin_c,
        safe_rho * np.cos(glat0) * cos_c - dy * np.sin(glat0) * sin_c,
    )
    glon = np.where(rho > 0.0, glon, glon0)
    glat = np.where(rho > 0.0, glat, glat0)
    return np.degrees(glon) % 360.0, np.degrees(glat)


def trigger_peak_arrays_for_policy(
    row: dict[str, Any],
    scores: dict[str, np.ndarray],
    peaks: dict[str, np.ndarray],
    trigger: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return peak coordinates and sortable margins for triggered tiles."""

    idxs = np.flatnonzero(trigger)
    peak_glon = np.asarray(scores["glon_deg"][idxs], dtype=np.float64).copy()
    peak_glat = np.asarray(scores["glat_deg"][idxs], dtype=np.float64).copy()
    margins = policy_margin(row, scores)
    if idxs.size == 0:
        return peak_glon, peak_glat, margins[idxs]

    thresholds = {key: float(value) for key, value in row["thresholds"].items()}
    best_ratio = np.full(idxs.shape[0], -np.inf, dtype=np.float64)
    best_i = np.full(idxs.shape[0], -1, dtype=np.int32)
    best_j = np.full(idxs.shape[0], -1, dtype=np.int32)
    for method in ML_METHODS:
        if method not in thresholds or method not in scores:
            continue
        ratio = np.asarray(scores[method][idxs], dtype=np.float64) / max(
            abs(float(thresholds[method])),
            1.0e-12,
        )
        update = ratio > best_ratio
        best_ratio[update] = ratio[update]
        best_i[update] = peaks[f"peak_i__{method}"][idxs][update]
        best_j[update] = peaks[f"peak_j__{method}"][idxs][update]

    has_ml_peak = best_i >= 0
    if np.any(has_ml_peak):
        dx_rad = (best_j[has_ml_peak].astype(np.float64) - PATCH_CENTER) * PATCH_RESO_RAD
        dy_rad = (PATCH_CENTER - best_i[has_ml_peak].astype(np.float64)) * PATCH_RESO_RAD
        peak_glon[has_ml_peak], peak_glat[has_ml_peak] = gnomonic_inverse_vectorized(
            peak_glon[has_ml_peak],
            peak_glat[has_ml_peak],
            dx_rad,
            dy_rad,
        )
    return peak_glon, peak_glat, margins[idxs]


def fast_greedy_cluster_count(
    peak_glon_deg: np.ndarray,
    peak_glat_deg: np.ndarray,
    score: np.ndarray,
    radius_deg: float,
) -> int:
    """Return exact greedy-cluster count with vectorized distance checks."""

    n = int(np.asarray(score).shape[0])
    if n == 0:
        return 0
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    lon = np.radians(np.asarray(peak_glon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(peak_glat_deg, dtype=np.float64))
    cluster_lon = np.empty(n, dtype=np.float64)
    cluster_lat = np.empty(n, dtype=np.float64)
    radius_rad = np.radians(float(radius_deg))
    n_clusters = 0
    for idx in order:
        if n_clusters == 0:
            cluster_lon[0] = lon[idx]
            cluster_lat[0] = lat[idx]
            n_clusters = 1
            continue
        d_lat = lat[idx] - cluster_lat[:n_clusters]
        d_lon = lon[idx] - cluster_lon[:n_clusters]
        hav = (
            np.sin(d_lat / 2.0) ** 2
            + np.cos(lat[idx]) * np.cos(cluster_lat[:n_clusters]) * np.sin(d_lon / 2.0) ** 2
        )
        dist = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(hav)))
        if np.any(dist <= radius_rad):
            continue
        cluster_lon[n_clusters] = lon[idx]
        cluster_lat[n_clusters] = lat[idx]
        n_clusters += 1
    return int(n_clusters)


def tile_burden_summary(
    row: dict[str, Any],
    tiles: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    cluster_radius_deg: float,
) -> dict[str, Any]:
    """Return tile trigger and cluster burden for a policy."""

    by_map = {}
    for map_name, (scores, peaks) in tiles.items():
        trigger = apply_policy(row, scores)
        peak_glon, peak_glat, margins = trigger_peak_arrays_for_policy(row, scores, peaks, trigger)
        n_clusters = fast_greedy_cluster_count(peak_glon, peak_glat, margins, float(cluster_radius_deg))
        n_tiles = int(trigger.shape[0])
        by_map[map_name] = {
            "num_tiles": n_tiles,
            "num_triggered_tiles": int(np.count_nonzero(trigger)),
            "trigger_fraction": float(np.count_nonzero(trigger) / max(n_tiles, 1)),
            "clusters": int(n_clusters),
        }
    max_cluster_map = max(by_map, key=lambda name: by_map[name]["clusters"])
    max_trigger_map = max(by_map, key=lambda name: by_map[name]["trigger_fraction"])
    return {
        "tile_burden_by_map": by_map,
        "max_clusters": by_map[max_cluster_map]["clusters"],
        "max_clusters_map": max_cluster_map,
        "max_trigger_fraction": by_map[max_trigger_map]["trigger_fraction"],
        "max_trigger_fraction_map": max_trigger_map,
        "mean_clusters": float(np.mean([value["clusters"] for value in by_map.values()])),
    }


def evaluate_rows(
    rows: list[dict[str, Any]],
    null_scores: dict[str, np.ndarray],
    tiles: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter policies with cheap checks, then evaluate tile clustering."""

    cheap_pass = []
    for row in rows:
        if row["camb_fpr"] > float(args.max_camb_fpr) + 1.0e-12:
            continue
        if row["real_fpr"] > float(args.max_real_fpr) + 1.0e-12:
            continue
        null = pooled_null_summary(row, null_scores, float(args.confidence))
        if null["pooled_null_fpr_ci_high"] > float(args.max_pooled_null_fpr_ci_high) + 1.0e-12:
            continue
        cheap_pass.append({**row, **null})

    cheap_pass.sort(
        key=lambda row: (
            row["real_recall"],
            row["camb_recall"],
            -row["pooled_null_fpr_ci_high"],
        ),
        reverse=True,
    )

    evaluated = []
    cluster_evals = 0
    num_after_trigger_fraction_filter = 0
    cluster_eval_limit_hit = False
    last_cluster_evaluated: dict[str, Any] | None = None
    for row in cheap_pass:
        cheap_tile_ok = True
        for scores, _peaks in tiles.values():
            trigger = apply_policy(row, scores)
            trigger_fraction = float(np.count_nonzero(trigger) / max(trigger.shape[0], 1))
            if trigger_fraction > float(args.max_trigger_fraction_any_map) + 1.0e-12:
                cheap_tile_ok = False
                break
        if not cheap_tile_ok:
            continue
        if args.max_cluster_evals and cluster_evals >= int(args.max_cluster_evals):
            cluster_eval_limit_hit = True
            break
        num_after_trigger_fraction_filter += 1
        tile = tile_burden_summary(row, tiles, float(args.cluster_radius_deg))
        cluster_evals += 1
        last_cluster_evaluated = row
        if tile["max_trigger_fraction"] > float(args.max_trigger_fraction_any_map) + 1.0e-12:
            continue
        if tile["max_clusters"] > int(args.max_clusters_any_map):
            continue
        evaluated.append({**row, **tile})

    evaluated.sort(
        key=lambda row: (
            row["real_recall"],
            row["camb_recall"],
            -row["max_clusters"],
            -row["max_trigger_fraction"],
            -row["pooled_null_fpr_ci_high"],
        ),
        reverse=True,
    )
    diagnostics = {
        "num_policy_rows": len(rows),
        "num_after_camb_real_null_filters": len(cheap_pass),
        "num_after_trigger_fraction_filter_evaluated_or_capped": num_after_trigger_fraction_filter,
        "num_cluster_evaluated": cluster_evals,
        "cluster_eval_limit": int(args.max_cluster_evals),
        "cluster_eval_limit_hit": bool(cluster_eval_limit_hit),
        "search_exhaustive": not bool(cluster_eval_limit_hit),
        "num_feasible": len(evaluated),
    }
    if cluster_eval_limit_hit and last_cluster_evaluated is not None:
        diagnostics["cluster_eval_frontier_real_recall"] = float(
            last_cluster_evaluated["real_recall"]
        )
        diagnostics["cluster_eval_frontier_camb_recall"] = float(
            last_cluster_evaluated["camb_recall"]
        )
    if cluster_eval_limit_hit and evaluated and last_cluster_evaluated is not None:
        diagnostics["best_feasible_real_recall_certified_by_frontier"] = bool(
            evaluated[0]["real_recall"] > last_cluster_evaluated["real_recall"]
        )
    return evaluated, diagnostics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact search CSV."""

    columns = [
        "policy",
        "family",
        "real_recall",
        "camb_recall",
        "real_fpr",
        "camb_fpr",
        "pooled_null_fpr",
        "pooled_null_fpr_ci_high",
        "real_recall_gain_vs_best_feasible_single",
        "max_clusters",
        "max_clusters_map",
        "mean_clusters",
        "max_trigger_fraction",
        "max_trigger_fraction_map",
        "thresholds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in columns}
            out["thresholds"] = json.dumps(row.get("thresholds", {}), sort_keys=True)
            writer.writerow(out)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write Markdown report."""

    lines = ["# Tile-Constrained Policy Search", ""]
    lines.append("This search optimizes diagnostic recall subject to cross-map tile burden constraints.")
    lines.append("")
    lines.append("## Constraints")
    lines.append("")
    for key, value in report["constraints"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    for key, value in report["diagnostics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Best Feasible Policies")
    lines.append("")
    baseline = report.get("best_feasible_single_row")
    if baseline:
        lines.append(
            "Best feasible single-score baseline: "
            f"`{baseline['policy']}` with real recall `{baseline['real_recall']:.4f}`, "
            f"real FPR `{baseline['real_fpr']:.4f}`, and max clusters "
            f"`{baseline['max_clusters']}`."
        )
        lines.append("")
    if not report["top_rows"]:
        lines.append("No feasible policy was found under the configured constraints.")
    else:
        lines.append(
            "| rank | policy | real recall | gain vs single | real FPR | pooled null FPR high | max clusters | max trigger frac |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for rank, row in enumerate(report["top_rows"], start=1):
            gain = row.get("real_recall_gain_vs_best_feasible_single")
            gain_text = "" if gain is None else f"{gain:+.4f}"
            lines.append(
                f"| {rank} | `{row['policy']}` | {row['real_recall']:.4f} | "
                f"{gain_text} | {row['real_fpr']:.4f} | "
                f"{row['pooled_null_fpr_ci_high']:.4f} | "
                f"{row['max_clusters']} ({row['max_clusters_map']}) | "
                f"{row['max_trigger_fraction']:.4f} ({row['max_trigger_fraction_map']}) |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_labels, sensitivity_scores = load_scores(Path(args.sensitivity_scores).expanduser().resolve())
    real_labels, real_scores = load_scores(Path(args.real_sky_scores).expanduser().resolve())
    sensitivity_labels = np.asarray(sensitivity_labels, dtype=bool)
    real_labels = np.asarray(real_labels, dtype=bool)
    null_scores = load_pooled_null_scores(args)
    tiles = load_all_tiles(args)
    grids = build_threshold_grids(
        sensitivity_labels,
        sensitivity_scores,
        real_labels,
        real_scores,
        null_scores,
        tiles,
        int(args.num_quantiles),
    )
    rows = build_policy_rows(
        sensitivity_labels,
        sensitivity_scores,
        real_labels,
        real_scores,
        grids,
        confidence=float(args.confidence),
        kof3_step=int(args.kof3_step),
    )
    add_exact_single_rows(
        rows,
        sensitivity_labels,
        sensitivity_scores,
        real_labels,
        real_scores,
        grids,
        float(args.confidence),
    )
    feasible, diagnostics = evaluate_rows(rows, null_scores, tiles, args)
    top_rows = feasible[: int(args.top_k)]
    best_feasible_single = next(
        (row for row in feasible if row.get("family") in {"single", "single_exact"}),
        None,
    )
    if best_feasible_single is not None:
        for row in top_rows:
            row["real_recall_gain_vs_best_feasible_single"] = float(
                row["real_recall"] - best_feasible_single["real_recall"]
            )
    report = {
        "inputs": {
            "sensitivity_scores": str(Path(args.sensitivity_scores).expanduser().resolve()),
            "real_sky_scores": str(Path(args.real_sky_scores).expanduser().resolve()),
            "tile_score_template": str(args.tile_score_template),
            "null_score_template": str(args.null_score_template),
            "maps": list(args.maps),
        },
        "constraints": {
            "max_camb_fpr": float(args.max_camb_fpr),
            "max_real_fpr": float(args.max_real_fpr),
            "max_pooled_null_fpr_ci_high": float(args.max_pooled_null_fpr_ci_high),
            "max_trigger_fraction_any_map": float(args.max_trigger_fraction_any_map),
            "max_clusters_any_map": int(args.max_clusters_any_map),
            "cluster_radius_deg": float(args.cluster_radius_deg),
        },
        "threshold_grid_sizes": {method: int(len(values)) for method, values in grids.items()},
        "diagnostics": diagnostics,
        "best_feasible_single_row": best_feasible_single,
        "top_rows": top_rows,
    }
    json_path = output_dir / "tile_constrained_policy_search.json"
    csv_path = output_dir / "tile_constrained_policy_search_top.csv"
    md_path = output_dir / "tile_constrained_policy_search.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, top_rows)
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
