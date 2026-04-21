"""Build deployment candidate-burden tables from Batch 6 full-sky tiles.

Assumptions
-----------
* This is a deployment accounting script. It does not retrain models and does
  not create a cosmological detection claim.
* The input tile-feature caches are fixed Batch 6 Nside=32 full-sky audits over
  Planck cleaned maps after the common-mask projection cut.
* Patch counts are counts of eligible full-sky tile centers. Cluster counts use
  greedy great-circle linkage of score-peak coordinates, matching
  ``phase3_fullsky_tile.py``.
* Bootstrap intervals are spatial block-bootstrap diagnostics over coarse
  HEALPix sky blocks. They account for map-position heterogeneity better than an
  iid patch bootstrap, but overlapping 55 deg gnomonic patches are still
  correlated; quote intervals as operational uncertainty, not fundamental
  cosmological uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
from scipy.stats import beta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from batch6_overnight_analysis import (  # noqa: E402
    ALL_FEATURE_NAMES,
    CACHE_DIR,
    FPR_TARGET,
    SCORE_FEATURE_NAMES,
    TILE_PATHS,
    fit_gbt,
    load_transform_npz,
    tile_scores_for_gbt,
    tile_scores_for_v6,
)
from phase3_fullsky_tile import greedy_cluster, peak_sky_coord  # noqa: E402


DEFAULT_CROSSMAP_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "batch6_fullsky_nside32_smica"
    / "crossmap_recalibration_nside32.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_deployment_burden"
)
METHODS = ("v6_only", "gbt_6", "gbt_14")
THRESHOLD_MODES = ("tile_recalibrated", "shipped")
THRESHOLD_KEYS = {
    "v6_only": "thr_v6_tile",
    "gbt_6": "thr_g6_tile",
    "gbt_14": "thr_g14_tile",
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute expected full-sky patch and clustered candidate burden "
            "from Batch 6 tile-feature caches."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--crossmap-json", type=str, default=str(DEFAULT_CROSSMAP_JSON))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--method", action="append", choices=METHODS, default=[])
    parser.add_argument(
        "--threshold-mode",
        action="append",
        choices=THRESHOLD_MODES,
        default=[],
        help="Threshold family to report. Default: both.",
    )
    parser.add_argument("--cluster-radii-deg", type=str, default="10,15,25")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-block-nside", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260420)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_resamples < 0:
        raise ValueError("--bootstrap-resamples must be non-negative.")
    if args.bootstrap_block_nside <= 0 or not hp.isnsideok(args.bootstrap_block_nside):
        raise ValueError("--bootstrap-block-nside must be a valid positive HEALPix Nside.")
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must lie in (0, 1).")
    for radius in parse_radii(args.cluster_radii_deg):
        if not math.isfinite(radius) or radius <= 0.0 or radius >= 180.0:
            raise ValueError("Cluster radii must be finite values in (0, 180) deg.")


def parse_radii(text: str) -> list[float]:
    radii = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not radii:
        raise ValueError("At least one cluster radius is required.")
    return radii


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def binomial_ci(k: int, n: int, confidence: float) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("Binomial interval requires n > 0.")
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def percentile_ci(
    samples: np.ndarray,
    confidence: float,
) -> tuple[float | None, float | None]:
    if samples.size == 0:
        return None, None
    alpha = 1.0 - confidence
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def sky_block_ids(glon_deg: np.ndarray, glat_deg: np.ndarray, nside: int) -> np.ndarray:
    theta = np.radians(90.0 - np.asarray(glat_deg, dtype=np.float64))
    phi = np.radians(np.asarray(glon_deg, dtype=np.float64) % 360.0)
    return hp.ang2pix(int(nside), theta, phi)


def block_bootstrap_counts(
    unit_block_ids: np.ndarray,
    all_block_ids: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_resamples <= 0:
        return np.asarray([], dtype=np.float64)
    all_blocks = np.asarray(
        sorted(set(int(block) for block in all_block_ids)),
        dtype=np.int64,
    )
    count_by_block = {int(block): 0 for block in all_blocks}
    for block in np.asarray(unit_block_ids, dtype=np.int64):
        count_by_block[int(block)] = count_by_block.get(int(block), 0) + 1
    counts = np.asarray([count_by_block[int(block)] for block in all_blocks], dtype=np.float64)
    draws = rng.integers(0, all_blocks.size, size=(int(n_resamples), all_blocks.size))
    return counts[draws].sum(axis=1)


def load_mixed_scores(fit6: Any, fit14: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    v6_inj_m = load_transform_npz(CACHE_DIR / "inj_mixed_v6_aux_only_transforms.npz")
    labels = np.asarray(v6_inj_m["labels"], dtype=np.uint8)
    return {
        "v6_only": (np.asarray(v6_inj_m["baseline"], dtype=np.float64), labels),
        "gbt_6": (np.asarray(fit6["mixed_scores"], dtype=np.float64), fit6["mixed_labels"]),
        "gbt_14": (np.asarray(fit14["mixed_scores"], dtype=np.float64), fit14["mixed_labels"]),
    }


def load_tile_scores(
    method: str,
    tile_path: Path,
    fit6: Any,
    fit14: Any,
) -> np.ndarray:
    if method == "v6_only":
        return tile_scores_for_v6(tile_path)
    if method == "gbt_6":
        return tile_scores_for_gbt(fit6["gbt"], list(SCORE_FEATURE_NAMES), tile_path)
    if method == "gbt_14":
        return tile_scores_for_gbt(fit14["gbt"], list(ALL_FEATURE_NAMES), tile_path)
    raise ValueError(f"Unknown method: {method}")


def make_trigger_records(
    method: str,
    trigger_mask: np.ndarray,
    scores: np.ndarray,
    tile_data: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records = []
    v6_baseline = np.asarray(tile_data["v6_baseline"], dtype=np.float64)
    v7_baseline = np.asarray(tile_data["v7_baseline"], dtype=np.float64)
    for idx in np.where(trigger_mask)[0]:
        if method == "v6_only":
            source = "v6"
        else:
            source = "v7" if v7_baseline[idx] >= v6_baseline[idx] else "v6"
        peak_i = int(tile_data[f"{source}_peak_i"][idx])
        peak_j = int(tile_data[f"{source}_peak_j"][idx])
        peak_glon, peak_glat = peak_sky_coord(
            float(tile_data["glon_deg"][idx]),
            float(tile_data["glat_deg"][idx]),
            peak_i,
            peak_j,
        )
        records.append(
            {
                "patch_index": int(idx),
                "patch_glon_deg": float(tile_data["glon_deg"][idx]),
                "patch_glat_deg": float(tile_data["glat_deg"][idx]),
                "peak_glon_deg": peak_glon,
                "peak_glat_deg": peak_glat,
                "peak_source_model": source,
                "gbt_score": float(scores[idx]),
                "mask_fraction": float(tile_data["mask_fraction"][idx]),
            }
        )
    return records


def threshold_for(
    method: str,
    threshold_mode: str,
    map_row: dict[str, Any],
    crossmap: dict[str, Any],
) -> float:
    if threshold_mode == "shipped":
        return float(crossmap["shipped_thresholds"][method])
    if threshold_mode == "tile_recalibrated":
        return float(map_row[THRESHOLD_KEYS[method]])
    raise ValueError(f"Unknown threshold mode: {threshold_mode}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    methods = tuple(args.method) if args.method else METHODS
    threshold_modes = tuple(args.threshold_mode) if args.threshold_mode else THRESHOLD_MODES
    radii = parse_radii(args.cluster_radii_deg)
    crossmap = load_json(Path(args.crossmap_json).resolve())
    per_map = {row["map"]: row for row in crossmap.get("per_map", [])}
    rng = np.random.default_rng(int(args.seed))
    fit6 = fit_gbt(list(SCORE_FEATURE_NAMES))
    fit14 = fit_gbt(list(ALL_FEATURE_NAMES))
    mixed_scores = load_mixed_scores(fit6, fit14)

    patch_rows = []
    cluster_rows = []
    for map_name, tile_path in TILE_PATHS.items():
        if map_name not in per_map or not Path(tile_path).exists():
            continue
        map_row = per_map[map_name]
        with np.load(tile_path) as loaded:
            tile_data = {key: loaded[key].copy() for key in loaded.files}
        n_tile = int(tile_data["glon_deg"].shape[0])
        block_ids = sky_block_ids(
            tile_data["glon_deg"],
            tile_data["glat_deg"],
            int(args.bootstrap_block_nside),
        )
        for method in methods:
            scores = load_tile_scores(method, tile_path, fit6, fit14)
            pos_scores, labels = mixed_scores[method]
            positive_scores = pos_scores[np.asarray(labels, dtype=np.uint8) == 1]
            n_pos = int(positive_scores.size)
            for threshold_mode in threshold_modes:
                threshold = threshold_for(method, threshold_mode, map_row, crossmap)
                trigger_mask = np.asarray(scores >= threshold, dtype=bool)
                n_triggered = int(trigger_mask.sum())
                patch_fpr = float(n_triggered / n_tile)
                trigger_blocks = block_ids[trigger_mask]
                patch_boot = block_bootstrap_counts(
                    trigger_blocks,
                    block_ids,
                    n_resamples=int(args.bootstrap_resamples),
                    rng=rng,
                )
                patch_low, patch_high = percentile_ci(patch_boot, float(args.confidence))
                fpr_low, fpr_high = binomial_ci(n_triggered, n_tile, float(args.confidence))
                recall_hits = int(np.count_nonzero(positive_scores >= threshold))
                recall = float(recall_hits / n_pos)
                recall_low, recall_high = binomial_ci(recall_hits, n_pos, float(args.confidence))
                patch_rows.append(
                    {
                        "map": map_name,
                        "method": method,
                        "threshold_mode": threshold_mode,
                        "threshold": threshold,
                        "n_tile": n_tile,
                        "n_patch_candidates": n_triggered,
                        "patch_fpr": patch_fpr,
                        "patch_fpr_ci_low": fpr_low,
                        "patch_fpr_ci_high": fpr_high,
                        "patch_candidates_boot_ci_low": patch_low,
                        "patch_candidates_boot_ci_high": patch_high,
                        "n_pos": n_pos,
                        "recall": recall,
                        "recall_ci_low": recall_low,
                        "recall_ci_high": recall_high,
                    }
                )
                trigger_records = make_trigger_records(method, trigger_mask, scores, tile_data)
                for radius in radii:
                    clusters, _assignment = greedy_cluster(trigger_records, float(radius))
                    cluster_glon = np.asarray(
                        [c["peak_glon_deg"] for c in clusters],
                        dtype=np.float64,
                    )
                    cluster_glat = np.asarray(
                        [c["peak_glat_deg"] for c in clusters],
                        dtype=np.float64,
                    )
                    if clusters:
                        cluster_blocks = sky_block_ids(
                            cluster_glon,
                            cluster_glat,
                            int(args.bootstrap_block_nside),
                        )
                    else:
                        cluster_blocks = np.asarray([], dtype=np.int64)
                    cluster_boot = block_bootstrap_counts(
                        cluster_blocks,
                        np.unique(block_ids),
                        n_resamples=int(args.bootstrap_resamples),
                        rng=rng,
                    )
                    cluster_low, cluster_high = percentile_ci(
                        cluster_boot,
                        float(args.confidence),
                    )
                    cluster_rows.append(
                        {
                            "map": map_name,
                            "method": method,
                            "threshold_mode": threshold_mode,
                            "cluster_radius_deg": float(radius),
                            "n_patch_candidates": n_triggered,
                            "n_clusters": int(len(clusters)),
                            "cluster_boot_ci_low": cluster_low,
                            "cluster_boot_ci_high": cluster_high,
                            "max_cluster_size": int(
                                max((c["n_members"] for c in clusters), default=0)
                            ),
                            "mean_cluster_size": float(
                                np.mean([c["n_members"] for c in clusters])
                                if clusters
                                else 0.0
                            ),
                        }
                    )

    return {
        "metadata": {
            "crossmap_json": str(Path(args.crossmap_json).resolve()),
            "fpr_target": float(crossmap.get("fpr_target", FPR_TARGET)),
            "methods": list(methods),
            "threshold_modes": list(threshold_modes),
            "cluster_radii_deg": radii,
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "bootstrap_block_nside": int(args.bootstrap_block_nside),
            "confidence": float(args.confidence),
            "seed": int(args.seed),
            "assumption_notes": [
                "Patch candidate count is the full eligible Nside=32 tile count.",
                "Cluster counts use greedy linkage of score-peak sky positions.",
                (
                    "Bootstrap intervals resample coarse HEALPix sky blocks "
                    "and are operational diagnostics."
                ),
            ],
        },
        "patch_rows": patch_rows,
        "cluster_rows": cluster_rows,
        "summary": summarize_rows(patch_rows, cluster_rows),
    }


def summarize_rows(
    patch_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"patch": [], "cluster": []}
    for threshold_mode in sorted({row["threshold_mode"] for row in patch_rows}):
        for method in sorted({row["method"] for row in patch_rows}):
            rows = [
                row
                for row in patch_rows
                if row["threshold_mode"] == threshold_mode and row["method"] == method
            ]
            if not rows:
                continue
            summary["patch"].append(
                {
                    "threshold_mode": threshold_mode,
                    "method": method,
                    "mean_patch_candidates": float(
                        np.mean([row["n_patch_candidates"] for row in rows])
                    ),
                    "mean_patch_fpr": float(np.mean([row["patch_fpr"] for row in rows])),
                    "mean_recall": float(np.mean([row["recall"] for row in rows])),
                }
            )
    for threshold_mode in sorted({row["threshold_mode"] for row in cluster_rows}):
        for method in sorted({row["method"] for row in cluster_rows}):
            for radius in sorted({row["cluster_radius_deg"] for row in cluster_rows}):
                rows = [
                    row
                    for row in cluster_rows
                    if row["threshold_mode"] == threshold_mode
                    and row["method"] == method
                    and row["cluster_radius_deg"] == radius
                ]
                if not rows:
                    continue
                summary["cluster"].append(
                    {
                        "threshold_mode": threshold_mode,
                        "method": method,
                        "cluster_radius_deg": radius,
                        "mean_clusters": float(np.mean([row["n_clusters"] for row in rows])),
                        "mean_max_cluster_size": float(
                            np.mean([row["max_cluster_size"] for row in rows])
                        ),
                    }
                )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_ci(low: float | None, high: float | None, decimals: int = 0) -> str:
    if low is None or high is None:
        return "n/a"
    return f"[{low:.{decimals}f}, {high:.{decimals}f}]"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    patch_rows = report["patch_rows"]
    cluster_rows = report["cluster_rows"]
    lines = ["# Remediated v1 Deployment Candidate Burden", ""]
    lines.append("## Assumptions")
    lines.append("")
    for note in metadata["assumption_notes"]:
        lines.append(f"- {note}")
    lines.append(f"- Confidence level: `{metadata['confidence']}`")
    lines.append(f"- Spatial bootstrap block Nside: `{metadata['bootstrap_block_nside']}`")
    lines.append(f"- Bootstrap resamples: `{metadata['bootstrap_resamples']}`")
    lines.append("")
    for threshold_mode in metadata["threshold_modes"]:
        lines.append(f"## Patch Candidates: `{threshold_mode}`")
        lines.append("")
        lines.append(
            "| map | method | threshold | patch candidates | block-bootstrap count CI | "
            "patch FPR | recall |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in patch_rows:
            if row["threshold_mode"] != threshold_mode:
                continue
            patch_count_ci = format_ci(
                row["patch_candidates_boot_ci_low"],
                row["patch_candidates_boot_ci_high"],
            )
            lines.append(
                f"| {row['map']} | `{row['method']}` | {row['threshold']:.6g} | "
                f"{row['n_patch_candidates']} | {patch_count_ci} | "
                f"{row['patch_fpr']:.4f} | {row['recall']:.4f} |"
            )
        lines.append("")
    for threshold_mode in metadata["threshold_modes"]:
        for radius in metadata["cluster_radii_deg"]:
            lines.append(f"## Cluster Candidates: `{threshold_mode}`, radius `{radius:g} deg`")
            lines.append("")
            lines.append(
                "| map | method | clusters | block-bootstrap cluster CI | "
                "max cluster size | mean cluster size |"
            )
            lines.append("|---|---|---:|---:|---:|---:|")
            for row in cluster_rows:
                if (
                    row["threshold_mode"] != threshold_mode
                    or float(row["cluster_radius_deg"]) != float(radius)
                ):
                    continue
                lines.append(
                    f"| {row['map']} | `{row['method']}` | {row['n_clusters']} | "
                    f"{format_ci(row['cluster_boot_ci_low'], row['cluster_boot_ci_high'])} | "
                    f"{row['max_cluster_size']} | {row['mean_cluster_size']:.2f} |"
                )
            lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- Use `tile_recalibrated` rows for deployment burden.")
    lines.append("- Use `shipped` rows only to diagnose clean-null threshold drift.")
    lines.append("- These are screening candidate counts, not detections.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    report = build_report(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deployment_burden.json"
    md_path = output_dir / "deployment_burden.md"
    patch_csv = output_dir / "deployment_patch_burden.csv"
    cluster_csv = output_dir / "deployment_cluster_burden.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(patch_csv, report["patch_rows"])
    write_csv(cluster_csv, report["cluster_rows"])
    write_markdown(md_path, report)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "patch_csv": str(patch_csv),
                "cluster_csv": str(cluster_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
