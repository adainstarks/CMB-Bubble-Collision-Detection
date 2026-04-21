"""Search fixed-score policies for recall under explicit false-positive limits.

Assumptions
-----------
* This is a policy-selection diagnostic over existing remediated score caches;
  it does not train a model and does not prove real-sky deployment calibration.
* The CAMB sensitivity grid supplies a 5000-negative simulation-null reference.
* The real-SMICA injection grid supplies a small 200-negative real-background
  stress test. Real-FPR constraints are therefore coarse and must be treated as
  screening diagnostics, not final p-values.
* Composite policies are only promoted when they improve recall while obeying
  explicit CAMB and real-background FPR budgets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta


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
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_policy_pareto"
)
METHOD_KEYS = {
    "random_b64_aux": "score__random_b64_aux",
    "imagenet_b64_aux": "score__imagenet_b64_aux",
    "circular_template_screen": "score__circular_template_screen",
}
DEFAULT_CONSTRAINTS = (
    "0.05:0.02,0.05:0.05,0.05:0.08,0.08:0.05,0.08:0.08"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search simple score-composite policies under CAMB and "
            "real-SMICA FPR constraints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sensitivity-scores",
        type=str,
        default=str(DEFAULT_SENSITIVITY_SCORES),
    )
    parser.add_argument(
        "--real-sky-scores",
        type=str,
        default=str(DEFAULT_REAL_SKY_SCORES),
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--constraints", type=str, default=DEFAULT_CONSTRAINTS)
    parser.add_argument("--num-quantiles", type=int, default=52)
    parser.add_argument("--kof3-step", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def parse_constraints(text: str) -> list[tuple[float, float]]:
    constraints = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError("Constraints must use CAMB_FPR:REAL_FPR comma syntax.")
        camb_limit, real_limit = float(parts[0]), float(parts[1])
        if not (0.0 <= camb_limit <= 1.0 and 0.0 <= real_limit <= 1.0):
            raise ValueError("FPR constraints must lie in [0, 1].")
        constraints.append((camb_limit, real_limit))
    if not constraints:
        raise ValueError("At least one FPR constraint pair is required.")
    return constraints


def validate_args(args: argparse.Namespace) -> None:
    if args.num_quantiles < 8:
        raise ValueError("--num-quantiles must be at least 8.")
    if args.kof3_step <= 0:
        raise ValueError("--kof3-step must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must lie in (0, 1).")
    parse_constraints(args.constraints)


def binomial_ci(k: int, n: int, confidence: float) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("Binomial interval requires n > 0.")
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def load_scores(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path) as data:
        labels = np.asarray(data["labels"], dtype=bool)
        scores = {
            method: np.asarray(data[key], dtype=np.float64)
            for method, key in METHOD_KEYS.items()
            if key in data.files
        }
    missing = sorted(set(METHOD_KEYS) - set(scores))
    if missing:
        raise ValueError(f"Missing score arrays in {path}: {missing}")
    if not bool(labels.any()) or not bool((~labels).any()):
        raise ValueError(f"Scores file must contain both positives and negatives: {path}")
    return labels, scores


def threshold_grid(
    method: str,
    sensitivity_labels: np.ndarray,
    sensitivity_scores: dict[str, np.ndarray],
    real_labels: np.ndarray,
    real_scores: dict[str, np.ndarray],
    num_quantiles: int,
) -> np.ndarray:
    quantiles = np.unique(
        np.r_[
            np.linspace(0.50, 0.95, max(int(num_quantiles) // 2, 4)),
            np.linspace(0.955, 0.999, max(int(num_quantiles) // 2, 4)),
            [0.9995, 0.9999],
        ]
    )
    values = np.concatenate(
        [
            sensitivity_scores[method][~sensitivity_labels],
            sensitivity_scores[method][sensitivity_labels],
            real_scores[method][~real_labels],
            real_scores[method][real_labels],
        ]
    )
    grid = np.unique(np.quantile(values, quantiles))
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError(f"Non-finite threshold grid for {method}.")
    return grid


def metric_row(
    name: str,
    family: str,
    mask_sensitivity: np.ndarray,
    mask_real: np.ndarray,
    sensitivity_labels: np.ndarray,
    real_labels: np.ndarray,
    *,
    confidence: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    camb_tp = int(np.count_nonzero(mask_sensitivity & sensitivity_labels))
    camb_fp = int(np.count_nonzero(mask_sensitivity & ~sensitivity_labels))
    camb_pos = int(np.count_nonzero(sensitivity_labels))
    camb_neg = int(np.count_nonzero(~sensitivity_labels))
    real_tp = int(np.count_nonzero(mask_real & real_labels))
    real_fp = int(np.count_nonzero(mask_real & ~real_labels))
    real_pos = int(np.count_nonzero(real_labels))
    real_neg = int(np.count_nonzero(~real_labels))
    camb_recall = camb_tp / camb_pos
    camb_fpr = camb_fp / camb_neg
    real_recall = real_tp / real_pos
    real_fpr = real_fp / real_neg
    real_precision = real_tp / max(real_tp + real_fp, 1)
    camb_recall_ci = binomial_ci(camb_tp, camb_pos, confidence)
    real_recall_ci = binomial_ci(real_tp, real_pos, confidence)
    camb_fpr_ci = binomial_ci(camb_fp, camb_neg, confidence)
    real_fpr_ci = binomial_ci(real_fp, real_neg, confidence)
    return {
        "policy": name,
        "family": family,
        "thresholds": thresholds,
        "camb_tp": camb_tp,
        "camb_fp": camb_fp,
        "camb_recall": camb_recall,
        "camb_recall_ci_low": camb_recall_ci[0],
        "camb_recall_ci_high": camb_recall_ci[1],
        "camb_fpr": camb_fpr,
        "camb_fpr_ci_low": camb_fpr_ci[0],
        "camb_fpr_ci_high": camb_fpr_ci[1],
        "real_tp": real_tp,
        "real_fp": real_fp,
        "real_recall": real_recall,
        "real_recall_ci_low": real_recall_ci[0],
        "real_recall_ci_high": real_recall_ci[1],
        "real_fpr": real_fpr,
        "real_fpr_ci_low": real_fpr_ci[0],
        "real_fpr_ci_high": real_fpr_ci[1],
        "real_precision_observed_grid": real_precision,
    }


def add_policy(
    rows: list[dict[str, Any]],
    seen: set[tuple[bytes, bytes]],
    row: dict[str, Any],
    mask_sensitivity: np.ndarray,
    mask_real: np.ndarray,
) -> None:
    key = (np.packbits(mask_sensitivity).tobytes(), np.packbits(mask_real).tobytes())
    if key in seen:
        return
    seen.add(key)
    rows.append(row)


def build_policy_rows(
    sensitivity_labels: np.ndarray,
    sensitivity_scores: dict[str, np.ndarray],
    real_labels: np.ndarray,
    real_scores: dict[str, np.ndarray],
    grids: dict[str, np.ndarray],
    *,
    confidence: float,
    kof3_step: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[bytes, bytes]] = set()
    methods = tuple(METHOD_KEYS)
    bool_s = {method: [] for method in methods}
    bool_r = {method: [] for method in methods}
    for method in methods:
        for threshold in grids[method]:
            bool_s[method].append(sensitivity_scores[method] >= threshold)
            bool_r[method].append(real_scores[method] >= threshold)

    for method in methods:
        for idx, threshold in enumerate(grids[method]):
            mask_s, mask_r = bool_s[method][idx], bool_r[method][idx]
            row = metric_row(
                f"{method} >= {threshold:.6g}",
                "single",
                mask_s,
                mask_r,
                sensitivity_labels,
                real_labels,
                confidence=confidence,
                thresholds={method: float(threshold)},
            )
            add_policy(rows, seen, row, mask_s, mask_r)

    for left_idx, left in enumerate(methods):
        for right in methods[left_idx + 1 :]:
            for i, left_threshold in enumerate(grids[left]):
                left_s, left_r = bool_s[left][i], bool_r[left][i]
                for j, right_threshold in enumerate(grids[right]):
                    right_s, right_r = bool_s[right][j], bool_r[right][j]
                    thresholds = {left: float(left_threshold), right: float(right_threshold)}
                    for op, family, mask_s, mask_r in (
                        ("AND", "pair_and", left_s & right_s, left_r & right_r),
                        ("OR", "pair_or", left_s | right_s, left_r | right_r),
                    ):
                        row = metric_row(
                            (
                                f"{left} >= {left_threshold:.6g} {op} "
                                f"{right} >= {right_threshold:.6g}"
                            ),
                            family,
                            mask_s,
                            mask_r,
                            sensitivity_labels,
                            real_labels,
                            confidence=confidence,
                            thresholds=thresholds,
                        )
                        add_policy(rows, seen, row, mask_s, mask_r)

    stepped = {method: range(0, len(grids[method]), int(kof3_step)) for method in methods}
    for i in stepped[methods[0]]:
        for j in stepped[methods[1]]:
            for k in stepped[methods[2]]:
                thresholds = {
                    methods[0]: float(grids[methods[0]][i]),
                    methods[1]: float(grids[methods[1]][j]),
                    methods[2]: float(grids[methods[2]][k]),
                }
                count_s = (
                    bool_s[methods[0]][i].astype(np.uint8)
                    + bool_s[methods[1]][j].astype(np.uint8)
                    + bool_s[methods[2]][k].astype(np.uint8)
                )
                count_r = (
                    bool_r[methods[0]][i].astype(np.uint8)
                    + bool_r[methods[1]][j].astype(np.uint8)
                    + bool_r[methods[2]][k].astype(np.uint8)
                )
                for votes in (2, 3):
                    mask_s = count_s >= votes
                    mask_r = count_r >= votes
                    threshold_text = ", ".join(
                        f"{method} >= {thresholds[method]:.6g}" for method in methods
                    )
                    row = metric_row(
                        f"{votes}-of-3({threshold_text})",
                        f"{votes}_of_3",
                        mask_s,
                        mask_r,
                        sensitivity_labels,
                        real_labels,
                        confidence=confidence,
                        thresholds=thresholds,
                    )
                    add_policy(rows, seen, row, mask_s, mask_r)
    return rows


def choose_top_rows(
    rows: list[dict[str, Any]],
    constraints: list[tuple[float, float]],
    top_k: int,
) -> list[dict[str, Any]]:
    selected = []
    for camb_limit, real_limit in constraints:
        feasible = [
            row
            for row in rows
            if row["camb_fpr"] <= camb_limit + 1.0e-12
            and row["real_fpr"] <= real_limit + 1.0e-12
        ]
        feasible.sort(
            key=lambda row: (
                row["real_recall"],
                row["camb_recall"],
                -row["real_fpr"],
                -row["camb_fpr"],
            ),
            reverse=True,
        )
        for rank, row in enumerate(feasible[: int(top_k)], start=1):
            out = {
                **row,
                "constraint_camb_fpr_max": float(camb_limit),
                "constraint_real_fpr_max": float(real_limit),
                "rank": rank,
            }
            selected.append(out)
    return selected


def rates_at_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return recall and FPR for score >= threshold without materializing masks."""

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp_cumulative = np.cumsum(sorted_labels)
    fp_cumulative = np.cumsum(~sorted_labels)
    counts = np.searchsorted(-sorted_scores, -thresholds, side="right")
    tp = np.where(counts > 0, tp_cumulative[counts - 1], 0)
    fp = np.where(counts > 0, fp_cumulative[counts - 1], 0)
    recall = tp / max(int(np.count_nonzero(labels)), 1)
    fpr = fp / max(int(np.count_nonzero(~labels)), 1)
    return recall, fpr


def choose_best_single_rows(
    sensitivity_labels: np.ndarray,
    sensitivity_scores: dict[str, np.ndarray],
    real_labels: np.ndarray,
    real_scores: dict[str, np.ndarray],
    constraints: list[tuple[float, float]],
    *,
    confidence: float,
) -> list[dict[str, Any]]:
    """Select the best single-score baseline over exact observed thresholds."""

    selected = []
    for camb_limit, real_limit in constraints:
        best: dict[str, Any] | None = None
        for method in METHOD_KEYS:
            thresholds = np.unique(
                np.concatenate([sensitivity_scores[method], real_scores[method]])
            )
            if thresholds.size == 0 or not np.all(np.isfinite(thresholds)):
                raise ValueError(f"Non-finite exact threshold set for {method}.")
            camb_recall, camb_fpr = rates_at_thresholds(
                sensitivity_scores[method],
                sensitivity_labels,
                thresholds,
            )
            real_recall, real_fpr = rates_at_thresholds(
                real_scores[method],
                real_labels,
                thresholds,
            )
            feasible = (camb_fpr <= camb_limit + 1.0e-12) & (
                real_fpr <= real_limit + 1.0e-12
            )
            if not np.any(feasible):
                continue
            feasible_idxs = np.flatnonzero(feasible)
            best_idx = max(
                feasible_idxs,
                key=lambda idx: (
                    real_recall[idx],
                    camb_recall[idx],
                    -real_fpr[idx],
                    -camb_fpr[idx],
                ),
            )
            threshold = thresholds[best_idx]
            row = metric_row(
                f"{method} >= {threshold:.6g}",
                "single_exact",
                sensitivity_scores[method] >= threshold,
                real_scores[method] >= threshold,
                sensitivity_labels,
                real_labels,
                confidence=confidence,
                thresholds={method: float(threshold)},
            )
            if best is None or (
                row["real_recall"],
                row["camb_recall"],
                -row["real_fpr"],
                -row["camb_fpr"],
            ) > (
                best["real_recall"],
                best["camb_recall"],
                -best["real_fpr"],
                -best["camb_fpr"],
            ):
                best = row
        if best is None:
            continue
        selected.append(
            {
                **best,
                "constraint_camb_fpr_max": float(camb_limit),
                "constraint_real_fpr_max": float(real_limit),
            }
        )
    return selected


def annotate_single_baseline(
    top_rows: list[dict[str, Any]],
    best_single_rows: list[dict[str, Any]],
) -> None:
    """Attach reproducible gain-vs-single fields to selected rows in place."""

    baseline_by_constraint = {
        (
            row["constraint_camb_fpr_max"],
            row["constraint_real_fpr_max"],
        ): row
        for row in best_single_rows
    }
    for row in top_rows:
        key = (row["constraint_camb_fpr_max"], row["constraint_real_fpr_max"])
        baseline = baseline_by_constraint.get(key)
        if baseline is None:
            continue
        row["best_single_policy"] = baseline["policy"]
        row["best_single_real_recall"] = baseline["real_recall"]
        row["best_single_camb_recall"] = baseline["camb_recall"]
        row["best_single_real_fpr"] = baseline["real_fpr"]
        row["real_recall_gain_vs_best_single"] = (
            row["real_recall"] - baseline["real_recall"]
        )
        row["camb_recall_gain_vs_best_single"] = (
            row["camb_recall"] - baseline["camb_recall"]
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            work = dict(row)
            if isinstance(work.get("thresholds"), dict):
                work["thresholds"] = json.dumps(work["thresholds"], sort_keys=True)
            writer.writerow(work)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    rows = report["top_rows"]
    best_single_rows = report["best_single_rows"]
    lines = ["# Remediated v1 Policy Pareto Search", ""]
    lines.append("## Assumptions")
    lines.append("")
    for note in metadata["assumption_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    for constraint in metadata["constraints"]:
        camb_limit = constraint["camb_fpr_max"]
        real_limit = constraint["real_fpr_max"]
        lines.append(f"## CAMB FPR <= `{camb_limit}`, Real FPR <= `{real_limit}`")
        lines.append("")
        baseline = next(
            (
                row
                for row in best_single_rows
                if row["constraint_camb_fpr_max"] == camb_limit
                and row["constraint_real_fpr_max"] == real_limit
            ),
            None,
        )
        rank1 = next(
            (
                row
                for row in rows
                if row["constraint_camb_fpr_max"] == camb_limit
                and row["constraint_real_fpr_max"] == real_limit
                and row["rank"] == 1
            ),
            None,
        )
        if baseline is not None and rank1 is not None:
            gain = rank1["real_recall"] - baseline["real_recall"]
            lines.append(
                "Best single-score baseline: "
                f"`{baseline['policy']}` with real recall "
                f"`{baseline['real_recall']:.4f}` and real FPR "
                f"`{baseline['real_fpr']:.4f}`. Rank-1 policy gain: "
                f"`{gain:+.4f}` real-recall points."
            )
            lines.append("")
        lines.append(
            "| rank | family | policy | CAMB FPR | real FPR | CAMB recall | "
            "real recall | gain vs single | real FP |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            if (
                row["constraint_camb_fpr_max"] != camb_limit
                or row["constraint_real_fpr_max"] != real_limit
            ):
                continue
            gain = row.get("real_recall_gain_vs_best_single")
            gain_text = "" if gain is None else f"{gain:+.4f}"
            lines.append(
                f"| {row['rank']} | `{row['family']}` | `{row['policy']}` | "
                f"{row['camb_fpr']:.4f} | {row['real_fpr']:.4f} | "
                f"{row['camb_recall']:.4f} | {row['real_recall']:.4f} | "
                f"{gain_text} | {row['real_fp']} |"
            )
        lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- Do not tune deployment thresholds directly on the 200 real-SMICA negatives.")
    lines.append("- Treat these policies as candidates for larger real-null tile calibration.")
    lines.append("- Require a Batch 6-style full-sky tile audit before promoting any policy.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    sensitivity_labels, sensitivity_scores = load_scores(Path(args.sensitivity_scores).resolve())
    real_labels, real_scores = load_scores(Path(args.real_sky_scores).resolve())
    grids = {
        method: threshold_grid(
            method,
            sensitivity_labels,
            sensitivity_scores,
            real_labels,
            real_scores,
            int(args.num_quantiles),
        )
        for method in METHOD_KEYS
    }
    policy_rows = build_policy_rows(
        sensitivity_labels,
        sensitivity_scores,
        real_labels,
        real_scores,
        grids,
        confidence=float(args.confidence),
        kof3_step=int(args.kof3_step),
    )
    constraints = parse_constraints(args.constraints)
    top_rows = choose_top_rows(policy_rows, constraints, int(args.top_k))
    best_single_rows = choose_best_single_rows(
        sensitivity_labels,
        sensitivity_scores,
        real_labels,
        real_scores,
        constraints,
        confidence=float(args.confidence),
    )
    annotate_single_baseline(top_rows, best_single_rows)
    return {
        "metadata": {
            "sensitivity_scores": str(Path(args.sensitivity_scores).resolve()),
            "real_sky_scores": str(Path(args.real_sky_scores).resolve()),
            "num_policy_rows_searched": int(len(policy_rows)),
            "methods": list(METHOD_KEYS),
            "threshold_grid_sizes": {method: int(len(values)) for method, values in grids.items()},
            "constraints": [
                {"camb_fpr_max": float(camb), "real_fpr_max": float(real)}
                for camb, real in constraints
            ],
            "confidence": float(args.confidence),
            "assumption_notes": [
                "CAMB negatives are the 5000 negatives from the remediated sensitivity grid.",
                "Real-SMICA negatives are only 200 real-background injection controls.",
                "Single-score baselines are optimized over exact observed thresholds.",
                "Composite policies must be tile-audited before deployment.",
            ],
        },
        "best_single_rows": best_single_rows,
        "top_rows": top_rows,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    report = build_report(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "policy_pareto.json"
    md_path = output_dir / "policy_pareto.md"
    csv_path = output_dir / "policy_pareto_top.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, report["top_rows"])
    write_markdown(md_path, report)
    print(
        json.dumps(
            {"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
