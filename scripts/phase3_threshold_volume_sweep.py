"""
Historical v6_aux_only threshold-volume sweep.

This pre-remediation diagnostic swept real-SMICA smoothed positives and SMICA
nulls for a given tolerated full-sky false-positive volume. Current deployment
calibration is the remediated Batch 6 tile audit; retain this script only for
provenance or explicitly reproduced legacy analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSITIVE_SCORES = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "real_sky_smoothed_sensitivity_v1"
    / "smoothed_v6_aux_only_scores.npz"
)
DEFAULT_NULL_SCORES = PROJECT_ROOT / "runs" / "phase3_unet" / "ensemble_eval_v1" / "score_cache" / "null_v6_aux_only_scores.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "threshold_volume_sweep_v1"


REGIMES = (
    ("dead_A_le_2e-6", None, 2e-6),
    ("contested_5e-6_to_2e-5", 5e-6, 2e-5),
    ("solved_A_ge_5e-5", 5e-5, None),
    ("contested_plus_solved_A_ge_5e-6", 5e-6, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold-volume sweep for cached v6 smoothed real-SMICA scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--positive-scores", type=str, default=str(DEFAULT_POSITIVE_SCORES))
    parser.add_argument("--null-scores", type=str, default=str(DEFAULT_NULL_SCORES))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--fine-step", type=float, default=0.001)
    parser.add_argument("--full-sky-independent-patches", type=float, default=3000.0)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def exact_ci(k: int, n: int) -> tuple[float, float]:
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def regime_mask(amplitudes: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    mask = np.ones_like(amplitudes, dtype=bool)
    if low is not None:
        mask &= amplitudes >= float(low) * (1.0 - 1e-5)
    if high is not None:
        mask &= amplitudes <= float(high) * (1.0 + 1e-5)
    return mask


def threshold_grid(args: argparse.Namespace) -> np.ndarray:
    count = int(np.floor((args.threshold_max - args.threshold_min) / args.threshold_step + 0.5)) + 1
    return np.round(args.threshold_min + np.arange(count) * args.threshold_step, 10)


def load_null_scores(path: Path) -> np.ndarray:
    with np.load(path) as loaded:
        if "scores" in loaded:
            return np.asarray(loaded["scores"], dtype=np.float32)
        if "negative_scores" in loaded:
            return np.asarray(loaded["negative_scores"], dtype=np.float32)
    raise KeyError(f"No scores array found in {path}")


def summarize_thresholds(
    thresholds: np.ndarray,
    positive_scores: np.ndarray,
    null_scores: np.ndarray,
    amplitudes: np.ndarray,
    radii: np.ndarray,
    full_sky_patches: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    radius_rows: list[dict] = []
    regime_masks = [(name, regime_mask(amplitudes, low, high)) for name, low, high in REGIMES]
    contested_mask = regime_mask(amplitudes, 5e-6, 2e-5)
    radius_values = [float(x) for x in sorted(np.unique(radii))]

    for tau in thresholds:
        pos_active = positive_scores > float(tau)
        null_active = null_scores > float(tau)
        fp = int(null_active.sum())
        n_null = int(null_scores.size)
        null_fpr = float(fp / max(n_null, 1))
        null_low, null_high = exact_ci(fp, n_null)
        global_k = int(pos_active.sum())
        global_n = int(pos_active.size)
        global_low, global_high = exact_ci(global_k, global_n)
        row = {
            "threshold": float(tau),
            "global_detected": global_k,
            "global_n": global_n,
            "global_recall": float(global_k / max(global_n, 1)),
            "global_ci95_low": global_low,
            "global_ci95_high": global_high,
            "null_fp": fp,
            "null_n": n_null,
            "null_fpr": null_fpr,
            "null_fpr_ci95_low": null_low,
            "null_fpr_ci95_high": null_high,
            "expected_fp_3000": float(null_fpr * full_sky_patches),
            "expected_fp_3000_ci95_low": float(null_low * full_sky_patches),
            "expected_fp_3000_ci95_high": float(null_high * full_sky_patches),
        }
        for name, mask in regime_masks:
            k = int(pos_active[mask].sum())
            n = int(mask.sum())
            low, high = exact_ci(k, n)
            prefix = name.replace("_A_", "_")
            row[f"{prefix}_detected"] = k
            row[f"{prefix}_n"] = n
            row[f"{prefix}_recall"] = float(k / max(n, 1))
            row[f"{prefix}_ci95_low"] = low
            row[f"{prefix}_ci95_high"] = high
        rows.append(row)

        for theta in radius_values:
            mask = contested_mask & np.isclose(radii, theta, rtol=1e-5, atol=0.0)
            k = int(pos_active[mask].sum())
            n = int(mask.sum())
            low, high = exact_ci(k, n)
            radius_rows.append(
                {
                    "threshold": float(tau),
                    "theta_crit_deg": theta,
                    "detected": k,
                    "n": n,
                    "recall": float(k / max(n, 1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "null_fpr": null_fpr,
                    "expected_fp_3000": float(null_fpr * full_sky_patches),
                }
            )
    return rows, radius_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_operating_points(fine_rows: list[dict]) -> dict:
    contested_key = "contested_5e-6_to_2e-5_recall"
    by_threshold = sorted(fine_rows, key=lambda row: row["threshold"])
    max_tau_contested_050 = None
    for row in by_threshold:
        if row[contested_key] >= 0.50:
            max_tau_contested_050 = row
    first_fp_le_800 = next((row for row in by_threshold if row["expected_fp_3000"] <= 800.0), None)
    first_fp_le_500 = next((row for row in by_threshold if row["expected_fp_3000"] <= 500.0), None)
    first_fp_le_300 = next((row for row in by_threshold if row["expected_fp_3000"] <= 300.0), None)
    return {
        "max_threshold_with_contested_recall_ge_0p50": max_tau_contested_050,
        "lowest_threshold_with_expected_fp_le_800": first_fp_le_800,
        "lowest_threshold_with_expected_fp_le_500": first_fp_le_500,
        "lowest_threshold_with_expected_fp_le_300": first_fp_le_300,
    }


def plot_sweep(path: Path, rows: list[dict], dpi: int) -> None:
    tau = np.asarray([row["threshold"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(tau, [row["dead_le_2e-6_recall"] for row in rows], marker="o", label="dead A<=2e-6")
    axes[0].plot(tau, [row["contested_5e-6_to_2e-5_recall"] for row in rows], marker="o", label="contested 5e-6..2e-5")
    axes[0].plot(tau, [row["solved_ge_5e-5_recall"] for row in rows], marker="o", label="solved A>=5e-5")
    axes[0].plot(tau, [row["global_recall"] for row in rows], marker="o", label="global", alpha=0.7)
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("recall")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(tau, [row["expected_fp_3000"] for row in rows], marker="o", color="black")
    axes[1].axhline(300, color="tab:green", linestyle="--", linewidth=1.0, label="300 FP")
    axes[1].axhline(500, color="tab:orange", linestyle="--", linewidth=1.0, label="500 FP")
    axes[1].axhline(800, color="tab:red", linestyle="--", linewidth=1.0, label="800 FP")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("expected false positives / 3000 patches")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("v6_aux_only threshold-volume sweep on smoothed real-SMICA positives")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_markdown(path: Path, coarse_rows: list[dict], operating_points: dict, full_sky_patches: float) -> None:
    lines = [
        "# Threshold-Volume Sweep",
        "",
        "`v6_aux_only` was swept over cached smoothed real-SMICA positive scores and cached 5000-patch SMICA null scores.",
        f"Expected false positives use `{full_sky_patches:.0f}` independent full-sky patches.",
        "",
        "## Coarse Sweep",
        "",
        "| threshold | dead recall | contested recall | solved recall | global recall | null FPR | expected FP |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in coarse_rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['dead_le_2e-6_recall']:.3f} | "
            f"{row['contested_5e-6_to_2e-5_recall']:.3f} | {row['solved_ge_5e-5_recall']:.3f} | "
            f"{row['global_recall']:.3f} | {row['null_fpr']:.3f} | {row['expected_fp_3000']:.0f} |"
        )
    lines.extend(["", "## Operating Points", ""])
    labels = [
        ("max_threshold_with_contested_recall_ge_0p50", "Highest threshold with contested recall >= 0.50"),
        ("lowest_threshold_with_expected_fp_le_800", "Lowest threshold with expected FP <= 800"),
        ("lowest_threshold_with_expected_fp_le_500", "Lowest threshold with expected FP <= 500"),
        ("lowest_threshold_with_expected_fp_le_300", "Lowest threshold with expected FP <= 300"),
    ]
    for key, label in labels:
        row = operating_points.get(key)
        if row is None:
            lines.append(f"- {label}: not reached in fine grid.")
            continue
        lines.append(
            f"- {label}: threshold `{row['threshold']:.3f}`, "
            f"contested recall `{row['contested_5e-6_to_2e-5_recall']:.3f}`, "
            f"global recall `{row['global_recall']:.3f}`, "
            f"null FPR `{row['null_fpr']:.3f}`, expected FP `{row['expected_fp_3000']:.0f}`."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The gallery read was correct: many contested positives live below the SMICA-calibrated 0.10-FPR threshold and can be recovered by lowering the threshold.",
            "",
            "However, contested recall `>= 0.50` requires about `1100` expected full-sky false positives for v6 alone. If the operational budget is closer to `800` candidates, contested recall is about `0.41`; if the budget is closer to `500`, contested recall is about `0.30`; if the budget is around `300`, it is about `0.23`.",
            "",
            "So the decision is no longer a model-blindness question. It is an explicit candidate-volume budget question.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(args.positive_scores).resolve()) as loaded:
        positive_scores = np.asarray(loaded["positive_scores"], dtype=np.float32)
        amplitudes = np.asarray(loaded["positive_amplitude"], dtype=np.float64)
        radii = np.asarray(loaded["positive_theta_crit_deg"], dtype=np.float64)
    null_scores = load_null_scores(Path(args.null_scores).resolve())

    coarse_thresholds = threshold_grid(args)
    fine_thresholds = np.round(np.arange(args.threshold_min, args.threshold_max + args.fine_step / 2.0, args.fine_step), 10)
    coarse_rows, coarse_radius_rows = summarize_thresholds(
        coarse_thresholds,
        positive_scores,
        null_scores,
        amplitudes,
        radii,
        args.full_sky_independent_patches,
    )
    fine_rows, _ = summarize_thresholds(
        fine_thresholds,
        positive_scores,
        null_scores,
        amplitudes,
        radii,
        args.full_sky_independent_patches,
    )
    operating_points = find_operating_points(fine_rows)

    coarse_csv = output_dir / "v6_smoothed_threshold_sweep.csv"
    radius_csv = output_dir / "v6_smoothed_contested_by_radius.csv"
    fine_csv = output_dir / "v6_smoothed_threshold_sweep_fine.csv"
    json_path = output_dir / "threshold_volume_sweep.json"
    md_path = output_dir / "threshold_volume_sweep.md"
    plot_path = output_dir / "threshold_volume_sweep.png"

    write_csv(coarse_csv, coarse_rows)
    write_csv(radius_csv, coarse_radius_rows)
    write_csv(fine_csv, fine_rows)
    plot_sweep(plot_path, coarse_rows, args.dpi)
    write_markdown(md_path, coarse_rows, operating_points, args.full_sky_independent_patches)
    json_path.write_text(
        json.dumps(
            {
                "positive_scores": str(Path(args.positive_scores).resolve()),
                "null_scores": str(Path(args.null_scores).resolve()),
                "full_sky_independent_patches": float(args.full_sky_independent_patches),
                "coarse_csv": str(coarse_csv),
                "fine_csv": str(fine_csv),
                "contested_by_radius_csv": str(radius_csv),
                "plot": str(plot_path),
                "markdown": str(md_path),
                "operating_points": operating_points,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "markdown": str(md_path),
                "coarse_csv": str(coarse_csv),
                "fine_csv": str(fine_csv),
                "contested_by_radius_csv": str(radius_csv),
                "plot": str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
