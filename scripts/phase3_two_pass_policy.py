"""
Evaluate a loose-ML proposal plus circular-template verifier policy.

Pass 1 keeps real-SMICA positive injections and SMICA nulls whose
`v6_aux_only` score exceeds a loose proposal threshold. Pass 2 applies a
circular-template threshold calibrated on marginal null activation over all
null rows, then also reports the conditional rate among ML-kept nulls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binomtest

import phase3_train_unet as p3
from phase2_generate_training import fwhm_arcmin_to_sigma_pixels
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN, T_CMB_K, bubble_collision_signal
from phase3_sensitivity_curve import (
    SIGN_QUADRANTS,
    make_feeney_template_kernel,
    score_circular_template_patch,
)
from phase3_method_registry import CIRCULAR_TEMPLATE_SCREEN, method_metadata
from phase_dataset_utils import make_angular_distance_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_H5 = PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_injection_v1" / "smica_real_sky_injection.h5"
DEFAULT_POSITIVE_SCORES = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "real_sky_smoothed_sensitivity_v1"
    / "smoothed_v6_aux_only_scores.npz"
)
DEFAULT_NULL_V6 = PROJECT_ROOT / "runs" / "phase3_unet" / "ensemble_eval_v1" / "score_cache" / "null_v6_aux_only_scores.npz"
DEFAULT_NULL_CLASSICAL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "real_sky_recalibration_v1"
    / "score_cache"
    / f"null_{CIRCULAR_TEMPLATE_SCREEN}_scores.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "two_pass_policy_v1"


REGIMES = (
    ("dead_A_le_2e-6", None, 2e-6),
    ("contested_5e-6_to_2e-5", 5e-6, 2e-5),
    ("solved_A_ge_5e-5", 5e-5, None),
    ("contested_plus_solved_A_ge_5e-6", 5e-6, None),
)


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v6 loose proposal followed by circular-template verifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-h5", type=str, default=str(DEFAULT_SOURCE_H5))
    parser.add_argument("--positive-scores", type=str, default=str(DEFAULT_POSITIVE_SCORES))
    parser.add_argument("--null-v6-scores", type=str, default=str(DEFAULT_NULL_V6))
    parser.add_argument("--null-classical-scores", type=str, default=str(DEFAULT_NULL_CLASSICAL))
    parser.add_argument(
        "--null-matched-scores",
        type=str,
        default="",
        help="Legacy alias for --null-classical-scores.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ml-threshold", type=float, default=0.75)
    parser.add_argument("--budgets", type=str, default="200,400,600,800")
    parser.add_argument("--full-sky-independent-patches", type=float, default=3000.0)
    parser.add_argument("--theta-grid-deg", type=str, default="5,10,15,20,25")
    parser.add_argument("--classical-beam-fwhm-arcmin", type=float, default=5.0)
    parser.add_argument(
        "--signal-beam-fwhm-arcmin",
        type=float,
        default=0.0,
        help="Optional legacy patch-space smoothing for reconstructed injected signals. Keep 0 for remediated products.",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def exact_ci(k: int, n: int) -> tuple[float, float]:
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def load_score_array(path: Path, preferred: str = "scores") -> np.ndarray:
    with np.load(path) as loaded:
        if preferred in loaded:
            return np.asarray(loaded[preferred], dtype=np.float32)
        if "scores" in loaded:
            return np.asarray(loaded["scores"], dtype=np.float32)
        if "positive_scores" in loaded:
            return np.asarray(loaded["positive_scores"], dtype=np.float32)
        if "negative_scores" in loaded:
            return np.asarray(loaded["negative_scores"], dtype=np.float32)
    raise KeyError(f"No usable score array in {path}")


def regime_mask(amplitudes: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    mask = np.ones_like(amplitudes, dtype=bool)
    if low is not None:
        mask &= amplitudes >= float(low) * (1.0 - 1e-5)
    if high is not None:
        mask &= amplitudes <= float(high) * (1.0 + 1e-5)
    return mask


def load_positive_artifact(path: Path) -> dict:
    with np.load(path) as loaded:
        return {
            "v6_scores": np.asarray(loaded["positive_scores"], dtype=np.float32),
            "source_index": np.asarray(loaded["positive_source_index"], dtype=np.int64),
            "amplitude": np.asarray(loaded["positive_amplitude"], dtype=np.float64),
            "theta_crit_deg": np.asarray(loaded["positive_theta_crit_deg"], dtype=np.float64),
            "edge_sigma_deg": np.asarray(loaded["positive_edge_sigma_deg"], dtype=np.float32),
        }


def build_circular_template_kernels(theta_grid: tuple[float, ...], beam_fwhm_arcmin: float) -> list[np.ndarray]:
    return [
        make_feeney_template_kernel(theta, z0_sign, zcrit_sign, beam_fwhm_arcmin=beam_fwhm_arcmin)
        for theta in theta_grid
        for z0_sign, zcrit_sign in SIGN_QUADRANTS
    ]


def reconstruct_patch(
    base: np.ndarray,
    center_x: float,
    center_y: float,
    theta_crit_deg: float,
    z0: float,
    zcrit: float,
    edge_sigma_deg: float,
    beam_sigma_pix: float,
) -> np.ndarray:
    theta_dist = make_angular_distance_grid(
        PATCH_PIX,
        RESO_ARCMIN,
        center_x_pix=float(center_x),
        center_y_pix=float(center_y),
    ).astype(np.float32)
    signal = bubble_collision_signal(
        theta_dist,
        float(z0),
        float(zcrit),
        np.radians(float(theta_crit_deg)),
        edge_sigma_deg=float(edge_sigma_deg),
    )
    signal_delta = np.asarray(signal * (T_CMB_K + base), dtype=np.float32)
    if beam_sigma_pix > 0.0:
        signal_delta = gaussian_filter(signal_delta, sigma=beam_sigma_pix, mode="reflect")
    return (base + signal_delta).astype(np.float32)


def score_circular_template_positives(args: argparse.Namespace, positives: dict, output_dir: Path, ml_keep: np.ndarray) -> np.ndarray:
    cache_dir = output_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{CIRCULAR_TEMPLATE_SCREEN}_positive_scores_ml_tau_{args.ml_threshold:.3f}.npz"
    if args.reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            scores = np.asarray(loaded["scores"], dtype=np.float32)
            if scores.shape == positives["v6_scores"].shape:
                return scores

    kernels = build_circular_template_kernels(parse_float_list(args.theta_grid_deg), args.classical_beam_fwhm_arcmin)
    beam_sigma_pix = fwhm_arcmin_to_sigma_pixels(args.signal_beam_fwhm_arcmin)
    rows_to_score = np.flatnonzero(ml_keep)
    scores = np.full_like(positives["v6_scores"], np.nan, dtype=np.float32)

    source_h5 = Path(args.source_h5).resolve()
    with h5py.File(source_h5, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        neg_idx = np.flatnonzero(labels == 0)
        base_patches = np.asarray(h5["patches"][neg_idx], dtype=np.float32)
        background_index = np.asarray(h5["metadata"]["background_index"][:], dtype=np.int64)
        z0 = np.asarray(h5["truth"]["z0"][:], dtype=np.float32)
        zcrit = np.asarray(h5["truth"]["zcrit"][:], dtype=np.float32)
        center_x = np.asarray(h5["truth"]["signal_center_x_pix"][:], dtype=np.float32)
        center_y = np.asarray(h5["truth"]["signal_center_y_pix"][:], dtype=np.float32)
        theta_crit = np.asarray(h5["truth"]["theta_crit_deg"][:], dtype=np.float32)
        source_rows = positives["source_index"]
        edge_sigma = positives["edge_sigma_deg"]

        progress = p3.ProgressPrinter(
            len(rows_to_score),
            f"{CIRCULAR_TEMPLATE_SCREEN} ML-kept positives ({args.workers} threads)",
        )
        completed = 0
        for start in range(0, len(rows_to_score), args.chunk_size):
            stop = min(start + args.chunk_size, len(rows_to_score))
            local_rows = rows_to_score[start:stop]
            patches = []
            for local_pos_idx in local_rows:
                source_row = int(source_rows[local_pos_idx])
                bg_idx = int(background_index[source_row])
                base = base_patches[bg_idx]
                patches.append(
                    reconstruct_patch(
                        base,
                        center_x=float(center_x[source_row]),
                        center_y=float(center_y[source_row]),
                        theta_crit_deg=float(theta_crit[source_row]),
                        z0=float(z0[source_row]),
                        zcrit=float(zcrit[source_row]),
                        edge_sigma_deg=float(edge_sigma[local_pos_idx]),
                        beam_sigma_pix=beam_sigma_pix,
                    )
                )
            patches = np.asarray(patches, dtype=np.float32)

            def score_one(chunk_idx: int) -> tuple[int, float]:
                return int(local_rows[chunk_idx]), score_circular_template_patch(patches[chunk_idx], kernels)

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(score_one, idx) for idx in range(len(local_rows))]
                for future in as_completed(futures):
                    local_pos_idx, score = future.result()
                    scores[local_pos_idx] = float(score)
                    completed += 1
                    if completed % 250 == 0 or completed == len(rows_to_score):
                        progress.update(completed)

    np.savez_compressed(
        cache_path,
        scores=scores,
        ml_keep=ml_keep.astype(np.uint8),
        ml_threshold=np.asarray(args.ml_threshold, dtype=np.float32),
        method=CIRCULAR_TEMPLATE_SCREEN,
        method_metadata=json.dumps(method_metadata(CIRCULAR_TEMPLATE_SCREEN), sort_keys=True),
        source_positive_scores=str(Path(args.positive_scores).resolve()),
    )
    return scores


def threshold_for_marginal_budget(
    ml_keep_null: np.ndarray,
    null_scores: np.ndarray,
    budget: float,
    full_sky_patches: float,
) -> tuple[float, int, int]:
    """Choose a verifier threshold by marginal full-null false-positive budget."""

    ml_keep_null = np.asarray(ml_keep_null, dtype=bool)
    null_scores = np.asarray(null_scores, dtype=np.float32)
    if ml_keep_null.shape != null_scores.shape:
        raise ValueError("ML proposal mask and null verifier scores must have the same shape.")
    n_total_null = int(null_scores.size)
    kept_scores = null_scores[ml_keep_null]
    if kept_scores.size == 0:
        return float("inf"), 0, 0

    max_fp = int(np.floor(float(budget) / float(full_sky_patches) * float(n_total_null)))
    if max_fp <= 0:
        return float(np.nextafter(np.max(kept_scores), np.inf)), 0, max_fp
    if max_fp >= kept_scores.size:
        return float(-np.inf), int(kept_scores.size), max_fp
    sorted_desc = np.sort(kept_scores)[::-1]
    threshold = float(sorted_desc[max_fp])
    fp = int(np.sum(ml_keep_null & (null_scores > threshold)))
    while fp > max_fp:
        threshold = float(np.nextafter(threshold, np.inf))
        fp = int(np.sum(ml_keep_null & (null_scores > threshold)))
    return threshold, fp, max_fp


def summarize_policy(
    positives: dict,
    null_v6: np.ndarray,
    null_classical: np.ndarray,
    classical_pos: np.ndarray,
    ml_keep_pos: np.ndarray,
    ml_keep_null: np.ndarray,
    budgets: tuple[float, ...],
    full_sky_patches: float,
) -> tuple[list[dict], list[dict]]:
    amplitudes = positives["amplitude"]
    radii = positives["theta_crit_deg"]
    n_null = int(null_v6.size)
    kept_null_classical_n = int(np.sum(ml_keep_null))
    rows: list[dict] = []
    radius_rows: list[dict] = []
    regime_masks = [(name, regime_mask(amplitudes, low, high)) for name, low, high in REGIMES]
    contested_mask = regime_mask(amplitudes, 5e-6, 2e-5)
    radius_values = [float(x) for x in sorted(np.unique(radii))]

    for budget in budgets:
        classical_threshold, fp_kept, max_fp = threshold_for_marginal_budget(
            ml_keep_null,
            null_classical,
            budget,
            full_sky_patches,
        )
        pos_active = ml_keep_pos & (classical_pos > classical_threshold)
        null_active = ml_keep_null & (null_classical > classical_threshold)
        fp = int(null_active.sum())
        null_fpr = float(fp / max(n_null, 1))
        conditional_fpr = float(fp / max(kept_null_classical_n, 1))
        fp_low, fp_high = exact_ci(fp, n_null)
        global_k = int(pos_active.sum())
        global_n = int(pos_active.size)
        global_low, global_high = exact_ci(global_k, global_n)
        row = {
            "budget": float(budget),
            "ml_threshold": float(np.nan),
            "classical_method": CIRCULAR_TEMPLATE_SCREEN,
            "classical_threshold": float(classical_threshold),
            "matched_threshold": float(classical_threshold),
            "null_fp": fp,
            "null_n": n_null,
            "null_fpr": null_fpr,
            "marginal_null_fpr": null_fpr,
            "conditional_null_fpr_after_ml_keep": conditional_fpr,
            "ml_kept_null_n": kept_null_classical_n,
            "max_marginal_fp_allowed": int(max_fp),
            "expected_fp_3000": float(null_fpr * full_sky_patches),
            "expected_fp_3000_ci95_low": float(fp_low * full_sky_patches),
            "expected_fp_3000_ci95_high": float(fp_high * full_sky_patches),
            "global_detected": global_k,
            "global_n": global_n,
            "global_recall": float(global_k / max(global_n, 1)),
            "global_ci95_low": global_low,
            "global_ci95_high": global_high,
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
            proposal_k = int(ml_keep_pos[mask].sum())
            row[f"{prefix}_proposal_recall"] = float(proposal_k / max(n, 1))
            row[f"{prefix}_retention_after_matched"] = float(k / max(proposal_k, 1))
        rows.append(row)

        for theta in radius_values:
            mask = contested_mask & np.isclose(radii, theta, rtol=1e-5, atol=0.0)
            k = int(pos_active[mask].sum())
            n = int(mask.sum())
            low, high = exact_ci(k, n)
            radius_rows.append(
                {
                    "budget": float(budget),
                    "theta_crit_deg": theta,
                    "detected": k,
                    "n": n,
                    "recall": float(k / max(n, 1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "classical_method": CIRCULAR_TEMPLATE_SCREEN,
                    "classical_threshold": float(classical_threshold),
                    "matched_threshold": float(classical_threshold),
                    "expected_fp_3000": float(null_fpr * full_sky_patches),
                }
            )
    return rows, radius_rows


def proposal_summary(positives: dict, null_v6: np.ndarray, ml_keep_pos: np.ndarray, ml_keep_null: np.ndarray, full_sky_patches: float) -> dict:
    amplitudes = positives["amplitude"]
    summary = {
        "positive_kept": int(ml_keep_pos.sum()),
        "positive_n": int(ml_keep_pos.size),
        "global_recall": float(ml_keep_pos.mean()),
        "null_kept": int(ml_keep_null.sum()),
        "null_n": int(null_v6.size),
        "null_fpr": float(ml_keep_null.mean()),
        "expected_fp_3000": float(ml_keep_null.mean() * full_sky_patches),
    }
    for name, low, high in REGIMES:
        mask = regime_mask(amplitudes, low, high)
        prefix = name.replace("_A_", "_")
        summary[f"{prefix}_recall"] = float(ml_keep_pos[mask].mean())
        summary[f"{prefix}_detected"] = int(ml_keep_pos[mask].sum())
        summary[f"{prefix}_n"] = int(mask.sum())
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_policy(path: Path, proposal: dict, rows: list[dict], dpi: int) -> None:
    budgets = np.asarray([row["expected_fp_3000"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [proposal["expected_fp_3000"]],
        [proposal["contested_5e-6_to_2e-5_recall"]],
        color="black",
        label=f"ML proposal only tau=0.75 ({proposal['expected_fp_3000']:.0f} FP)",
        zorder=3,
    )
    ax.plot(budgets, [row["contested_5e-6_to_2e-5_recall"] for row in rows], marker="o", label="two-pass contested")
    ax.plot(budgets, [row["solved_ge_5e-5_recall"] for row in rows], marker="o", label="two-pass solved")
    ax.plot(budgets, [row["global_recall"] for row in rows], marker="o", label="two-pass global")
    ax.set_xlabel("expected false positives / 3000 patches")
    ax.set_ylabel("recall")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Loose v6 proposal plus circular-template verifier")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_markdown(path: Path, proposal: dict, rows: list[dict]) -> None:
    lines = [
        "# Two-Pass Policy",
        "",
        "Policy: `v6_aux_only >= 0.75` proposal, followed by `circular_template_screen` verification calibrated on marginal SMICA-null activation.",
        "",
        "## Pass 1 Proposal",
        "",
        f"- Positive kept: `{proposal['positive_kept']} / {proposal['positive_n']}`; global recall `{proposal['global_recall']:.3f}`.",
        f"- Null kept: `{proposal['null_kept']} / {proposal['null_n']}`; expected FP / 3000 `{proposal['expected_fp_3000']:.0f}`.",
        f"- Dead recall: `{proposal['dead_le_2e-6_recall']:.3f}`.",
        f"- Contested recall: `{proposal['contested_5e-6_to_2e-5_recall']:.3f}`.",
        f"- Solved recall: `{proposal['solved_ge_5e-5_recall']:.3f}`.",
        "",
        "## Pass 2 Circular-Template Verifier",
        "",
        "| FP budget | threshold | marginal expected FP | conditional kept-null FPR | global recall | contested recall | solved recall | contested retention | solved retention |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['budget']:.0f} | {row['classical_threshold']:.3f} | {row['expected_fp_3000']:.0f} | "
            f"{row['conditional_null_fpr_after_ml_keep']:.3f} | {row['global_recall']:.3f} | "
            f"{row['contested_5e-6_to_2e-5_recall']:.3f} | "
            f"{row['solved_ge_5e-5_recall']:.3f} | {row['contested_5e-6_to_2e-5_retention_after_matched']:.3f} | "
            f"{row['solved_ge_5e-5_retention_after_matched']:.3f} |"
        )
    lines.extend(["", "## Decision", ""])
    best_800 = next((row for row in rows if abs(row["budget"] - 800.0) < 1e-9), rows[-1] if rows else None)
    best_400 = next((row for row in rows if abs(row["budget"] - 400.0) < 1e-9), None)
    if best_800 is not None:
        lines.append(
            f"At the 800-FP operating point, the two-pass policy has contested recall `{best_800['contested_5e-6_to_2e-5_recall']:.3f}` "
            f"and solved recall `{best_800['solved_ge_5e-5_recall']:.3f}`."
        )
    if best_400 is not None:
        lines.append(
            f"At the 400-FP operating point, contested recall is `{best_400['contested_5e-6_to_2e-5_recall']:.3f}`."
        )
    lines.append(
        "Compare these directly to the loose ML proposal: if circular-template verification removes many nulls but also removes most contested positives, then the false positives are not easy classical-template rejects."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.null_matched_scores:
        args.null_classical_scores = args.null_matched_scores
    if args.ml_threshold < 0.0 or args.ml_threshold > 1.0:
        raise ValueError("--ml-threshold must lie in [0, 1].")
    if args.classical_beam_fwhm_arcmin < 0.0:
        raise ValueError("--classical-beam-fwhm-arcmin must be non-negative.")
    if args.signal_beam_fwhm_arcmin < 0.0:
        raise ValueError("--signal-beam-fwhm-arcmin must be non-negative.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = parse_float_list(args.budgets)
    positives = load_positive_artifact(Path(args.positive_scores).resolve())
    null_v6 = load_score_array(Path(args.null_v6_scores).resolve())
    null_classical = load_score_array(Path(args.null_classical_scores).resolve())
    if null_v6.shape != null_classical.shape:
        raise ValueError(f"Null score shape mismatch: v6={null_v6.shape}, classical={null_classical.shape}")

    ml_keep_pos = positives["v6_scores"] >= float(args.ml_threshold)
    ml_keep_null = null_v6 >= float(args.ml_threshold)
    classical_pos = score_circular_template_positives(args, positives, output_dir, ml_keep_pos)
    proposal = proposal_summary(positives, null_v6, ml_keep_pos, ml_keep_null, args.full_sky_independent_patches)
    rows, radius_rows = summarize_policy(
        positives,
        null_v6,
        null_classical,
        classical_pos,
        ml_keep_pos,
        ml_keep_null,
        budgets,
        args.full_sky_independent_patches,
    )
    for row in rows:
        row["ml_threshold"] = float(args.ml_threshold)

    summary_csv = output_dir / "two_pass_policy_summary.csv"
    radius_csv = output_dir / "two_pass_contested_by_radius.csv"
    md_path = output_dir / "two_pass_policy.md"
    json_path = output_dir / "two_pass_policy.json"
    plot_path = output_dir / "two_pass_policy.png"
    write_csv(summary_csv, rows)
    write_csv(radius_csv, radius_rows)
    write_markdown(md_path, proposal, rows)
    plot_policy(plot_path, proposal, rows, args.dpi)
    json_path.write_text(
        json.dumps(
            {
                "proposal": proposal,
                "rows": rows,
                "classical_method": CIRCULAR_TEMPLATE_SCREEN,
                "classical_method_metadata": method_metadata(CIRCULAR_TEMPLATE_SCREEN),
                "positive_scores": str(Path(args.positive_scores).resolve()),
                "null_v6_scores": str(Path(args.null_v6_scores).resolve()),
                "null_classical_scores": str(Path(args.null_classical_scores).resolve()),
                "ml_threshold": float(args.ml_threshold),
                "classical_beam_fwhm_arcmin": float(args.classical_beam_fwhm_arcmin),
                "signal_beam_fwhm_arcmin": float(args.signal_beam_fwhm_arcmin),
                "injection_reconstruction_policy": (
                    "no_patch_space_injection_smoothing"
                    if args.signal_beam_fwhm_arcmin == 0.0
                    else "legacy_signal_delta_only_patch_space_smoothing"
                ),
                "summary_csv": str(summary_csv),
                "contested_by_radius_csv": str(radius_csv),
                "markdown": str(md_path),
                "plot": str(plot_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"markdown": str(md_path), "summary_csv": str(summary_csv), "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
