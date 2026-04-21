"""
Evaluate classical baselines on the external stratified validation set.

This complements phase3_eval_stratified_external.py by putting the classical
matched-template and centered-disc baselines through the same matched-FPR,
bootstrap-CI, and per-bin reporting protocol as the ML checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve
from sklearn.metrics import average_precision_score, roc_auc_score

import phase3_train_unet as p3
from phase3_eval_stratified_external import (
    bootstrap_auc,
    choose_threshold_at_fpr,
    group_masks,
    load_stratification,
    summarize_groups,
)
from phase3_sensitivity_curve import (
    SIGN_QUADRANTS,
    make_centered_disc_kernel,
    make_feeney_template_kernel,
    score_centered_disc_patch,
    standardize_patch,
)
from phase_dataset_utils import make_angular_distance_grid
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "test_data.h5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_classical_baselines" / "test"


def parse_float_list(text):
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical matched-template/centered-disc evaluation on the stratified external HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-h5", type=str, default=str(DEFAULT_DATA_H5))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--matched-fpr", type=float, default=0.08)
    parser.add_argument("--theta-grid-deg", type=str, default="5,10,15,20,25")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--classical-workers", type=int, default=8)
    return parser.parse_args()


def validate_args(args):
    args.theta_grid_deg = parse_float_list(args.theta_grid_deg)
    if not args.theta_grid_deg:
        raise ValueError("--theta-grid-deg must not be empty.")
    if not (0.0 < args.matched_fpr < 1.0):
        raise ValueError("--matched-fpr must be in (0, 1).")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive.")
    if args.classical_workers <= 0:
        raise ValueError("--classical-workers must be positive.")


def make_candidate_mask(center_x_pix, center_y_pix, radius_deg):
    theta = make_angular_distance_grid(PATCH_PIX, RESO_ARCMIN, center_x_pix=center_x_pix, center_y_pix=center_y_pix)
    return (theta <= np.radians(radius_deg)).astype(np.uint8)


def build_matched_kernel_bank(theta_grid_deg, beam_fwhm_arcmin):
    bank = []
    for theta in theta_grid_deg:
        for z0_sign, zcrit_sign in SIGN_QUADRANTS:
            bank.append(
                {
                    "theta_deg": float(theta),
                    "z0_sign": float(z0_sign),
                    "zcrit_sign": float(zcrit_sign),
                    "kernel": make_feeney_template_kernel(
                        theta,
                        z0_sign,
                        zcrit_sign,
                        beam_fwhm_arcmin=beam_fwhm_arcmin,
                    ),
                }
            )
    return bank


def build_centered_bank(theta_grid_deg):
    return [{"theta_deg": float(theta), "kernel": make_centered_disc_kernel(theta)} for theta in theta_grid_deg]


def score_matched_candidate(patch, kernel_bank):
    patch = standardize_patch(patch)
    best_score = -np.inf
    best_peak = (PATCH_PIX // 2, PATCH_PIX // 2)
    best_theta = float(kernel_bank[0]["theta_deg"])
    for item in kernel_bank:
        kernel = item["kernel"]
        response = fftconvolve(patch, kernel[::-1, ::-1], mode="same")
        peak = np.unravel_index(np.argmax(response), response.shape)
        score = float(response[peak])
        if score > best_score:
            best_score = score
            best_peak = peak
            best_theta = float(item["theta_deg"])
    center_y, center_x = best_peak
    return best_score, make_candidate_mask(float(center_x), float(center_y), best_theta)


def score_centered_candidate(patch, kernel_bank):
    patch = standardize_patch(patch)
    best_score = -np.inf
    best_theta = float(kernel_bank[0]["theta_deg"])
    for item in kernel_bank:
        score = abs(float(np.sum(patch * item["kernel"])))
        if score > best_score:
            best_score = score
            best_theta = float(item["theta_deg"])
    center = (PATCH_PIX - 1) / 2.0
    return best_score, make_candidate_mask(center, center, best_theta)


def score_method(patches, method_name, theta_grid_deg, beam_fwhm_arcmin, workers):
    if method_name == "matched_template":
        bank = build_matched_kernel_bank(theta_grid_deg, beam_fwhm_arcmin)
        scorer = lambda patch: score_matched_candidate(patch, bank)
    elif method_name == "centered_disc":
        bank = build_centered_bank(theta_grid_deg)
        scorer = lambda patch: score_centered_candidate(patch, bank)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    n = int(patches.shape[0])
    scores = np.zeros(n, dtype=np.float32)
    pred_masks = np.zeros((n, PATCH_PIX, PATCH_PIX), dtype=np.uint8)

    def score_one(idx):
        score, mask = scorer(patches[idx])
        return idx, score, mask

    progress = p3.ProgressPrinter(n, f"Classical stratified {method_name} ({workers} threads)")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(score_one, idx) for idx in range(n)]
        for future in as_completed(futures):
            idx, score, mask = future.result()
            scores[idx] = score
            pred_masks[idx] = mask
            completed += 1
            if completed % 250 == 0 or completed == n:
                progress.update(completed)
    return scores, pred_masks


def dice_from_masks(pred_masks, truth_masks, labels, active):
    dices = np.full(len(labels), np.nan, dtype=np.float32)
    positive = np.flatnonzero(labels == 1)
    for idx in positive:
        pred = pred_masks[idx].astype(bool) if active[idx] else np.zeros_like(pred_masks[idx], dtype=bool)
        truth = truth_masks[idx].astype(bool)
        intersection = float(np.logical_and(pred, truth).sum())
        pred_sum = float(pred.sum())
        truth_sum = float(truth.sum())
        if pred_sum == 0.0 and truth_sum == 0.0:
            dices[idx] = 1.0
        else:
            dices[idx] = (2.0 * intersection + 1e-8) / (pred_sum + truth_sum + 1e-8)
    return dices


def write_markdown(path, report):
    lines = ["# Classical Stratified External Evaluation", ""]
    lines.append(f"Dataset: `{report['data_h5']}`")
    lines.append(f"Matched FPR: `{report['matched_fpr']}`")
    lines.append("")
    lines.append("| model | AUROC | AUPRC | threshold | precision | recall | FPR | F1 | weak recall | Dice+ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["models"]:
        weak = row["groups"].get("weak_family_union", {})
        all_pos = row["groups"].get("all_positive", {})
        lines.append(
            "| {name} | {auroc:.3f} [{auroc_lo:.3f},{auroc_hi:.3f}] | "
            "{auprc:.3f} [{auprc_lo:.3f},{auprc_hi:.3f}] | {threshold:.6f} | "
            "{precision:.3f} | {recall:.3f} | {fpr:.3f} | {f1:.3f} | "
            "{weak_recall:.3f} [{weak_lo:.3f},{weak_hi:.3f}] | "
            "{dice:.3f} [{dice_lo:.3f},{dice_hi:.3f}] |".format(
                name=row["name"],
                auroc=row["auroc"],
                auroc_lo=row["auroc_ci95"][0],
                auroc_hi=row["auroc_ci95"][1],
                auprc=row["auprc"],
                auprc_lo=row["auprc_ci95"][0],
                auprc_hi=row["auprc_ci95"][1],
                threshold=row["matched_threshold"],
                precision=row["matched_metrics"]["precision"],
                recall=row["matched_metrics"]["recall"],
                fpr=row["matched_metrics"]["fpr"],
                f1=row["matched_metrics"]["f1"],
                weak_recall=weak.get("recall", float("nan")),
                weak_lo=weak.get("recall_ci95", [float("nan"), float("nan")])[0],
                weak_hi=weak.get("recall_ci95", [float("nan"), float("nan")])[1],
                dice=all_pos.get("dice_mean", float("nan")),
                dice_lo=all_pos.get("dice_ci95", [float("nan"), float("nan")])[0],
                dice_hi=all_pos.get("dice_ci95", [float("nan"), float("nan")])[1],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    data_h5 = Path(args.data_h5).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_h5, "r") as h5:
        patches = np.asarray(h5["patches"][:], dtype=np.float32)
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        truth_masks = np.asarray(h5["masks"][:], dtype=np.uint8)
        beam_fwhm_arcmin = float(h5["summary"].attrs.get("beam_fwhm_arcmin", 0.0)) if "summary" in h5 else 0.0

    strat = load_stratification(data_h5, len(labels))
    groups = group_masks(strat)
    rng = np.random.default_rng(args.bootstrap_seed)
    report = {
        "data_h5": str(data_h5),
        "num_samples": int(len(labels)),
        "num_positive": int(labels.sum()),
        "num_negative": int(len(labels) - labels.sum()),
        "matched_fpr": float(args.matched_fpr),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "theta_grid_deg": list(args.theta_grid_deg),
        "beam_fwhm_arcmin": float(beam_fwhm_arcmin),
        "models": [],
    }

    for method_name in ("matched_template", "centered_disc"):
        scores, pred_masks = score_method(
            patches,
            method_name=method_name,
            theta_grid_deg=args.theta_grid_deg,
            beam_fwhm_arcmin=beam_fwhm_arcmin,
            workers=args.classical_workers,
        )
        threshold, matched_metrics = choose_threshold_at_fpr(scores, labels, args.matched_fpr)
        active = scores >= threshold
        dices = dice_from_masks(pred_masks, truth_masks, labels, active)
        group_report = summarize_groups(groups, labels, active, dices, rng, args.bootstrap_resamples)
        auc_ci = bootstrap_auc(scores, labels, args.bootstrap_resamples, rng)
        np.savez_compressed(
            output_dir / f"{method_name}_scores_masks.npz",
            scores=scores,
            pred_masks=pred_masks,
            labels=labels,
        )
        report["models"].append(
            {
                "name": method_name,
                "score_kind": method_name,
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
                **auc_ci,
                "matched_threshold": float(threshold),
                "matched_metrics": matched_metrics,
                "groups": group_report,
            }
        )

    json_path = output_dir / "classical_stratified_eval_report.json"
    md_path = output_dir / "classical_stratified_eval_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
