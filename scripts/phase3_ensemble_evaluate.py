"""
Evaluate score-level and mask-level ensembles of ML branches.

Assumptions
-----------
* The default inputs are remediated-v1 candidate-screening artifacts.
* Model thresholds are scalar image-screening thresholds calibrated on the
  sensitivity grid; mask-level Dice is a diagnostic, not a detection claim.
* Policies are count-based for an arbitrary number of model branches:
  ``union_or`` means at least one branch passes, ``intersect`` means all pass,
  and ``majority`` means strictly more than half pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

import phase3_train_unet as p3
from phase3_eval_stratified_external import DEFAULT_MODELS, parse_model_spec
from phase3_sensitivity_curve import threshold_from_negatives
from phase3_evaluate_run import load_json, resolve_checkpoint_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENS_REPORT = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_report.json"
)
DEFAULT_SENS_SCORES = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_scores.npz"
)
DEFAULT_STRAT_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "test_data.h5"
DEFAULT_NULL_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "null_controls_smica_mask090.h5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_ensemble_eval"
POLICIES = ("union_or", "intersect", "majority", "score_avg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 3 model ensembles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_SENS_REPORT))
    parser.add_argument("--sensitivity-scores", type=str, default=str(DEFAULT_SENS_SCORES))
    parser.add_argument("--stratified-h5", type=str, default=str(DEFAULT_STRAT_H5))
    parser.add_argument("--null-h5", type=str, default=str(DEFAULT_NULL_H5))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", action="append", default=[], help="Model spec as in phase3_eval_stratified_external.py.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--skip-dice", action="store_true")
    return parser.parse_args()


def exact_ci(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def load_model(run_dir, checkpoint_arg, device):
    run_config = load_json(Path(run_dir) / "run_config.json")
    model_args = p3.model_args_from_run_config(run_config)
    model = p3.build_model(model_args).to(device)
    checkpoint_path, _ = resolve_checkpoint_path(Path(run_dir), checkpoint_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model, run_config


def score_model(spec, h5_path, output_dir, args, device, cache_prefix):
    cache_dir = output_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_prefix}_{spec.name}_scores.npz"
    if args.reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            return np.asarray(loaded["scores"], dtype=np.float32)

    model, run_config = load_model(spec.run_dir.resolve(), spec.checkpoint, device)
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["labels"].shape[0])
    dataset = p3.H5BubbleDataset(
        h5_path=h5_path,
        indices=np.arange(n, dtype=np.int64),
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 77,
        max_translate_pixels=0,
    )
    loader = p3.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    scores = np.zeros(n, dtype=np.float32)
    offset = 0
    progress = p3.ProgressPrinter(len(loader), f"Ensemble scores {cache_prefix}:{spec.name}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            mask_logits, _ = p3.unpack_model_output(model(images))
            probs = torch.sigmoid(mask_logits)
            batch_scores = probs.flatten(1).max(dim=1).values.detach().cpu().numpy()
            batch_size = int(images.shape[0])
            scores[offset : offset + batch_size] = batch_scores
            offset += batch_size
            progress.update(batch_idx)
    np.savez_compressed(cache_path, scores=scores)
    return scores


def load_sensitivity_scores(path, methods):
    with np.load(path) as loaded:
        labels = np.asarray(loaded["labels"], dtype=np.uint8)
        scores = {method: np.asarray(loaded[f"score__{method}"], dtype=np.float32) for method in methods}
    return labels, scores


def ensemble_scores(scores_by_model, thresholds):
    methods = list(scores_by_model)
    matrix = np.vstack([scores_by_model[method] for method in methods])
    votes = np.vstack([scores_by_model[method] > thresholds[method] for method in methods]).sum(axis=0)
    return {
        "union_or": votes.astype(np.float32),
        "intersect": votes.astype(np.float32),
        "majority": votes.astype(np.float32),
        "score_avg": matrix.mean(axis=0).astype(np.float32),
    }


def vote_thresholds(num_models):
    majority_count = (int(num_models) // 2) + 1
    return {
        "union_or": 0.5,
        "majority": float(majority_count) - 0.5,
        "intersect": float(num_models) - 0.5,
    }


def policy_thresholds(policy_scores, labels, fpr_target, num_models):
    thresholds = {
        **vote_thresholds(num_models),
    }
    avg_thr, _, _ = threshold_from_negatives(policy_scores["score_avg"], labels, fpr_target)
    thresholds["score_avg"] = float(avg_thr)
    return thresholds


def active_for_policy(policy, score, threshold):
    return score > float(threshold)


def summarize_binary(active, labels):
    labels = np.asarray(labels, dtype=np.uint8)
    active = np.asarray(active, dtype=bool)
    tp = int(np.logical_and(active, labels == 1).sum())
    fp = int(np.logical_and(active, labels == 0).sum())
    tn = int(np.logical_and(~active, labels == 0).sum())
    fn = int(np.logical_and(~active, labels == 1).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "fpr": fpr, "f1": f1}


def sensitivity_rows(h5_path, policy_scores, thresholds):
    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amplitude_grid = json.loads(h5["summary"].attrs["amplitude_grid"])
        theta_grid = json.loads(h5["summary"].attrs["theta_grid_deg"])
    rows = []
    for policy, scores in policy_scores.items():
        active = active_for_policy(policy, scores, thresholds[policy])
        for ai, amp in enumerate(amplitude_grid):
            for ti, theta in enumerate(theta_grid):
                mask = (labels == 1) & (amp_idx == ai) & (theta_idx == ti)
                n = int(mask.sum())
                k = int(active[mask].sum())
                ci = exact_ci(k, n)
                rows.append(
                    {
                        "policy": policy,
                        "amplitude": float(amp),
                        "theta_crit_deg": float(theta),
                        "num_positive": n,
                        "detected": k,
                        "p_det": float(k / max(n, 1)),
                        "ci95_low": ci[0],
                        "ci95_high": ci[1],
                    }
                )
    return rows


def compute_ensemble_masks_and_scores(specs, h5_path, thresholds, policy_thresholds_map, args, device):
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["labels"].shape[0])
        truth_masks = np.asarray(h5["masks"][:], dtype=np.uint8)
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    vote_counts = np.zeros((n, 256, 256), dtype=np.uint8)
    avg_probs = np.zeros((n, 256, 256), dtype=np.float32)

    for spec in specs:
        model, run_config = load_model(spec.run_dir.resolve(), spec.checkpoint, device)
        dataset = p3.H5BubbleDataset(
            h5_path=h5_path,
            indices=np.arange(n, dtype=np.int64),
            **p3.dataset_kwargs_from_run_config(run_config),
            augment=False,
            seed=int(run_config["args"]["seed"]) + 101,
            max_translate_pixels=0,
        )
        loader = p3.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        offset = 0
        progress = p3.ProgressPrinter(len(loader), f"Ensemble masks {spec.name}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                images = batch["image"].to(device, non_blocking=True)
                mask_logits, _ = p3.unpack_model_output(model(images))
                probs = torch.sigmoid(mask_logits).detach().cpu().numpy()[:, 0]
                b = int(probs.shape[0])
                vote_counts[offset : offset + b] += (probs > thresholds[spec.name]).astype(np.uint8)
                avg_probs[offset : offset + b] += (probs / float(len(specs))).astype(np.float32)
                offset += b
                progress.update(batch_idx)

    majority_count = (len(specs) // 2) + 1
    masks = {
        "union_or": vote_counts >= 1,
        "majority": vote_counts >= majority_count,
        "intersect": vote_counts >= len(specs),
        "score_avg": avg_probs > float(policy_thresholds_map["score_avg"]),
    }
    dices = {}
    positive = np.flatnonzero(labels == 1)
    for policy, pred_masks in masks.items():
        values = []
        for idx in positive:
            pred = pred_masks[idx]
            truth = truth_masks[idx].astype(bool)
            inter = float(np.logical_and(pred, truth).sum())
            pred_sum = float(pred.sum())
            truth_sum = float(truth.sum())
            if pred_sum == 0.0 and truth_sum == 0.0:
                values.append(1.0)
            else:
                values.append((2.0 * inter + 1e-8) / (pred_sum + truth_sum + 1e-8))
        dices[policy] = float(np.mean(values))
    return dices


def write_csv(path, rows):
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sens_report = load_json(Path(args.sensitivity_report))
    specs = [parse_model_spec(text) for text in (args.model or DEFAULT_MODELS)]
    model_names = [spec.name for spec in specs]
    model_thresholds = {}
    missing_thresholds = []
    for name in model_names:
        if name in sens_report["thresholds"]:
            model_thresholds[name] = float(sens_report["thresholds"][name]["threshold"])
        else:
            missing_thresholds.append(name)
    if missing_thresholds:
        raise KeyError(f"Sensitivity report is missing model thresholds: {missing_thresholds}")
    labels_sens, sens_model_scores = load_sensitivity_scores(args.sensitivity_scores, model_names)
    sens_policy_scores = ensemble_scores(sens_model_scores, model_thresholds)
    ens_thresholds = policy_thresholds(
        sens_policy_scores,
        labels_sens,
        float(sens_report["fpr_target"]),
        len(model_names),
    )

    sens_rows = sensitivity_rows(sens_report["data_h5"], sens_policy_scores, ens_thresholds)
    sens_metrics = {policy: summarize_binary(active_for_policy(policy, scores, ens_thresholds[policy]), labels_sens) for policy, scores in sens_policy_scores.items()}

    device = p3.resolve_device(args.device)
    null_labels = None
    null_model_scores = {}
    for spec in specs:
        null_model_scores[spec.name] = score_model(spec, Path(args.null_h5).resolve(), output_dir, args, device, "null")
    with h5py.File(args.null_h5, "r") as h5:
        null_labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    null_policy_scores = ensemble_scores(null_model_scores, model_thresholds)
    null_metrics = {policy: summarize_binary(active_for_policy(policy, scores, ens_thresholds[policy]), null_labels) for policy, scores in null_policy_scores.items()}

    strat_model_scores = {}
    for spec in specs:
        strat_model_scores[spec.name] = score_model(spec, Path(args.stratified_h5).resolve(), output_dir, args, device, "stratified")
    with h5py.File(args.stratified_h5, "r") as h5:
        strat_labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    strat_policy_scores = ensemble_scores(strat_model_scores, model_thresholds)
    strat_metrics = {}
    for policy, scores in strat_policy_scores.items():
        active = active_for_policy(policy, scores, ens_thresholds[policy])
        row = summarize_binary(active, strat_labels)
        row["auroc"] = float(roc_auc_score(strat_labels, scores))
        row["auprc"] = float(average_precision_score(strat_labels, scores))
        strat_metrics[policy] = row

    dice = {}
    if not args.skip_dice:
        dice = compute_ensemble_masks_and_scores(specs, Path(args.stratified_h5).resolve(), model_thresholds, ens_thresholds, args, device)
        for policy, value in dice.items():
            strat_metrics[policy]["dice_pos"] = value

    report = {
        "sensitivity_report": str(Path(args.sensitivity_report).resolve()),
        "sensitivity_scores": str(Path(args.sensitivity_scores).resolve()),
        "stratified_h5": str(Path(args.stratified_h5).resolve()),
        "null_h5": str(Path(args.null_h5).resolve()),
        "model_thresholds": model_thresholds,
        "ensemble_thresholds": ens_thresholds,
        "sensitivity_metrics": sens_metrics,
        "sensitivity_rows": sens_rows,
        "null_metrics": null_metrics,
        "stratified_metrics": strat_metrics,
    }
    json_path = output_dir / "ensemble_eval.json"
    sens_csv = output_dir / "ensemble_sensitivity.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(sens_csv, sens_rows)
    print(json.dumps({k: report[k] for k in ("ensemble_thresholds", "sensitivity_metrics", "null_metrics", "stratified_metrics")}, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV:  {sens_csv}")


if __name__ == "__main__":
    main()
