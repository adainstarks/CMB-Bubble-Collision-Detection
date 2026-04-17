"""
Test-time-augmentation (D4) and multi-model ensemble evaluator.

Reuses phase3_eval_stratified_external for stratification, bootstrap CIs, and
report writing. Scores are computed from per-sample D4-averaged probability
masks, optionally averaged across multiple model checkpoints. No retraining.

Each --model spec uses the same format as phase3_eval_stratified_external:
    name:run_dir:checkpoint:score_kind[:mask_threshold]
score_kind is accepted but only the mask-probability branch is used for
ensemble averaging (D4 TTA over auxiliary image logits would mix scales
across models and is not well defined).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import phase3_train_unet as p3
from phase3_evaluate_run import load_json, resolve_checkpoint_path
from phase3_eval_stratified_external import (
    ModelSpec,
    bootstrap_auc,
    build_model_from_run,
    ci,
    group_masks,
    load_all_labels,
    load_stratification,
    make_loader,
    parse_model_spec,
    save_curves,
    summarize_groups,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="D4 TTA and multi-model ensemble evaluator over the stratified external HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-h5", type=str, required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model spec name:run_dir:checkpoint:score_kind[:mask_threshold]. "
            "Ensemble: repeat --model. TTA alone: pass a single --model."
        ),
    )
    parser.add_argument("--ensemble-name", type=str, default="ensemble_tta")
    parser.add_argument(
        "--tta",
        type=str,
        default="d4",
        choices=["none", "flip", "d4"],
        help="none: no TTA. flip: identity + horizontal flip (2x). d4: 4 rotations x 2 flips (8x).",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--matched-fpr", type=float, default=0.08)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-samples", type=int, default=0, help="Debug only. 0 means all samples.")
    parser.add_argument(
        "--dice-threshold-mode",
        type=str,
        default="matched",
        choices=["matched", "fixed"],
        help="matched: use the FPR-calibrated score threshold for Dice. fixed: use --dice-threshold.",
    )
    parser.add_argument("--dice-threshold", type=float, default=0.5)
    return parser.parse_args()


def tta_transforms(mode):
    if mode == "none":
        return [("identity", lambda x: x, lambda y: y)]
    if mode == "flip":
        return [
            ("identity", lambda x: x, lambda y: y),
            ("flipW", lambda x: torch.flip(x, dims=[-1]), lambda y: torch.flip(y, dims=[-1])),
        ]
    if mode == "d4":
        ops = []
        for k in range(4):
            def fwd(x, k=k):
                return torch.rot90(x, k=k, dims=[-2, -1])

            def inv(y, k=k):
                return torch.rot90(y, k=-k, dims=[-2, -1])

            ops.append((f"rot{k*90}", fwd, inv))
            def fwd_flip(x, k=k):
                return torch.rot90(torch.flip(x, dims=[-1]), k=k, dims=[-2, -1])

            def inv_flip(y, k=k):
                return torch.flip(torch.rot90(y, k=-k, dims=[-2, -1]), dims=[-1])

            ops.append((f"rot{k*90}_flipW", fwd_flip, inv_flip))
        return ops
    raise ValueError(f"Unknown TTA mode: {mode}")


def tta_model_prob_mask(model, images, transforms):
    """Return the TTA-averaged sigmoid mask for a batch of images (on device)."""
    acc = None
    for _, fwd, inv in transforms:
        augmented = fwd(images)
        mask_logits, _ = p3.unpack_model_output(model(augmented))
        probs = torch.sigmoid(mask_logits)
        probs_back = inv(probs)
        acc = probs_back if acc is None else acc + probs_back
    return acc / float(len(transforms))


def collect_ensemble_scores(specs, data_h5, indices, args, device):
    """Average TTA probability masks across models, return per-sample max scores and per-sample probability masks stored to disk."""
    transforms = tta_transforms(args.tta)
    print(f"\n  TTA mode: {args.tta} ({len(transforms)} augmentations per forward pass)", flush=True)

    # Pre-build per-model handles, each with its own loader (run_config may differ across models)
    model_bundles = []
    for spec in specs:
        model, run_config, checkpoint_path, checkpoint_label = build_model_from_run(
            spec.run_dir.resolve(), spec.checkpoint, device
        )
        model_args = p3.model_args_from_run_config(run_config)
        model_bundles.append({
            "spec": spec,
            "model": model,
            "run_config": run_config,
            "model_args": model_args,
            "checkpoint_path": checkpoint_path,
            "checkpoint_label": checkpoint_label,
        })

    n = len(indices)
    scores = np.zeros(n, dtype=np.float32)

    # Loaders: each model gets its own loader because H5BubbleDataset uses
    # run-config normalization. We iterate synchronously by sample index.
    loaders = [make_loader(data_h5, indices, b["run_config"], args, device) for b in model_bundles]
    progress = p3.ProgressPrinter(len(loaders[0]), f"Score {args.ensemble_name}")
    offset = 0
    with torch.no_grad():
        for batch_idx, batches in enumerate(zip(*loaders), start=1):
            batch_size = int(batches[0]["image"].shape[0])
            ensemble_probs = None
            for bundle, batch in zip(model_bundles, batches):
                images = batch["image"].to(device, non_blocking=True)
                probs = tta_model_prob_mask(bundle["model"], images, transforms)
                ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs
            ensemble_probs = ensemble_probs / float(len(model_bundles))
            batch_scores = ensemble_probs.flatten(1).max(dim=1).values.detach().cpu().numpy()
            scores[offset : offset + batch_size] = batch_scores
            offset += batch_size
            progress.update(batch_idx)

    return scores, model_bundles, transforms


def collect_ensemble_dice(specs_bundles, data_h5, indices, transforms, score_threshold, args, device):
    """Dice pass: recompute the TTA-averaged ensemble mask and threshold at score_threshold per-pixel."""
    loaders = [make_loader(data_h5, indices, b["run_config"], args, device) for b in specs_bundles]
    n = len(indices)
    dices = np.zeros(n, dtype=np.float32)
    image_pred = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=np.uint8)
    progress = p3.ProgressPrinter(len(loaders[0]), "Dice pass")
    offset = 0
    with torch.no_grad():
        for batch_idx, batches in enumerate(zip(*loaders), start=1):
            batch_size = int(batches[0]["image"].shape[0])
            targets = batches[0]["mask"].to(device, non_blocking=True) >= 0.5
            ensemble_probs = None
            for bundle, batch in zip(specs_bundles, batches):
                images = batch["image"].to(device, non_blocking=True)
                probs = tta_model_prob_mask(bundle["model"], images, transforms)
                ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs
            ensemble_probs = ensemble_probs / float(len(specs_bundles))
            preds = ensemble_probs >= float(score_threshold)
            intersection = (preds & targets).flatten(1).sum(dim=1).float()
            pred_sum = preds.flatten(1).sum(dim=1).float()
            truth_sum = targets.flatten(1).sum(dim=1).float()
            empty_both = (pred_sum == 0) & (truth_sum == 0)
            dice = torch.where(
                empty_both,
                torch.ones_like(intersection),
                (2.0 * intersection + p3.EPS) / (pred_sum + truth_sum + p3.EPS),
            )
            dices[offset : offset + batch_size] = dice.detach().cpu().numpy()
            image_pred[offset : offset + batch_size] = (pred_sum > 0).detach().cpu().numpy()
            labels[offset : offset + batch_size] = batches[0]["label"].detach().cpu().numpy().astype(np.uint8)
            offset += batch_size
            progress.update(batch_idx)
    dices[labels == 0] = np.nan
    return dices, image_pred, labels


def choose_threshold_at_fpr(scores, labels, target_fpr):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    thresholds = np.unique(scores)[::-1]
    if thresholds.size == 0:
        return 1.0, {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "fpr": 0.0, "f1": 0.0}
    thresholds = np.concatenate(([np.nextafter(thresholds[0], np.inf)], thresholds))
    best = None
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(np.logical_and(pred, labels == 1).sum())
        fp = int(np.logical_and(pred, labels == 0).sum())
        fn = int(np.logical_and(~pred, labels == 1).sum())
        tn = int(np.logical_and(~pred, labels == 0).sum())
        fpr = fp / max(fp + tn, 1)
        if fpr > target_fpr + 1e-12:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        row = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": float(precision), "recall": float(recall),
            "fpr": float(fpr), "f1": float(f1),
        }
        if best is None or (row["recall"], row["f1"], row["precision"], threshold) > (
            best[1]["recall"], best[1]["f1"], best[1]["precision"], best[0],
        ):
            best = (float(threshold), row)
    if best is None:
        threshold = float(np.nextafter(np.max(scores), np.inf))
        pred = scores >= threshold
        fp = int(np.logical_and(pred, labels == 0).sum())
        tn = int(np.logical_and(~pred, labels == 0).sum())
        return threshold, {
            "tp": 0, "fp": fp, "tn": tn, "fn": int((labels == 1).sum()),
            "precision": 0.0, "recall": 0.0,
            "fpr": fp / max(fp + tn, 1), "f1": 0.0,
        }
    return best


def write_markdown(output_path, report):
    lines = [
        "# TTA + Ensemble Stratified Evaluation",
        "",
        f"Dataset: `{report['data_h5']}`",
        f"Matched FPR: `{report['matched_fpr']}`",
        f"TTA mode: `{report['tta']}`",
        f"Models: {', '.join(m['name'] for m in report['models'])}",
        "",
        "| config | AUROC | AUPRC | threshold | precision | recall | FPR | F1 | "
        "weak recall | contained recall | truncated recall | Dice+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    row = report
    weak = row["groups"].get("weak_family_union", {})
    all_pos = row["groups"].get("all_positive", {})
    contained = row["groups"].get("geometry_contained", {})
    truncated = row["groups"].get("geometry_truncated", {})
    lines.append(
        "| {name} | {auroc:.3f} [{auroc_lo:.3f},{auroc_hi:.3f}] | "
        "{auprc:.3f} [{auprc_lo:.3f},{auprc_hi:.3f}] | {threshold:.6f} | "
        "{precision:.3f} | {recall:.3f} | {fpr:.3f} | {f1:.3f} | "
        "{weak_recall:.3f} [{weak_lo:.3f},{weak_hi:.3f}] | "
        "{contained_recall:.3f} | {truncated_recall:.3f} | "
        "{dice:.3f} [{dice_lo:.3f},{dice_hi:.3f}] |".format(
            name=row["ensemble_name"],
            auroc=row["auroc"], auroc_lo=row["auroc_ci95"][0], auroc_hi=row["auroc_ci95"][1],
            auprc=row["auprc"], auprc_lo=row["auprc_ci95"][0], auprc_hi=row["auprc_ci95"][1],
            threshold=row["matched_threshold"],
            precision=row["matched_metrics"]["precision"], recall=row["matched_metrics"]["recall"],
            fpr=row["matched_metrics"]["fpr"], f1=row["matched_metrics"]["f1"],
            weak_recall=weak.get("recall", float("nan")),
            weak_lo=weak.get("recall_ci95", [float("nan"), float("nan")])[0],
            weak_hi=weak.get("recall_ci95", [float("nan"), float("nan")])[1],
            contained_recall=contained.get("recall", float("nan")),
            truncated_recall=truncated.get("recall", float("nan")),
            dice=all_pos.get("dice_mean", float("nan")),
            dice_lo=all_pos.get("dice_ci95", [float("nan"), float("nan")])[0],
            dice_hi=all_pos.get("dice_ci95", [float("nan"), float("nan")])[1],
        )
    )
    extra_rows = []
    for key in ("z0_amp_bin_0", "z0_amp_bin_1", "z0_amp_bin_2",
                "visible_fraction_low", "visible_fraction_mid", "visible_fraction_high",
                "center_inside_patch", "center_outside_patch"):
        g = row["groups"].get(key)
        if g is not None:
            extra_rows.append(f"- `{key}` (n={g['num_positive']}): recall {g['recall']:.3f} [{g['recall_ci95'][0]:.3f},{g['recall_ci95'][1]:.3f}], Dice {g['dice_mean']:.3f}")
    if extra_rows:
        lines.append("")
        lines.append("## Selected group breakdowns")
        lines.extend(extra_rows)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not args.model:
        raise SystemExit("Pass at least one --model spec.")
    data_h5 = Path(args.data_h5).resolve()
    if not data_h5.exists():
        raise FileNotFoundError(f"External validation HDF5 not found: {data_h5}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [parse_model_spec(text) for text in args.model]
    device = p3.resolve_device(args.device)
    rng = np.random.default_rng(args.bootstrap_seed)

    indices, labels = load_all_labels(data_h5, args.max_samples)
    strat = load_stratification(data_h5, len(indices))
    groups = group_masks(strat)

    print(f"\n=== Evaluating {args.ensemble_name} (n_models={len(specs)}, tta={args.tta}) ===", flush=True)
    scores, bundles, transforms = collect_ensemble_scores(specs, data_h5, indices, args, device)
    threshold, matched_metrics = choose_threshold_at_fpr(scores, labels, args.matched_fpr)
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    ci_bundle = bootstrap_auc(scores, labels, args.bootstrap_resamples, rng)
    save_curves(output_dir, args.ensemble_name, labels, scores)

    dice_threshold = float(threshold) if args.dice_threshold_mode == "matched" else float(args.dice_threshold)
    print(f"\n  Matched threshold: {threshold:.6f} | FPR {matched_metrics['fpr']:.3f} | recall {matched_metrics['recall']:.3f}", flush=True)
    print(f"  Dice threshold:    {dice_threshold:.6f} (mode={args.dice_threshold_mode})", flush=True)
    dices, image_pred, _ = collect_ensemble_dice(
        bundles, data_h5, indices, transforms, dice_threshold, args, device,
    )
    group_summary = summarize_groups(groups, labels, image_pred, dices, rng, args.bootstrap_resamples)

    report = {
        "data_h5": str(data_h5),
        "num_samples": int(len(indices)),
        "num_positive": int(labels.sum()),
        "matched_fpr": float(args.matched_fpr),
        "tta": args.tta,
        "ensemble_name": args.ensemble_name,
        "models": [
            {
                "name": b["spec"].name,
                "run_dir": str(b["spec"].run_dir.resolve()),
                "checkpoint": b["spec"].checkpoint,
                "checkpoint_path": b["checkpoint_path"],
                "checkpoint_label": b["checkpoint_label"],
                "score_kind": b["spec"].score_kind,
            }
            for b in bundles
        ],
        "auroc": auroc,
        "auprc": auprc,
        "auroc_ci95": ci_bundle["auroc_ci95"],
        "auprc_ci95": ci_bundle["auprc_ci95"],
        "matched_threshold": float(threshold),
        "matched_metrics": matched_metrics,
        "dice_threshold": float(dice_threshold),
        "dice_threshold_mode": args.dice_threshold_mode,
        "groups": group_summary,
    }

    json_path = output_dir / "tta_ensemble_eval_report.json"
    md_path = output_dir / "tta_ensemble_eval_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"\n=== Saved ===\n  JSON: {json_path}\n  MD:   {md_path}", flush=True)


if __name__ == "__main__":
    main()
