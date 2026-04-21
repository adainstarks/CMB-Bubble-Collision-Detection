"""
Evaluate Phase 3 checkpoints on an external stratified validation HDF5.

This is the high-power comparison harness.  It deliberately does not reuse the
1000-sample split saved inside training run directories.  Each model is scored
on the same external HDF5, at a matched image-level FPR, with bootstrap
uncertainty and per-bin physics breakdowns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import phase3_train_unet as p3
from phase3_evaluate_run import load_json, resolve_checkpoint_path


DEFAULT_MODELS = (
    "random_b64_aux:runs/phase3_unet/remediated_v1_unet_random_b64_aux:best:mask_max",
    "imagenet_b64_aux:runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux:best:mask_max",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    run_dir: Path
    checkpoint: str
    score_kind: str
    mask_threshold: float | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched-FPR, bootstrap-CI evaluation on an external stratified validation HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-h5", type=str, default="data/remediated_v1/test_data.h5")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model spec as name:run_dir:checkpoint:score_kind[:mask_threshold]. "
            "score_kind is one of mask_max, aux, aux_mask_min. Can be repeated. "
            "mask_threshold is optional and only affects pixel-mask Dice."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="runs/phase3_unet/remediated_v1_stratified_external_eval")
    parser.add_argument("--matched-fpr", type=float, default=0.08)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-samples", type=int, default=0, help="Debug only. Use 0 for all samples.")
    return parser.parse_args()


def validate_args(args):
    if not (0.0 < args.matched_fpr < 1.0):
        raise ValueError("--matched-fpr must be in (0, 1).")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")


def parse_model_spec(text):
    parts = text.split(":")
    if len(parts) not in {4, 5}:
        raise ValueError(f"Invalid --model spec: {text}")
    name, run_dir, checkpoint, score_kind = parts[:4]
    if score_kind not in {"mask_max", "aux", "aux_mask_min"}:
        raise ValueError(f"Unsupported score_kind in {text}")
    mask_threshold = None
    if len(parts) == 5:
        mask_threshold = float(parts[4])
        if not (0.0 <= mask_threshold <= 1.0):
            raise ValueError(f"mask_threshold must be in [0, 1] in {text}")
    return ModelSpec(
        name=name,
        run_dir=Path(run_dir),
        checkpoint=checkpoint,
        score_kind=score_kind,
        mask_threshold=mask_threshold,
    )


def load_all_labels(data_h5, max_samples=0):
    with h5py.File(data_h5, "r") as h5:
        n = int(h5["labels"].shape[0])
        if max_samples:
            n = min(n, int(max_samples))
        indices = np.arange(n, dtype=np.int64)
        labels = np.asarray(h5["labels"][:n], dtype=np.uint8)
    return indices, labels


def make_loader(data_h5, indices, run_config, args, device):
    dataset = p3.H5BubbleDataset(
        h5_path=data_h5,
        indices=indices,
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 10007,
        max_translate_pixels=0,
    )
    return p3.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def build_model_from_run(run_dir, checkpoint_arg, device):
    run_config = load_json(run_dir / "run_config.json")
    model_args = p3.model_args_from_run_config(run_config)
    model = p3.build_model(model_args).to(device)
    checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, checkpoint_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model, run_config, str(checkpoint_path), checkpoint_label


def score_from_outputs(mask_logits, image_logits, score_kind):
    mask_score = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values
    if score_kind == "mask_max":
        return mask_score.detach().cpu().numpy(), mask_score.detach().cpu().numpy()
    if image_logits is None:
        raise RuntimeError(f"score_kind={score_kind} requires an auxiliary image head.")
    aux_score = torch.sigmoid(image_logits.view(-1))
    if score_kind == "aux":
        score = aux_score
    elif score_kind == "aux_mask_min":
        score = torch.minimum(aux_score, mask_score)
    else:
        raise ValueError(f"Unknown score_kind: {score_kind}")
    return score.detach().cpu().numpy(), mask_score.detach().cpu().numpy()


def collect_scores(spec, data_h5, indices, args, device):
    model, run_config, checkpoint_path, checkpoint_label = build_model_from_run(spec.run_dir, spec.checkpoint, device)
    model_args = p3.model_args_from_run_config(run_config)
    radius_bin_edges = p3.parse_radius_bin_edges(getattr(model_args, "radius_bin_edges_deg", "5,10,15,20,25"))
    radius_bin_count = p3.radius_bin_count_from_edges(radius_bin_edges) if getattr(model_args, "radius_head_weight", 0.0) > 0.0 else 0
    loader = make_loader(data_h5, indices, run_config, args, device)
    scores = np.zeros(len(indices), dtype=np.float32)
    mask_scores = np.zeros(len(indices), dtype=np.float32)
    labels = np.zeros(len(indices), dtype=np.uint8)

    progress = p3.ProgressPrinter(len(loader), f"Score {spec.name}")
    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            mask_logits, aux_logits = p3.unpack_model_output(model(images))
            image_logits, _ = p3.split_aux_logits(
                aux_logits,
                use_image_aux=getattr(model_args, "aux_head_weight", 0.0) > 0.0,
                radius_bin_count=radius_bin_count,
            )
            batch_scores, batch_mask_scores = score_from_outputs(mask_logits, image_logits, spec.score_kind)
            batch_size = int(images.shape[0])
            scores[offset : offset + batch_size] = batch_scores
            mask_scores[offset : offset + batch_size] = batch_mask_scores
            labels[offset : offset + batch_size] = batch["label"].detach().cpu().numpy().astype(np.uint8)
            offset += batch_size
            progress.update(batch_idx)

    return {
        "scores": scores,
        "mask_scores": mask_scores,
        "labels": labels,
        "run_config": run_config,
        "checkpoint_path": checkpoint_path,
        "checkpoint_label": checkpoint_label,
    }


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
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "fpr": float(fpr),
            "f1": float(f1),
        }
        if best is None or (row["recall"], row["f1"], row["precision"], threshold) > (
            best[1]["recall"],
            best[1]["f1"],
            best[1]["precision"],
            best[0],
        ):
            best = (float(threshold), row)
    if best is None:
        threshold = float(np.nextafter(np.max(scores), np.inf))
        pred = scores >= threshold
        fp = int(np.logical_and(pred, labels == 0).sum())
        tn = int(np.logical_and(~pred, labels == 0).sum())
        return threshold, {
            "tp": 0,
            "fp": fp,
            "tn": tn,
            "fn": int((labels == 1).sum()),
            "precision": 0.0,
            "recall": 0.0,
            "fpr": fp / max(fp + tn, 1),
            "f1": 0.0,
        }
    return best


def ci(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_auc(scores, labels, resamples, rng):
    n = len(labels)
    roc_values = []
    pr_values = []
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    for _ in range(resamples):
        sample = rng.integers(0, n, size=n)
        y = labels[sample]
        if np.unique(y).size < 2:
            continue
        s = scores[sample]
        roc_values.append(float(roc_auc_score(y, s)))
        pr_values.append(float(average_precision_score(y, s)))
    return {"auroc_ci95": ci(roc_values), "auprc_ci95": ci(pr_values)}


def load_stratification(data_h5, n):
    out = {}
    with h5py.File(data_h5, "r") as h5:
        labels = np.asarray(h5["labels"][:n], dtype=np.uint8)
        truth = h5["truth"]
        out["z0_abs"] = np.abs(np.asarray(truth["z0"][:n], dtype=np.float64))
        out["zcrit_abs"] = np.abs(np.asarray(truth["zcrit"][:n], dtype=np.float64))
        out["theta_crit_deg"] = np.asarray(truth["theta_crit_deg"][:n], dtype=np.float64)
        out["edge_sigma_deg"] = np.asarray(truth["edge_sigma_deg"][:n], dtype=np.float64)
        for key in (
            "geometry_mode_code",
            "fully_contained",
            "target_touches_edge",
            "visible_target_fraction",
            "signal_center_in_patch",
        ):
            if key in truth:
                out[key] = np.asarray(truth[key][:n])
        if "stratification" in h5:
            for key in h5["stratification"].keys():
                out[key] = np.asarray(h5["stratification"][key][:n])
        out["labels"] = labels
    return out


def group_masks(strat):
    labels = strat["labels"]
    positive = labels == 1
    groups = {"all_positive": positive}

    def add_bin_groups(field, prefix):
        if field not in strat:
            return
        values = strat[field]
        for value in sorted(np.unique(values[positive]).tolist()):
            if int(value) < 0:
                continue
            groups[f"{prefix}_{int(value)}"] = positive & (values == value)

    add_bin_groups("z0_amp_bin", "z0_amp_bin")
    add_bin_groups("zcrit_abs_bin", "zcrit_abs_bin")
    add_bin_groups("theta_bin", "theta_bin")
    add_bin_groups("edge_sigma_bin", "edge_sigma_bin")
    if "fully_contained" in strat:
        fully_contained = np.asarray(strat["fully_contained"], dtype=bool)
        groups["geometry_contained"] = positive & fully_contained
        groups["geometry_truncated"] = positive & ~fully_contained
    if "signal_center_in_patch" in strat:
        center_in_patch = np.asarray(strat["signal_center_in_patch"], dtype=bool)
        groups["center_inside_patch"] = positive & center_in_patch
        groups["center_outside_patch"] = positive & ~center_in_patch
    if "visible_target_fraction" in strat:
        visible = np.asarray(strat["visible_target_fraction"], dtype=np.float64)
        groups["visible_fraction_low"] = positive & (visible > 0.0) & (visible < 0.35)
        groups["visible_fraction_mid"] = positive & (visible >= 0.35) & (visible < 0.70)
        groups["visible_fraction_high"] = positive & (visible >= 0.70)
    if all(key in strat for key in ("z0_amp_bin", "zcrit_abs_bin", "theta_bin")):
        groups["weak_family_union"] = positive & (
            (strat["z0_amp_bin"] == 0) | (strat["zcrit_abs_bin"] == 0) | (strat["theta_bin"] == 0)
        )
    return groups


def dice_per_sample_at_threshold(data_h5, indices, run_config, model, score_threshold, args, device):
    loader = make_loader(data_h5, indices, run_config, args, device)
    dices = np.zeros(len(indices), dtype=np.float32)
    image_pred = np.zeros(len(indices), dtype=bool)
    labels = np.zeros(len(indices), dtype=np.uint8)
    progress = p3.ProgressPrinter(len(loader), "Dice pass")
    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True) >= 0.5
            mask_logits, _ = p3.unpack_model_output(model(images))
            probs = torch.sigmoid(mask_logits)
            preds = probs >= float(score_threshold)
            intersection = (preds & targets).flatten(1).sum(dim=1).float()
            pred_sum = preds.flatten(1).sum(dim=1).float()
            truth_sum = targets.flatten(1).sum(dim=1).float()
            empty_both = (pred_sum == 0) & (truth_sum == 0)
            dice = torch.where(
                empty_both,
                torch.ones_like(intersection),
                (2.0 * intersection + p3.EPS) / (pred_sum + truth_sum + p3.EPS),
            )
            batch_size = int(images.shape[0])
            dices[offset : offset + batch_size] = dice.detach().cpu().numpy()
            image_pred[offset : offset + batch_size] = (pred_sum > 0).detach().cpu().numpy()
            labels[offset : offset + batch_size] = batch["label"].detach().cpu().numpy().astype(np.uint8)
            offset += batch_size
            progress.update(batch_idx)
    dices[labels == 0] = np.nan
    return dices, image_pred


def summarize_groups(groups, labels, image_pred, dices, rng, resamples):
    report = {}
    for group_name, mask in groups.items():
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        detected = image_pred[idx]
        recall = float(detected.mean())
        dice_values = dices[idx]
        dice_mean = float(np.nanmean(dice_values)) if np.isfinite(dice_values).any() else float("nan")
        recall_bs = []
        dice_bs = []
        for _ in range(resamples):
            sample = rng.choice(idx, size=idx.size, replace=True)
            recall_bs.append(float(image_pred[sample].mean()))
            sample_dice = dices[sample]
            dice_bs.append(float(np.nanmean(sample_dice)) if np.isfinite(sample_dice).any() else float("nan"))
        report[group_name] = {
            "num_positive": int(idx.size),
            "recall": recall,
            "recall_ci95": ci(recall_bs),
            "dice_mean": dice_mean,
            "dice_ci95": ci(dice_bs),
        }
    return report


def save_curves(output_dir, name, labels, scores):
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    np.savez_compressed(
        output_dir / f"{name}_curves.npz",
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=roc_thresholds,
        precision=precision,
        recall=recall,
        pr_thresholds=pr_thresholds,
    )


def write_markdown(output_path, report):
    lines = ["# Stratified External Evaluation", ""]
    lines.append(f"Dataset: `{report['data_h5']}`")
    lines.append(f"Matched FPR: `{report['matched_fpr']}`")
    lines.append("")
    lines.append(
        "| model | AUROC | AUPRC | threshold | precision | recall | FPR | F1 | "
        "weak recall | contained recall | truncated recall | Dice+ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["models"]:
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
                contained_recall=contained.get("recall", float("nan")),
                truncated_recall=truncated.get("recall", float("nan")),
                dice=all_pos.get("dice_mean", float("nan")),
                dice_lo=all_pos.get("dice_ci95", [float("nan"), float("nan")])[0],
                dice_hi=all_pos.get("dice_ci95", [float("nan"), float("nan")])[1],
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    data_h5 = Path(args.data_h5).resolve()
    if not data_h5.exists():
        raise FileNotFoundError(f"External validation HDF5 not found: {data_h5}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [parse_model_spec(text) for text in (args.model or DEFAULT_MODELS)]
    indices, labels = load_all_labels(data_h5, args.max_samples)
    strat = load_stratification(data_h5, len(indices))
    groups = group_masks(strat)
    device = p3.resolve_device(args.device)
    rng = np.random.default_rng(args.bootstrap_seed)

    report = {
        "data_h5": str(data_h5),
        "num_samples": int(len(indices)),
        "num_positive": int(labels.sum()),
        "num_negative": int(len(labels) - labels.sum()),
        "matched_fpr": float(args.matched_fpr),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "models": [],
    }

    for spec in specs:
        print(f"\n=== Evaluating {spec.name} ===")
        scored = collect_scores(spec, data_h5, indices, args, device)
        scores = scored["scores"]
        model_labels = scored["labels"]
        if not np.array_equal(model_labels, labels):
            raise RuntimeError(f"Label mismatch while scoring {spec.name}.")
        threshold, matched_metrics = choose_threshold_at_fpr(scores, labels, args.matched_fpr)
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
        auc_ci = bootstrap_auc(scores, labels, args.bootstrap_resamples, rng)

        # Rebuild model for the dice pass because collect_scores releases no model handle.
        model, run_config, checkpoint_path, checkpoint_label = build_model_from_run(spec.run_dir.resolve(), spec.checkpoint, device)
        # Dice is computed from pixel masks, while detection may be scored from an
        # auxiliary image head. Keep those thresholds separate so scalar-gated
        # branches are not assigned an arbitrary mask quantile as their pixel
        # threshold.
        if spec.mask_threshold is not None:
            dice_threshold = float(spec.mask_threshold)
        elif spec.score_kind == "mask_max":
            dice_threshold = float(threshold)
        else:
            dice_threshold = 0.5
        dices, image_pred = dice_per_sample_at_threshold(data_h5, indices, run_config, model, dice_threshold, args, device)
        if spec.score_kind != "mask_max":
            image_pred = scores >= threshold
        group_report = summarize_groups(groups, labels, image_pred, dices, rng, args.bootstrap_resamples)
        save_curves(output_dir, spec.name, labels, scores)

        row = {
            "name": spec.name,
            "run_dir": str(spec.run_dir.resolve()),
            "checkpoint": spec.checkpoint,
            "checkpoint_path": checkpoint_path,
            "checkpoint_label": checkpoint_label,
            "score_kind": spec.score_kind,
            "mask_threshold_source": "explicit" if spec.mask_threshold is not None else ("matched_score" if spec.score_kind == "mask_max" else "default_0.5"),
            "auroc": auroc,
            "auprc": auprc,
            **auc_ci,
            "matched_threshold": float(threshold),
            "matched_metrics": matched_metrics,
            "dice_threshold": float(dice_threshold),
            "groups": group_report,
        }
        report["models"].append(row)

    report_path = output_dir / "stratified_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = output_dir / "stratified_eval_report.md"
    write_markdown(md_path, report)
    print("\n=== Stratified external evaluation saved ===")
    print(f"  JSON: {report_path}")
    print(f"  MD:   {md_path}")


if __name__ == "__main__":
    main()
