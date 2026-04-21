"""
Evaluate one ML branch against matched_template with map-domain recalibration.

This is the post-recalibration evaluation path for feature-injection models:
score one run on CAMB sensitivity, real-SMICA injection, and real-SMICA null
controls; calibrate thresholds separately on CAMB negatives and SMICA nulls;
then report cell and regime recall for ML-only, matched-only, OR, and AND.
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

import phase3_train_unet as p3
from phase3_evaluate_run import load_json, resolve_checkpoint_path
from phase3_sensitivity_curve import threshold_from_negatives


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENS_H5 = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_data.h5"
DEFAULT_SENS_SCORES = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_scores.npz"
DEFAULT_REAL_H5 = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_real_sky_injection_smica_mask090"
    / "smica_real_sky_injection.h5"
)
DEFAULT_REAL_SCORES = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_real_sky_injection_smica_mask090" / "real_sky_scores.npz"
)
DEFAULT_NULL_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "null_controls_smica_mask090.h5"
DEFAULT_NULL_MATCHED = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_real_sky_recalibration"
    / "score_cache"
    / "null_matched_template_scores.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_single_model_recalibrated"
POLICIES = ("model_only", "matched_template", "either_model_or_matched", "both_model_and_matched")
REGIMES = (
    ("dead_A_le_2e-6", None, 2e-6),
    ("contested_5e-6_to_2e-5", 5e-6, 2e-5),
    ("solved_A_ge_5e-5", 5e-5, None),
    ("contested_plus_solved_A_ge_5e-6", 5e-6, None),
)


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one ML branch with SMICA-calibrated thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sensitivity-h5", type=str, default=str(DEFAULT_SENS_H5))
    parser.add_argument("--sensitivity-scores", type=str, default=str(DEFAULT_SENS_SCORES))
    parser.add_argument("--real-h5", type=str, default=str(DEFAULT_REAL_H5))
    parser.add_argument("--real-scores", type=str, default=str(DEFAULT_REAL_SCORES))
    parser.add_argument("--null-h5", type=str, default=str(DEFAULT_NULL_H5))
    parser.add_argument("--null-matched-scores", type=str, default=str(DEFAULT_NULL_MATCHED))
    parser.add_argument("--fpr-targets", type=str, default="0.05,0.10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--full-sky-independent-patches", type=float, default=3000.0)
    parser.add_argument("--reuse-scores", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.fpr_targets = parse_float_list(args.fpr_targets)
    if not args.fpr_targets or any(x <= 0.0 or x >= 1.0 for x in args.fpr_targets):
        raise ValueError("--fpr-targets must contain values in (0, 1).")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")


def exact_ci(k: int, n: int) -> list[float]:
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def load_labels(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5["labels"][:], dtype=np.uint8)


def load_stratification(path: Path) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    with h5py.File(path, "r") as h5:
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amp_grid = [float(x) for x in json.loads(h5["summary"].attrs["amplitude_grid"])]
        theta_grid = [float(x) for x in json.loads(h5["summary"].attrs["theta_grid_deg"])]
    return amp_idx, theta_idx, amp_grid, theta_grid


def load_npz_score(path: Path, key: str) -> np.ndarray:
    with np.load(path) as loaded:
        return np.asarray(loaded[key], dtype=np.float32)


def load_matched_scores(sensitivity_scores: Path, real_scores: Path, null_matched_scores: Path) -> dict[str, np.ndarray]:
    return {
        "sensitivity": load_npz_score(sensitivity_scores, "score__matched_template"),
        "real": load_npz_score(real_scores, "score__matched_template"),
        "null": load_npz_score(null_matched_scores, "scores"),
    }


def build_model(run_dir: Path, checkpoint_arg: str, device: torch.device):
    run_config = load_json(run_dir / "run_config.json")
    model = p3.build_model(p3.model_args_from_run_config(run_config)).to(device)
    checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, checkpoint_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model, run_config, checkpoint_path, checkpoint_label


def score_model_h5(
    model_name: str,
    model,
    run_config: dict,
    h5_path: Path,
    cache_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    if args.reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            return np.asarray(loaded["scores"], dtype=np.float32)
    input_config = p3.input_config_from_run_config(run_config)
    means = np.asarray(input_config["channel_means"], dtype=np.float32)
    stds = np.maximum(np.asarray(input_config["channel_stds"], dtype=np.float32), 1e-8)
    if means.size != input_config["input_channels"]:
        raise RuntimeError(f"Run config normalization does not match input channel count: {run_config['normalization']}")
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["labels"].shape[0])
        scores = np.zeros(n, dtype=np.float32)
        progress = p3.ProgressPrinter(int(np.ceil(n / args.batch_size)), f"Score {model_name}:{h5_path.name}")
        batch_counter = 0
        extra_datasets = []
        for dataset_path in input_config["extra_channel_datasets"]:
            if dataset_path not in h5:
                raise KeyError(f"Input feature dataset missing in {h5_path}: {dataset_path}")
            extra_datasets.append(h5[dataset_path])
        with torch.no_grad():
            for start in range(0, n, args.batch_size):
                stop = min(start + args.batch_size, n)
                channels = [np.asarray(h5["patches"][start:stop], dtype=np.float32)]
                channels.extend(np.asarray(dataset[start:stop], dtype=np.float32) for dataset in extra_datasets)
                batch = np.stack(channels, axis=1)
                batch = (batch - means[None, :, None, None]) / stds[None, :, None, None]
                images = torch.from_numpy(batch).to(device, non_blocking=True)
                mask_logits, _ = p3.unpack_model_output(model(images))
                batch_scores = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values.detach().cpu().numpy()
                scores[start:stop] = batch_scores.astype(np.float32)
                batch_counter += 1
                progress.update(batch_counter)
    np.savez_compressed(cache_path, scores=scores)
    return scores


def binary_metrics(active: np.ndarray, labels: np.ndarray) -> dict:
    active = np.asarray(active, dtype=bool)
    labels = np.asarray(labels, dtype=np.uint8)
    tp = int(np.logical_and(active, labels == 1).sum())
    fp = int(np.logical_and(active, labels == 0).sum())
    tn = int(np.logical_and(~active, labels == 0).sum())
    fn = int(np.logical_and(~active, labels == 1).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
        "recall_ci95": exact_ci(tp, tp + fn) if tp + fn else [0.0, 0.0],
        "fpr_ci95": exact_ci(fp, fp + tn) if fp + tn else [0.0, 0.0],
    }


def active_policies(model_scores: np.ndarray, matched_scores: np.ndarray, model_threshold: float, matched_threshold: float) -> dict[str, np.ndarray]:
    model = model_scores > float(model_threshold)
    matched = matched_scores > float(matched_threshold)
    return {
        "model_only": model,
        "matched_template": matched,
        "either_model_or_matched": model | matched,
        "both_model_and_matched": model & matched,
    }


def regime_mask(amplitudes: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    mask = np.ones_like(amplitudes, dtype=bool)
    if low is not None:
        mask &= amplitudes >= float(low)
    if high is not None:
        mask &= amplitudes <= float(high)
    return mask


def summarize_regimes(labels, amplitude_idx, amp_grid, active_by_policy, fpr_target, domain) -> list[dict]:
    amplitudes = np.asarray([amp_grid[idx] if idx >= 0 else 0.0 for idx in amplitude_idx], dtype=np.float64)
    rows = []
    for policy, active in active_by_policy.items():
        for regime_name, low, high in REGIMES:
            mask = (labels == 1) & regime_mask(amplitudes, low, high)
            k = int(active[mask].sum())
            n = int(mask.sum())
            ci = exact_ci(k, n) if n else [0.0, 0.0]
            rows.append(
                {
                    "fpr_target": float(fpr_target),
                    "domain": domain,
                    "policy": policy,
                    "regime": regime_name,
                    "detected": k,
                    "n": n,
                    "recall": float(k / max(n, 1)),
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                }
            )
    return rows


def summarize_cells(labels, amplitude_idx, theta_idx, amp_grid, theta_grid, active_by_policy, fpr_target, domain) -> list[dict]:
    rows = []
    for policy, active in active_by_policy.items():
        for amp_i, amp in enumerate(amp_grid):
            for theta_i, theta in enumerate(theta_grid):
                mask = (labels == 1) & (amplitude_idx == amp_i) & (theta_idx == theta_i)
                k = int(active[mask].sum())
                n = int(mask.sum())
                ci = exact_ci(k, n) if n else [0.0, 0.0]
                rows.append(
                    {
                        "fpr_target": float(fpr_target),
                        "domain": domain,
                        "policy": policy,
                        "amplitude": float(amp),
                        "theta_crit_deg": float(theta),
                        "detected": k,
                        "n": n,
                        "p_det": float(k / max(n, 1)),
                        "ci95_low": ci[0],
                        "ci95_high": ci[1],
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Single-Model SMICA-Recalibrated Evaluation",
        "",
        f"- Model: `{report['model']['name']}`",
        f"- Run dir: `{report['model']['run_dir']}`",
        f"- Checkpoint: `{report['model']['checkpoint_path']}`",
        "",
        "## Global Metrics",
        "",
        "| FPR target | domain | policy | recall | FPR | precision | expected FP / 3000 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in report["global_rows"]:
        lines.append(
            f"| {row['fpr_target']:.2f} | {row['domain']} | `{row['policy']}` | "
            f"{row['recall']:.3f} | {row['fpr']:.3f} | {row['precision']:.3f} | "
            f"{row['expected_fp_full_sky']:.1f} |"
        )
    lines.extend(["", "## Real-SMICA Regime Recall", "", "| FPR target | policy | regime | recall | 95% CI | detected / n |", "|---:|---|---|---:|---:|---:|"])
    for row in report["regime_rows"]:
        if row["domain"] != "real_smica_recalibrated":
            continue
        lines.append(
            f"| {row['fpr_target']:.2f} | `{row['policy']}` | `{row['regime']}` | "
            f"{row['recall']:.3f} | [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | "
            f"{row['detected']} / {row['n']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir).resolve()
    device = p3.resolve_device(args.device)
    model, run_config, checkpoint_path, checkpoint_label = build_model(run_dir, args.checkpoint, device)
    matched_scores = load_matched_scores(
        Path(args.sensitivity_scores).resolve(),
        Path(args.real_scores).resolve(),
        Path(args.null_matched_scores).resolve(),
    )
    h5_paths = {
        "sensitivity": Path(args.sensitivity_h5).resolve(),
        "real": Path(args.real_h5).resolve(),
        "null": Path(args.null_h5).resolve(),
    }
    labels = {name: load_labels(path) for name, path in h5_paths.items()}
    model_scores = {
        name: score_model_h5(
            model_name=args.model_name,
            model=model,
            run_config=run_config,
            h5_path=path,
            cache_path=cache_dir / f"{name}_{args.model_name}_scores.npz",
            args=args,
            device=device,
        )
        for name, path in h5_paths.items()
    }
    real_amp_idx, real_theta_idx, real_amp_grid, real_theta_grid = load_stratification(h5_paths["real"])
    sens_amp_idx, sens_theta_idx, sens_amp_grid, sens_theta_grid = load_stratification(h5_paths["sensitivity"])
    if real_amp_grid != sens_amp_grid or real_theta_grid != sens_theta_grid:
        raise RuntimeError("Real and sensitivity grids do not match.")

    global_rows = []
    regime_rows = []
    cell_rows = []
    threshold_rows = []
    null_labels = np.zeros_like(labels["null"], dtype=np.uint8)

    for fpr_target in args.fpr_targets:
        model_camb_threshold, model_camb_fp, model_camb_fpr = threshold_from_negatives(model_scores["sensitivity"], labels["sensitivity"], fpr_target)
        model_smica_threshold, model_smica_fp, model_smica_fpr = threshold_from_negatives(model_scores["null"], null_labels, fpr_target)
        matched_camb_threshold, matched_camb_fp, matched_camb_fpr = threshold_from_negatives(matched_scores["sensitivity"], labels["sensitivity"], fpr_target)
        matched_smica_threshold, matched_smica_fp, matched_smica_fpr = threshold_from_negatives(matched_scores["null"], null_labels, fpr_target)
        threshold_rows.extend(
            [
                {
                    "fpr_target": float(fpr_target),
                    "method": args.model_name,
                    "camb_threshold": float(model_camb_threshold),
                    "camb_negative_fp": int(model_camb_fp),
                    "camb_negative_fpr": float(model_camb_fpr),
                    "smica_threshold": float(model_smica_threshold),
                    "smica_negative_fp": int(model_smica_fp),
                    "smica_negative_fpr": float(model_smica_fpr),
                },
                {
                    "fpr_target": float(fpr_target),
                    "method": "matched_template",
                    "camb_threshold": float(matched_camb_threshold),
                    "camb_negative_fp": int(matched_camb_fp),
                    "camb_negative_fpr": float(matched_camb_fpr),
                    "smica_threshold": float(matched_smica_threshold),
                    "smica_negative_fp": int(matched_smica_fp),
                    "smica_negative_fpr": float(matched_smica_fpr),
                },
            ]
        )

        active = {
            "real_smica_recalibrated": active_policies(
                model_scores["real"],
                matched_scores["real"],
                model_smica_threshold,
                matched_smica_threshold,
            ),
            "camb_reference": active_policies(
                model_scores["sensitivity"],
                matched_scores["sensitivity"],
                model_camb_threshold,
                matched_camb_threshold,
            ),
            "smica_null_recalibrated": active_policies(
                model_scores["null"],
                matched_scores["null"],
                model_smica_threshold,
                matched_smica_threshold,
            ),
        }
        for domain, domain_labels in (
            ("real_smica_recalibrated", labels["real"]),
            ("camb_reference", labels["sensitivity"]),
            ("smica_null_recalibrated", null_labels),
        ):
            for policy, flags in active[domain].items():
                metrics = binary_metrics(flags, domain_labels)
                global_rows.append(
                    {
                        "fpr_target": float(fpr_target),
                        "domain": domain,
                        "policy": policy,
                        "expected_fp_full_sky": float(metrics["fpr"] * args.full_sky_independent_patches),
                        **metrics,
                    }
                )
        regime_rows.extend(
            summarize_regimes(labels["real"], real_amp_idx, real_amp_grid, active["real_smica_recalibrated"], fpr_target, "real_smica_recalibrated")
        )
        regime_rows.extend(
            summarize_regimes(labels["sensitivity"], sens_amp_idx, sens_amp_grid, active["camb_reference"], fpr_target, "camb_reference")
        )
        cell_rows.extend(
            summarize_cells(labels["real"], real_amp_idx, real_theta_idx, real_amp_grid, real_theta_grid, active["real_smica_recalibrated"], fpr_target, "real_smica_recalibrated")
        )
        cell_rows.extend(
            summarize_cells(labels["sensitivity"], sens_amp_idx, sens_theta_idx, sens_amp_grid, sens_theta_grid, active["camb_reference"], fpr_target, "camb_reference")
        )

    report = {
        "model": {
            "name": args.model_name,
            "run_dir": str(run_dir),
            "checkpoint_arg": args.checkpoint,
            "checkpoint_label": checkpoint_label,
            "checkpoint_path": str(checkpoint_path),
        },
        "inputs": {name: str(path) for name, path in h5_paths.items()},
        "matched_score_sources": {
            "sensitivity": str(Path(args.sensitivity_scores).resolve()),
            "real": str(Path(args.real_scores).resolve()),
            "null": str(Path(args.null_matched_scores).resolve()),
        },
        "fpr_targets": [float(x) for x in args.fpr_targets],
        "threshold_rows": threshold_rows,
        "global_rows": global_rows,
        "regime_rows": regime_rows,
        "cell_rows": cell_rows,
    }
    json_path = output_dir / f"{args.model_name}_recalibrated_eval.json"
    md_path = output_dir / f"{args.model_name}_recalibrated_eval.md"
    regime_csv = output_dir / f"{args.model_name}_regime_recall.csv"
    cell_csv = output_dir / f"{args.model_name}_cell_recall.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    write_csv(regime_csv, regime_rows)
    write_csv(cell_csv, cell_rows)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "regime_csv": str(regime_csv), "cell_csv": str(cell_csv)}, indent=2))


if __name__ == "__main__":
    main()
