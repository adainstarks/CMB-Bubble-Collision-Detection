"""
Historical v6_aux_only real-SMICA smoothed-edge sensitivity diagnostic.

This tests whether the hard-edge sensitivity grid is artificially depressing
contested-regime recall. It reuses the existing real-SMICA injection HDF5 for
background patches, coordinates, amplitudes, radii, signs, and centers, but
replaces the hard Feeney boundary with edge_sigma sampled uniformly from the
training distribution.

The default paths are pre-remediation v6 artifacts. The current active flow is
remediated-v1 plus Batch 6 deployment calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import binomtest

import phase3_train_unet as p3
from phase_config import (
    DEFAULT_INJECTION_CONVENTION,
    INJECTION_CONVENTIONS,
    INJECTION_CONVENTION_NOTES,
    PROVENANCE_SCHEMA_VERSION,
)
from phase2_generate_training import fwhm_arcmin_to_sigma_pixels
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN, bubble_collision_signal, fractional_signal_delta
from phase3_evaluate_run import load_json, resolve_checkpoint_path
from phase_dataset_utils import make_angular_distance_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_H5 = PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_injection_v1" / "smica_real_sky_injection.h5"
DEFAULT_RECAL_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_recalibration_v1" / "real_sky_recalibration_report.json"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "phase3_v6_aux_only_w4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_smoothed_sensitivity_v1"
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
        description="Score smoothed real-SMICA injections with v6_aux_only and SMICA-calibrated thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-h5", type=str, default=str(DEFAULT_SOURCE_H5))
    parser.add_argument("--recalibration-report", type=str, default=str(DEFAULT_RECAL_REPORT))
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--edge-sigma-min-deg", type=float, default=0.3)
    parser.add_argument("--edge-sigma-max-deg", type=float, default=1.0)
    parser.add_argument("--signal-beam-fwhm-arcmin", type=float, default=15.0)
    parser.add_argument(
        "--injection-convention",
        type=str,
        default=DEFAULT_INJECTION_CONVENTION,
        choices=INJECTION_CONVENTIONS,
        help="Signal convention for reconstructed positives.",
    )
    parser.add_argument("--fpr-targets", type=str, default="0.05,0.10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reuse-scores", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.fpr_targets = parse_float_list(args.fpr_targets)
    if not args.fpr_targets or any(target <= 0.0 or target >= 1.0 for target in args.fpr_targets):
        raise ValueError("--fpr-targets must contain values in (0, 1).")
    if args.edge_sigma_min_deg < 0.0 or args.edge_sigma_max_deg < args.edge_sigma_min_deg:
        raise ValueError("Invalid edge-sigma range.")
    if args.signal_beam_fwhm_arcmin < 0.0:
        raise ValueError("--signal-beam-fwhm-arcmin must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")


def exact_ci(k: int, n: int) -> list[float]:
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def build_model(run_dir: Path, checkpoint_arg: str, device: torch.device):
    run_config = load_json(run_dir / "run_config.json")
    model = p3.build_model(p3.model_args_from_run_config(run_config)).to(device)
    checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, checkpoint_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model, run_config, checkpoint_path, checkpoint_label


def load_thresholds(report_path: Path, method: str, fpr_targets: tuple[float, ...]) -> dict[float, float]:
    report = load_json(report_path)
    thresholds = {}
    for target in fpr_targets:
        matches = [
            row
            for row in report["threshold_rows"]
            if row["method"] == method and abs(float(row["fpr_target"]) - float(target)) < 1e-12
        ]
        if not matches:
            raise KeyError(f"Missing threshold for {method} at FPR target {target} in {report_path}")
        thresholds[float(target)] = float(matches[0]["real_threshold"])
    return thresholds


def sigmoid_scores(model, run_config: dict, patches: np.ndarray, device: torch.device) -> np.ndarray:
    norm = p3.dataset_kwargs_from_run_config(run_config)
    means = np.asarray(norm["channel_means"], dtype=np.float32)
    stds = np.maximum(np.asarray(norm["channel_stds"], dtype=np.float32), 1e-8)
    if len(norm["extra_channel_datasets"]) != 0:
        raise RuntimeError("This smoothed sensitivity runner currently expects a one-channel model.")
    batch = ((patches[:, None, :, :] - means[None, :, None, None]) / stds[None, :, None, None]).astype(np.float32)
    with torch.no_grad():
        images = torch.from_numpy(batch).to(device, non_blocking=True)
        logits, _ = p3.unpack_model_output(model(images))
        return torch.sigmoid(logits).flatten(1).max(dim=1).values.detach().cpu().numpy().astype(np.float32)


def score_smoothed(args: argparse.Namespace, output_dir: Path):
    score_path = output_dir / "smoothed_v6_aux_only_scores.npz"
    if args.reuse_scores and score_path.exists():
        with np.load(score_path) as loaded:
            return {key: np.asarray(loaded[key]) for key in loaded.files}

    source_h5 = Path(args.source_h5).resolve()
    run_dir = Path(args.run_dir).resolve()
    device = p3.resolve_device(args.device)
    model, run_config, checkpoint_path, checkpoint_label = build_model(run_dir, args.checkpoint, device)
    rng = np.random.default_rng(args.seed)
    beam_sigma_pix = fwhm_arcmin_to_sigma_pixels(args.signal_beam_fwhm_arcmin)

    with h5py.File(source_h5, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        neg_idx = np.flatnonzero(labels == 0)
        pos_idx = np.flatnonzero(labels == 1)
        if neg_idx.size == 0 or pos_idx.size == 0:
            raise RuntimeError("Source HDF5 must contain both negative backgrounds and positive injections.")
        base_patches = np.asarray(h5["patches"][neg_idx], dtype=np.float32)
        base_by_row = {int(row): local for local, row in enumerate(neg_idx)}
        background_index = np.asarray(h5["metadata"]["background_index"][:], dtype=np.int64)
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        amplitude = np.asarray(h5["truth"]["amplitude"][:], dtype=np.float32)
        theta_crit_deg = np.asarray(h5["truth"]["theta_crit_deg"][:], dtype=np.float32)
        z0 = np.asarray(h5["truth"]["z0"][:], dtype=np.float32)
        zcrit = np.asarray(h5["truth"]["zcrit"][:], dtype=np.float32)
        center_x = np.asarray(h5["truth"]["signal_center_x_pix"][:], dtype=np.float32)
        center_y = np.asarray(h5["truth"]["signal_center_y_pix"][:], dtype=np.float32)
        amp_grid = [float(x) for x in json.loads(h5["summary"].attrs["amplitude_grid"])]
        theta_grid = [float(x) for x in json.loads(h5["summary"].attrs["theta_grid_deg"])]

        neg_scores = np.zeros(neg_idx.size, dtype=np.float32)
        progress = p3.ProgressPrinter(int(np.ceil(neg_idx.size / args.batch_size)), "Score smoothed negatives")
        batch_counter = 0
        for start in range(0, neg_idx.size, args.batch_size):
            stop = min(start + args.batch_size, neg_idx.size)
            neg_scores[start:stop] = sigmoid_scores(model, run_config, base_patches[start:stop], device)
            batch_counter += 1
            progress.update(batch_counter)

        pos_scores = np.zeros(pos_idx.size, dtype=np.float32)
        edge_sigma = rng.uniform(args.edge_sigma_min_deg, args.edge_sigma_max_deg, size=pos_idx.size).astype(np.float32)
        progress = p3.ProgressPrinter(int(np.ceil(pos_idx.size / args.batch_size)), "Score smoothed positives")
        batch_counter = 0
        for start in range(0, pos_idx.size, args.batch_size):
            stop = min(start + args.batch_size, pos_idx.size)
            rows = pos_idx[start:stop]
            patches = np.empty((rows.size, PATCH_PIX, PATCH_PIX), dtype=np.float32)
            for local, row in enumerate(rows):
                bg_local = int(background_index[row])
                if bg_local >= base_patches.shape[0]:
                    bg_local = base_by_row.get(int(background_index[row]), bg_local)
                base = base_patches[bg_local]
                theta_dist = make_angular_distance_grid(
                    PATCH_PIX,
                    RESO_ARCMIN,
                    center_x_pix=float(center_x[row]),
                    center_y_pix=float(center_y[row]),
                ).astype(np.float32)
                signal = bubble_collision_signal(
                    theta_dist,
                    float(z0[row]),
                    float(zcrit[row]),
                    np.radians(float(theta_crit_deg[row])),
                    edge_sigma_deg=float(edge_sigma[start + local]),
                )
                signal_delta = np.asarray(
                    fractional_signal_delta(
                        base,
                        signal,
                        injection_convention=args.injection_convention,
                    ),
                    dtype=np.float32,
                )
                if beam_sigma_pix > 0.0:
                    signal_delta = gaussian_filter(signal_delta, sigma=beam_sigma_pix, mode="reflect")
                patches[local] = (base + signal_delta).astype(np.float32)
            pos_scores[start:stop] = sigmoid_scores(model, run_config, patches, device)
            batch_counter += 1
            progress.update(batch_counter)

    out = {
        "negative_scores": neg_scores,
        "positive_scores": pos_scores,
        "positive_source_index": pos_idx.astype(np.int64),
        "positive_amplitude_idx": amp_idx[pos_idx].astype(np.int16),
        "positive_theta_idx": theta_idx[pos_idx].astype(np.int16),
        "positive_amplitude": amplitude[pos_idx].astype(np.float32),
        "positive_theta_crit_deg": theta_crit_deg[pos_idx].astype(np.float32),
        "positive_edge_sigma_deg": edge_sigma.astype(np.float32),
        "amplitude_grid": np.asarray(amp_grid, dtype=np.float32),
        "theta_grid_deg": np.asarray(theta_grid, dtype=np.float32),
        "checkpoint_path": np.asarray(str(checkpoint_path)),
        "checkpoint_label": np.asarray(str(checkpoint_label)),
    }
    np.savez_compressed(score_path, **out)
    return out


def regime_mask(amplitudes: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    mask = np.ones_like(amplitudes, dtype=bool)
    if low is not None:
        mask &= amplitudes >= float(low) * (1.0 - 1e-5)
    if high is not None:
        mask &= amplitudes <= float(high) * (1.0 + 1e-5)
    return mask


def summarize(args: argparse.Namespace, scores: dict, thresholds: dict[float, float], output_dir: Path) -> dict:
    positive_scores = np.asarray(scores["positive_scores"], dtype=np.float32)
    negative_scores = np.asarray(scores["negative_scores"], dtype=np.float32)
    amplitudes = np.asarray(scores["positive_amplitude"], dtype=np.float64)
    amp_idx = np.asarray(scores["positive_amplitude_idx"], dtype=np.int16)
    theta_idx = np.asarray(scores["positive_theta_idx"], dtype=np.int16)
    amp_grid = [float(x) for x in np.asarray(scores["amplitude_grid"], dtype=np.float64)]
    theta_grid = [float(x) for x in np.asarray(scores["theta_grid_deg"], dtype=np.float64)]

    global_rows = []
    regime_rows = []
    cell_rows = []
    for fpr_target, threshold in thresholds.items():
        neg_active = negative_scores > threshold
        pos_active = positive_scores > threshold
        fp = int(neg_active.sum())
        tp = int(pos_active.sum())
        n_neg = int(negative_scores.size)
        n_pos = int(positive_scores.size)
        global_rows.append(
            {
                "fpr_target": float(fpr_target),
                "threshold": float(threshold),
                "positive_detected": tp,
                "positive_n": n_pos,
                "recall": float(tp / max(n_pos, 1)),
                "recall_ci95": exact_ci(tp, n_pos),
                "negative_fp": fp,
                "negative_n": n_neg,
                "real_injection_background_fpr": float(fp / max(n_neg, 1)),
                "background_fpr_ci95": exact_ci(fp, n_neg),
            }
        )
        for name, low, high in REGIMES:
            mask = regime_mask(amplitudes, low, high)
            k = int(pos_active[mask].sum())
            n = int(mask.sum())
            regime_rows.append(
                {
                    "fpr_target": float(fpr_target),
                    "regime": name,
                    "detected": k,
                    "n": n,
                    "recall": float(k / max(n, 1)),
                    "ci95_low": exact_ci(k, n)[0],
                    "ci95_high": exact_ci(k, n)[1],
                }
            )
        for a_i, amp in enumerate(amp_grid):
            for t_i, theta in enumerate(theta_grid):
                mask = (amp_idx == a_i) & (theta_idx == t_i)
                k = int(pos_active[mask].sum())
                n = int(mask.sum())
                cell_rows.append(
                    {
                        "fpr_target": float(fpr_target),
                        "amplitude": float(amp),
                        "theta_crit_deg": float(theta),
                        "detected": k,
                        "n": n,
                        "p_det": float(k / max(n, 1)),
                        "ci95_low": exact_ci(k, n)[0],
                        "ci95_high": exact_ci(k, n)[1],
                    }
                )
    report = {
        "source_h5": str(Path(args.source_h5).resolve()),
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint": args.checkpoint,
        "threshold_source": str(Path(args.recalibration_report).resolve()),
        "edge_sigma_min_deg": float(args.edge_sigma_min_deg),
        "edge_sigma_max_deg": float(args.edge_sigma_max_deg),
        "signal_beam_fwhm_arcmin": float(args.signal_beam_fwhm_arcmin),
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "injection_convention": args.injection_convention,
        "injection_convention_note": INJECTION_CONVENTION_NOTES[args.injection_convention],
        "score_npz": str((output_dir / "smoothed_v6_aux_only_scores.npz").resolve()),
        "global_rows": global_rows,
        "regime_rows": regime_rows,
        "cell_rows": cell_rows,
    }
    return report


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
        "# Real-SMICA Smoothed Sensitivity",
        "",
        f"- Model: `v6_aux_only`",
        f"- Run dir: `{report['run_dir']}`",
        f"- Edge smoothing: uniform `{report['edge_sigma_min_deg']:.2f}` to `{report['edge_sigma_max_deg']:.2f}` deg",
        f"- Threshold source: `{report['threshold_source']}`",
        "",
        "## Global Recall",
        "",
        "| FPR target | threshold | recall | 95% CI | negative FP / n | background FPR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["global_rows"]:
        ci = row["recall_ci95"]
        bg_ci = row["background_fpr_ci95"]
        lines.append(
            f"| {row['fpr_target']:.2f} | {row['threshold']:.8g} | {row['recall']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {row['negative_fp']} / {row['negative_n']} | "
            f"{row['real_injection_background_fpr']:.3f} [{bg_ci[0]:.3f}, {bg_ci[1]:.3f}] |"
        )
    lines.extend(["", "## Regime Recall", "", "| FPR target | regime | recall | 95% CI | detected / n |", "|---:|---|---:|---:|---:|"])
    for row in report["regime_rows"]:
        lines.append(
            f"| {row['fpr_target']:.2f} | `{row['regime']}` | {row['recall']:.3f} | "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | {row['detected']} / {row['n']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(Path(args.recalibration_report).resolve(), "v6_aux_only", args.fpr_targets)
    scores = score_smoothed(args, output_dir)
    report = summarize(args, scores, thresholds, output_dir)
    json_path = output_dir / "smoothed_sensitivity_report.json"
    md_path = output_dir / "smoothed_sensitivity_report.md"
    regime_csv = output_dir / "smoothed_regime_recall.csv"
    cell_csv = output_dir / "smoothed_cell_recall.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    write_csv(regime_csv, report["regime_rows"])
    write_csv(cell_csv, report["cell_rows"])
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "regime_csv": str(regime_csv), "cell_csv": str(cell_csv)}, indent=2))


if __name__ == "__main__":
    main()
