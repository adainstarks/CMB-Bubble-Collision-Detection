"""Estimate CMB-noise-floor diagnostics for remediated sensitivity results.

Assumptions
-----------
* This script analyzes candidate-screening sensitivity, not Bayesian evidence.
* Signal amplitudes are dimensionless fractional ``Delta T / T`` parameters
  following Feeney et al. Phys. Rev. D 84, 043507 (2011), arXiv:1012.3667.
* The reported SNR proxies are diagnostics. They are not information-theoretic
  impossibility proofs because a true bound requires the full CMB covariance,
  foreground residual model, mask coupling, and the downstream Bayesian
  likelihood.
* ``white_independent_recall_bound`` is intentionally optimistic: it treats
  correlated CMB structure as independent white noise with the same patch RMS.
* ``correlated_area_recall_proxy`` is a conservative heuristic that compresses
  the disc support by a configurable correlation length.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from phase2_signal_model import bubble_collision_signal
from phase_config import PATCH_PIX, RESO_ARCMIN, T_CMB
from phase_dataset_utils import make_angular_distance_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENSITIVITY_H5 = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_data.h5"
)
DEFAULT_SCORES_NPZ = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_scores.npz"
)
DEFAULT_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_report.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_noise_floor"
METHODS = ("imagenet_b64_aux", "random_b64_aux", "circular_template_screen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate low-amplitude CMB noise-floor diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sensitivity-h5", type=str, default=str(DEFAULT_SENSITIVITY_H5))
    parser.add_argument("--scores-npz", type=str, default=str(DEFAULT_SCORES_NPZ))
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--max-templates-per-cell",
        type=int,
        default=32,
        help="Truth templates sampled per amplitude-radius cell for fast deterministic diagnostics.",
    )
    parser.add_argument(
        "--max-negative-patches",
        type=int,
        default=512,
        help="Negative patches sampled for empirical CMB RMS estimation.",
    )
    parser.add_argument("--correlation-length-deg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260420)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_templates_per_cell <= 0:
        raise ValueError("--max-templates-per-cell must be positive.")
    if args.max_negative_patches <= 0:
        raise ValueError("--max-negative-patches must be positive.")
    if args.correlation_length_deg <= 0.0:
        raise ValueError("--correlation-length-deg must be positive.")


def negative_patch_stats(h5: h5py.File, max_negative_patches: int, rng: np.random.Generator) -> dict[str, float]:
    labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    if not np.any(labels == 0):
        raise ValueError("Sensitivity HDF5 contains no negative rows.")
    patches = h5["patches"]
    chunk_rows = int(patches.chunks[0]) if patches.chunks else 128
    row_stds = []
    row_means = []
    # The dataset is already globally permuted. Scanning compressed HDF5 chunks
    # is much faster than random single-row reads and remains representative for
    # this RMS diagnostic.
    for start in range(0, int(labels.size), chunk_rows):
        if len(row_stds) >= int(max_negative_patches):
            break
        stop = min(start + chunk_rows, int(labels.size))
        rel = np.flatnonzero(labels[start:stop] == 0)
        if rel.size == 0:
            continue
        need = int(max_negative_patches) - len(row_stds)
        if rel.size > need:
            rel = np.sort(rng.choice(rel, size=need, replace=False))
        block = np.asarray(patches[start:stop], dtype=np.float64)
        neg_block = block[rel]
        if not np.all(np.isfinite(neg_block)):
            raise ValueError(f"Non-finite negative patch in rows {start}:{stop}.")
        flat = neg_block.reshape(neg_block.shape[0], -1)
        row_stds.extend(np.std(flat, axis=1).astype(float).tolist())
        row_means.extend(np.mean(flat, axis=1).astype(float).tolist())
    row_stds_arr = np.asarray(row_stds, dtype=np.float64)
    row_means_arr = np.asarray(row_means, dtype=np.float64)
    return {
        "num_negative_patches": int(row_stds_arr.size),
        "patch_std_median_k": float(np.median(row_stds_arr)),
        "patch_std_mean_k": float(np.mean(row_stds_arr)),
        "patch_std_p16_k": float(np.percentile(row_stds_arr, 16)),
        "patch_std_p84_k": float(np.percentile(row_stds_arr, 84)),
        "patch_mean_abs_median_k": float(np.median(np.abs(row_means_arr))),
    }


def method_recall_lookup(report: dict[str, Any]) -> dict[tuple[str, float, float], dict[str, Any]]:
    lookup = {}
    for row in report.get("rows", []):
        key = (str(row["method"]), float(row["amplitude"]), float(row["theta_crit_deg"]))
        lookup[key] = row
    return lookup


def ideal_recall_from_snr(snr: float, fpr: float) -> float:
    threshold = float(norm.isf(float(fpr)))
    return float(norm.sf(threshold - float(snr)))


def cell_signal_stats(
    h5: h5py.File,
    indices: np.ndarray,
    *,
    max_templates: int,
    rng: np.random.Generator,
    background_std_k: float,
    instrument_noise_k: float,
    correlation_length_deg: float,
    fpr_target: float,
) -> dict[str, float]:
    if indices.size > max_templates:
        indices = np.sort(rng.choice(indices, size=int(max_templates), replace=False))
    truth = h5["truth"]
    peak_abs = []
    support_rms = []
    patch_rms = []
    support_pixels = []
    white_cmb_snr = []
    white_inst_snr = []
    corr_snr = []
    pixel_deg = RESO_ARCMIN.to_value(u.deg)
    corr_area_pix = np.pi * (float(correlation_length_deg) / pixel_deg) ** 2
    t_cmb_k = T_CMB.to_value(u.K)
    for idx in indices:
        row = int(idx)
        theta_deg = float(truth["theta_crit_deg"][row])
        theta_grid = make_angular_distance_grid(
            PATCH_PIX,
            RESO_ARCMIN.to_value(u.arcmin),
            center_x_pix=float(truth["signal_center_x_pix"][row]),
            center_y_pix=float(truth["signal_center_y_pix"][row]),
        )
        frac = bubble_collision_signal(
            theta_grid,
            float(truth["z0"][row]),
            float(truth["zcrit"][row]),
            np.radians(theta_deg),
            edge_sigma_deg=float(truth["edge_sigma_deg"][row]) if "edge_sigma_deg" in truth else 0.0,
        )
        delta_k = np.asarray(frac * t_cmb_k, dtype=np.float64)
        support = np.abs(frac) > 0.0
        if not bool(support.any()):
            continue
        support_delta = delta_k[support]
        peak_abs.append(float(np.max(np.abs(delta_k))))
        support_rms_i = float(np.sqrt(np.mean(support_delta**2)))
        patch_rms_i = float(np.sqrt(np.mean(delta_k**2)))
        n_support = int(support_delta.size)
        support_rms.append(support_rms_i)
        patch_rms.append(patch_rms_i)
        support_pixels.append(n_support)
        energy = float(np.sqrt(np.sum(delta_k**2)))
        white_cmb_snr.append(energy / max(float(background_std_k), 1.0e-30))
        white_inst_snr.append(energy / max(float(instrument_noise_k), 1.0e-30))
        n_eff = max(1.0, float(n_support) / max(corr_area_pix, 1.0))
        corr_snr.append(support_rms_i * np.sqrt(n_eff) / max(float(background_std_k), 1.0e-30))
    if not support_rms:
        raise RuntimeError("No signal templates were reconstructed for a cell.")
    white_snr_med = float(np.median(white_cmb_snr))
    corr_snr_med = float(np.median(corr_snr))
    return {
        "num_templates_sampled": int(len(support_rms)),
        "peak_abs_signal_uk_median": float(np.median(peak_abs) * 1.0e6),
        "support_rms_signal_uk_median": float(np.median(support_rms) * 1.0e6),
        "patch_rms_signal_uk_median": float(np.median(patch_rms) * 1.0e6),
        "support_pixels_median": float(np.median(support_pixels)),
        "signal_to_cmb_patch_std": float(np.median(support_rms) / max(float(background_std_k), 1.0e-30)),
        "white_independent_snr_cmb": white_snr_med,
        "white_independent_snr_instrument": float(np.median(white_inst_snr)),
        "correlated_area_snr_proxy": corr_snr_med,
        "white_independent_recall_bound": ideal_recall_from_snr(white_snr_med, fpr_target),
        "correlated_area_recall_proxy": ideal_recall_from_snr(corr_snr_med, fpr_target),
    }


def score_separation(
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
    positive_indices: np.ndarray,
    methods: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    out = {}
    negative_mask = labels == 0
    for method in methods:
        if method not in scores:
            continue
        method_scores = scores[method]
        y = np.concatenate(
            [
                np.ones(int(positive_indices.size), dtype=np.uint8),
                np.zeros(int(negative_mask.sum()), dtype=np.uint8),
            ]
        )
        s = np.concatenate([method_scores[positive_indices], method_scores[negative_mask]])
        out[method] = {
            "cell_vs_negative_auroc": float(roc_auc_score(y, s)),
            "positive_score_median": float(np.median(method_scores[positive_indices])),
            "negative_score_median": float(np.median(method_scores[negative_mask])),
        }
    return out


def verdict(row: dict[str, Any]) -> str:
    imagenet = row.get("imagenet_b64_aux_p_det")
    ratio = float(row["signal_to_cmb_patch_std"])
    corr_recall = float(row["correlated_area_recall_proxy"])
    if ratio < 0.05 and (imagenet is None or imagenet < 0.15):
        return "CMB-confusion dominated at current thresholds"
    if corr_recall < 0.15 and (imagenet is None or imagenet < 0.20):
        return "weak evidence regime"
    if imagenet is not None and imagenet >= 0.80:
        return "high-recall regime"
    return "intermediate regime"


def build_rows(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(args.seed))
    report = load_json(Path(args.sensitivity_report).resolve())
    recall_lookup = method_recall_lookup(report)
    with np.load(Path(args.scores_npz).resolve()) as loaded:
        labels_scores = np.asarray(loaded["labels"], dtype=np.uint8)
        scores = {
            key.removeprefix("score__"): np.asarray(loaded[key], dtype=np.float32)
            for key in loaded.files
            if key.startswith("score__")
        }
    with h5py.File(Path(args.sensitivity_h5).resolve(), "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        if not np.array_equal(labels, labels_scores):
            raise RuntimeError("HDF5 labels and score labels do not match.")
        neg_stats = negative_patch_stats(h5, int(args.max_negative_patches), rng)
        summary = h5["summary"].attrs
        amplitude_grid = [float(x) for x in json.loads(summary["amplitude_grid"])]
        theta_grid = [float(x) for x in json.loads(summary["theta_grid_deg"])]
        amp_idx = np.asarray(h5["stratification"]["amplitude_idx"][:], dtype=np.int16)
        theta_idx = np.asarray(h5["stratification"]["theta_idx"][:], dtype=np.int16)
        noise_sigma_uk_arcmin = float(summary.get("noise_sigma_uk_arcmin", 0.0))
        instrument_noise_k = (noise_sigma_uk_arcmin / RESO_ARCMIN.to_value(u.arcmin)) * 1.0e-6
        fpr_target = float(report.get("fpr_target", summary.get("fpr_target", 0.05)))
        rows: list[dict[str, Any]] = []
        for ai, amp in enumerate(amplitude_grid):
            for ti, theta in enumerate(theta_grid):
                positive = np.flatnonzero((labels == 1) & (amp_idx == ai) & (theta_idx == ti))
                signal = cell_signal_stats(
                    h5,
                    positive,
                    max_templates=int(args.max_templates_per_cell),
                    rng=rng,
                    background_std_k=float(neg_stats["patch_std_median_k"]),
                    instrument_noise_k=instrument_noise_k,
                    correlation_length_deg=float(args.correlation_length_deg),
                    fpr_target=fpr_target,
                )
                row: dict[str, Any] = {
                    "amplitude": float(amp),
                    "theta_crit_deg": float(theta),
                    "num_positive": int(positive.size),
                    **signal,
                }
                for method in METHODS:
                    rec = recall_lookup.get((method, float(amp), float(theta)))
                    if rec:
                        row[f"{method}_p_det"] = float(rec["p_det"])
                        row[f"{method}_ci95_low"] = float(rec["ci95_low"])
                        row[f"{method}_ci95_high"] = float(rec["ci95_high"])
                for method, sep in score_separation(scores, labels, positive, METHODS).items():
                    for key, value in sep.items():
                        row[f"{method}_{key}"] = value
                row["verdict"] = verdict(row)
                rows.append(row)
        metadata_out = {
            "sensitivity_h5": str(Path(args.sensitivity_h5).resolve()),
            "scores_npz": str(Path(args.scores_npz).resolve()),
            "sensitivity_report": str(Path(args.sensitivity_report).resolve()),
            "fpr_target": fpr_target,
            "background": {
                **neg_stats,
                "patch_std_median_uk": float(neg_stats["patch_std_median_k"] * 1.0e6),
                "instrument_noise_sigma_uk_per_pixel": float(instrument_noise_k * 1.0e6),
                "correlation_length_deg": float(args.correlation_length_deg),
            },
            "assumption_warnings": [
                "Recall proxies are not impossibility proofs.",
                "The correlated-area proxy depends on the chosen correlation length.",
                "The white independent bound overestimates detectability for correlated CMB backgrounds.",
            ],
        }
    return metadata_out, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = report["rows"]
    lines = ["# Remediated v1 Noise-Floor Diagnostics", ""]
    lines.append("These diagnostics estimate why low-amplitude recall is hard. They are not Bayesian evidence or an information-theoretic impossibility proof.")
    lines.append("")
    background = report["background"]
    lines.append("## Background Scale")
    lines.append("")
    lines.append(f"- Median negative-patch RMS: `{background['patch_std_median_uk']:.2f} uK`")
    lines.append(f"- Instrument white-noise scale: `{background['instrument_noise_sigma_uk_per_pixel']:.2f} uK/pixel`")
    lines.append(f"- Correlation-length heuristic: `{background['correlation_length_deg']:.2f} deg`")
    lines.append("")
    lines.append("## Cell Summary")
    lines.append("")
    lines.append("| A | theta_deg | signal RMS/support uK | signal/CMB RMS | corr recall proxy | ImageNet P_det | circular P_det | verdict |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['amplitude']:.0e} | {row['theta_crit_deg']:.1f} | "
            f"{row['support_rms_signal_uk_median']:.2f} | "
            f"{row['signal_to_cmb_patch_std']:.3f} | "
            f"{row['correlated_area_recall_proxy']:.3f} | "
            f"{row.get('imagenet_b64_aux_p_det', float('nan')):.3f} | "
            f"{row.get('circular_template_screen_p_det', float('nan')):.3f} | "
            f"{row['verdict']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `A <= 5e-6` generally produces support RMS well below the empirical CMB patch RMS, so low recall is expected without stronger priors or a full covariance-aware likelihood.")
    lines.append("- The instrument white-noise scale is much smaller than the CMB patch RMS; CMB confusion and foreground/domain transfer dominate this diagnostic.")
    lines.append("- To prove an actual impossibility theorem, the next stage needs the full covariance and Bayesian/template-fit likelihood, not only patch-level screeners.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    amplitudes = sorted({row["amplitude"] for row in rows})
    theta_values = sorted({row["theta_crit_deg"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    for theta in theta_values:
        theta_rows = sorted([row for row in rows if row["theta_crit_deg"] == theta], key=lambda row: row["amplitude"])
        x = np.asarray([row["amplitude"] for row in theta_rows], dtype=np.float64)
        axes[0].plot(
            x,
            [row["signal_to_cmb_patch_std"] for row in theta_rows],
            marker="o",
            linewidth=1.5,
            label=f"{theta:g} deg",
        )
        axes[1].plot(
            x,
            [row.get("imagenet_b64_aux_p_det", np.nan) for row in theta_rows],
            marker="s",
            linewidth=1.5,
            label=f"{theta:g} deg",
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_xlabel("A = |z0|")
    axes[0].set_ylabel("support RMS / negative-patch RMS")
    axes[1].set_ylabel("ImageNet P_det at calibrated FPR")
    axes[0].set_title("Signal Scale")
    axes[1].set_title("Observed Recall")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_out, rows = build_rows(args)
    report = {**metadata_out, "rows": rows}
    json_path = output_dir / "noise_floor_report.json"
    csv_path = output_dir / "noise_floor_cells.csv"
    md_path = output_dir / "noise_floor_report.md"
    png_path = output_dir / "noise_floor_recall.png"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, report)
    plot_rows(png_path, rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path), "plot": str(png_path)}, indent=2))


if __name__ == "__main__":
    main()
