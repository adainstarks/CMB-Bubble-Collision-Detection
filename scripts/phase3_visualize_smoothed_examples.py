"""
Historical example panels for real-SMICA smoothed positive injections.

The gallery shows positive patches, truth masks, v6 probability maps, thresholded
model guesses, and truth/prediction overlays. It uses the saved smoothed-score
artifact to select a deterministic mix of true positives and false negatives.

The default paths are pre-remediation v6 diagnostics. Current paper-facing
examples should be regenerated from remediated-v1 or Batch 6 products.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import phase3_train_unet as p3
from phase_config import DEFAULT_INJECTION_CONVENTION
from phase2_generate_training import fwhm_arcmin_to_sigma_pixels
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN, bubble_collision_signal, fractional_signal_delta
from phase3_evaluate_run import load_json, resolve_checkpoint_path
from phase_dataset_utils import make_angular_distance_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_H5 = PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_injection_v1" / "smica_real_sky_injection.h5"
DEFAULT_SCORES = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "real_sky_smoothed_sensitivity_v1"
    / "smoothed_v6_aux_only_scores.npz"
)
DEFAULT_RECAL_REPORT = PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_recalibration_v1" / "real_sky_recalibration_report.json"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "phase3_v6_aux_only_w4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "visual_examples_smoothed_v1"

EXAMPLE_TARGETS = (
    {
        "label": "FN low-amplitude large-radius",
        "detected": False,
        "amplitude": 1e-6,
        "theta_crit_deg": 20.0,
        "score_quantile": 0.70,
    },
    {
        "label": "FN contested small-radius",
        "detected": False,
        "amplitude": 2e-5,
        "theta_crit_deg": 5.0,
        "score_quantile": 0.85,
    },
    {
        "label": "TP contested large-radius",
        "detected": True,
        "amplitude": 2e-5,
        "theta_crit_deg": 25.0,
        "score_quantile": 0.50,
    },
    {
        "label": "TP solved small-radius",
        "detected": True,
        "amplitude": 5e-5,
        "theta_crit_deg": 5.0,
        "score_quantile": 0.50,
    },
    {
        "label": "TP high-amplitude medium-radius",
        "detected": True,
        "amplitude": 1e-4,
        "theta_crit_deg": 15.0,
        "score_quantile": 0.50,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render five smoothed positive injection examples with model guesses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-h5", type=str, default=str(DEFAULT_SOURCE_H5))
    parser.add_argument("--scores-npz", type=str, default=str(DEFAULT_SCORES))
    parser.add_argument("--recalibration-report", type=str, default=str(DEFAULT_RECAL_REPORT))
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fpr-target", type=float, default=0.10)
    parser.add_argument("--signal-beam-fwhm-arcmin", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def load_threshold(report_path: Path, method: str, fpr_target: float) -> float:
    report = load_json(report_path)
    for row in report["threshold_rows"]:
        if row["method"] == method and abs(float(row["fpr_target"]) - float(fpr_target)) < 1e-12:
            return float(row["real_threshold"])
    raise KeyError(f"Missing {method} threshold at FPR target {fpr_target} in {report_path}")


def build_model(run_dir: Path, checkpoint_arg: str, device: torch.device):
    run_config = load_json(run_dir / "run_config.json")
    model = p3.build_model(p3.model_args_from_run_config(run_config)).to(device)
    checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, checkpoint_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model, run_config, checkpoint_path, checkpoint_label


def select_examples(scores: np.lib.npyio.NpzFile, threshold: float) -> list[dict]:
    positive_scores = np.asarray(scores["positive_scores"], dtype=np.float32)
    amplitudes = np.asarray(scores["positive_amplitude"], dtype=np.float64)
    radii = np.asarray(scores["positive_theta_crit_deg"], dtype=np.float64)
    source_rows = np.asarray(scores["positive_source_index"], dtype=np.int64)
    edge_sigma = np.asarray(scores["positive_edge_sigma_deg"], dtype=np.float64)

    selected: list[dict] = []
    used: set[int] = set()
    for target in EXAMPLE_TARGETS:
        detected = positive_scores > threshold
        mask = (
            np.isclose(amplitudes, float(target["amplitude"]), rtol=1e-5, atol=0.0)
            & np.isclose(radii, float(target["theta_crit_deg"]), rtol=1e-5, atol=0.0)
            & (detected == bool(target["detected"]))
        )
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            raise RuntimeError(f"No candidate matched target {target}")
        candidate_scores = positive_scores[candidates]
        goal = float(np.quantile(candidate_scores, float(target["score_quantile"])))
        local = int(np.argmin(np.abs(candidate_scores - goal)))
        pos_local = int(candidates[local])
        while pos_local in used and candidates.size > 1:
            candidates = candidates[candidates != pos_local]
            candidate_scores = positive_scores[candidates]
            goal = float(np.quantile(candidate_scores, float(target["score_quantile"])))
            local = int(np.argmin(np.abs(candidate_scores - goal)))
            pos_local = int(candidates[local])
        used.add(pos_local)
        selected.append(
            {
                "label": str(target["label"]),
                "expected_detected": bool(target["detected"]),
                "positive_local_index": pos_local,
                "source_row": int(source_rows[pos_local]),
                "score": float(positive_scores[pos_local]),
                "edge_sigma_deg": float(edge_sigma[pos_local]),
                "amplitude": float(amplitudes[pos_local]),
                "theta_crit_deg": float(radii[pos_local]),
            }
        )
    return selected


def normalize_patch(model_patch: np.ndarray, run_config: dict) -> np.ndarray:
    norm = p3.dataset_kwargs_from_run_config(run_config)
    means = np.asarray(norm["channel_means"], dtype=np.float32)
    stds = np.maximum(np.asarray(norm["channel_stds"], dtype=np.float32), 1e-8)
    if len(norm["extra_channel_datasets"]) != 0:
        raise RuntimeError("This gallery expects the one-channel v6 model.")
    return ((model_patch[None, None, :, :] - means[None, :, None, None]) / stds[None, :, None, None]).astype(np.float32)


def reconstruct_example(h5: h5py.File, row: int, edge_sigma_deg: float, beam_sigma_pix: float) -> dict:
    background_index = int(h5["metadata"]["background_index"][row])
    base = np.asarray(h5["patches"][background_index], dtype=np.float32)
    center_x = float(h5["truth"]["signal_center_x_pix"][row])
    center_y = float(h5["truth"]["signal_center_y_pix"][row])
    theta_crit_deg = float(h5["truth"]["theta_crit_deg"][row])
    theta_dist = make_angular_distance_grid(
        PATCH_PIX,
        RESO_ARCMIN,
        center_x_pix=center_x,
        center_y_pix=center_y,
    ).astype(np.float32)
    signal = bubble_collision_signal(
        theta_dist,
        float(h5["truth"]["z0"][row]),
        float(h5["truth"]["zcrit"][row]),
        np.radians(theta_crit_deg),
        edge_sigma_deg=float(edge_sigma_deg),
    )
    signal_delta = np.asarray(
        fractional_signal_delta(
            base,
            signal,
            injection_convention=DEFAULT_INJECTION_CONVENTION,
        ),
        dtype=np.float32,
    )
    if beam_sigma_pix > 0.0:
        signal_delta = gaussian_filter(signal_delta, sigma=beam_sigma_pix, mode="reflect")
    patch = (base + signal_delta).astype(np.float32)
    return {
        "patch": patch,
        "true_mask": np.asarray(h5["masks"][row], dtype=bool),
        "signal_delta": signal_delta.astype(np.float32),
        "background_index": background_index,
        "center_x": center_x,
        "center_y": center_y,
    }


def score_probability_map(model, run_config: dict, patch: np.ndarray, device: torch.device) -> np.ndarray:
    batch = normalize_patch(patch, run_config)
    with torch.no_grad():
        tensor = torch.from_numpy(batch).to(device)
        logits, _ = p3.unpack_model_output(model(tensor))
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)


def add_image(ax, image, title: str, cmap: str, vmin=None, vmax=None):
    im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def render_example_axes(fig, axes, example: dict, threshold: float, show_colorbar: bool = False) -> None:
    patch_uk = example["patch"] * 1e6
    prob = example["probability"]
    true_mask = example["true_mask"]
    pred_mask = prob > threshold
    lo, hi = np.percentile(patch_uk, [1.0, 99.0])
    lim = max(abs(float(lo)), abs(float(hi)), 1e-6)

    im_patch = add_image(axes[0], patch_uk, "positive patch [uK]", "coolwarm", -lim, lim)
    add_image(axes[1], true_mask.astype(float), "truth mask", "Greens", 0.0, 1.0)
    im_prob = add_image(axes[2], prob, "model probability", "magma", 0.0, 1.0)
    add_image(axes[3], pred_mask.astype(float), "model guess mask", "Purples", 0.0, 1.0)
    axes[4].imshow(patch_uk, origin="lower", cmap="gray", vmin=-lim, vmax=lim)
    axes[4].contour(true_mask.astype(float), levels=[0.5], colors=["lime"], linewidths=1.0)
    if pred_mask.any():
        axes[4].contour(pred_mask.astype(float), levels=[0.5], colors=["magenta"], linewidths=1.0)
    axes[4].set_title("overlay: truth green, pred magenta", fontsize=9)
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    if show_colorbar:
        fig.colorbar(im_patch, ax=axes[0], fraction=0.046, pad=0.02)
        fig.colorbar(im_prob, ax=axes[2], fraction=0.046, pad=0.02)


def metadata_line(example: dict, threshold: float) -> str:
    outcome = "TP" if example["score"] > threshold else "FN"
    return (
        f"{outcome} | {example['label']} | "
        f"A={example['amplitude']:.1e}, theta_c={example['theta_crit_deg']:.1f} deg, "
        f"edge_sigma={example['edge_sigma_deg']:.3f} deg, score={example['score']:.3f}, "
        f"threshold={threshold:.3f}, z0={example['z0']:.2e}, zcrit={example['zcrit']:.2e}, "
        f"center=({example['center_x']:.1f},{example['center_y']:.1f}) px, "
        f"mask_frac={example['mask_fraction']:.3f}, touches_edge={example['target_touches_edge']}"
    )


def render_gallery(output_dir: Path, examples: list[dict], threshold: float, dpi: int) -> Path:
    gallery_path = output_dir / "v6_smoothed_positive_examples_gallery.png"
    fig = plt.figure(figsize=(22, 19), constrained_layout=False)
    gs = fig.add_gridspec(nrows=len(examples) * 2, ncols=5, height_ratios=[0.18, 1.0] * len(examples))
    for row, example in enumerate(examples):
        text_ax = fig.add_subplot(gs[row * 2, :])
        text_ax.axis("off")
        text_ax.text(0.01, 0.45, metadata_line(example, threshold), ha="left", va="center", fontsize=10)
        axes = [fig.add_subplot(gs[row * 2 + 1, col]) for col in range(5)]
        render_example_axes(fig, axes, example, threshold)
    fig.suptitle("Real-SMICA smoothed positive injections with v6_aux_only guesses", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(gallery_path, dpi=dpi)
    plt.close(fig)
    return gallery_path


def render_individual(output_dir: Path, example: dict, example_number: int, threshold: float, dpi: int) -> Path:
    outcome = "TP" if example["score"] > threshold else "FN"
    stem = f"example_{example_number:02d}_{outcome}_A{example['amplitude']:.0e}_theta{example['theta_crit_deg']:.0f}"
    path = output_dir / f"{stem}.png"
    fig = plt.figure(figsize=(20, 4.8), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=5, height_ratios=[0.23, 1.0])
    text_ax = fig.add_subplot(gs[0, :])
    text_ax.axis("off")
    text_ax.text(0.01, 0.45, metadata_line(example, threshold), ha="left", va="center", fontsize=10)
    axes = [fig.add_subplot(gs[1, col]) for col in range(5)]
    render_example_axes(fig, axes, example, threshold, show_colorbar=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_selection_csv(path: Path, examples: list[dict], threshold: float) -> None:
    fieldnames = [
        "example_number",
        "outcome",
        "label",
        "source_row",
        "positive_local_index",
        "background_index",
        "score",
        "threshold",
        "amplitude",
        "theta_crit_deg",
        "edge_sigma_deg",
        "z0",
        "zcrit",
        "center_x",
        "center_y",
        "glon_deg",
        "glat_deg",
        "mask_fraction",
        "target_touches_edge",
        "fully_contained",
        "image_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, example in enumerate(examples, start=1):
            row = {key: example.get(key) for key in fieldnames}
            row["example_number"] = idx
            row["outcome"] = "TP" if example["score"] > threshold else "FN"
            row["threshold"] = threshold
            row["image_path"] = str(example["image_path"])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = load_threshold(Path(args.recalibration_report).resolve(), "v6_aux_only", float(args.fpr_target))
    device = p3.resolve_device(args.device)
    model, run_config, checkpoint_path, checkpoint_label = build_model(Path(args.run_dir).resolve(), args.checkpoint, device)
    beam_sigma_pix = fwhm_arcmin_to_sigma_pixels(args.signal_beam_fwhm_arcmin)

    with np.load(Path(args.scores_npz).resolve()) as scores:
        selected = select_examples(scores, threshold)

    examples: list[dict] = []
    with h5py.File(Path(args.source_h5).resolve(), "r") as h5:
        for item in selected:
            row = int(item["source_row"])
            reconstructed = reconstruct_example(h5, row, float(item["edge_sigma_deg"]), beam_sigma_pix)
            prob = score_probability_map(model, run_config, reconstructed["patch"], device)
            example = {
                **item,
                **reconstructed,
                "probability": prob,
                "z0": float(h5["truth"]["z0"][row]),
                "zcrit": float(h5["truth"]["zcrit"][row]),
                "glon_deg": float(h5["metadata"]["glon_deg"][row]),
                "glat_deg": float(h5["metadata"]["glat_deg"][row]),
                "mask_fraction": float(h5["metadata"]["coord_mask_fraction"][row]),
                "target_touches_edge": int(h5["truth"]["target_touches_edge"][row]),
                "fully_contained": int(h5["truth"]["fully_contained"][row]),
            }
            examples.append(example)

    for idx, example in enumerate(examples, start=1):
        example["image_path"] = render_individual(output_dir, example, idx, threshold, args.dpi)
    gallery_path = render_gallery(output_dir, examples, threshold, args.dpi)
    csv_path = output_dir / "selected_examples.csv"
    write_selection_csv(csv_path, examples, threshold)
    manifest = {
        "gallery": str(gallery_path),
        "selection_csv": str(csv_path),
        "threshold": threshold,
        "fpr_target": float(args.fpr_target),
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": str(checkpoint_label),
        "source_h5": str(Path(args.source_h5).resolve()),
        "scores_npz": str(Path(args.scores_npz).resolve()),
        "examples": [
            {
                "label": ex["label"],
                "outcome": "TP" if ex["score"] > threshold else "FN",
                "score": ex["score"],
                "amplitude": ex["amplitude"],
                "theta_crit_deg": ex["theta_crit_deg"],
                "edge_sigma_deg": ex["edge_sigma_deg"],
                "source_row": ex["source_row"],
                "image_path": str(ex["image_path"]),
            }
            for ex in examples
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
