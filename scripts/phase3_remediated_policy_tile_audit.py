"""Full-sky tile burden audit for remediated score-composite policies.

Assumptions
-----------
* Existing Batch 6 tile HDF5 files contain signal-free Planck cleaned-map
  patches and are valid for measuring deployment candidate burden.
* The remediated ML policy scores are patch-level maximum segmentation
  probabilities, matching ``phase3_sensitivity_curve.py`` and
  ``phase3_policy_pareto_search.py``.
* The circular-template score is the patch-space Feeney-template screen, not a
  Wiener matched filter.
* Trigger fractions on tiled maps are deployment-burden diagnostics, not
  p-values, because overlapping tiles produce correlated triggers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.signal import fftconvolve

import phase3_train_unet as p3
from phase_config import DEFAULTS
from phase2_signal_model import PATCH_PIX
from phase3_fullsky_tile import CLUSTER_RADII_DEG, greedy_cluster, peak_sky_coord
from phase3_sensitivity_curve import (
    DEFAULT_THETA_GRID_DEG,
    SIGN_QUADRANTS,
    build_model_from_run,
    make_feeney_template_kernel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_policy_pareto"
    / "policy_pareto.json"
)
DEFAULT_TILE_ROOT_TEMPLATE = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "batch6_fullsky_nside32_{map}"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_policy_tile_audit"
)
DEFAULT_MODELS = (
    "random_b64_aux:runs/phase3_unet/remediated_v1_unet_random_b64_aux:best",
    "imagenet_b64_aux:runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux:best",
)
DEFAULT_MAPS = "smica,nilc,sevem,commander"
ML_METHODS = ("random_b64_aux", "imagenet_b64_aux")
CIRCULAR_METHOD = "circular_template_screen"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved remediated U-Net checkpoint specification."""

    name: str
    run_dir: Path
    checkpoint: str


def parse_model_spec(text: str) -> ModelSpec:
    """Parse ``name:run_dir:checkpoint``."""

    parts = str(text).split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid model spec {text!r}; expected name:run_dir:checkpoint.")
    name, run_dir, checkpoint = parts
    if name not in ML_METHODS:
        raise ValueError(f"Unsupported model name {name!r}; expected one of {ML_METHODS}.")
    return ModelSpec(name=name, run_dir=Path(run_dir), checkpoint=checkpoint)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Audit remediated policy-Pareto candidates on full-sky tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy-json", type=str, default=str(DEFAULT_POLICY_JSON))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--maps", type=str, default=DEFAULT_MAPS)
    parser.add_argument("--tile-nside", type=int, default=32)
    parser.add_argument(
        "--tile-root-template",
        type=str,
        default=str(DEFAULT_TILE_ROOT_TEMPLATE),
        help="Format string containing {map}; must contain tile_patches_<map>_nside<N>.h5.",
    )
    parser.add_argument("--model", action="append", default=[], help="name:run_dir:checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--circular-batch-size", type=int, default=64)
    parser.add_argument("--circular-kernel-chunk", type=int, default=4)
    parser.add_argument("--circular-engine", type=str, default="auto", choices=("auto", "torch", "scipy"))
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--top-rank-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--theta-grid-deg",
        type=str,
        default=",".join(f"{x:g}" for x in DEFAULT_THETA_GRID_DEG),
    )
    parser.add_argument("--beam-fwhm-arcmin", type=float, default=DEFAULTS.beam_fwhm_arcmin)
    return parser.parse_args()


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse comma-separated floats."""

    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one float in list.")
    return values


def validate_args(args: argparse.Namespace) -> None:
    """Validate non-physical or unsupported arguments."""

    args.maps = tuple(item.strip().lower() for item in str(args.maps).split(",") if item.strip())
    if not args.maps:
        raise ValueError("--maps must contain at least one map name.")
    if args.tile_nside <= 0:
        raise ValueError("--tile-nside must be positive.")
    if args.batch_size <= 0 or args.circular_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.circular_kernel_chunk <= 0:
        raise ValueError("--circular-kernel-chunk must be positive.")
    args.theta_grid_deg = parse_float_list(args.theta_grid_deg)
    if any(theta <= 0.0 for theta in args.theta_grid_deg):
        raise ValueError("Template radii must be positive.")
    if args.beam_fwhm_arcmin < 0.0:
        raise ValueError("--beam-fwhm-arcmin must be non-negative.")
    args.models = tuple(parse_model_spec(text) for text in (args.model or DEFAULT_MODELS))


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slugify(text: str) -> str:
    """Return a filesystem-safe identifier."""

    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return text[:96] or "policy"


def load_ranked_policies(path: Path, top_rank_only: bool) -> list[dict[str, Any]]:
    """Load policy rows emitted by ``phase3_policy_pareto_search.py``."""

    report = load_json(path)
    rows = list(report.get("top_rows", []))
    if top_rank_only:
        rows = [row for row in rows if int(row.get("rank", -1)) == 1]
    if not rows:
        raise ValueError(f"No policy rows found in {path}.")
    for row in rows:
        thresholds = row.get("thresholds", {})
        if not isinstance(thresholds, dict) or not thresholds:
            raise ValueError(f"Policy row has no thresholds: {row.get('policy')}")
        for method, threshold in thresholds.items():
            if method not in (*ML_METHODS, CIRCULAR_METHOD):
                raise ValueError(f"Unsupported policy method: {method}")
            if not np.isfinite(float(threshold)):
                raise ValueError(f"Non-finite threshold for {method}: {threshold}")
    return rows


def tile_h5_path(args: argparse.Namespace, map_name: str) -> Path:
    """Resolve the existing Batch 6 tile HDF5 path for a map."""

    root = Path(str(args.tile_root_template).format(map=map_name)).resolve()
    path = root / f"tile_patches_{map_name}_nside{int(args.tile_nside)}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Missing tile HDF5: {path}")
    return path


def load_tile_metadata(tile_h5: Path) -> dict[str, np.ndarray]:
    """Load tile coordinates and mask fractions."""

    with h5py.File(tile_h5, "r") as h5:
        if "metadata" not in h5:
            raise KeyError(f"{tile_h5} missing metadata group.")
        meta = h5["metadata"]
        required = ("glon_deg", "glat_deg", "mask_fraction")
        missing = [key for key in required if key not in meta]
        if missing:
            raise KeyError(f"{tile_h5} missing metadata arrays: {missing}")
        out = {key: np.asarray(meta[key][:], dtype=np.float64) for key in required}
        n = int(h5["patches"].shape[0])
    if any(arr.shape[0] != n for arr in out.values()):
        raise ValueError(f"Metadata length mismatch in {tile_h5}.")
    return out


def score_ml_tiles(
    spec: ModelSpec,
    tile_h5: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Score tile patches with one remediated U-Net checkpoint."""

    model, run_config, _checkpoint_path, _checkpoint_label = build_model_from_run(
        spec.run_dir.resolve(),
        spec.checkpoint,
        device,
    )
    with h5py.File(tile_h5, "r") as h5:
        n = int(h5["patches"].shape[0])
    dataset = p3.H5BubbleDataset(
        h5_path=str(tile_h5),
        indices=np.arange(n, dtype=np.int64),
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 20260420,
        max_translate_pixels=0,
        cache_data=True,
    )
    loader = p3.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    score = np.zeros(n, dtype=np.float32)
    peak_i = np.zeros(n, dtype=np.int32)
    peak_j = np.zeros(n, dtype=np.int32)
    offset = 0
    model.eval()
    progress = p3.ProgressPrinter(len(loader), f"Tile ML scores {spec.name}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            mask_logits, _aux_logits = p3.unpack_model_output(model(images))
            probs = torch.sigmoid(mask_logits).squeeze(1).float().cpu().numpy()
            batch_n = int(probs.shape[0])
            flat = probs.reshape(batch_n, -1)
            flat_idx = np.argmax(flat, axis=1)
            score[offset : offset + batch_n] = np.max(flat, axis=1)
            peak_i[offset : offset + batch_n] = flat_idx // PATCH_PIX
            peak_j[offset : offset + batch_n] = flat_idx % PATCH_PIX
            offset += batch_n
            progress.update(batch_idx)
    if not np.all(np.isfinite(score)):
        raise ValueError(f"Non-finite ML scores for {spec.name} on {tile_h5}.")
    return {
        f"score__{spec.name}": score,
        f"peak_i__{spec.name}": peak_i,
        f"peak_j__{spec.name}": peak_j,
    }


def circular_kernels(theta_grid_deg: tuple[float, ...], beam_fwhm_arcmin: float) -> np.ndarray:
    """Build circular-template kernels with the sensitivity-curve recipe."""

    kernels = [
        make_feeney_template_kernel(theta, z0_sign, zcrit_sign, beam_fwhm_arcmin)
        for theta in theta_grid_deg
        for z0_sign, zcrit_sign in SIGN_QUADRANTS
    ]
    return np.stack(kernels, axis=0).astype(np.float32)


def standardize_patch_batch(patches: np.ndarray) -> np.ndarray:
    """Standardize patches independently in float64, returning float32."""

    work = np.asarray(patches, dtype=np.float64)
    flat = work.reshape(work.shape[0], -1)
    mean = np.mean(flat, axis=1)[:, None, None]
    std = np.std(flat, axis=1)[:, None, None]
    std = np.where(std > 0.0, std, 1.0)
    return ((work - mean) / std).astype(np.float32)


def score_circular_tiles(
    tile_h5: Path,
    *,
    theta_grid_deg: tuple[float, ...],
    beam_fwhm_arcmin: float,
    batch_size: int,
    kernel_chunk: int,
    device: torch.device,
    engine: str,
) -> np.ndarray:
    """Vectorized circular-template tile scoring."""

    kernels = circular_kernels(theta_grid_deg, beam_fwhm_arcmin)[:, ::-1, ::-1]
    if engine == "auto":
        engine = "torch" if device.type == "cuda" else "scipy"
    if engine == "torch":
        return score_circular_tiles_torch(
            tile_h5,
            kernels=kernels,
            batch_size=batch_size,
            kernel_chunk=kernel_chunk,
            device=device,
        )
    return score_circular_tiles_scipy(tile_h5, kernels=kernels, batch_size=batch_size)


def score_circular_tiles_scipy(
    tile_h5: Path,
    *,
    kernels: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """CPU fallback using scipy's exact FFT convolution."""

    with h5py.File(tile_h5, "r") as h5:
        patches = h5["patches"]
        n = int(patches.shape[0])
        scores = np.zeros(n, dtype=np.float32)
        progress = p3.ProgressPrinter((n + batch_size - 1) // batch_size, "Tile circular scores")
        batch_idx = 0
        for start in range(0, n, int(batch_size)):
            stop = min(start + int(batch_size), n)
            patch_batch = standardize_patch_batch(np.asarray(patches[start:stop], dtype=np.float32))
            best = np.full(patch_batch.shape[0], -np.inf, dtype=np.float32)
            for kernel in kernels:
                response = fftconvolve(
                    patch_batch,
                    kernel[None, :, :],
                    mode="same",
                    axes=(-2, -1),
                )
                best = np.maximum(best, np.max(response, axis=(1, 2)).astype(np.float32))
            scores[start:stop] = best
            batch_idx += 1
            progress.update(batch_idx)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite circular scores for {tile_h5}.")
    return scores


def score_circular_tiles_torch(
    tile_h5: Path,
    *,
    kernels: np.ndarray,
    batch_size: int,
    kernel_chunk: int,
    device: torch.device,
) -> np.ndarray:
    """CUDA/torch FFT implementation of scipy ``fftconvolve(..., mode='same')``."""

    if device.type != "cuda":
        raise ValueError("Torch circular engine requires a CUDA device.")
    full_shape = (2 * PATCH_PIX - 1, 2 * PATCH_PIX - 1)
    crop_i = (PATCH_PIX - 1) // 2
    crop_j = (PATCH_PIX - 1) // 2
    kernel_tensor = torch.zeros(
        (kernels.shape[0], full_shape[0], full_shape[1]),
        dtype=torch.float32,
        device=device,
    )
    kernel_tensor[:, :PATCH_PIX, :PATCH_PIX] = torch.as_tensor(
        np.ascontiguousarray(kernels),
        dtype=torch.float32,
        device=device,
    )
    kernel_fft = torch.fft.rfft2(kernel_tensor, s=full_shape)
    with h5py.File(tile_h5, "r") as h5:
        patches = h5["patches"]
        n = int(patches.shape[0])
        scores = np.zeros(n, dtype=np.float32)
        progress = p3.ProgressPrinter((n + batch_size - 1) // batch_size, "Tile circular scores")
        batch_idx = 0
        for start in range(0, n, int(batch_size)):
            stop = min(start + int(batch_size), n)
            patch_batch = standardize_patch_batch(np.asarray(patches[start:stop], dtype=np.float32))
            batch_tensor = torch.zeros(
                (patch_batch.shape[0], full_shape[0], full_shape[1]),
                dtype=torch.float32,
                device=device,
            )
            batch_tensor[:, :PATCH_PIX, :PATCH_PIX] = torch.as_tensor(
                np.ascontiguousarray(patch_batch),
                dtype=torch.float32,
                device=device,
            )
            batch_fft = torch.fft.rfft2(batch_tensor, s=full_shape)
            best = torch.full((patch_batch.shape[0],), -torch.inf, dtype=torch.float32, device=device)
            for k0 in range(0, kernels.shape[0], int(kernel_chunk)):
                k1 = min(k0 + int(kernel_chunk), kernels.shape[0])
                conv = torch.fft.irfft2(
                    batch_fft[:, None, :, :] * kernel_fft[None, k0:k1, :, :],
                    s=full_shape,
                )
                same = conv[
                    :,
                    :,
                    crop_i : crop_i + PATCH_PIX,
                    crop_j : crop_j + PATCH_PIX,
                ]
                best = torch.maximum(best, torch.amax(same, dim=(1, 2, 3)))
                del conv, same
            scores[start:stop] = best.detach().cpu().numpy()
            del batch_tensor, batch_fft, best
            batch_idx += 1
            progress.update(batch_idx)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite circular scores for {tile_h5}.")
    return scores


def load_or_score_map(
    args: argparse.Namespace,
    map_name: str,
    output_dir: Path,
    device: torch.device,
) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load or produce score cache for one full-sky tile map."""

    tile_h5 = tile_h5_path(args, map_name)
    metadata = load_tile_metadata(tile_h5)
    cache_path = output_dir / f"tile_scores_{map_name}_nside{int(args.tile_nside)}.npz"
    if args.reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            scores = {key.removeprefix("score__"): np.asarray(loaded[key]) for key in loaded.files if key.startswith("score__")}
            peaks = {key: np.asarray(loaded[key]) for key in loaded.files if key.startswith("peak_")}
        return tile_h5, {**metadata, **scores}, peaks

    payload: dict[str, np.ndarray] = {}
    for spec in args.models:
        payload.update(
            score_ml_tiles(
                spec,
                tile_h5,
                batch_size=int(args.batch_size),
                device=device,
            )
        )
    payload[f"score__{CIRCULAR_METHOD}"] = score_circular_tiles(
        tile_h5,
        theta_grid_deg=args.theta_grid_deg,
        beam_fwhm_arcmin=float(args.beam_fwhm_arcmin),
        batch_size=int(args.circular_batch_size),
        kernel_chunk=int(args.circular_kernel_chunk),
        device=device,
        engine=str(args.circular_engine),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **metadata, **payload)
    scores = {
        key.removeprefix("score__"): np.asarray(value)
        for key, value in payload.items()
        if key.startswith("score__")
    }
    peaks = {key: np.asarray(value) for key, value in payload.items() if key.startswith("peak_")}
    return tile_h5, {**metadata, **scores}, peaks


def apply_policy(row: dict[str, Any], scores: dict[str, np.ndarray]) -> np.ndarray:
    """Return tile-trigger mask for one policy row."""

    thresholds = {key: float(value) for key, value in row["thresholds"].items()}
    bools = {method: np.asarray(scores[method]) >= threshold for method, threshold in thresholds.items()}
    family = str(row["family"])
    if family in {"single", "single_exact"}:
        return next(iter(bools.values()))
    if family == "pair_and":
        return np.logical_and.reduce(tuple(bools.values()))
    if family == "pair_or":
        return np.logical_or.reduce(tuple(bools.values()))
    if family in {"2_of_3", "3_of_3"}:
        votes = int(family[0])
        count = np.zeros_like(next(iter(bools.values())), dtype=np.uint8)
        for value in bools.values():
            count += value.astype(np.uint8)
        return count >= votes
    raise ValueError(f"Unsupported policy family: {family}")


def policy_margin(row: dict[str, Any], scores: dict[str, np.ndarray]) -> np.ndarray:
    """Return a sortable threshold-relative score for clustering."""

    ratios = []
    for method, threshold in row["thresholds"].items():
        denom = max(abs(float(threshold)), 1.0e-12)
        ratios.append(np.asarray(scores[method], dtype=np.float64) / denom)
    return np.max(np.stack(ratios, axis=0), axis=0)


def best_ml_peak(
    idx: int,
    row: dict[str, Any],
    scores: dict[str, np.ndarray],
    peaks: dict[str, np.ndarray],
) -> tuple[str | None, int | None, int | None]:
    """Choose the ML model peak with largest threshold-relative margin."""

    best_method = None
    best_ratio = -np.inf
    thresholds = {key: float(value) for key, value in row["thresholds"].items()}
    for method in ML_METHODS:
        if method not in thresholds or method not in scores:
            continue
        ratio = float(scores[method][idx]) / max(abs(thresholds[method]), 1.0e-12)
        if ratio > best_ratio:
            best_method = method
            best_ratio = ratio
    if best_method is None:
        return None, None, None
    return (
        best_method,
        int(peaks[f"peak_i__{best_method}"][idx]),
        int(peaks[f"peak_j__{best_method}"][idx]),
    )


def evaluate_policy_on_map(
    map_name: str,
    row: dict[str, Any],
    scores: dict[str, np.ndarray],
    peaks: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate one policy on one full-sky tile map and write candidate files."""

    trigger = apply_policy(row, scores)
    n = int(trigger.shape[0])
    margins = policy_margin(row, scores)
    policy_slug = slugify(
        f"camb{row['constraint_camb_fpr_max']}_real{row['constraint_real_fpr_max']}_rank{row['rank']}"
    )
    trigger_records = []
    for idx in np.flatnonzero(trigger):
        peak_model, peak_i, peak_j = best_ml_peak(idx, row, scores, peaks)
        if peak_model is None:
            peak_glon = float(scores["glon_deg"][idx])
            peak_glat = float(scores["glat_deg"][idx])
        else:
            peak_glon, peak_glat = peak_sky_coord(
                float(scores["glon_deg"][idx]),
                float(scores["glat_deg"][idx]),
                int(peak_i),
                int(peak_j),
            )
        record = {
            "map": map_name,
            "policy_slug": policy_slug,
            "policy": row["policy"],
            "patch_index": int(idx),
            "patch_glon_deg": float(scores["glon_deg"][idx]),
            "patch_glat_deg": float(scores["glat_deg"][idx]),
            "peak_glon_deg": float(peak_glon),
            "peak_glat_deg": float(peak_glat),
            "peak_source_model": peak_model,
            "peak_pixel_i": peak_i,
            "peak_pixel_j": peak_j,
            "gbt_score": float(margins[idx]),
            "policy_margin": float(margins[idx]),
            "mask_fraction": float(scores["mask_fraction"][idx]),
            **{f"score__{method}": float(scores[method][idx]) for method in row["thresholds"]},
        }
        trigger_records.append(record)

    map_dir = output_dir / map_name
    map_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = map_dir / f"candidates_{policy_slug}.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for record in trigger_records:
            handle.write(json.dumps(record) + "\n")

    cluster_summary = {}
    for radius in CLUSTER_RADII_DEG:
        clusters, _assignment = greedy_cluster(trigger_records, float(radius))
        cluster_path = map_dir / f"clusters_{policy_slug}_{int(radius)}deg.jsonl"
        with cluster_path.open("w", encoding="utf-8") as handle:
            for cluster in clusters:
                handle.write(json.dumps(cluster) + "\n")
        cluster_summary[f"{float(radius):.1f}"] = {
            "n_clusters": int(len(clusters)),
            "max_cluster_size": int(max((c["n_members"] for c in clusters), default=0)),
            "mean_cluster_size": (
                float(np.mean([c["n_members"] for c in clusters])) if clusters else 0.0
            ),
        }

    return {
        "map": map_name,
        "policy_slug": policy_slug,
        "policy": row["policy"],
        "family": row["family"],
        "constraint_camb_fpr_max": float(row["constraint_camb_fpr_max"]),
        "constraint_real_fpr_max": float(row["constraint_real_fpr_max"]),
        "diagnostic_real_recall": float(row["real_recall"]),
        "diagnostic_real_fpr": float(row["real_fpr"]),
        "diagnostic_gain_vs_best_single": float(row.get("real_recall_gain_vs_best_single", np.nan)),
        "num_tiles": n,
        "num_triggered_tiles": int(np.count_nonzero(trigger)),
        "trigger_fraction": float(np.count_nonzero(trigger) / max(n, 1)),
        "candidate_jsonl": str(candidates_path),
        "cluster_summary": cluster_summary,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact summary CSV."""

    columns = [
        "map",
        "policy_slug",
        "constraint_camb_fpr_max",
        "constraint_real_fpr_max",
        "diagnostic_real_recall",
        "diagnostic_real_fpr",
        "diagnostic_gain_vs_best_single",
        "num_tiles",
        "num_triggered_tiles",
        "trigger_fraction",
        "clusters_15deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in columns}
            out["clusters_15deg"] = row["cluster_summary"].get("15.0", {}).get("n_clusters")
            writer.writerow(out)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write human-readable summary."""

    lines = ["# Remediated v1 Full-Sky Policy Tile Audit", ""]
    lines.append("These are candidate-burden diagnostics on overlapping full-sky tiles.")
    lines.append("They are not p-values and are not half-mission null significances.")
    lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    for note in report["assumption_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| map | policy | real recall diag | real FPR diag | tiles triggered | trigger frac | clusters 15 deg |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in report["rows"]:
        clusters_15 = row["cluster_summary"].get("15.0", {}).get("n_clusters")
        lines.append(
            f"| {row['map']} | `{row['policy_slug']}` | "
            f"{row['diagnostic_real_recall']:.4f} | {row['diagnostic_real_fpr']:.4f} | "
            f"{row['num_triggered_tiles']} / {row['num_tiles']} | "
            f"{row['trigger_fraction']:.4f} | {clusters_15} |"
        )
    lines.append("")
    lines.append("Policy text lives in the JSON report and candidate JSONL files.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = p3.resolve_device(args.device)
    policies = load_ranked_policies(Path(args.policy_json).resolve(), bool(args.top_rank_only))

    rows = []
    score_caches = {}
    for map_name in args.maps:
        tile_h5, scores, peaks = load_or_score_map(args, map_name, output_dir, device)
        score_caches[map_name] = str(tile_h5)
        for policy in policies:
            rows.append(evaluate_policy_on_map(map_name, policy, scores, peaks, output_dir))

    report = {
        "policy_json": str(Path(args.policy_json).resolve()),
        "tile_nside": int(args.tile_nside),
        "maps": list(args.maps),
        "score_caches": score_caches,
        "assumption_notes": [
            "Existing Batch 6 tile HDF5s are treated as signal-free cleaned-map deployment tiles.",
            "Scores are remediated ML max-probability scores plus circular-template patch scores.",
            "Overlapping tile trigger fractions are candidate-burden diagnostics, not independent FPRs.",
        ],
        "rows": rows,
    }
    json_path = output_dir / "policy_tile_audit.json"
    csv_path = output_dir / "policy_tile_audit.csv"
    md_path = output_dir / "policy_tile_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
