"""Tile-burden audit for the legacy matched-filter-channel U-Net branch.

Assumptions
-----------
* This is a deployment-burden audit, not a model-promotion decision by itself.
* The v7 matched-filter-channel checkpoint was trained with a second input
  channel named ``features/matched_filter_response``. The current full-sky tile
  HDF5 files do not contain that channel, so this script reconstructs the
  circular-template response in memory without modifying the tile files.
* The reconstructed feature uses the same radius/sign bank and default
  ``15 arcmin`` feature beam recorded in the legacy training-v4 feature attrs.
  This deliberately audits the existing checkpoint as trained; it is not a
  remediated-v1 retrain.
* Thresholds are read from an existing SMICA/CAMB recalibrated single-model
  report. They are screening thresholds, not p-values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

import phase3_train_unet as p3
from phase2_signal_model import PATCH_PIX
from phase3_circular_template_features import (
    circular_template_kernels,
    circular_template_response_maps_scipy,
    circular_template_response_maps_torch,
    prepare_circular_template_kernel_fft,
)
from phase3_evaluate_run import load_json, resolve_checkpoint_path
from phase3_fullsky_tile import greedy_cluster, peak_sky_coord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "phase3_v7_mf_channel_aux_w4"
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "phase3_v7_mf_channel_recalibrated_last"
    / "v7_mf_channel_last_recalibrated_eval.json"
)
DEFAULT_TILE_ROOT_TEMPLATE = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "batch6_fullsky_nside32_{map}"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_mf_channel_tile_audit"
)
DEFAULT_MAPS = "smica,nilc,sevem,commander"
DEFAULT_RADII_DEG = (5.0, 8.0, 12.0, 16.0, 20.0, 25.0)
DEFAULT_FEATURE_BEAM_FWHM_ARCMIN = 15.0


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse a comma-separated float list."""

    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Audit full-sky tile burden for a matched-filter-channel U-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="v7_mf_channel_last")
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint", type=str, default="last")
    parser.add_argument("--recalibrated-report", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--fpr-target", type=float, default=0.05)
    parser.add_argument("--threshold-domain", type=str, default="smica", choices=("smica", "camb"))
    parser.add_argument("--maps", type=str, default=DEFAULT_MAPS)
    parser.add_argument("--tile-nside", type=int, default=32)
    parser.add_argument("--tile-root-template", type=str, default=str(DEFAULT_TILE_ROOT_TEMPLATE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-kernel-chunk", type=int, default=4)
    parser.add_argument(
        "--feature-engine",
        type=str,
        default="auto",
        choices=("auto", "torch", "scipy"),
    )
    parser.add_argument(
        "--feature-radii-deg",
        type=str,
        default=",".join(f"{x:g}" for x in DEFAULT_RADII_DEG),
    )
    parser.add_argument(
        "--feature-beam-fwhm-arcmin",
        type=float,
        default=DEFAULT_FEATURE_BEAM_FWHM_ARCMIN,
    )
    parser.add_argument("--mask-threshold", type=float, default=0.9)
    parser.add_argument("--cluster-radius-deg", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--reuse-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--score-limit",
        type=int,
        default=0,
        help="Optional smoke-test tile limit per map.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate physical and numerical arguments."""

    args.maps = tuple(item.strip().lower() for item in str(args.maps).split(",") if item.strip())
    if not args.maps:
        raise ValueError("--maps must contain at least one map.")
    if not (0.0 < args.fpr_target < 1.0):
        raise ValueError("--fpr-target must lie in (0, 1).")
    if args.tile_nside <= 0:
        raise ValueError("--tile-nside must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.feature_kernel_chunk <= 0:
        raise ValueError("--feature-kernel-chunk must be positive.")
    args.feature_radii_deg = parse_float_list(args.feature_radii_deg)
    if any(radius <= 0.0 for radius in args.feature_radii_deg):
        raise ValueError("--feature-radii-deg values must be positive.")
    if args.feature_beam_fwhm_arcmin < 0.0:
        raise ValueError("--feature-beam-fwhm-arcmin must be non-negative.")
    if not (0.0 <= args.mask_threshold <= 1.0):
        raise ValueError("--mask-threshold must lie in [0, 1].")
    if args.cluster_radius_deg <= 0.0:
        raise ValueError("--cluster-radius-deg must be positive.")
    if args.score_limit < 0:
        raise ValueError("--score-limit must be non-negative.")
    for path_text in (args.run_dir, args.recalibrated_report):
        if not Path(path_text).expanduser().exists():
            raise FileNotFoundError(f"Missing input path: {path_text}")


def tile_h5_path(args: argparse.Namespace, map_name: str) -> Path:
    """Resolve one existing Batch 6 tile HDF5 path."""

    root = Path(str(args.tile_root_template).format(map=map_name)).expanduser().resolve()
    path = root / f"tile_patches_{map_name}_nside{int(args.tile_nside)}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Missing tile HDF5: {path}")
    return path


def threshold_from_report(
    path: Path,
    model_name: str,
    fpr_target: float,
    domain: str,
) -> dict[str, Any]:
    """Load the requested recalibrated threshold row."""

    report = load_json(path)
    rows = [
        row
        for row in report.get("threshold_rows", [])
        if str(row.get("method")) == str(model_name)
        and np.isclose(float(row.get("fpr_target")), float(fpr_target), rtol=0.0, atol=1.0e-12)
    ]
    if not rows:
        raise ValueError(
            f"No threshold row for model={model_name!r}, fpr_target={fpr_target} in {path}."
        )
    row = dict(rows[0])
    key = "smica_threshold" if domain == "smica" else "camb_threshold"
    if key not in row:
        raise KeyError(f"Threshold row missing {key}.")
    row["selected_threshold"] = float(row[key])
    row["selected_threshold_key"] = key
    return row


def diagnostic_context_from_report(path: Path, fpr_target: float) -> dict[str, Any]:
    """Extract synthetic/real-null diagnostic rows for the selected checkpoint."""

    report = load_json(path)
    global_rows = [
        row
        for row in report.get("global_rows", [])
        if np.isclose(float(row.get("fpr_target")), float(fpr_target), rtol=0.0, atol=1.0e-12)
        and row.get("policy") == "model_only"
    ]
    regime_rows = [
        row
        for row in report.get("regime_rows", [])
        if np.isclose(float(row.get("fpr_target")), float(fpr_target), rtol=0.0, atol=1.0e-12)
        and row.get("domain") == "real_smica_recalibrated"
        and row.get("policy") == "model_only"
    ]
    return {
        "global_model_only_rows": global_rows,
        "real_smica_model_only_regime_rows": regime_rows,
    }


def build_model(run_dir: Path, checkpoint_arg: str, device: torch.device):
    """Load the U-Net and run config."""

    run_config = load_json(run_dir / "run_config.json")
    model = p3.build_model(p3.model_args_from_run_config(run_config)).to(device)
    checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, checkpoint_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    input_config = p3.input_config_from_run_config(run_config)
    if int(input_config["input_channels"]) != 2:
        raise ValueError(
            "Expected a two-channel matched-filter model; found "
            f"{input_config['input_channels']} channels."
        )
    return model, run_config, input_config, checkpoint_path, checkpoint_label


def score_map_tiles(
    args: argparse.Namespace,
    map_name: str,
    model: torch.nn.Module,
    input_config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """Load or compute tile scores and peak locations for one map."""

    cache_path = output_dir / f"mf_channel_scores_{map_name}_nside{int(args.tile_nside)}.npz"
    tile_h5 = tile_h5_path(args, map_name)
    with h5py.File(tile_h5, "r") as h5:
        n_total = int(h5["patches"].shape[0])
    expected_n = min(n_total, int(args.score_limit)) if args.score_limit else n_total
    if args.reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            payload = {key: np.asarray(loaded[key]) for key in loaded.files}
        cached_n = int(np.asarray(payload["score"]).shape[0])
        if cached_n == expected_n:
            return payload
        print(
            f"Cache length mismatch for {map_name}: expected {expected_n}, "
            f"found {cached_n}; recomputing.",
            flush=True,
        )

    kernels = circular_template_kernels(
        tuple(args.feature_radii_deg),
        float(args.feature_beam_fwhm_arcmin),
    )
    engine = str(args.feature_engine)
    if engine == "auto":
        engine = "torch" if device.type == "cuda" else "scipy"
    if engine == "torch" and device.type != "cuda":
        raise ValueError("Torch feature engine requires CUDA.")
    kernel_fft = (
        prepare_circular_template_kernel_fft(kernels, device)
        if engine == "torch"
        else None
    )
    means = np.asarray(input_config["channel_means"], dtype=np.float32)
    stds = np.maximum(np.asarray(input_config["channel_stds"], dtype=np.float32), 1.0e-8)

    with h5py.File(tile_h5, "r") as h5:
        patches = h5["patches"]
        meta = h5["metadata"]
        n = expected_n
        score = np.zeros(n, dtype=np.float32)
        peak_i = np.zeros(n, dtype=np.int32)
        peak_j = np.zeros(n, dtype=np.int32)
        progress = p3.ProgressPrinter(
            int(np.ceil(n / args.batch_size)),
            f"MF-channel tile scores {map_name}",
        )
        model.eval()
        with torch.no_grad():
            batch_idx = 0
            for start in range(0, n, int(args.batch_size)):
                stop = min(start + int(args.batch_size), n)
                patch_batch = np.asarray(patches[start:stop], dtype=np.float32)
                if engine == "torch":
                    feature_batch = circular_template_response_maps_torch(
                        patch_batch,
                        kernel_fft,
                        kernel_chunk=int(args.feature_kernel_chunk),
                        device=device,
                    )
                else:
                    feature_batch = circular_template_response_maps_scipy(patch_batch, kernels)
                batch = np.stack((patch_batch, feature_batch), axis=1)
                batch = (batch - means[None, :, None, None]) / stds[None, :, None, None]
                images = torch.as_tensor(batch, dtype=torch.float32, device=device)
                mask_logits, _aux_logits = p3.unpack_model_output(model(images))
                probs = torch.sigmoid(mask_logits).squeeze(1).detach().cpu().numpy()
                flat = probs.reshape(probs.shape[0], -1)
                flat_idx = np.argmax(flat, axis=1)
                score[start:stop] = np.max(flat, axis=1).astype(np.float32)
                peak_i[start:stop] = (flat_idx // PATCH_PIX).astype(np.int32)
                peak_j[start:stop] = (flat_idx % PATCH_PIX).astype(np.int32)
                batch_idx += 1
                progress.update(batch_idx)
        payload = {
            "score": score,
            "peak_i": peak_i,
            "peak_j": peak_j,
            "glon_deg": np.asarray(meta["glon_deg"][:n], dtype=np.float64),
            "glat_deg": np.asarray(meta["glat_deg"][:n], dtype=np.float64),
            "mask_fraction": np.asarray(meta["mask_fraction"][:n], dtype=np.float64),
        }
    if not np.all(np.isfinite(payload["score"])):
        raise ValueError(f"Non-finite scores for {map_name}.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def candidate_records(
    map_name: str,
    scores: dict[str, np.ndarray],
    threshold: float,
) -> list[dict[str, Any]]:
    """Build triggered tile candidate records."""

    records = []
    for idx in np.flatnonzero(np.asarray(scores["score"]) > float(threshold)):
        peak_glon, peak_glat = peak_sky_coord(
            float(scores["glon_deg"][idx]),
            float(scores["glat_deg"][idx]),
            int(scores["peak_i"][idx]),
            int(scores["peak_j"][idx]),
        )
        records.append(
            {
                "map": map_name,
                "patch_index": int(idx),
                "patch_glon_deg": float(scores["glon_deg"][idx]),
                "patch_glat_deg": float(scores["glat_deg"][idx]),
                "peak_glon_deg": float(peak_glon),
                "peak_glat_deg": float(peak_glat),
                "peak_pixel_i": int(scores["peak_i"][idx]),
                "peak_pixel_j": int(scores["peak_j"][idx]),
                "score": float(scores["score"][idx]),
                "gbt_score": float(scores["score"][idx]),
                "threshold": float(threshold),
                "score_margin": float(scores["score"][idx] / max(abs(float(threshold)), 1.0e-12)),
                "mask_fraction": float(scores["mask_fraction"][idx]),
            }
        )
    records.sort(key=lambda row: row["score_margin"], reverse=True)
    return records


def summarize_map(
    map_name: str,
    scores: dict[str, np.ndarray],
    threshold: float,
    *,
    mask_threshold: float,
    cluster_radius_deg: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Summarize tile and clustered burden for one map."""

    all_records = candidate_records(map_name, scores, threshold)
    eligible_records = [
        row for row in all_records if float(row["mask_fraction"]) >= float(mask_threshold)
    ]
    clusters, _assignment = greedy_cluster(eligible_records, float(cluster_radius_deg))
    map_dir = output_dir / map_name
    map_dir.mkdir(parents=True, exist_ok=True)
    all_path = map_dir / "mf_channel_candidates_all.jsonl"
    eligible_path = map_dir / "mf_channel_candidates_masked.jsonl"
    cluster_path = map_dir / f"mf_channel_clusters_{int(cluster_radius_deg)}deg.jsonl"
    for path, rows in (
        (all_path, all_records),
        (eligible_path, eligible_records),
        (cluster_path, clusters),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    n = int(np.asarray(scores["score"]).shape[0])
    eligible = np.asarray(scores["mask_fraction"], dtype=np.float64) >= float(mask_threshold)
    n_eligible = int(np.count_nonzero(eligible))
    trigger = np.asarray(scores["score"], dtype=np.float64) > float(threshold)
    return {
        "map": map_name,
        "num_tiles": n,
        "num_triggered_tiles": int(np.count_nonzero(trigger)),
        "trigger_fraction": float(np.count_nonzero(trigger) / max(n, 1)),
        "num_tiles_passing_mask_fraction": n_eligible,
        "num_triggered_eligible_tiles": int(np.count_nonzero(trigger & eligible)),
        "eligible_trigger_fraction": float(
            np.count_nonzero(trigger & eligible) / max(n_eligible, 1)
        ),
        "cluster_radius_deg": float(cluster_radius_deg),
        "num_eligible_clusters": int(len(clusters)),
        "max_cluster_size": int(max((row.get("n_members", 0) for row in clusters), default=0)),
        "candidates_all_jsonl": str(all_path),
        "candidates_masked_jsonl": str(eligible_path),
        "clusters_jsonl": str(cluster_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact map summary CSV."""

    columns = [
        "map",
        "num_tiles",
        "num_triggered_tiles",
        "trigger_fraction",
        "num_tiles_passing_mask_fraction",
        "num_triggered_eligible_tiles",
        "eligible_trigger_fraction",
        "num_eligible_clusters",
        "max_cluster_size",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a human-readable report."""

    lines = ["# Matched-Filter-Channel Tile Audit", ""]
    lines.append("This audits deployment burden for an existing two-channel checkpoint.")
    lines.append("It is not a promotion decision and not a retrain.")
    lines.append("")
    lines.append("## Threshold")
    lines.append("")
    threshold = report["threshold"]
    for key, value in threshold.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Diagnostic Context")
    lines.append("")
    global_rows = report.get("diagnostic_context", {}).get("global_model_only_rows", [])
    if global_rows:
        lines.append("| domain | recall | FPR | precision | expected FP full sky |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in global_rows:
            lines.append(
                f"| {row.get('domain')} | {float(row.get('recall', 0.0)):.4f} | "
                f"{float(row.get('fpr', 0.0)):.4f} | "
                f"{float(row.get('precision', 0.0)):.4f} | "
                f"{float(row.get('expected_fp_full_sky', 0.0)):.1f} |"
            )
    regime_rows = report.get("diagnostic_context", {}).get("real_smica_model_only_regime_rows", [])
    if regime_rows:
        lines.append("")
        lines.append("| real-SMICA regime | detected / n | recall |")
        lines.append("|---|---:|---:|")
        for row in regime_rows:
            lines.append(
                f"| {row.get('regime')} | {row.get('detected')} / {row.get('n')} | "
                f"{float(row.get('recall', 0.0)):.4f} |"
            )
    lines.append("")
    lines.append("## Map Burden")
    lines.append("")
    lines.append(
        "| map | triggered / tiles | trigger frac | eligible triggered / eligible | "
        "eligible frac | clusters | max cluster size |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in report["map_rows"]:
        lines.append(
            f"| {row['map']} | {row['num_triggered_tiles']} / {row['num_tiles']} | "
            f"{row['trigger_fraction']:.4f} | "
            f"{row['num_triggered_eligible_tiles']} / {row['num_tiles_passing_mask_fraction']} | "
            f"{row['eligible_trigger_fraction']:.4f} | "
            f"{row['num_eligible_clusters']} | {row['max_cluster_size']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Promotion requires comparing this burden against the current tile-constrained "
        "policy and held-out multi-map null controls. High recall alone is insufficient."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = p3.resolve_device(args.device)
    threshold = threshold_from_report(
        Path(args.recalibrated_report).expanduser().resolve(),
        str(args.model_name),
        float(args.fpr_target),
        str(args.threshold_domain),
    )
    diagnostic_context = diagnostic_context_from_report(
        Path(args.recalibrated_report).expanduser().resolve(),
        float(args.fpr_target),
    )
    model, run_config, input_config, checkpoint_path, checkpoint_label = build_model(
        Path(args.run_dir).expanduser().resolve(),
        str(args.checkpoint),
        device,
    )
    map_rows = []
    for map_name in args.maps:
        scores = score_map_tiles(args, map_name, model, input_config, device, output_dir)
        map_rows.append(
            summarize_map(
                map_name,
                scores,
                float(threshold["selected_threshold"]),
                mask_threshold=float(args.mask_threshold),
                cluster_radius_deg=float(args.cluster_radius_deg),
                output_dir=output_dir,
            )
        )
    report = {
        "model": {
            "name": str(args.model_name),
            "run_dir": str(Path(args.run_dir).expanduser().resolve()),
            "checkpoint_arg": str(args.checkpoint),
            "checkpoint_label": checkpoint_label,
            "checkpoint_path": str(checkpoint_path),
            "input_channels": int(input_config["input_channels"]),
            "extra_channel_datasets": list(input_config.get("extra_channel_datasets", [])),
        },
        "threshold": threshold,
        "diagnostic_context": diagnostic_context,
        "settings": {
            "maps": list(args.maps),
            "tile_nside": int(args.tile_nside),
            "mask_threshold": float(args.mask_threshold),
            "cluster_radius_deg": float(args.cluster_radius_deg),
            "feature_radii_deg": [float(x) for x in args.feature_radii_deg],
            "feature_beam_fwhm_arcmin": float(args.feature_beam_fwhm_arcmin),
            "score_limit": int(args.score_limit),
        },
        "assumption_notes": [
            "This audits the existing legacy v7 matched-filter-channel checkpoint as trained.",
            (
                "The second channel is reconstructed in memory and the tile HDF5 "
                "files are not modified."
            ),
            (
                "Full-sky overlapping tile burden is a deployment diagnostic, "
                "not an independent-pixel FPR."
            ),
        ],
        "map_rows": map_rows,
    }
    json_path = output_dir / "mf_channel_tile_audit.json"
    csv_path = output_dir / "mf_channel_tile_audit.csv"
    md_path = output_dir / "mf_channel_tile_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, map_rows)
    write_markdown(md_path, report)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
