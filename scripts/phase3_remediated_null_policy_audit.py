"""Audit remediated policy-Pareto candidates on real-map null controls.

Assumptions
-----------
* The null-control HDF5 files are signal-free cleaned-map patches sampled with
  the remediated mask and split protocol.
* Scores are the same patch-level quantities used by the remediated sensitivity
  and policy-Pareto diagnostics: U-Net maximum segmentation probability and the
  circular-template screen.
* The test split estimates held-out real-map false-positive burden. It is not a
  half-mission noise randomization and not a final discovery p-value.
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
from scipy.stats import beta

import phase3_train_unet as p3
from phase3_remediated_policy_tile_audit import (
    CIRCULAR_METHOD,
    DEFAULT_MODELS,
    DEFAULT_POLICY_JSON,
    DEFAULT_THETA_GRID_DEG,
    ML_METHODS,
    ModelSpec,
    apply_policy,
    circular_kernels,
    parse_model_spec,
    standardize_patch_batch,
)
from phase3_score_null_controls import load_null_split_indices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_null_policy_audit"
)
DEFAULT_MAPS = "smica,nilc,sevem,commander"
DEFAULT_MASK_TAG = "mask090"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Audit remediated composite policies on real null-control splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy-json", type=str, default=str(DEFAULT_POLICY_JSON))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--maps", type=str, default=DEFAULT_MAPS)
    parser.add_argument("--mask-tag", type=str, default=DEFAULT_MASK_TAG, choices=("mask090", "mask050"))
    parser.add_argument("--split", type=str, default="test", choices=("calibration", "test", "val", "all"))
    parser.add_argument("--model", action="append", default=[], help="name:run_dir:checkpoint")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--circular-batch-size", type=int, default=64)
    parser.add_argument("--circular-kernel-chunk", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--reuse-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments and attach parsed fields."""

    args.maps = tuple(item.strip().lower() for item in str(args.maps).split(",") if item.strip())
    if not args.maps:
        raise ValueError("--maps must contain at least one map name.")
    if args.batch_size <= 0 or args.circular_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.circular_kernel_chunk <= 0:
        raise ValueError("--circular-kernel-chunk must be positive.")
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must lie in (0, 1).")
    args.models = tuple(parse_model_spec(text) for text in (args.model or DEFAULT_MODELS))


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def binomial_ci(k: int, n: int, confidence: float) -> tuple[float, float]:
    """Exact Clopper-Pearson interval."""

    if n <= 0:
        raise ValueError("Binomial interval requires n > 0.")
    alpha = 1.0 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def null_h5_path(map_name: str, mask_tag: str) -> Path:
    """Resolve null-control HDF5 path."""

    path = PROJECT_ROOT / "data" / "remediated_v1" / f"null_controls_{map_name}_{mask_tag}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Missing null-control HDF5: {path}")
    return path


def load_rank1_policies(path: Path) -> list[dict[str, Any]]:
    """Load rank-1 policy rows from the policy-Pareto report."""

    rows = [row for row in load_json(path).get("top_rows", []) if int(row.get("rank", -1)) == 1]
    if not rows:
        raise ValueError(f"No rank-1 policy rows found in {path}.")
    return rows


def score_ml_indices(
    spec: ModelSpec,
    h5_path: Path,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Score selected HDF5 rows with one remediated U-Net."""

    from phase3_sensitivity_curve import build_model_from_run

    model, run_config, _checkpoint_path, _checkpoint_label = build_model_from_run(
        spec.run_dir.resolve(),
        spec.checkpoint,
        device,
    )
    dataset = p3.H5BubbleDataset(
        h5_path=str(h5_path),
        indices=np.asarray(indices, dtype=np.int64),
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 20260421,
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
    scores = np.zeros(len(indices), dtype=np.float32)
    offset = 0
    model.eval()
    progress = p3.ProgressPrinter(len(loader), f"Null ML scores {spec.name}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            mask_logits, _aux_logits = p3.unpack_model_output(model(images))
            score = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values
            batch_n = int(images.shape[0])
            scores[offset : offset + batch_n] = score.detach().cpu().numpy()
            offset += batch_n
            progress.update(batch_idx)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite ML scores for {spec.name} on {h5_path}.")
    return scores


def score_circular_batch_torch(
    patch_batch: np.ndarray,
    kernel_fft: torch.Tensor,
    *,
    kernel_count: int,
    kernel_chunk: int,
    device: torch.device,
) -> np.ndarray:
    """Score one patch batch against all circular kernels with CUDA FFTs."""

    patch_pix = int(patch_batch.shape[-1])
    full_shape = (2 * patch_pix - 1, 2 * patch_pix - 1)
    crop = (patch_pix - 1) // 2
    batch_tensor = torch.zeros(
        (patch_batch.shape[0], full_shape[0], full_shape[1]),
        dtype=torch.float32,
        device=device,
    )
    batch_tensor[:, :patch_pix, :patch_pix] = torch.as_tensor(
        np.ascontiguousarray(patch_batch),
        dtype=torch.float32,
        device=device,
    )
    batch_fft = torch.fft.rfft2(batch_tensor, s=full_shape)
    best = torch.full((patch_batch.shape[0],), -torch.inf, dtype=torch.float32, device=device)
    for k0 in range(0, kernel_count, int(kernel_chunk)):
        k1 = min(k0 + int(kernel_chunk), kernel_count)
        conv = torch.fft.irfft2(
            batch_fft[:, None, :, :] * kernel_fft[None, k0:k1, :, :],
            s=full_shape,
        )
        same = conv[:, :, crop : crop + patch_pix, crop : crop + patch_pix]
        best = torch.maximum(best, torch.amax(same, dim=(1, 2, 3)))
        del conv, same
    out = best.detach().cpu().numpy().astype(np.float32)
    del batch_tensor, batch_fft, best
    return out


def prepare_kernel_fft(device: torch.device, beam_fwhm_arcmin: float = 5.0) -> torch.Tensor:
    """Prepare circular kernel FFTs for CUDA scoring."""

    kernels = circular_kernels(DEFAULT_THETA_GRID_DEG, beam_fwhm_arcmin)[:, ::-1, ::-1]
    patch_pix = int(kernels.shape[-1])
    full_shape = (2 * patch_pix - 1, 2 * patch_pix - 1)
    kernel_tensor = torch.zeros(
        (kernels.shape[0], full_shape[0], full_shape[1]),
        dtype=torch.float32,
        device=device,
    )
    kernel_tensor[:, :patch_pix, :patch_pix] = torch.as_tensor(
        np.ascontiguousarray(kernels),
        dtype=torch.float32,
        device=device,
    )
    return torch.fft.rfft2(kernel_tensor, s=full_shape)


def score_circular_indices(
    h5_path: Path,
    indices: np.ndarray,
    *,
    batch_size: int,
    kernel_chunk: int,
    device: torch.device,
) -> np.ndarray:
    """Score selected null-control rows with the circular-template screen."""

    if device.type != "cuda":
        raise ValueError("This null policy audit currently requires CUDA for circular scoring.")
    kernel_fft = prepare_kernel_fft(device)
    kernel_count = int(kernel_fft.shape[0])
    scores = np.zeros(len(indices), dtype=np.float32)
    progress = p3.ProgressPrinter((len(indices) + batch_size - 1) // batch_size, "Null circular scores")
    batch_idx = 0
    with h5py.File(h5_path, "r") as h5:
        patches = h5["patches"]
        for start in range(0, len(indices), int(batch_size)):
            stop = min(start + int(batch_size), len(indices))
            batch_indices = np.asarray(indices[start:stop], dtype=np.int64)
            patch_batch = standardize_patch_batch(np.asarray(patches[batch_indices], dtype=np.float32))
            scores[start:stop] = score_circular_batch_torch(
                patch_batch,
                kernel_fft,
                kernel_count=kernel_count,
                kernel_chunk=kernel_chunk,
                device=device,
            )
            batch_idx += 1
            progress.update(batch_idx)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite circular scores for {h5_path}.")
    return scores


def load_or_score_null(
    args: argparse.Namespace,
    map_name: str,
    output_dir: Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Load or build per-sample score cache for one null-control split."""

    h5_path = null_h5_path(map_name, args.mask_tag)
    indices = np.sort(load_null_split_indices(str(h5_path), args.split))
    cache_path = output_dir / f"null_policy_scores_{map_name}_{args.mask_tag}_{args.split}.npz"
    if args.reuse_scores and cache_path.exists():
        with np.load(cache_path) as loaded:
            return {key.removeprefix("score__"): np.asarray(loaded[key]) for key in loaded.files if key.startswith("score__")} | {
                "sample_indices": np.asarray(loaded["sample_indices"], dtype=np.int64)
            }

    payload: dict[str, np.ndarray] = {"sample_indices": indices}
    for spec in args.models:
        payload[f"score__{spec.name}"] = score_ml_indices(
            spec,
            h5_path,
            indices,
            batch_size=int(args.batch_size),
            device=device,
        )
    payload[f"score__{CIRCULAR_METHOD}"] = score_circular_indices(
        h5_path,
        indices,
        batch_size=int(args.circular_batch_size),
        kernel_chunk=int(args.circular_kernel_chunk),
        device=device,
    )
    np.savez_compressed(cache_path, **payload)
    return {key.removeprefix("score__"): value for key, value in payload.items() if key.startswith("score__")} | {
        "sample_indices": indices
    }


def summarize_policy(
    row: dict[str, Any],
    scores: dict[str, np.ndarray],
    *,
    map_name: str,
    confidence: float,
) -> dict[str, Any]:
    """Compute false-positive count/rate for a policy on one null set."""

    mask = apply_policy(row, scores)
    n = int(mask.shape[0])
    fp = int(np.count_nonzero(mask))
    lo, hi = binomial_ci(fp, n, confidence)
    return {
        "map": map_name,
        "policy": row["policy"],
        "family": row["family"],
        "constraint_camb_fpr_max": float(row["constraint_camb_fpr_max"]),
        "constraint_real_fpr_max": float(row["constraint_real_fpr_max"]),
        "diagnostic_real_recall": float(row["real_recall"]),
        "diagnostic_real_fpr_200": float(row["real_fpr"]),
        "num_null": n,
        "false_positive_count": fp,
        "false_positive_rate": float(fp / max(n, 1)),
        "fpr_ci_low": lo,
        "fpr_ci_high": hi,
    }


def pooled_rows(rows: list[dict[str, Any]], policies: list[dict[str, Any]], confidence: float) -> list[dict[str, Any]]:
    """Aggregate per-map rows into pooled rows by policy."""

    out = []
    for policy in policies:
        matching = [
            row
            for row in rows
            if row["constraint_camb_fpr_max"] == float(policy["constraint_camb_fpr_max"])
            and row["constraint_real_fpr_max"] == float(policy["constraint_real_fpr_max"])
        ]
        n = int(sum(row["num_null"] for row in matching))
        fp = int(sum(row["false_positive_count"] for row in matching))
        lo, hi = binomial_ci(fp, n, confidence)
        out.append(
            {
                "map": "pooled",
                "policy": policy["policy"],
                "family": policy["family"],
                "constraint_camb_fpr_max": float(policy["constraint_camb_fpr_max"]),
                "constraint_real_fpr_max": float(policy["constraint_real_fpr_max"]),
                "diagnostic_real_recall": float(policy["real_recall"]),
                "diagnostic_real_fpr_200": float(policy["real_fpr"]),
                "num_null": n,
                "false_positive_count": fp,
                "false_positive_rate": float(fp / max(n, 1)),
                "fpr_ci_low": lo,
                "fpr_ci_high": hi,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write summary CSV."""

    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write Markdown report."""

    lines = ["# Remediated v1 Null-Control Policy Audit", ""]
    lines.append("Held-out real-map null controls provide a larger false-positive stress test than the 200-negative injection diagnostic.")
    lines.append("")
    lines.append("## Pooled Test")
    lines.append("")
    lines.append("| policy budget | recall diag | FPR on 200 diag | null FP | null FPR | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report["pooled_rows"]:
        lines.append(
            f"| CAMB<={row['constraint_camb_fpr_max']:.2f}, real<={row['constraint_real_fpr_max']:.2f} | "
            f"{row['diagnostic_real_recall']:.4f} | {row['diagnostic_real_fpr_200']:.4f} | "
            f"{row['false_positive_count']} / {row['num_null']} | {row['false_positive_rate']:.4f} | "
            f"[{row['fpr_ci_low']:.4f}, {row['fpr_ci_high']:.4f}] |"
        )
    lines.append("")
    lines.append("## By Map")
    lines.append("")
    lines.append("| map | policy budget | null FP | null FPR | 95% CI |")
    lines.append("|---|---|---:|---:|---:|")
    for row in report["rows"]:
        lines.append(
            f"| {row['map']} | CAMB<={row['constraint_camb_fpr_max']:.2f}, "
            f"real<={row['constraint_real_fpr_max']:.2f} | "
            f"{row['false_positive_count']} / {row['num_null']} | "
            f"{row['false_positive_rate']:.4f} | "
            f"[{row['fpr_ci_low']:.4f}, {row['fpr_ci_high']:.4f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = p3.resolve_device(args.device)
    policies = load_rank1_policies(Path(args.policy_json).resolve())
    rows = []
    for map_name in args.maps:
        scores = load_or_score_null(args, map_name, output_dir, device)
        for policy in policies:
            rows.append(
                summarize_policy(
                    policy,
                    scores,
                    map_name=map_name,
                    confidence=float(args.confidence),
                )
            )
    pooled = pooled_rows(rows, policies, float(args.confidence))
    report = {
        "policy_json": str(Path(args.policy_json).resolve()),
        "split": args.split,
        "mask_tag": args.mask_tag,
        "maps": list(args.maps),
        "rows": rows,
        "pooled_rows": pooled,
    }
    json_path = output_dir / "null_policy_audit.json"
    csv_path = output_dir / "null_policy_audit.csv"
    md_path = output_dir / "null_policy_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, rows + pooled)
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
