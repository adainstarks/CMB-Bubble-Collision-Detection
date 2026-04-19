"""
Cache a circular-template response map as an HDF5 input feature channel.

For each patch, this computes the per-pixel maximum correlation response over a
coarse Feeney-template bank. The result is intended as a second U-Net input
channel that injects circularity/profile prior information without replacing
the learned segmentation objective.

This filename is retained as a legacy entry point. The cached feature is not a
Wiener or CMB/noise-whitened matched filter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve

import phase3_train_unet as p3
from phase_config import DEFAULTS
from phase3_method_registry import CIRCULAR_TEMPLATE_SCREEN, method_metadata
from phase3_sensitivity_curve import SIGN_QUADRANTS, make_feeney_template_kernel, standardize_patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    PROJECT_ROOT / "data" / "training_v4" / "training_data.h5",
    PROJECT_ROOT / "runs" / "phase3_unet" / "sensitivity_curve_v1" / "sensitivity_data.h5",
    PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_injection_v1" / "smica_real_sky_injection.h5",
    PROJECT_ROOT / "data" / "training_v4" / "smica_null_controls_all.h5",
)
DEFAULT_RADII = (5.0, 8.0, 12.0, 16.0, 20.0, 25.0)


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache circular-template response maps into HDF5 feature datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5", action="append", default=[], help="HDF5 file to update. Can be repeated.")
    parser.add_argument("--dataset", type=str, default="features/circular_template_response")
    parser.add_argument("--radii-deg", type=str, default=",".join(f"{x:g}" for x in DEFAULT_RADII))
    parser.add_argument("--beam-fwhm-arcmin", type=float, default=DEFAULTS.beam_fwhm_arcmin)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compression", type=str, default="lzf", choices=["none", "lzf", "gzip"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.radii_deg = parse_float_list(args.radii_deg)
    if not args.radii_deg:
        raise ValueError("--radii-deg must contain at least one radius.")
    if any(radius <= 0.0 for radius in args.radii_deg):
        raise ValueError("--radii-deg values must be positive.")
    if args.beam_fwhm_arcmin < 0.0:
        raise ValueError("--beam-fwhm-arcmin must be non-negative.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    args.h5_paths = [Path(path).resolve() for path in (args.h5 or DEFAULT_DATASETS)]
    missing = [str(path) for path in args.h5_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing HDF5 input(s): {missing}")


def make_kernels(radii_deg: tuple[float, ...], beam_fwhm_arcmin: float) -> list[np.ndarray]:
    return [
        make_feeney_template_kernel(radius, z0_sign, zcrit_sign, beam_fwhm_arcmin=beam_fwhm_arcmin)
        for radius in radii_deg
        for z0_sign, zcrit_sign in SIGN_QUADRANTS
    ]


def response_map_for_patch(patch: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    patch = standardize_patch(np.asarray(patch, dtype=np.float32))
    best = None
    for kernel in kernels:
        response = fftconvolve(patch, kernel[::-1, ::-1], mode="same").astype(np.float32, copy=False)
        if best is None:
            best = response
        else:
            np.maximum(best, response, out=best)
    return best.astype(np.float32, copy=False)


def compression_kwargs(name: str) -> dict:
    if name == "none":
        return {}
    if name == "gzip":
        return {"compression": "gzip", "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


def ensure_parent_group(h5: h5py.File, dataset_path: str) -> None:
    parent = str(Path(dataset_path).parent).strip(".")
    if parent and parent != "/":
        h5.require_group(parent)


def cache_one(path: Path, args: argparse.Namespace, kernels: list[np.ndarray]) -> dict:
    dataset_path = args.dataset.strip("/")
    with h5py.File(path, "r+") as h5:
        patches = h5["patches"]
        n, height, width = patches.shape
        if dataset_path in h5 and not args.overwrite:
            existing = h5[dataset_path]
            if existing.shape != patches.shape:
                raise RuntimeError(f"{path}:{dataset_path} exists with wrong shape {existing.shape}; expected {patches.shape}.")
            print(f"Reusing existing feature channel: {path}:{dataset_path}")
            return {
                "h5": str(path),
                "dataset": dataset_path,
                "status": "reused",
                "num_samples": int(n),
                "shape": list(existing.shape),
                "dtype": str(existing.dtype),
            }
        if args.dry_run:
            return {
                "h5": str(path),
                "dataset": dataset_path,
                "status": "dry_run",
                "num_samples": int(n),
                "shape": [int(n), int(height), int(width)],
                "dtype": args.dtype,
            }
        if dataset_path in h5:
            del h5[dataset_path]
        ensure_parent_group(h5, dataset_path)
        chunks = (min(args.chunk_size, n), height, width)
        out = h5.create_dataset(
            dataset_path,
            shape=patches.shape,
            dtype=np.dtype(args.dtype),
            chunks=chunks,
            **compression_kwargs(args.compression),
        )
        out.attrs["created_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        out.attrs["description"] = "Per-pixel max Feeney circular-template response over radius and sign bank."
        out.attrs["method_metadata"] = json.dumps(method_metadata(CIRCULAR_TEMPLATE_SCREEN), sort_keys=True)
        out.attrs["radii_deg"] = json.dumps([float(x) for x in args.radii_deg])
        out.attrs["beam_fwhm_arcmin"] = float(args.beam_fwhm_arcmin)
        out.attrs["sign_quadrants"] = json.dumps([[float(a), float(b)] for a, b in SIGN_QUADRANTS])
        out.attrs["patch_standardization"] = "per-patch mean/std before convolution"

        progress = p3.ProgressPrinter(n, f"Cache circular channel {path.name}")
        completed = 0
        for start in range(0, n, args.chunk_size):
            stop = min(start + args.chunk_size, n)
            batch = np.asarray(patches[start:stop], dtype=np.float32)
            responses = np.empty((stop - start, height, width), dtype=np.float32)

            def score_one(local_idx: int) -> tuple[int, np.ndarray]:
                return local_idx, response_map_for_patch(batch[local_idx], kernels)

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(score_one, idx) for idx in range(stop - start)]
                for future in as_completed(futures):
                    local_idx, response = future.result()
                    responses[local_idx] = response
                    completed += 1
                    if completed % 250 == 0 or completed == n:
                        progress.update(completed)
            out[start:stop] = responses.astype(np.dtype(args.dtype), copy=False)
            h5.flush()

    return {
        "h5": str(path),
        "dataset": dataset_path,
        "status": "created",
        "num_samples": int(n),
        "shape": [int(n), int(height), int(width)],
        "dtype": args.dtype,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    kernels = make_kernels(args.radii_deg, args.beam_fwhm_arcmin)
    rows = []
    for path in args.h5_paths:
        rows.append(cache_one(path, args, kernels))
    print(json.dumps({"dataset": args.dataset, "radii_deg": args.radii_deg, "beam_fwhm_arcmin": args.beam_fwhm_arcmin, "outputs": rows}, indent=2))


if __name__ == "__main__":
    main()
