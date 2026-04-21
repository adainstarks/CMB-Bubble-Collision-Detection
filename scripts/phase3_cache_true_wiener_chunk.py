"""Compute true-Wiener response patches for a selected row chunk.

Assumptions
-----------
* The source HDF5 is read-only during chunk generation. This script does not
  mutate the source file, which avoids HDF5 writer-lock contention when
  multiple chunk workers run in parallel.
* The output is a chunk HDF5 containing explicit `rows` and `patches`
  datasets. A separate stitch step writes those patches back into the source
  HDF5 once all chunk jobs complete.
* The Wiener response is the blind maximum over the sign-collapsed spherical
  Wiener/Feeney bank used elsewhere in the remediated Phase 3 benchmarking
  stack. The output is an auxiliary response channel, not a temperature map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from phase2_observing_model import camb_tt_cls
from phase3_cache_true_wiener_channel import (
    load_summary,
    parse_float_list,
    reconstruct_response_patch,
    resolve_camb_params,
    resolve_theta_grid,
)
from phase3_classical_filters import precompute_wiener_feeney_filter_bank
from phase3_same_grid_fullsky_benchmark import CmbMapCache
from phase_config import PATCH_PIX


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_H5 = PROJECT_ROOT / "data" / "remediated_v1" / "training_data.h5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "true_wiener_chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a true-Wiener response chunk into a standalone HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-h5", type=str, default=str(DEFAULT_SOURCE_H5))
    parser.add_argument("--output-h5", type=str, required=True)
    parser.add_argument("--rows-json", type=str, default="")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-stop", type=int, default=0, help="0 means end of dataset.")
    parser.add_argument("--theta-grid-deg", type=str, default="")
    parser.add_argument("--lmax", type=int, default=0, help="0 means 3*nside-1.")
    parser.add_argument("--quadrature-order", type=int, default=1024)
    parser.add_argument("--cmb-realization-cache", type=int, default=4)
    parser.add_argument("--include-surrogate-noise", action="store_true")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    return parser.parse_args()


def load_rows(source_h5: Path, path: Path | None, start: int, stop: int) -> np.ndarray:
    if path:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"] if isinstance(payload, dict) else payload
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0:
            raise ValueError(f"Row JSON {path} produced no valid rows.")
        return rows
    with h5py.File(source_h5, "r") as h5:
        total = int(h5["labels"].shape[0])
    row_stop = total if int(stop) == 0 else int(stop)
    if start < 0 or row_stop <= int(start):
        raise ValueError("Invalid row range.")
    return np.arange(int(start), int(row_stop), dtype=np.int64)


def main() -> None:
    args = parse_args()
    source_h5 = Path(args.source_h5).expanduser().resolve()
    output_h5 = Path(args.output_h5).expanduser().resolve()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    summary = load_summary(source_h5)
    camb_params = resolve_camb_params(source_h5, summary)
    seeds = list(camb_params.get("seeds", []))
    camb_meta = dict(camb_params.get("camb", {}))
    nside = int(summary.get("nside", 256))
    lmax = int(args.lmax) if int(args.lmax) else 3 * nside - 1
    theta_grid_deg = resolve_theta_grid(
        summary,
        parse_float_list(args.theta_grid_deg) if args.theta_grid_deg else tuple(),
    )
    beam_fwhm_arcmin = float(summary.get("beam_fwhm_arcmin", 5.0))
    noise_sigma_uk_arcmin = float(summary.get("noise_sigma_uk_arcmin", 0.0))
    pixel_window_policy = str(summary.get("pixel_window_policy", "synfast_pixwin_true"))
    injection_convention = str(summary.get("injection_convention", "feeney2011_full_temperature_modulation"))
    rows = load_rows(
        source_h5,
        Path(args.rows_json).expanduser().resolve() if args.rows_json else None,
        int(args.row_start),
        int(args.row_stop),
    )

    cls_tt, _ = camb_tt_cls(
        lmax=int(lmax),
        params=camb_meta.get("params"),
        lens_potential_accuracy=int(camb_meta.get("lens_potential_accuracy", 1)),
    )
    weights_bank, bank_meta = precompute_wiener_feeney_filter_bank(
        nside=int(nside),
        cmb_cl=cls_tt,
        theta_grid_deg=tuple(theta_grid_deg),
        lmax=int(lmax),
        beam_fwhm_arcmin=float(beam_fwhm_arcmin),
        noise_sigma_uk_arcmin=float(noise_sigma_uk_arcmin),
        pixel_window_policy=str(pixel_window_policy),
        quadrature_order=int(args.quadrature_order),
        collapse_sign_pairs=True,
    )
    cache = CmbMapCache(
        seeds,
        nside=int(nside),
        lmax=int(lmax),
        cls_tt=np.asarray(cls_tt, dtype=np.float64),
        max_items=int(args.cmb_realization_cache),
    )

    with h5py.File(source_h5, "r") as h5:
        meta = h5["metadata"]
        truth = h5["truth"]
        processing_rows = sorted(
            [int(row) for row in rows],
            key=lambda row: int(meta["cmb_realization_idx"][row]),
        )
        out_rows = np.asarray(rows, dtype=np.int64)
        row_to_slot = {int(row): idx for idx, row in enumerate(out_rows.tolist())}
        patches = np.empty((out_rows.size, PATCH_PIX, PATCH_PIX), dtype=np.dtype(args.dtype))

        for rows_done, row in enumerate(processing_rows, start=1):
            patch = reconstruct_response_patch(
                row=int(row),
                glon_deg=float(meta["glon_deg"][row]),
                glat_deg=float(meta["glat_deg"][row]),
                cmb_realization_idx=int(meta["cmb_realization_idx"][row]),
                has_signal=bool(truth["has_signal"][row]),
                z0=float(truth["z0"][row]),
                zcrit=float(truth["zcrit"][row]),
                theta_crit_deg=float(truth["theta_crit_deg"][row]),
                edge_sigma_deg=float(truth["edge_sigma_deg"][row]) if "edge_sigma_deg" in truth else 0.0,
                center_x_pix=float(truth["signal_center_x_pix"][row]),
                center_y_pix=float(truth["signal_center_y_pix"][row]),
                injection_convention=str(injection_convention),
                cache=cache,
                nside=int(nside),
                lmax=int(lmax),
                beam_fwhm_arcmin=float(beam_fwhm_arcmin),
                pixel_window_policy=str(pixel_window_policy),
                noise_sigma_uk_arcmin=float(noise_sigma_uk_arcmin),
                include_surrogate_noise=bool(args.include_surrogate_noise),
                weights_bank=weights_bank,
                bank_meta=bank_meta,
            )
            patches[row_to_slot[int(row)]] = np.asarray(patch, dtype=np.dtype(args.dtype))
            if rows_done % 25 == 0 or rows_done == len(processing_rows):
                print(f"  cached {rows_done} / {len(processing_rows)} rows into {output_h5.name}", flush=True)

    with h5py.File(output_h5, "w") as h5:
        h5.create_dataset("rows", data=out_rows, compression="gzip", shuffle=True)
        h5.create_dataset("patches", data=patches, compression="lzf", shuffle=True)
        h5.attrs["source_h5"] = str(source_h5)
        h5.attrs["num_rows"] = int(out_rows.size)

    print(json.dumps({"output_h5": str(output_h5), "num_rows": int(out_rows.size)}, indent=2))


if __name__ == "__main__":
    main()
