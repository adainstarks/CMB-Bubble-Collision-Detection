"""
Extract negative-only Planck null-control patches with remediated split metadata.

Assumptions
-----------
* Null-control products are map-specific calibration/evaluation controls, not
  independent cosmological skies.
* Calibration and test/evaluation rows must be separated by coordinate cluster
  rather than random patch row.
* Canonical science controls use the Planck mask threshold 0.9; threshold 0.5
  products are deployment stress-test artifacts and are named separately.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import healpy as hp
import h5py
import numpy as np

from phase1_explore import DATA_DIR, download_file
from phase_config import CANONICAL_MASK_THRESHOLD
from phase2_generate_training import (
    NSIDE_WORKING,
    PATCH_PIX,
    RESO_ARCMIN,
    coordinate_cluster_ids,
    coordinate_cluster_pixels,
    load_mask,
    project_patch,
)
from phase2_observing_model import remove_real_map_low_modes
from phase_dataset_utils import stable_group_id


PLANCK_CMB_MAP_BASE_URL = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb"
PLANCK_CLEANED_MAPS = {
    "smica": {
        "filename": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "product": "Planck 2018 SMICA component-separated CMB map",
    },
    "nilc": {
        "filename": "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
        "product": "Planck 2018 NILC component-separated CMB map",
    },
    "sevem": {
        "filename": "COM_CMB_IQU-sevem_2048_R3.00_full.fits",
        "product": "Planck 2018 SEVEM component-separated CMB map",
    },
    "commander": {
        "filename": "COM_CMB_IQU-commander_2048_R3.00_full.fits",
        "product": "Planck 2018 Commander component-separated CMB map",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract real-map null-control patches using a generated dataset's coordinate pool and split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training-h5", type=str, required=True)
    parser.add_argument("--output-h5", type=str, default="")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "calibration", "val", "test"])
    parser.add_argument(
        "--map-name",
        type=str,
        default="smica",
        choices=sorted(PLANCK_CLEANED_MAPS),
        help="Planck 2018 component-separated cleaned CMB map to extract null controls from.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for smoke tests. Use 0 to extract every selected coordinate.",
    )
    parser.add_argument("--mask-threshold", type=float, default=CANONICAL_MASK_THRESHOLD)
    parser.add_argument("--chunk-size", type=int, default=8)
    return parser.parse_args()


def map_product(map_name):
    info = PLANCK_CLEANED_MAPS[map_name]
    filename = info["filename"]
    return {
        **info,
        "url": f"{PLANCK_CMB_MAP_BASE_URL}/{filename}",
        "path": os.path.join(DATA_DIR, filename),
    }


def ensure_map_input(map_name):
    os.makedirs(DATA_DIR, exist_ok=True)
    product = map_product(map_name)
    download_file(product["url"], product["path"])
    return product


def mask_tag(mask_threshold):
    """Return a stable mask-threshold tag such as mask090 or mask050."""

    return f"mask{int(round(float(mask_threshold) * 100)):03d}"


def default_output_path(training_h5, map_name, mask_threshold):
    training_h5 = os.path.abspath(training_h5)
    return os.path.join(os.path.dirname(training_h5), f"null_controls_{map_name}_{mask_tag(mask_threshold)}.h5")


def main():
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if not (0.0 < args.mask_threshold <= 1.0):
        raise ValueError("--mask-threshold must lie in (0, 1].")

    training_h5 = os.path.abspath(args.training_h5)
    if not os.path.exists(training_h5):
        raise FileNotFoundError(f"Training dataset not found: {training_h5}")

    output_h5 = (
        os.path.abspath(args.output_h5)
        if args.output_h5
        else default_output_path(training_h5, args.map_name, args.mask_threshold)
    )
    product = ensure_map_input(args.map_name)

    print(f"Loading {args.map_name.upper()} map...")
    cleaned_map = hp.read_map(product["path"], field=0)
    cleaned_map_256 = hp.ud_grade(cleaned_map, NSIDE_WORKING)
    mask_256, sky_fraction = load_mask(threshold=float(args.mask_threshold))
    cleaned_map_256 = remove_real_map_low_modes(cleaned_map_256, mask=mask_256)
    cleaned_map_256[np.asarray(mask_256) <= 0] = 0.0

    print("Loading coordinate pool and split metadata...")
    with h5py.File(training_h5, "r") as src:
        coord_pool = src["coordinate_pool"]
        glon_pool = np.asarray(coord_pool["glon_deg"][:], dtype=np.float32)
        glat_pool = np.asarray(coord_pool["glat_deg"][:], dtype=np.float32)
        mask_fraction_pool = np.asarray(coord_pool["mask_fraction"][:], dtype=np.float32)
        summary_attrs = dict(src["summary"].attrs.items())
        if "coord_cluster_id" in coord_pool:
            coord_cluster_id_pool = np.asarray(coord_pool["coord_cluster_id"][:], dtype=np.uint64)
        else:
            coord_cluster_nside = int(summary_attrs.get("coord_cluster_nside", 4))
            coords = np.column_stack((glon_pool, glat_pool))
            coord_cluster_pix_pool = coordinate_cluster_pixels(coords, coord_cluster_nside).astype(np.int32)
            coord_cluster_id_pool = coordinate_cluster_ids(coord_cluster_pix_pool, coord_cluster_nside)
        splits = src["splits"]
        coord_train_idx = np.asarray(splits["coord_train_idx"][:], dtype=np.int32)
        if "coord_calibration_idx" in splits:
            coord_calibration_idx = np.asarray(splits["coord_calibration_idx"][:], dtype=np.int32)
        else:
            coord_calibration_idx = np.asarray(splits["coord_val_idx"][:], dtype=np.int32)
        coord_val_idx = coord_calibration_idx
        coord_test_idx = (
            np.asarray(splits["coord_test_idx"][:], dtype=np.int32)
            if "coord_test_idx" in splits
            else np.asarray([], dtype=np.int32)
        )

    split_sources = []
    if args.split in {"all", "train"}:
        split_sources.append((0, coord_train_idx))
    if args.split in {"all", "calibration", "val"}:
        split_sources.append((1, coord_calibration_idx))
    if args.split in {"all", "test"} and coord_test_idx.size:
        split_sources.append((2, coord_test_idx))

    sample_plan = []
    for split_tag, coord_indices in split_sources:
        for coord_idx in coord_indices:
            sample_plan.append((int(split_tag), int(coord_idx)))
    if args.max_samples:
        sample_plan = sample_plan[: int(args.max_samples)]

    num_samples = len(sample_plan)
    if num_samples == 0:
        raise RuntimeError("No null-control samples selected.")

    metadata = {
        "sample_index": np.arange(num_samples, dtype=np.int32),
        "split_tag": np.zeros(num_samples, dtype=np.uint8),
        "glon_deg": np.empty(num_samples, dtype=np.float32),
        "glat_deg": np.empty(num_samples, dtype=np.float32),
        "coord_pool_idx": np.empty(num_samples, dtype=np.int32),
        "coord_mask_fraction": np.empty(num_samples, dtype=np.float32),
        "cmb_realization_idx": np.full(num_samples, -1, dtype=np.int32),
        "background_id": np.zeros(num_samples, dtype=np.uint64),
        "split_group_id": np.zeros(num_samples, dtype=np.uint64),
        "coord_cluster_id": np.zeros(num_samples, dtype=np.uint64),
    }
    truth = {
        "has_signal": np.zeros(num_samples, dtype=np.uint8),
        "event_id": np.zeros(num_samples, dtype=np.uint64),
        "theta_crit_deg": np.zeros(num_samples, dtype=np.float32),
        "z0": np.zeros(num_samples, dtype=np.float32),
        "zcrit": np.zeros(num_samples, dtype=np.float32),
        "edge_sigma_deg": np.zeros(num_samples, dtype=np.float32),
        "signal_center_x_pix": np.full(num_samples, (PATCH_PIX - 1) / 2.0, dtype=np.float32),
        "signal_center_y_pix": np.full(num_samples, (PATCH_PIX - 1) / 2.0, dtype=np.float32),
        "signal_center_dx_deg": np.zeros(num_samples, dtype=np.float32),
        "signal_center_dy_deg": np.zeros(num_samples, dtype=np.float32),
        "geometry_mode_code": np.zeros(num_samples, dtype=np.uint8),
        "fully_contained": np.zeros(num_samples, dtype=np.uint8),
        "target_touches_edge": np.zeros(num_samples, dtype=np.uint8),
    }

    with h5py.File(output_h5, "w") as dst:
        patch_chunk = (min(int(args.chunk_size), num_samples), PATCH_PIX, PATCH_PIX)
        patches = dst.create_dataset(
            "patches",
            shape=(num_samples, PATCH_PIX, PATCH_PIX),
            dtype=np.float32,
            chunks=patch_chunk,
            compression="gzip",
            shuffle=True,
        )
        dst.create_dataset(
            "labels",
            shape=(num_samples,),
            dtype=np.uint8,
            chunks=(min(max(int(args.chunk_size), 1), num_samples),),
            compression="gzip",
            shuffle=True,
            fillvalue=0,
        )
        dst.create_dataset(
            "masks",
            shape=(num_samples, PATCH_PIX, PATCH_PIX),
            dtype=np.uint8,
            chunks=patch_chunk,
            compression="gzip",
            shuffle=True,
            fillvalue=0,
        )

        print(f"Extracting {num_samples} {args.map_name.upper()} null-control patches...", flush=True)
        for sample_idx, (split_tag, coord_idx) in enumerate(sample_plan):
            lon = float(glon_pool[coord_idx])
            lat = float(glat_pool[coord_idx])
            patch = np.asarray(project_patch(cleaned_map_256, lon, lat), dtype=np.float32)
            background_id = stable_group_id(args.map_name, coord_idx)
            coord_cluster_id = coord_cluster_id_pool[coord_idx]

            patches[sample_idx] = patch
            metadata["split_tag"][sample_idx] = split_tag
            metadata["glon_deg"][sample_idx] = lon
            metadata["glat_deg"][sample_idx] = lat
            metadata["coord_pool_idx"][sample_idx] = coord_idx
            metadata["coord_mask_fraction"][sample_idx] = float(mask_fraction_pool[coord_idx])
            metadata["background_id"][sample_idx] = np.uint64(background_id)
            metadata["split_group_id"][sample_idx] = np.uint64(background_id)
            metadata["coord_cluster_id"][sample_idx] = np.uint64(coord_cluster_id)

            if (sample_idx + 1) % 100 == 0 or sample_idx + 1 == num_samples:
                dst.flush()
                print(f"  Extracted {sample_idx + 1:5d} / {num_samples}", flush=True)

        metadata_group = dst.create_group("metadata")
        for key, value in metadata.items():
            metadata_group.create_dataset(key, data=value, compression="gzip", shuffle=True)

        truth_group = dst.create_group("truth")
        for key, value in truth.items():
            truth_group.create_dataset(key, data=value, compression="gzip", shuffle=True)

        splits_group = dst.create_group("splits")
        train_idx = np.flatnonzero(metadata["split_tag"] == 0).astype(np.int64)
        calibration_idx = np.flatnonzero(metadata["split_tag"] == 1).astype(np.int64)
        test_idx = np.flatnonzero(metadata["split_tag"] == 2).astype(np.int64)
        val_idx = calibration_idx
        splits_group.create_dataset("train_idx", data=train_idx, compression="gzip", shuffle=True)
        splits_group.create_dataset("calibration_idx", data=calibration_idx, compression="gzip", shuffle=True)
        splits_group.create_dataset("test_idx", data=test_idx, compression="gzip", shuffle=True)
        splits_group.create_dataset("val_idx", data=val_idx, compression="gzip", shuffle=True)
        splits_group.create_dataset("coord_train_idx", data=coord_train_idx, compression="gzip", shuffle=True)
        splits_group.create_dataset("coord_calibration_idx", data=coord_calibration_idx, compression="gzip", shuffle=True)
        splits_group.create_dataset("coord_test_idx", data=coord_test_idx, compression="gzip", shuffle=True)
        splits_group.create_dataset("coord_val_idx", data=coord_val_idx, compression="gzip", shuffle=True)

        summary_group = dst.create_group("summary")
        payload = {
            "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "source_training_h5": training_h5,
            "source_training_created_utc": summary_attrs.get("created_utc", ""),
            "num_samples": int(num_samples),
            "num_train_samples": int(len(train_idx)),
            "num_calibration_samples": int(len(calibration_idx)),
            "num_test_samples": int(len(test_idx)),
            "num_val_samples": int(len(val_idx)),
            "selected_split": args.split,
            "max_samples": int(args.max_samples),
            "nside": int(NSIDE_WORKING),
            "patch_pixels": int(PATCH_PIX),
            "reso_arcmin": float(RESO_ARCMIN),
            "mask_threshold": float(args.mask_threshold),
            "mask_tag": mask_tag(args.mask_threshold),
            "sky_fraction": float(sky_fraction),
            "source_map": os.path.basename(product["path"]),
            "source_map_url": product["url"],
            "source_map_product": product["product"],
            "null_control_kind": f"real_{args.map_name}_cleaned_map",
            "split_method": "coordinate_cluster_disjoint_calibration_test",
            "coord_cluster_nside": int(summary_attrs.get("coord_cluster_nside", 4)),
            "num_coordinate_clusters": int(len(np.unique(coord_cluster_id_pool[np.asarray([idx for _, idx in sample_plan])]))),
            "low_mode_policy": "remove_monopole_dipole_with_mask",
        }
        for key, value in payload.items():
            summary_group.attrs[key] = value

    summary_path = os.path.splitext(output_h5)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved null controls: {output_h5}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
