"""Fit Feeney templates to screened real-map candidates and emit handoff rows.

Assumptions
-----------
* Input candidates are screening outputs from the remediated full-sky tile
  policy. They are not detections and are not posterior samples.
* Candidate patches live in the Batch 6 tile HDF5s indexed by `map` and
  `patch_index`.
* The fit statistic is the local template-vs-plane delta-chi2 from
  ``phase3_template_fit_candidates.py``. It is a deterministic handoff metric
  for downstream Bayesian/model-comparison tooling, not Bayesian evidence by
  itself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import h5py

from phase_dataset_utils import pixel_to_patch_offsets_deg
from phase3_template_fit_candidates import fit_one_candidate, summarize, validate_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_tile_constrained_candidates"
    / "cluster_representatives_15deg.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_template_fit_handoff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Feeney templates to screened candidate JSONL rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-jsonl", type=str, default=str(DEFAULT_CANDIDATE_JSONL))
    parser.add_argument("--tile-h5-template", type=str, default=str(
        PROJECT_ROOT / "runs" / "phase3_unet" / "batch6_fullsky_nside32_{map}" / "tile_patches_{map}_nside32.h5"
    ))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-jsonl", type=str, default="")
    parser.add_argument("--output-summary", type=str, default="")
    parser.add_argument("--radius-window-deg", type=float, default=3.0)
    parser.add_argument("--radius-step-deg", type=float, default=0.5)
    parser.add_argument("--min-radius-deg", type=float, default=5.0)
    parser.add_argument("--max-radius-deg", type=float, default=25.0)
    parser.add_argument("--support-extra-deg", type=float, default=5.0)
    parser.add_argument("--support-factor", type=float, default=1.5)
    parser.add_argument("--edge-sigma-deg", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def adapt_record(record: dict) -> dict:
    peak_i = int(record["peak_pixel_i"])
    peak_j = int(record["peak_pixel_j"])
    dx_deg, dy_deg = pixel_to_patch_offsets_deg(float(peak_j), float(peak_i))
    return {
        **record,
        "sample_index": int(record["patch_index"]),
        "patch_center_glon_deg": float(record["patch_center_glon_deg"]),
        "patch_center_glat_deg": float(record["patch_center_glat_deg"]),
        "candidate_x_pix": float(peak_j),
        "candidate_y_pix": float(peak_i),
        "candidate_dx_deg": float(dx_deg),
        "candidate_dy_deg": float(dy_deg),
        "radius_est_deg": float(record.get("radius_est_deg", 0.0) or 0.0),
        "has_candidate": True,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    candidate_jsonl = Path(args.candidate_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else output_dir / "template_fit_records.jsonl"
    output_summary = Path(args.output_summary).expanduser().resolve() if args.output_summary else output_dir / "template_fit_summary.json"

    records = load_jsonl(candidate_jsonl)
    if args.max_candidates:
        records = records[: int(args.max_candidates)]

    rows = []
    h5_handles: dict[str, h5py.File] = {}
    try:
        for idx, raw_record in enumerate(records, start=1):
            map_name = str(raw_record["map"])
            if map_name not in h5_handles:
                tile_path = Path(str(args.tile_h5_template).format(map=map_name)).expanduser().resolve()
                if not tile_path.exists():
                    raise FileNotFoundError(f"Missing tile HDF5 for map {map_name}: {tile_path}")
                h5_handles[map_name] = h5py.File(tile_path, "r")
            record = adapt_record(raw_record)
            patch = h5_handles[map_name]["patches"][int(record["sample_index"])]
            fit_row = fit_one_candidate(patch, record, args)
            fit_row["map"] = map_name
            fit_row["source_kind"] = raw_record.get("source_kind", "")
            fit_row["policy_slug"] = raw_record.get("policy_slug", "")
            fit_row["cluster_id"] = raw_record.get("cluster_id")
            fit_row["cluster_radius_deg"] = raw_record.get("cluster_radius_deg")
            fit_row["global_cluster_rank"] = raw_record.get("global_cluster_rank")
            fit_row["rank_score"] = raw_record.get("rank_score")
            rows.append(fit_row)
            if idx % 25 == 0 or idx == len(records):
                print(f"  Fit {idx:4d} / {len(records)} screened candidates", flush=True)
    finally:
        for handle in h5_handles.values():
            handle.close()

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "candidate_jsonl": str(candidate_jsonl),
        "output_jsonl": str(output_jsonl),
        "fit_policy": {
            "radius_window_deg": float(args.radius_window_deg),
            "radius_step_deg": float(args.radius_step_deg),
            "min_radius_deg": float(args.min_radius_deg),
            "max_radius_deg": float(args.max_radius_deg),
            "support_extra_deg": float(args.support_extra_deg),
            "support_factor": float(args.support_factor),
            "edge_sigma_deg": float(args.edge_sigma_deg),
            "max_candidates": int(args.max_candidates),
        },
        "metrics": summarize(rows),
        "maps": sorted({str(row["map"]) for row in rows}),
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Template-fit records: {output_jsonl}")
    print(f"Template-fit summary: {output_summary}")


if __name__ == "__main__":
    main()
