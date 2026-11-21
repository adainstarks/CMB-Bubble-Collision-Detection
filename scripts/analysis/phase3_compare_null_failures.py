"""
Compare null-control false positives emitted by two Phase 3 screeners.

This is a red-team diagnostic for real-map nuisance behavior. It answers:
    - do two models fail on the same null patches?
    - are false positives spatially clustered on the sky?
    - do they live in unusual mask/latitude/patch-statistic environments?
    - are emitted masks broad filled blobs, ring-like, or small fragments?
"""

from __future__ import annotations
# Bootstrap organized script folders for direct CLI execution.
import sys as _cmb_sys
from pathlib import Path as _cmb_Path

_cmb_scripts_root = _cmb_Path(__file__).resolve().parents[1]
_cmb_paths = [_cmb_scripts_root]
_cmb_paths.extend(
    sorted(
        _cmb_path
        for _cmb_path in _cmb_scripts_root.iterdir()
        if _cmb_path.is_dir() and _cmb_path.name != "__pycache__"
    )
)
for _cmb_path in reversed(_cmb_paths):
    _cmb_path_text = str(_cmb_path)
    if _cmb_path_text not in _cmb_sys.path:
        _cmb_sys.path.insert(0, _cmb_path_text)
del _cmb_Path, _cmb_sys, _cmb_scripts_root, _cmb_paths, _cmb_path, _cmb_path_text

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi


PIXEL_SCALE_DEG = 13.0 / 60.0
EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare false-positive families from two real-SMICA null-control output directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--null-h5", type=str, required=True)
    parser.add_argument("--first-dir", type=str, required=True)
    parser.add_argument("--second-dir", type=str, required=True)
    parser.add_argument("--first-label", type=str, default="first")
    parser.add_argument("--second-label", type=str, default="second")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, default="")
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_masks(path):
    data = np.load(path)
    shape = tuple(int(x) for x in data["mask_shape"])
    unpacked = np.unpackbits(data["mask_bits"], axis=1)[:, : int(np.prod(shape))]
    return unpacked.reshape((-1, *shape)).astype(bool)


def load_output_bundle(output_dir):
    output_dir = Path(output_dir)
    records = read_jsonl(output_dir / "null_candidate_records.jsonl")
    masks = load_masks(output_dir / "null_candidate_masks.npz")
    if len(records) != int(masks.shape[0]):
        raise RuntimeError(f"Record/mask count mismatch in {output_dir}: {len(records)} vs {masks.shape[0]}")
    by_sample = {int(record["sample_index"]): idx for idx, record in enumerate(records)}
    candidate_indices = [idx for idx, record in enumerate(records) if bool(record.get("has_candidate", False))]
    return {
        "dir": str(output_dir.resolve()),
        "records": records,
        "masks": masks,
        "by_sample": by_sample,
        "candidate_indices": candidate_indices,
    }


def finite_float(value, default=np.nan):
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def lonlat_to_unit(lon_deg, lat_deg):
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def angular_distance_matrix_deg(lon_a, lat_a, lon_b, lat_b):
    if len(lon_a) == 0 or len(lon_b) == 0:
        return np.zeros((len(lon_a), len(lon_b)), dtype=np.float64)
    unit_a = lonlat_to_unit(lon_a, lat_a)
    unit_b = lonlat_to_unit(lon_b, lat_b)
    cosang = np.clip(unit_a @ unit_b.T, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def summarize_values(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def summarize_candidate_sky(records, candidate_indices):
    if not candidate_indices:
        return {
            "candidate_count": 0,
            "nearest_neighbor_deg": summarize_values([]),
            "pairs_within_5deg": 0,
            "pairs_within_10deg": 0,
            "pairs_within_20deg": 0,
            "max_neighbors_within_10deg": 0,
        }

    lon = np.asarray(
        [
            finite_float(records[idx].get("candidate_glon_deg"), records[idx].get("patch_center_glon_deg"))
            for idx in candidate_indices
        ],
        dtype=np.float64,
    )
    lat = np.asarray(
        [
            finite_float(records[idx].get("candidate_glat_deg"), records[idx].get("patch_center_glat_deg"))
            for idx in candidate_indices
        ],
        dtype=np.float64,
    )
    dist = angular_distance_matrix_deg(lon, lat, lon, lat)
    if len(candidate_indices) > 1:
        dist_no_self = dist.copy()
        np.fill_diagonal(dist_no_self, np.inf)
        nearest = np.min(dist_no_self, axis=1)
        upper = dist[np.triu_indices_from(dist, k=1)]
        neighbor_counts_10 = (dist_no_self <= 10.0).sum(axis=1)
    else:
        nearest = np.asarray([], dtype=np.float64)
        upper = np.asarray([], dtype=np.float64)
        neighbor_counts_10 = np.asarray([0], dtype=np.int64)

    clusters = []
    if len(candidate_indices) > 1:
        adjacency = dist <= 10.0
        seen = np.zeros(len(candidate_indices), dtype=bool)
        units = lonlat_to_unit(lon, lat)
        for start in range(len(candidate_indices)):
            if seen[start]:
                continue
            stack = [start]
            members = []
            seen[start] = True
            while stack:
                node = stack.pop()
                members.append(node)
                for neighbor in np.flatnonzero(adjacency[node] & ~seen):
                    seen[neighbor] = True
                    stack.append(int(neighbor))
            member_units = units[members]
            center_vec = member_units.mean(axis=0)
            norm = np.linalg.norm(center_vec)
            if norm > 0.0:
                center_vec = center_vec / norm
                center_lon = math.degrees(math.atan2(center_vec[1], center_vec[0])) % 360.0
                center_lat = math.degrees(math.asin(float(np.clip(center_vec[2], -1.0, 1.0))))
            else:
                center_lon = float(np.mean(lon[members]))
                center_lat = float(np.mean(lat[members]))
            clusters.append(
                {
                    "size": int(len(members)),
                    "center_glon_deg": float(center_lon),
                    "center_glat_deg": float(center_lat),
                    "sample_indices": [int(records[candidate_indices[m]]["sample_index"]) for m in members[:25]],
                }
            )
    elif len(candidate_indices) == 1:
        clusters = [
            {
                "size": 1,
                "center_glon_deg": float(lon[0]),
                "center_glat_deg": float(lat[0]),
                "sample_indices": [int(records[candidate_indices[0]]["sample_index"])],
            }
        ]

    clusters.sort(key=lambda row: row["size"], reverse=True)

    return {
        "candidate_count": int(len(candidate_indices)),
        "nearest_neighbor_deg": summarize_values(nearest),
        "pairs_within_5deg": int((upper <= 5.0).sum()),
        "pairs_within_10deg": int((upper <= 10.0).sum()),
        "pairs_within_20deg": int((upper <= 20.0).sum()),
        "max_neighbors_within_10deg": int(neighbor_counts_10.max()) if neighbor_counts_10.size else 0,
        "clusters_within_10deg": {
            "num_clusters": int(len(clusters)),
            "largest_cluster_size": int(clusters[0]["size"]) if clusters else 0,
            "top_clusters": clusters[:10],
        },
        "abs_candidate_glat_deg": summarize_values(np.abs(lat)),
    }


def mask_boundary(mask):
    if not mask.any():
        return mask
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def shape_features(mask, record):
    area = int(mask.sum())
    if area == 0:
        return {
            "area_pixels": 0,
            "perimeter_pixels": 0,
            "compactness": 0.0,
            "fill_fraction": 0.0,
            "outer_band_fraction": 0.0,
            "inner_fill_fraction": 0.0,
            "shape_class": "empty",
        }

    perimeter = int(mask_boundary(mask).sum())
    compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1))

    ys, xs = np.nonzero(mask)
    cx = finite_float(record.get("candidate_x_pix"), np.mean(xs))
    cy = finite_float(record.get("candidate_y_pix"), np.mean(ys))
    yy, xx = np.indices(mask.shape)
    radial_pix = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius_pix = max(finite_float(record.get("radius_est_deg"), 0.0) / PIXEL_SCALE_DEG, 1.0)
    inside = radial_pix <= radius_pix
    inner = radial_pix <= 0.5 * radius_pix
    outer = (radial_pix >= 0.8 * radius_pix) & inside
    fill_fraction = float(area / max(float(inside.sum()), 1.0))
    outer_band_fraction = float((mask & outer).sum() / max(area, 1))
    inner_fill_fraction = float((mask & inner).sum() / max(float(inner.sum()), 1.0))

    if area < 25:
        shape_class = "tiny_fragment"
    elif fill_fraction < 0.35 and outer_band_fraction > 0.45 and inner_fill_fraction < 0.25:
        shape_class = "ring_or_arc_like"
    elif fill_fraction > 0.45 and inner_fill_fraction > 0.35:
        shape_class = "broad_filled_blob"
    else:
        shape_class = "irregular_fragment"

    return {
        "area_pixels": area,
        "perimeter_pixels": perimeter,
        "compactness": compactness,
        "fill_fraction": fill_fraction,
        "outer_band_fraction": outer_band_fraction,
        "inner_fill_fraction": inner_fill_fraction,
        "shape_class": shape_class,
    }


def patch_plane_gradient_features(patch):
    patch = np.asarray(patch, dtype=np.float64)
    n_y, n_x = patch.shape
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    x_deg = (xx - (n_x - 1) / 2.0) * PIXEL_SCALE_DEG
    y_deg = (yy - (n_y - 1) / 2.0) * PIXEL_SCALE_DEG
    design = np.column_stack([np.ones(patch.size), x_deg.ravel(), y_deg.ravel()])
    coeff, *_ = np.linalg.lstsq(design, patch.ravel(), rcond=None)
    fitted = design @ coeff
    residual = patch.ravel() - fitted
    total_var = float(np.var(patch.ravel()))
    residual_var = float(np.var(residual))
    return {
        "patch_mean_k": float(np.mean(patch)),
        "patch_std_k": float(np.std(patch)),
        "plane_gradient_k_per_deg": float(math.hypot(coeff[1], coeff[2])),
        "plane_r2": float(1.0 - residual_var / max(total_var, EPS)),
    }


def collect_candidate_features(bundle, null_h5):
    rows = []
    with h5py.File(null_h5, "r") as h5:
        patches = h5["patches"]
        for idx in bundle["candidate_indices"]:
            record = bundle["records"][idx]
            sample_idx = int(record["sample_index"])
            shape = shape_features(bundle["masks"][idx], record)
            patch_features = patch_plane_gradient_features(patches[sample_idx])
            row = {
                "sample_index": sample_idx,
                "coord_pool_idx": int(record.get("coord_pool_idx", -1)),
                "patch_center_glon_deg": finite_float(record.get("patch_center_glon_deg")),
                "patch_center_glat_deg": finite_float(record.get("patch_center_glat_deg")),
                "candidate_glon_deg": finite_float(record.get("candidate_glon_deg"), record.get("patch_center_glon_deg")),
                "candidate_glat_deg": finite_float(record.get("candidate_glat_deg"), record.get("patch_center_glat_deg")),
                "coord_mask_fraction": finite_float(record.get("coord_mask_fraction")),
                "score_max": finite_float(record.get("score_max")),
                "score_mean": finite_float(record.get("score_mean")),
                "positive_fraction": finite_float(record.get("positive_fraction")),
                "radius_est_deg": finite_float(record.get("radius_est_deg")),
            }
            row.update(shape)
            row.update(patch_features)
            rows.append(row)
    return rows


def summarize_feature_rows(rows):
    classes = {}
    for row in rows:
        classes[row["shape_class"]] = classes.get(row["shape_class"], 0) + 1
    return {
        "count": int(len(rows)),
        "shape_class_counts": classes,
        "coord_mask_fraction": summarize_values([row["coord_mask_fraction"] for row in rows]),
        "abs_patch_center_glat_deg": summarize_values([abs(row["patch_center_glat_deg"]) for row in rows]),
        "radius_est_deg": summarize_values([row["radius_est_deg"] for row in rows]),
        "area_pixels": summarize_values([row["area_pixels"] for row in rows]),
        "compactness": summarize_values([row["compactness"] for row in rows]),
        "fill_fraction": summarize_values([row["fill_fraction"] for row in rows]),
        "outer_band_fraction": summarize_values([row["outer_band_fraction"] for row in rows]),
        "inner_fill_fraction": summarize_values([row["inner_fill_fraction"] for row in rows]),
        "patch_std_k": summarize_values([row["patch_std_k"] for row in rows]),
        "plane_gradient_k_per_deg": summarize_values([row["plane_gradient_k_per_deg"] for row in rows]),
        "plane_r2": summarize_values([row["plane_r2"] for row in rows]),
    }


def overlap_summary(first_bundle, second_bundle):
    first_samples = {int(first_bundle["records"][idx]["sample_index"]) for idx in first_bundle["candidate_indices"]}
    second_samples = {int(second_bundle["records"][idx]["sample_index"]) for idx in second_bundle["candidate_indices"]}
    shared = sorted(first_samples & second_samples)

    first_only = sorted(first_samples - second_samples)
    second_only = sorted(second_samples - first_samples)
    first_candidate_records = [first_bundle["records"][first_bundle["by_sample"][idx]] for idx in first_samples]
    second_candidate_records = [second_bundle["records"][second_bundle["by_sample"][idx]] for idx in second_samples]

    if first_candidate_records and second_candidate_records:
        dist = angular_distance_matrix_deg(
            [finite_float(r.get("candidate_glon_deg"), r.get("patch_center_glon_deg")) for r in first_candidate_records],
            [finite_float(r.get("candidate_glat_deg"), r.get("patch_center_glat_deg")) for r in first_candidate_records],
            [finite_float(r.get("candidate_glon_deg"), r.get("patch_center_glon_deg")) for r in second_candidate_records],
            [finite_float(r.get("candidate_glat_deg"), r.get("patch_center_glat_deg")) for r in second_candidate_records],
        )
        nearest_cross = np.min(dist, axis=1)
        cross_pairs_5 = int((dist <= 5.0).sum())
        cross_pairs_10 = int((dist <= 10.0).sum())
        cross_pairs_20 = int((dist <= 20.0).sum())
    else:
        nearest_cross = np.asarray([], dtype=np.float64)
        cross_pairs_5 = cross_pairs_10 = cross_pairs_20 = 0

    return {
        "first_candidate_count": int(len(first_samples)),
        "second_candidate_count": int(len(second_samples)),
        "shared_sample_count": int(len(shared)),
        "first_only_count": int(len(first_only)),
        "second_only_count": int(len(second_only)),
        "shared_sample_indices": shared[:50],
        "first_only_sample_indices": first_only[:50],
        "second_only_sample_indices": second_only[:50],
        "cross_model_nearest_deg_for_first_candidates": summarize_values(nearest_cross),
        "cross_model_pairs_within_5deg": cross_pairs_5,
        "cross_model_pairs_within_10deg": cross_pairs_10,
        "cross_model_pairs_within_20deg": cross_pairs_20,
    }


def environment_baseline(null_h5):
    with h5py.File(null_h5, "r") as h5:
        meta = h5["metadata"]
        glat = np.asarray(meta["glat_deg"][:], dtype=np.float64)
        mask_fraction = np.asarray(meta["coord_mask_fraction"][:], dtype=np.float64)
        sample_indices = np.linspace(0, h5["patches"].shape[0] - 1, num=min(512, h5["patches"].shape[0]), dtype=np.int64)
        patch_std = [float(np.std(h5["patches"][int(idx)])) for idx in sample_indices]
    return {
        "num_null_patches": int(len(glat)),
        "coord_mask_fraction_all": summarize_values(mask_fraction),
        "abs_patch_center_glat_deg_all": summarize_values(np.abs(glat)),
        "patch_std_k_sampled": summarize_values(patch_std),
    }


def write_markdown(path, payload):
    lines = []
    labels = payload["labels"]
    lines.append("# Null Failure Comparison")
    lines.append("")
    lines.append(f"- First model: `{labels['first']}`")
    lines.append(f"- Second model: `{labels['second']}`")
    lines.append(f"- Null patches: `{payload['environment_baseline']['num_null_patches']}`")
    overlap = payload["overlap"]
    lines.append(
        f"- Exact sample overlap: `{overlap['shared_sample_count']}` "
        f"of `{overlap['first_candidate_count']}` first-model and `{overlap['second_candidate_count']}` second-model candidates."
    )
    lines.append("")
    lines.append("## Candidate Counts")
    for key in ("first", "second"):
        label = labels[key]
        summary = payload[key]["features_summary"]
        sky = payload[key]["sky_clustering"]
        lines.append(
            f"- `{label}`: `{summary['count']}` candidates; shape classes `{summary['shape_class_counts']}`; "
            f"pairs within 10 deg `{sky['pairs_within_10deg']}`."
        )
    lines.append("")
    lines.append("## Interpretation Flags")
    lines.append(
        "- Exact-overlap and cross-distance metrics show whether an ensemble would be redundant or complementary."
    )
    lines.append(
        "- Shape classes are heuristics from emitted masks only: broad filled blobs indicate smooth-disc/gradient sensitivity; ring_or_arc_like indicates thin contour-like false positives."
    )
    lines.append(
        "- Plane-gradient metrics are patch-level diagnostics, not proof of causal mechanism."
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    null_h5 = Path(args.null_h5)
    first = load_output_bundle(args.first_dir)
    second = load_output_bundle(args.second_dir)

    first_rows = collect_candidate_features(first, null_h5)
    second_rows = collect_candidate_features(second, null_h5)
    payload = {
        "null_h5": str(null_h5.resolve()),
        "labels": {"first": args.first_label, "second": args.second_label},
        "environment_baseline": environment_baseline(null_h5),
        "overlap": overlap_summary(first, second),
        "first": {
            "output_dir": first["dir"],
            "sky_clustering": summarize_candidate_sky(first["records"], first["candidate_indices"]),
            "features_summary": summarize_feature_rows(first_rows),
            "candidate_features": first_rows,
        },
        "second": {
            "output_dir": second["dir"],
            "sky_clustering": summarize_candidate_sky(second["records"], second["candidate_indices"]),
            "features_summary": summarize_feature_rows(second_rows),
            "candidate_features": second_rows,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if args.output_md:
        write_markdown(args.output_md, payload)

    print(json.dumps({k: payload[k] for k in ("labels", "environment_baseline", "overlap")}, indent=2))
    print(f"Saved comparison JSON: {output_json}")
    if args.output_md:
        print(f"Saved comparison MD:   {args.output_md}")


if __name__ == "__main__":
    main()
