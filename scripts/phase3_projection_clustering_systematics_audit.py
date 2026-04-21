"""Audit projection mismatch and clustering sensitivity for remediated v1.

Assumptions
-----------
* The training HDF5s are patch-generated products, while the true Wiener
  benchmark and auxiliary channel live on the sphere. Projection mismatch must
  therefore be measured explicitly rather than assumed negligible.
* Great-circle clustering is candidate-volume accounting on overlapping sky
  tiles, not a detection statistic. The scientifically relevant question is how
  strongly candidate burden depends on the chosen clustering radius.
* The top deployed policy is taken from the emitted candidate summary; this
  audit does not search for a new policy.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import healpy as hp
import numpy as np

from phase2_observing_model import inject_signal_on_sphere, project_patch
from phase2_signal_model import PATCH_PIX, RESO_ARCMIN, inject_signal_into_patch
from phase_dataset_utils import patch_center_pixel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_projection_clustering_audit"
DEFAULT_CANDIDATE_SUMMARY = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_tile_constrained_candidates" / "candidate_emission_summary.json"
)
DEFAULT_POLICY_TILE_AUDIT = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_policy_tile_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit projection mismatch and clustering-radius sensitivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate-summary-json", type=str, default=str(DEFAULT_CANDIDATE_SUMMARY))
    parser.add_argument("--policy-tile-audit-dir", type=str, default=str(DEFAULT_POLICY_TILE_AUDIT))
    parser.add_argument("--theta-grid-deg", type=str, default="5,10,15,20,25")
    parser.add_argument("--offset-grid-deg", type=str, default="0,5,10,15")
    parser.add_argument("--cluster-radii-deg", type=str, default="5,10,15,25,40")
    parser.add_argument("--nside", type=int, default=256)
    return parser.parse_args()


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(text).split(",") if item.strip())


def validate_args(args: argparse.Namespace) -> None:
    args.theta_values = parse_float_list(args.theta_grid_deg)
    args.offset_values = parse_float_list(args.offset_grid_deg)
    args.cluster_values = parse_float_list(args.cluster_radii_deg)
    if not args.theta_values:
        raise ValueError("--theta-grid-deg must contain at least one radius.")
    if not args.offset_values:
        raise ValueError("--offset-grid-deg must contain at least one offset.")
    if not args.cluster_values:
        raise ValueError("--cluster-radii-deg must contain at least one clustering radius.")
    if any(theta <= 0.0 for theta in args.theta_values):
        raise ValueError("All theta-grid values must be positive.")
    if any(offset < 0.0 for offset in args.offset_values):
        raise ValueError("All offset-grid values must be non-negative.")
    if any(radius <= 0.0 for radius in args.cluster_values):
        raise ValueError("All cluster radii must be positive.")
    if int(args.nside) <= 0 or not hp.isnsideok(int(args.nside)):
        raise ValueError("--nside must be a valid positive HEALPix Nside.")


def compare_planar_vs_spherical(theta_crit_deg: float, offset_deg: float, *, nside: int) -> dict[str, float]:
    zero_patch = np.zeros((PATCH_PIX, PATCH_PIX), dtype=np.float64)
    zero_map = np.zeros(hp.nside2npix(int(nside)), dtype=np.float64)
    center_pix = patch_center_pixel(PATCH_PIX)
    offset_pix = float(offset_deg) * 60.0 / RESO_ARCMIN
    center_x = float(center_pix + offset_pix)
    center_y = float(center_pix)

    planar_injected, planar_signal = inject_signal_into_patch(
        zero_patch,
        z0=1.0e-5,
        zcrit=0.0,
        theta_crit_deg=float(theta_crit_deg),
        edge_sigma_deg=0.0,
        center_x_pix=center_x,
        center_y_pix=center_y,
    )
    del planar_injected  # Signal-only comparison; zero background makes the delta equal the patch.
    spherical_map = inject_signal_on_sphere(
        zero_map,
        glon_deg=180.0,
        glat_deg=0.0,
        z0=1.0e-5,
        zcrit=0.0,
        theta_crit_deg=float(theta_crit_deg),
        edge_sigma_deg=0.0,
        center_x_pix=float((PATCH_PIX - 1) - center_x),
        center_y_pix=center_y,
    )
    spherical_patch = np.asarray(project_patch(spherical_map, 180.0, 0.0), dtype=np.float64)
    planar_signal = np.asarray(planar_signal, dtype=np.float64)

    ref_norm = float(np.linalg.norm(spherical_patch))
    diff = planar_signal - spherical_patch
    rel_l2 = float(np.linalg.norm(diff) / max(ref_norm, 1e-30))
    cosine = float(
        np.dot(planar_signal.ravel(), spherical_patch.ravel())
        / max(np.linalg.norm(planar_signal.ravel()) * np.linalg.norm(spherical_patch.ravel()), 1e-30)
    )
    spherical_peak = float(np.max(np.abs(spherical_patch)))
    planar_peak = float(np.max(np.abs(planar_signal)))
    peak_abs_frac_error = float(np.abs(planar_peak - spherical_peak) / max(spherical_peak, 1e-30))
    threshold = 0.01 * max(spherical_peak, planar_peak, 1e-30)
    planar_support = np.abs(planar_signal) >= threshold
    spherical_support = np.abs(spherical_patch) >= threshold
    intersection = int(np.count_nonzero(planar_support & spherical_support))
    union = int(np.count_nonzero(planar_support | spherical_support))
    support_iou = float(intersection / max(union, 1))
    return {
        "theta_crit_deg": float(theta_crit_deg),
        "offset_deg": float(offset_deg),
        "relative_l2_error": rel_l2,
        "cosine_similarity": cosine,
        "peak_abs_frac_error": peak_abs_frac_error,
        "support_iou": support_iou,
        "x_mirror_correction_applied": True,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decimal_slug(value: float) -> str:
    text = f"{float(value):.2f}"
    return text.replace(".", "_")


def clustering_rows(candidate_summary: dict, policy_tile_audit_dir: Path, cluster_values: tuple[float, ...]) -> list[dict]:
    constraints = candidate_summary["constraints"]
    policy_rank = int(candidate_summary["policy_rank"])
    slug = f"camb{decimal_slug(float(constraints['max_camb_fpr']))}_real{decimal_slug(float(constraints['max_real_fpr']))}_rank{policy_rank}"
    rows = []
    for map_row in candidate_summary["map_rows"]:
        map_name = str(map_row["map"])
        candidate_path = policy_tile_audit_dir / map_name / f"candidates_{slug}.jsonl"
        candidate_count = len(load_jsonl(candidate_path))
        for radius in cluster_values:
            cluster_path = policy_tile_audit_dir / map_name / f"clusters_{slug}_{int(radius):d}deg.jsonl"
            clusters = load_jsonl(cluster_path)
            cluster_sizes = [int(row["n_members"]) for row in clusters]
            n_clusters = int(len(clusters))
            rows.append(
                {
                    "map": map_name,
                    "cluster_radius_deg": float(radius),
                    "num_candidates": int(candidate_count),
                    "num_clusters": n_clusters,
                    "reduction_factor": float(candidate_count / max(n_clusters, 1)),
                    "max_cluster_size": int(max(cluster_sizes) if cluster_sizes else 0),
                    "mean_cluster_size": float(np.mean(cluster_sizes) if cluster_sizes else 0.0),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict) -> None:
    lines = ["# Projection And Clustering Systematics Audit", ""]
    lines.append("## Assumptions")
    lines.append("")
    for note in report["assumption_warnings"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Projection Mismatch")
    lines.append("")
    lines.append("| theta_deg | offset_deg | rel L2 | cosine | peak frac err | support IoU |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in report["projection_rows"]:
        lines.append(
            f"| {row['theta_crit_deg']:.1f} | {row['offset_deg']:.1f} | "
            f"{row['relative_l2_error']:.4f} | {row['cosine_similarity']:.4f} | "
            f"{row['peak_abs_frac_error']:.4f} | {row['support_iou']:.4f} |"
        )
    lines.append("")
    lines.append("## Clustering Sensitivity")
    lines.append("")
    lines.append("| map | cluster radius deg | candidates | clusters | reduction x | max cluster size |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report["clustering_rows"]:
        lines.append(
            f"| {row['map']} | {row['cluster_radius_deg']:.1f} | {row['num_candidates']} | "
            f"{row['num_clusters']} | {row['reduction_factor']:.2f} | {row['max_cluster_size']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    projection_rows = [
        compare_planar_vs_spherical(theta, offset, nside=int(args.nside))
        for theta in args.theta_values
        for offset in args.offset_values
    ]
    candidate_summary = json.loads(Path(args.candidate_summary_json).expanduser().resolve().read_text(encoding="utf-8"))
    cluster_rows = clustering_rows(
        candidate_summary,
        Path(args.policy_tile_audit_dir).expanduser().resolve(),
        args.cluster_values,
    )

    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "candidate_summary_json": str(Path(args.candidate_summary_json).expanduser().resolve()),
        "policy_tile_audit_dir": str(Path(args.policy_tile_audit_dir).expanduser().resolve()),
        "projection_rows": projection_rows,
        "clustering_rows": cluster_rows,
        "assumption_warnings": [
            "Projection rows compare patch injection against spherical injection projected back to the same gnomonic patch center.",
            "The spherical comparison applies the repo's required x-axis mirror so the injected sky position matches the patch-generated source geometry.",
            "The comparison isolates geometry mismatch on a zero background; it does not include CMB or noise realizations.",
            "Clustering rows summarize candidate-volume sensitivity for the deployed policy. They are not detection significances.",
        ],
    }
    json_path = output_dir / "projection_clustering_audit.json"
    md_path = output_dir / "projection_clustering_audit.md"
    projection_csv = output_dir / "projection_rows.csv"
    clustering_csv = output_dir / "clustering_rows.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    write_csv(projection_csv, projection_rows)
    write_csv(clustering_csv, cluster_rows)
    print(
        json.dumps(
            {
                "report": str(json_path),
                "markdown": str(md_path),
                "projection_csv": str(projection_csv),
                "clustering_csv": str(clustering_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
