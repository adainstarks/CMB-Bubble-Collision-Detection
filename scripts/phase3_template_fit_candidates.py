"""
Fit Feeney Eq. 1 templates to structured Phase 3 candidate records.

This is a classical handoff product, not a Bayesian detection claim. It converts
candidate masks/provenance into deterministic template-fit records containing
candidate center, angular size, fitted z0/zcrit amplitudes, and a null-vs-template
delta-chi2 score suitable for downstream posterior/evidence tooling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from phase2_signal_model import T_CMB_K, bubble_collision_signal
from phase3_audit_outputs import load_jsonl
from phase_dataset_utils import make_angular_distance_grid, patch_offsets_deg_to_sky


PATCH_PIX = 256
RESO_ARCMIN = 13.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit Feeney Eq. 1 template parameters for Phase 3 candidate records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--data-h5", type=str, default="")
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
    parser.add_argument("--include-empty", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.radius_window_deg < 0.0:
        raise ValueError("--radius-window-deg must be non-negative.")
    if args.radius_step_deg <= 0.0:
        raise ValueError("--radius-step-deg must be positive.")
    if args.min_radius_deg <= 0.0:
        raise ValueError("--min-radius-deg must be positive.")
    if args.max_radius_deg <= args.min_radius_deg:
        raise ValueError("--max-radius-deg must be larger than --min-radius-deg.")
    if args.support_extra_deg < 0.0:
        raise ValueError("--support-extra-deg must be non-negative.")
    if args.support_factor < 1.0:
        raise ValueError("--support-factor must be >= 1.")
    if args.edge_sigma_deg < 0.0:
        raise ValueError("--edge-sigma-deg must be non-negative.")
    if args.max_candidates < 0:
        raise ValueError("--max-candidates must be non-negative.")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_h5(eval_dir, data_h5_arg):
    if data_h5_arg:
        return Path(data_h5_arg).resolve()
    summary = load_json(eval_dir / "evaluation_summary.json")
    run_config = load_json(Path(summary["run_dir"]) / "run_config.json")
    return Path(run_config["data_h5"]).resolve()


def radius_grid(radius_est_deg, args):
    """Return the radius search grid for one candidate.

    When a screening-stage radius estimate is unavailable, do not collapse the
    template search around an arbitrary default scale. Search the full
    configured radius interval instead.
    """

    radius_est_deg = float(radius_est_deg)
    if not np.isfinite(radius_est_deg) or radius_est_deg <= 0.0:
        lo = float(args.min_radius_deg)
        hi = float(args.max_radius_deg)
    else:
        radius_est_deg = float(np.clip(radius_est_deg, float(args.min_radius_deg), float(args.max_radius_deg)))
        lo = max(float(args.min_radius_deg), radius_est_deg - float(args.radius_window_deg))
        hi = min(float(args.max_radius_deg), radius_est_deg + float(args.radius_window_deg))
    grid = np.arange(lo, hi + 0.5 * float(args.radius_step_deg), float(args.radius_step_deg))
    grid = np.clip(grid, float(args.min_radius_deg), float(args.max_radius_deg))
    return np.unique(np.round(grid, 6))


def nuisance_design(theta_grid, support):
    y_pix, x_pix = np.nonzero(support)
    center = (PATCH_PIX - 1) / 2.0
    x = (x_pix.astype(np.float64) - center) / center
    y = (y_pix.astype(np.float64) - center) / center
    return np.column_stack([np.ones(len(x), dtype=np.float64), x, y])


def fit_least_squares(y, design):
    coeff, residuals, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coeff
    resid = y - fitted
    sse = float(np.dot(resid, resid))
    dof = max(int(len(y) - design.shape[1]), 1)
    return coeff, sse, dof, rank, singular_values


def build_template_columns(theta_grid, theta_crit_deg, support, edge_sigma_deg):
    theta_crit_rad = np.radians(float(theta_crit_deg))
    basis_z0 = T_CMB_K * bubble_collision_signal(
        theta_grid,
        z0=1.0,
        zcrit=0.0,
        theta_crit=theta_crit_rad,
        edge_sigma_deg=float(edge_sigma_deg),
    )
    basis_zcrit = T_CMB_K * bubble_collision_signal(
        theta_grid,
        z0=0.0,
        zcrit=1.0,
        theta_crit=theta_crit_rad,
        edge_sigma_deg=float(edge_sigma_deg),
    )
    return basis_z0[support], basis_zcrit[support]


def fit_one_candidate(patch, record, args):
    if not bool(record.get("has_candidate", False)):
        return {
            "fit_status": "skipped_no_candidate",
            "sample_index": int(record["sample_index"]),
            "has_candidate": False,
        }

    center_x = float(record["candidate_x_pix"])
    center_y = float(record["candidate_y_pix"])
    theta_grid = make_angular_distance_grid(
        PATCH_PIX,
        RESO_ARCMIN,
        center_x_pix=center_x,
        center_y_pix=center_y,
    )
    theta_grid_deg = np.degrees(theta_grid)
    candidate_radius = float(record.get("radius_est_deg") or 0.0)
    has_radius_seed = bool(np.isfinite(candidate_radius) and candidate_radius > 0.0)
    if has_radius_seed:
        support_radius = min(
            float(args.max_radius_deg) + float(args.support_extra_deg),
            max(candidate_radius + float(args.support_extra_deg), candidate_radius * float(args.support_factor)),
        )
    else:
        # Real-map screened candidates currently do not carry a reliable radius
        # estimate. Use the full configured search support so the follow-up fit
        # is not biased toward an arbitrary default scale.
        support_radius = float(args.max_radius_deg) + float(args.support_extra_deg)
    support = np.isfinite(patch) & (theta_grid_deg <= support_radius)
    if int(support.sum()) < 32:
        return {
            "fit_status": "skipped_insufficient_support",
            "sample_index": int(record["sample_index"]),
            "has_candidate": True,
            "support_pixels": int(support.sum()),
        }

    y = np.asarray(patch[support], dtype=np.float64)
    nuisance = nuisance_design(theta_grid, support)
    nuisance_coeff, null_sse, null_dof, _, _ = fit_least_squares(y, nuisance)

    best = None
    for radius_deg in radius_grid(candidate_radius, args):
        basis_z0, basis_zcrit = build_template_columns(
            theta_grid=theta_grid,
            theta_crit_deg=radius_deg,
            support=support,
            edge_sigma_deg=float(args.edge_sigma_deg),
        )
        design = np.column_stack([nuisance, basis_z0, basis_zcrit])
        coeff, template_sse, template_dof, rank, singular_values = fit_least_squares(y, design)
        delta_chi2 = null_sse - template_sse
        candidate = {
            "theta_crit_fit_deg": float(radius_deg),
            "z0_fit": float(coeff[-2]),
            "zcrit_fit": float(coeff[-1]),
            "nuisance_offset_k": float(coeff[0]),
            "nuisance_x_slope_k": float(coeff[1]),
            "nuisance_y_slope_k": float(coeff[2]),
            "template_sse": float(template_sse),
            "template_dof": int(template_dof),
            "template_reduced_sse": float(template_sse / max(template_dof, 1)),
            "delta_chi2_vs_plane_null": float(delta_chi2),
            "rank": int(rank),
            "condition_proxy": float(np.max(singular_values) / max(np.min(singular_values), 1e-30)),
        }
        if best is None or candidate["delta_chi2_vs_plane_null"] > best["delta_chi2_vs_plane_null"]:
            best = candidate

    denom = 1.0 - np.cos(np.radians(best["theta_crit_fit_deg"]))
    c0_fit = (best["zcrit_fit"] - best["z0_fit"] * np.cos(np.radians(best["theta_crit_fit_deg"]))) / denom
    c1_fit = (best["z0_fit"] - best["zcrit_fit"]) / denom
    center_glon, center_glat = patch_offsets_deg_to_sky(
        float(record["patch_center_glon_deg"]),
        float(record["patch_center_glat_deg"]),
        float(record["candidate_dx_deg"]),
        float(record["candidate_dy_deg"]),
    )

    result = {
        "fit_status": "fit",
        "sample_index": int(record["sample_index"]),
        "has_candidate": True,
        "patch_center_glon_deg": float(record["patch_center_glon_deg"]),
        "patch_center_glat_deg": float(record["patch_center_glat_deg"]),
        "candidate_x_pix": center_x,
        "candidate_y_pix": center_y,
        "candidate_dx_deg": float(record["candidate_dx_deg"]),
        "candidate_dy_deg": float(record["candidate_dy_deg"]),
        "candidate_glon_deg": float(center_glon),
        "candidate_glat_deg": float(center_glat),
        "candidate_radius_est_deg": candidate_radius,
        "candidate_radius_seed_available": has_radius_seed,
        "support_radius_deg": float(support_radius),
        "support_pixels": int(support.sum()),
        "plane_null_sse": float(null_sse),
        "plane_null_dof": int(null_dof),
        "plane_null_reduced_sse": float(null_sse / max(null_dof, 1)),
        "plane_null_offset_k": float(nuisance_coeff[0]),
        "edge_sigma_fit_deg": float(args.edge_sigma_deg),
        "c0_fit": float(c0_fit),
        "c1_fit": float(c1_fit),
    }
    result.update(best)
    for key in (
        "truth_label",
        "truth_theta_crit_deg",
        "truth_z0",
        "truth_zcrit",
        "truth_edge_sigma_deg",
        "coord_pool_idx",
        "cmb_realization_idx",
        "background_id",
    ):
        if key in record:
            result[key] = record[key]
    return result


def summarize(rows):
    fit_rows = [row for row in rows if row.get("fit_status") == "fit"]
    summary = {
        "num_records": int(len(rows)),
        "num_fit": int(len(fit_rows)),
        "num_skipped": int(len(rows) - len(fit_rows)),
    }
    if not fit_rows:
        return summary
    for key in ("delta_chi2_vs_plane_null", "theta_crit_fit_deg", "z0_fit", "zcrit_fit"):
        values = np.asarray([float(row[key]) for row in fit_rows], dtype=np.float64)
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_p90"] = float(np.percentile(values, 90.0))
        summary[f"{key}_max"] = float(np.max(values))
    if "truth_label" in fit_rows[0]:
        pos = [row for row in fit_rows if int(row.get("truth_label", 0)) == 1]
        neg = [row for row in fit_rows if int(row.get("truth_label", 0)) == 0]
        summary["num_fit_truth_positive"] = int(len(pos))
        summary["num_fit_truth_negative"] = int(len(neg))
        for label, subset in (("truth_positive", pos), ("truth_negative", neg)):
            if subset:
                values = np.asarray([float(row["delta_chi2_vs_plane_null"]) for row in subset], dtype=np.float64)
                summary[f"{label}_delta_chi2_median"] = float(np.median(values))
                summary[f"{label}_delta_chi2_p90"] = float(np.percentile(values, 90.0))
    return summary


def main():
    args = parse_args()
    validate_args(args)
    eval_dir = Path(args.eval_dir).resolve()
    data_h5 = resolve_data_h5(eval_dir, args.data_h5)
    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else eval_dir / "template_fit_records.jsonl"
    output_summary = Path(args.output_summary).resolve() if args.output_summary else eval_dir / "template_fit_summary.json"

    records = load_jsonl(eval_dir / "candidate_records.jsonl")
    if not args.include_empty:
        records = [record for record in records if bool(record.get("has_candidate", False))]
    if args.max_candidates:
        records = records[: int(args.max_candidates)]

    rows = []
    with h5py.File(data_h5, "r") as h5:
        patches = h5["patches"]
        for idx, record in enumerate(records, start=1):
            sample_idx = int(record["sample_index"])
            patch = np.asarray(patches[sample_idx], dtype=np.float64)
            rows.append(fit_one_candidate(patch, record, args))
            if idx % 100 == 0 or idx == len(records):
                print(f"  Fit {idx:5d} / {len(records)} candidates", flush=True)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "eval_dir": str(eval_dir),
        "data_h5": str(data_h5),
        "output_jsonl": str(output_jsonl),
        "fit_policy": {
            "radius_window_deg": float(args.radius_window_deg),
            "radius_step_deg": float(args.radius_step_deg),
            "min_radius_deg": float(args.min_radius_deg),
            "max_radius_deg": float(args.max_radius_deg),
            "support_extra_deg": float(args.support_extra_deg),
            "support_factor": float(args.support_factor),
            "edge_sigma_deg": float(args.edge_sigma_deg),
            "include_empty": bool(args.include_empty),
            "max_candidates": int(args.max_candidates),
        },
        "metrics": summarize(rows),
    }
    with open(output_summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Template-fit records: {output_jsonl}")
    print(f"Template-fit summary: {output_summary}")


if __name__ == "__main__":
    main()
