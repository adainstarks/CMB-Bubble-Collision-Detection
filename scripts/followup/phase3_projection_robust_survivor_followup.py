"""Projection-robust follow-up for adaptive systems that survive real-sky gates.

Assumptions
-----------
* Input systems are screening-derived adaptive follow-up systems, not
  detections or posterior samples.
* Candidate-centered Lambert azimuthal equal-area re-extraction is used as a
  practical lower-distortion follow-up geometry for large-angular-scale
  candidates. It is not a replacement for a full native-sphere evidence
  calculation.
* The deterministic Feeney-style template fit is reused on both the canonical
  gnomonic patch and the equal-area patch so the comparison isolates geometry
  sensitivity instead of changing the fit family.
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
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np

from phase_dataset_utils import make_angular_distance_grid
from phase2_generate_training import projected_unmasked_fraction
from phase2_observing_model import project_patch, remove_real_map_low_modes
from phase5_half_mission_signflip_null import degrade_map_if_needed, load_analysis_mask, load_map
from phase3_template_fit_candidates import (
    PATCH_PIX,
    RESO_ARCMIN,
    fit_candidate_with_theta_grid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HANDOFF_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_bayesian_template_handoff"
    / "bayesian_template_handoff.jsonl"
)
DEFAULT_SYSTEMS_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_scale_dependent_followup_table"
    / "scale_dependent_followup_table.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_projection_robust_survivor_followup"
)
DEFAULT_COMMON_MASK = PROJECT_ROOT / "data" / "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
DEFAULT_CLEANED_MAP_TEMPLATE = PROJECT_ROOT / "data" / "COM_CMB_IQU-{map}_2048_R3.00_full.fits"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Rerun the surviving adaptive systems on candidate-centered gnomonic and equal-area patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--handoff-jsonl", type=str, default=str(DEFAULT_HANDOFF_JSONL))
    parser.add_argument("--systems-json", type=str, default=str(DEFAULT_SYSTEMS_JSON))
    parser.add_argument("--cleaned-map-template", type=str, default=str(DEFAULT_CLEANED_MAP_TEMPLATE))
    parser.add_argument("--common-mask", type=str, default=str(DEFAULT_COMMON_MASK))
    parser.add_argument("--fits-field", type=int, default=0)
    parser.add_argument("--target-nside", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.9)
    parser.add_argument("--min-valid-fraction", type=float, default=0.9)
    parser.add_argument(
        "--remove-low-modes",
        action="store_true",
        help="Remove monopole/dipole from the cleaned maps before re-extraction.",
    )
    parser.add_argument(
        "--theta-shift-threshold-deg",
        type=float,
        default=2.0,
        help="Maximum allowed equal-area versus gnomonic theta_crit shift.",
    )
    parser.add_argument(
        "--z0-retention-min",
        type=float,
        default=0.5,
        help="Minimum |z0_equal_area| / |z0_gnomonic| retention counted as geometry-consistent.",
    )
    parser.add_argument(
        "--z0-retention-max",
        type=float,
        default=2.0,
        help="Maximum |z0_equal_area| / |z0_gnomonic| retention counted as geometry-consistent.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-systems", type=int, default=0)
    parser.add_argument("--radius-window-deg", type=float, default=3.0)
    parser.add_argument("--radius-step-deg", type=float, default=0.5)
    parser.add_argument("--min-radius-deg", type=float, default=5.0)
    parser.add_argument("--max-radius-deg", type=float, default=25.0)
    parser.add_argument("--support-extra-deg", type=float, default=5.0)
    parser.add_argument("--support-factor", type=float, default=1.5)
    parser.add_argument("--edge-sigma-deg", type=float, default=0.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate non-physical arguments and required inputs."""

    for label, value in (
        ("--mask-threshold", args.mask_threshold),
        ("--min-valid-fraction", args.min_valid_fraction),
        ("--theta-shift-threshold-deg", args.theta_shift_threshold_deg),
        ("--z0-retention-min", args.z0_retention_min),
        ("--z0-retention-max", args.z0_retention_max),
        ("--radius-window-deg", args.radius_window_deg),
        ("--radius-step-deg", args.radius_step_deg),
        ("--min-radius-deg", args.min_radius_deg),
        ("--max-radius-deg", args.max_radius_deg),
        ("--support-extra-deg", args.support_extra_deg),
        ("--support-factor", args.support_factor),
        ("--edge-sigma-deg", args.edge_sigma_deg),
    ):
        if not np.isfinite(float(value)):
            raise ValueError(f"{label} must be finite.")
    if not (0.0 <= float(args.mask_threshold) <= 1.0):
        raise ValueError("--mask-threshold must lie in [0, 1].")
    if not (0.0 <= float(args.min_valid_fraction) <= 1.0):
        raise ValueError("--min-valid-fraction must lie in [0, 1].")
    if float(args.theta_shift_threshold_deg) < 0.0:
        raise ValueError("--theta-shift-threshold-deg must be non-negative.")
    if float(args.z0_retention_min) <= 0.0:
        raise ValueError("--z0-retention-min must be positive.")
    if float(args.z0_retention_max) < float(args.z0_retention_min):
        raise ValueError("--z0-retention-max must be >= --z0-retention-min.")
    if int(args.target_nside) <= 0 or not hp.isnsideok(int(args.target_nside)):
        raise ValueError("--target-nside must be a valid HEALPix Nside.")
    if int(args.max_systems) < 0:
        raise ValueError("--max-systems must be non-negative.")
    for label, value in (
        ("handoff JSONL", args.handoff_jsonl),
        ("systems JSON", args.systems_json),
    ):
        path = Path(str(value)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def candidate_id(row: dict[str, Any]) -> str:
    """Return the canonical candidate identifier."""

    return f"{str(row['map']).lower()}:{int(row['patch_index'])}"


def index_handoff_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index handoff rows by canonical candidate id."""

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = candidate_id(row)
        if key in indexed:
            raise ValueError(f"Duplicate handoff row for {key}.")
        indexed[key] = row
    return indexed


def sign_label(value: float | None) -> int:
    """Return the sign label in ``{-1, 0, +1}``."""

    if value is None or not np.isfinite(float(value)):
        return 0
    return int(np.sign(float(value)))


def patch_center_pixel() -> float:
    """Return the central pixel coordinate of the canonical patch."""

    return (int(PATCH_PIX) - 1) / 2.0


def centered_record(row: dict[str, Any]) -> dict[str, Any]:
    """Build a candidate-centered template-fit record from one handoff row."""

    theta_seed = row.get("template_seed", {}).get("theta_crit_deg")
    return {
        "map": str(row["map"]).lower(),
        "patch_index": int(row["patch_index"]),
        "sample_index": int(row["patch_index"]),
        "has_candidate": True,
        "patch_center_glon_deg": float(row["candidate_glon_deg"]),
        "patch_center_glat_deg": float(row["candidate_glat_deg"]),
        "candidate_x_pix": float(patch_center_pixel()),
        "candidate_y_pix": float(patch_center_pixel()),
        "candidate_dx_deg": 0.0,
        "candidate_dy_deg": 0.0,
        "radius_est_deg": float(theta_seed if theta_seed is not None else 0.0),
    }


def plane_angle_grids() -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel plane coordinates in radians around the patch center."""

    center = patch_center_pixel()
    iy, ix = np.mgrid[0:PATCH_PIX, 0:PATCH_PIX]
    x = np.radians((ix - center) * float(RESO_ARCMIN) / 60.0)
    y = np.radians((iy - center) * float(RESO_ARCMIN) / 60.0)
    return x.astype(np.float64), y.astype(np.float64)


def equal_area_lonlat_and_theta(
    glon_deg: float,
    glat_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-area lon/lat and central-angle grids for one patch center."""

    x, y = plane_angle_grids()
    rho = np.hypot(x, y)
    if np.any(rho > 2.0 + 1.0e-9):
        raise ValueError("Equal-area patch extends beyond the valid Lambert disc.")
    rho_clipped = np.clip(rho, 0.0, 2.0)
    c = 2.0 * np.arcsin(0.5 * rho_clipped)

    lon0 = np.radians(float(glon_deg))
    lat0 = np.radians(float(glat_deg))
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    lat = np.full_like(c, lat0, dtype=np.float64)
    lon = np.full_like(c, lon0, dtype=np.float64)
    use = rho > 1.0e-12
    lat[use] = np.arcsin(
        cos_c[use] * sin_lat0 + (y[use] * sin_c[use] * cos_lat0 / rho[use])
    )
    lon[use] = lon0 + np.arctan2(
        x[use] * sin_c[use],
        rho[use] * cos_lat0 * cos_c[use] - y[use] * sin_lat0 * sin_c[use],
    )
    return np.mod(lon, 2.0 * np.pi), lat, c


def equal_area_project_patch(
    hp_map: np.ndarray,
    glon_deg: float,
    glat_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one map to a candidate-centered Lambert azimuthal equal-area patch."""

    lon, lat, theta_grid = equal_area_lonlat_and_theta(glon_deg, glat_deg)
    colat = 0.5 * np.pi - lat
    patch = hp.get_interp_val(
        np.asarray(hp_map, dtype=np.float64),
        np.asarray(colat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
    )
    return np.asarray(patch, dtype=np.float64), np.asarray(theta_grid, dtype=np.float64)


def finite_float(value: Any) -> float | None:
    """Return a finite float or ``None``."""

    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def prepare_cleaned_map(
    args: argparse.Namespace,
    map_name: str,
    mask_map: np.ndarray,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Load, degrade, and optionally low-mode clean one full cleaned map."""

    key = str(map_name).lower()
    if key in cache:
        return cache[key]
    path = Path(str(args.cleaned_map_template).format(map=key)).expanduser().resolve()
    hp_map = degrade_map_if_needed(load_map(path, int(args.fits_field)), int(args.target_nside))
    if hp.get_nside(hp_map) != int(args.target_nside):
        raise ValueError(f"{path} has wrong Nside after degradation: {hp.get_nside(hp_map)}")
    if args.remove_low_modes:
        hp_map = remove_real_map_low_modes(hp_map, mask=mask_map)
    cache[key] = np.asarray(hp_map, dtype=np.float64)
    return cache[key]


def build_fit_payload(
    hp_map: np.ndarray,
    mask_map: np.ndarray,
    row: dict[str, Any],
    *,
    projection: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Extract and fit one candidate-centered patch in the requested projection."""

    record = centered_record(row)
    patch_glon = float(record["patch_center_glon_deg"])
    patch_glat = float(record["patch_center_glat_deg"])
    if projection == "gnomonic":
        mask_patch = np.asarray(project_patch(mask_map, patch_glon, patch_glat), dtype=np.float64)
        theta_grid = None
        patch = np.asarray(project_patch(hp_map, patch_glon, patch_glat), dtype=np.float64)
    elif projection == "equal_area":
        mask_patch, theta_grid = equal_area_project_patch(mask_map, patch_glon, patch_glat)
        patch, theta_grid = equal_area_project_patch(hp_map, patch_glon, patch_glat)
    else:
        raise ValueError(f"Unsupported projection: {projection}")

    valid_patch = np.isfinite(mask_patch) & (mask_patch > 0.5)
    valid_fraction = float(projected_unmasked_fraction(mask_patch))
    result: dict[str, Any] = {
        "projection": projection,
        "patch_valid_fraction": valid_fraction,
    }
    if valid_fraction < float(args.min_valid_fraction):
        result.update(
            {
                "fit_status": "skipped_invalid_mask_fraction",
                "reason": (
                    f"Projected valid fraction {valid_fraction:.4f} is below "
                    f"--min-valid-fraction {float(args.min_valid_fraction):.4f}."
                ),
            }
        )
        return result

    patch = np.asarray(patch, dtype=np.float64)
    patch[~valid_patch] = 0.0
    if projection == "gnomonic":
        theta_gnomonic = make_angular_distance_grid(
            PATCH_PIX,
            RESO_ARCMIN,
            center_x_pix=float(record["candidate_x_pix"]),
            center_y_pix=float(record["candidate_y_pix"]),
        )
        fit_row = fit_candidate_with_theta_grid(patch, theta_gnomonic, record, args)
    else:
        fit_row = fit_candidate_with_theta_grid(patch, theta_grid, record, args)
    result.update(fit_row)
    return result


def compare_fits(
    gnomonic_row: dict[str, Any],
    equal_area_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Compare gnomonic and equal-area fits for one surviving member."""

    status = "compare_failed"
    failures: list[str] = []
    if str(gnomonic_row.get("fit_status")) != "fit":
        failures.append(f"gnomonic_status={gnomonic_row.get('fit_status')}")
    if str(equal_area_row.get("fit_status")) != "fit":
        failures.append(f"equal_area_status={equal_area_row.get('fit_status')}")

    theta_shift = None
    z0_retention = None
    z0_sign_consistent = None
    delta_chi2_retention = None
    geometry_consistent = None
    if not failures:
        theta_shift = abs(
            float(equal_area_row["theta_crit_fit_deg"]) - float(gnomonic_row["theta_crit_fit_deg"])
        )
        gnomonic_abs_z0 = abs(float(gnomonic_row["z0_fit"]))
        equal_abs_z0 = abs(float(equal_area_row["z0_fit"]))
        z0_retention = equal_abs_z0 / max(gnomonic_abs_z0, 1.0e-12)
        z0_sign_consistent = sign_label(equal_area_row["z0_fit"]) == sign_label(gnomonic_row["z0_fit"])
        delta_chi2_retention = float(equal_area_row["delta_chi2_vs_plane_null"]) / max(
            float(gnomonic_row["delta_chi2_vs_plane_null"]),
            1.0e-12,
        )
        geometry_consistent = bool(
            theta_shift <= float(args.theta_shift_threshold_deg)
            and float(args.z0_retention_min) <= z0_retention <= float(args.z0_retention_max)
            and z0_sign_consistent
        )
        status = "projection_consistent" if geometry_consistent else "projection_shifted"

    return {
        "status": status,
        "failures": failures,
        "theta_shift_deg": theta_shift,
        "z0_retention": z0_retention,
        "z0_sign_consistent": z0_sign_consistent,
        "delta_chi2_retention": delta_chi2_retention,
        "geometry_consistent": geometry_consistent,
        "theta_shift_threshold_deg": float(args.theta_shift_threshold_deg),
        "z0_retention_min": float(args.z0_retention_min),
        "z0_retention_max": float(args.z0_retention_max),
    }


def build_member_result(
    system_row: dict[str, Any],
    handoff_row: dict[str, Any],
    *,
    hp_map: np.ndarray,
    mask_map: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the projection-robust follow-up record for one surviving member."""

    gnomonic_row = build_fit_payload(hp_map, mask_map, handoff_row, projection="gnomonic", args=args)
    equal_area_row = build_fit_payload(hp_map, mask_map, handoff_row, projection="equal_area", args=args)
    comparison = compare_fits(gnomonic_row, equal_area_row, args)
    return {
        "candidate_id": candidate_id(handoff_row),
        "system_rank": int(system_row["system_rank"]),
        "system_status": str(system_row["system_status"]),
        "map": str(handoff_row["map"]).lower(),
        "patch_index": int(handoff_row["patch_index"]),
        "global_cluster_rank": int(handoff_row["global_cluster_rank"]),
        "candidate_glon_deg": float(handoff_row["candidate_glon_deg"]),
        "candidate_glat_deg": float(handoff_row["candidate_glat_deg"]),
        "screening_priority_tier": str(handoff_row.get("screening_priority_tier", "")),
        "screening_pooled_bh_q": finite_float(handoff_row.get("screening_pooled_bh_q")),
        "real_sky_followup_status": str(handoff_row.get("real_sky_followup_status", "")),
        "hm_signflip_p_value": finite_float(handoff_row.get("hm_signflip_followup", {}).get("empirical_p_value")),
        "frequency_jackknife_stable": bool(handoff_row.get("frequency_jackknife_followup", {}).get("stable")),
        "projection_systematics_caution": bool(handoff_row.get("projection_systematics_caution", False)),
        "projection_followup_severity": str(handoff_row.get("projection_followup", {}).get("severity", "none")),
        "original_template_seed": {
            "theta_crit_deg": finite_float(handoff_row.get("template_seed", {}).get("theta_crit_deg")),
            "z0": finite_float(handoff_row.get("template_seed", {}).get("z0")),
            "zcrit": finite_float(handoff_row.get("template_seed", {}).get("zcrit")),
            "candidate_offset_deg": finite_float(handoff_row.get("template_seed", {}).get("candidate_offset_deg")),
        },
        "candidate_centered_gnomonic_fit": gnomonic_row,
        "candidate_centered_equal_area_fit": equal_area_row,
        "projection_comparison": comparison,
    }


def summarize_system(
    system_row: dict[str, Any],
    member_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize projection-robust follow-up for one adaptive system."""

    representative = min(
        member_rows,
        key=lambda row: (
            float(row.get("screening_pooled_bh_q", np.inf) or np.inf),
            int(row.get("global_cluster_rank", 10**9)),
        ),
    )
    consistent_ids = [
        row["candidate_id"]
        for row in member_rows
        if bool(row.get("projection_comparison", {}).get("geometry_consistent"))
    ]
    if consistent_ids:
        projection_status = "projection_robust_ready"
    elif any(not row.get("projection_comparison", {}).get("failures") for row in member_rows):
        projection_status = "projection_shift_detected"
    else:
        projection_status = "projection_followup_failed"
    return {
        "system_rank": int(system_row["system_rank"]),
        "system_status": str(system_row["system_status"]),
        "representative_candidate_id": str(representative["candidate_id"]),
        "representative_map": str(representative["map"]),
        "representative_patch_index": int(representative["patch_index"]),
        "representative_glon_deg": float(representative["candidate_glon_deg"]),
        "representative_glat_deg": float(representative["candidate_glat_deg"]),
        "num_surviving_members": int(len(member_rows)),
        "surviving_member_ids": [row["candidate_id"] for row in member_rows],
        "projection_robust_status": projection_status,
        "geometry_consistent_member_ids": consistent_ids,
        "representative_equal_area_theta_crit_deg": finite_float(
            representative.get("candidate_centered_equal_area_fit", {}).get("theta_crit_fit_deg")
        ),
        "representative_equal_area_z0": finite_float(
            representative.get("candidate_centered_equal_area_fit", {}).get("z0_fit")
        ),
        "representative_equal_area_zcrit": finite_float(
            representative.get("candidate_centered_equal_area_fit", {}).get("zcrit_fit")
        ),
        "representative_theta_shift_deg": finite_float(
            representative.get("projection_comparison", {}).get("theta_shift_deg")
        ),
        "representative_z0_retention": finite_float(
            representative.get("projection_comparison", {}).get("z0_retention")
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a flat CSV summary for the member rows."""

    flat_rows = [
        {
            "candidate_id": row["candidate_id"],
            "system_rank": row["system_rank"],
            "map": row["map"],
            "patch_index": row["patch_index"],
            "global_cluster_rank": row["global_cluster_rank"],
            "screening_priority_tier": row["screening_priority_tier"],
            "screening_pooled_bh_q": row["screening_pooled_bh_q"],
            "hm_signflip_p_value": row["hm_signflip_p_value"],
            "frequency_jackknife_stable": row["frequency_jackknife_stable"],
            "projection_systematics_caution": row["projection_systematics_caution"],
            "gnomonic_theta_crit_deg": finite_float(row["candidate_centered_gnomonic_fit"].get("theta_crit_fit_deg")),
            "equal_area_theta_crit_deg": finite_float(row["candidate_centered_equal_area_fit"].get("theta_crit_fit_deg")),
            "gnomonic_z0": finite_float(row["candidate_centered_gnomonic_fit"].get("z0_fit")),
            "equal_area_z0": finite_float(row["candidate_centered_equal_area_fit"].get("z0_fit")),
            "theta_shift_deg": finite_float(row["projection_comparison"].get("theta_shift_deg")),
            "z0_retention": finite_float(row["projection_comparison"].get("z0_retention")),
            "projection_compare_status": row["projection_comparison"]["status"],
        }
        for row in rows
    ]
    if not flat_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    system_rows: list[dict[str, Any]],
    member_rows: list[dict[str, Any]],
) -> None:
    """Write a paper-facing markdown report."""

    lines = ["# Projection-Robust Survivor Follow-Up", ""]
    lines.append("## Assumptions")
    lines.append("")
    for note in summary["assumption_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- `num_surviving_systems`: `{summary['num_surviving_systems']}`")
    lines.append(f"- `num_surviving_members`: `{summary['num_surviving_members']}`")
    lines.append(f"- `projection_robust_status_counts`: `{summary['projection_robust_status_counts']}`")
    lines.append("")
    lines.append("## Ranked Systems")
    lines.append("")
    lines.append("| system | rep id | projection status | equal-area theta | equal-area z0 | theta shift | z0 retention | consistent members |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    for row in system_rows:
        members = ", ".join(row["geometry_consistent_member_ids"]) if row["geometry_consistent_member_ids"] else "none"
        theta = 0.0 if row["representative_equal_area_theta_crit_deg"] is None else row["representative_equal_area_theta_crit_deg"]
        z0 = 0.0 if row["representative_equal_area_z0"] is None else row["representative_equal_area_z0"]
        theta_shift = 0.0 if row["representative_theta_shift_deg"] is None else row["representative_theta_shift_deg"]
        retention = 0.0 if row["representative_z0_retention"] is None else row["representative_z0_retention"]
        lines.append(
            f"| {row['system_rank']} | {row['representative_candidate_id']} | {row['projection_robust_status']} | "
            f"{theta:.2f} | {z0:.3e} | {theta_shift:.2f} | {retention:.3f} | {members} |"
        )
    lines.append("")
    lines.append("## Surviving Members")
    lines.append("")
    lines.append("| id | system | HM p | freq stable | gnomonic theta | equal-area theta | theta shift | gnomonic z0 | equal-area z0 | compare |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in member_rows:
        g_theta = finite_float(row["candidate_centered_gnomonic_fit"].get("theta_crit_fit_deg")) or 0.0
        e_theta = finite_float(row["candidate_centered_equal_area_fit"].get("theta_crit_fit_deg")) or 0.0
        g_z0 = finite_float(row["candidate_centered_gnomonic_fit"].get("z0_fit")) or 0.0
        e_z0 = finite_float(row["candidate_centered_equal_area_fit"].get("z0_fit")) or 0.0
        theta_shift = finite_float(row["projection_comparison"].get("theta_shift_deg")) or 0.0
        lines.append(
            f"| {row['candidate_id']} | {row['system_rank']} | {float(row['hm_signflip_p_value'] or np.nan):.4f} | "
            f"{row['frequency_jackknife_stable']} | {g_theta:.2f} | {e_theta:.2f} | {theta_shift:.2f} | "
            f"{g_z0:.3e} | {e_z0:.3e} | {row['projection_comparison']['status']} |"
        )
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- These are deterministic follow-up fits on the surviving screening targets, not posterior constraints.")
    lines.append("- Use the equal-area result only as a geometry-robust seed check before any downstream likelihood statement.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the projection-robust survivor follow-up."""

    args = parse_args()
    validate_args(args)
    handoff_path = Path(args.handoff_jsonl).expanduser().resolve()
    systems_path = Path(args.systems_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    handoff_rows = load_jsonl(handoff_path)
    handoff_index = index_handoff_rows(handoff_rows)
    systems = load_json(systems_path).get("systems", [])
    survivor_systems = [
        row for row in systems if str(row.get("system_status")) == "survives_real_sky_followup"
    ]
    if int(args.max_systems) > 0:
        survivor_systems = survivor_systems[: int(args.max_systems)]
    if not survivor_systems:
        raise ValueError("No surviving adaptive systems found in the input systems JSON.")

    mask_map = load_analysis_mask(args)
    map_cache: dict[str, np.ndarray] = {}
    member_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []

    for system_row in survivor_systems:
        system_member_rows: list[dict[str, Any]] = []
        for member_id in system_row.get("surviving_member_ids", []):
            if member_id not in handoff_index:
                raise KeyError(f"Surviving member {member_id!r} is missing from the handoff JSONL.")
            handoff_row = handoff_index[member_id]
            hp_map = prepare_cleaned_map(args, str(handoff_row["map"]), mask_map, map_cache)
            result = build_member_result(system_row, handoff_row, hp_map=hp_map, mask_map=mask_map, args=args)
            member_rows.append(result)
            system_member_rows.append(result)
        system_rows.append(summarize_system(system_row, system_member_rows))

    summary = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "handoff_jsonl": str(handoff_path),
        "systems_json": str(systems_path),
        "cleaned_map_template": str(Path(str(args.cleaned_map_template)).expanduser().resolve()),
        "num_surviving_systems": int(len(system_rows)),
        "num_surviving_members": int(len(member_rows)),
        "projection_robust_status_counts": {
            key: int(sum(str(row["projection_robust_status"]) == key for row in system_rows))
            for key in sorted({str(row["projection_robust_status"]) for row in system_rows})
        },
        "assumption_notes": [
            "Only adaptive systems that already survive HM sign-flip and frequency jackknife are rerun here.",
            "Equal-area re-extraction is a lower-distortion geometry check, not a replacement for native-sphere evidence.",
            "The deterministic Feeney-style fit is reused on both projections so fit shifts are attributable to geometry rather than model-family changes.",
        ],
    }

    json_path = output_dir / "projection_robust_survivor_followup.json"
    member_jsonl = output_dir / "projection_robust_surviving_members.jsonl"
    system_jsonl = output_dir / "projection_robust_surviving_systems.jsonl"
    csv_path = output_dir / "projection_robust_surviving_members.csv"
    md_path = output_dir / "projection_robust_survivor_followup.md"
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "systems": system_rows,
                "members": member_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_jsonl(member_jsonl, member_rows)
    write_jsonl(system_jsonl, system_rows)
    write_csv(csv_path, member_rows)
    write_markdown(md_path, summary, system_rows, member_rows)
    print(
        json.dumps(
            {
                "output_json": str(json_path),
                "output_member_jsonl": str(member_jsonl),
                "output_system_jsonl": str(system_jsonl),
                "output_csv": str(csv_path),
                "output_markdown": str(md_path),
                "num_surviving_systems": int(len(system_rows)),
                "num_surviving_members": int(len(member_rows)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
