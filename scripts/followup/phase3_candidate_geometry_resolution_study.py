"""Resolve center and mask geometry stability for one screened survivor.

Assumptions
-----------
* This study evaluates a deterministic follow-up target from the screening
  pipeline. It does not produce a cosmological posterior or evidence ratio.
* The same local Feeney-style template fit is applied after re-extraction on
  gnomonic and Lambert azimuthal equal-area patches so projection differences
  are isolated from fit-family changes.
* The key scientific question is whether a candidate is stable under reasonable
  center and mask choices. A target that only survives after relaxing mask
  support or by moving far from the screening candidate center remains
  screening-only.
* Center separations are measured on the sphere with ``astropy`` sky geometry,
  and all fit calculations use float64 arithmetic to avoid avoidable precision
  loss in low-contrast template-difference calculations.
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

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from phase3_projection_robust_survivor_followup import (
    DEFAULT_CLEANED_MAP_TEMPLATE,
    DEFAULT_COMMON_MASK,
    candidate_id,
    compare_fits,
    finite_float,
    load_jsonl,
    prepare_cleaned_map,
    build_fit_payload,
)
from phase5_half_mission_signflip_null import load_analysis_mask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HANDOFF_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_bayesian_template_handoff"
    / "bayesian_template_handoff.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_candidate_geometry_resolution"
)
DEFAULT_CANDIDATE_ID = "nilc:9628"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run a center- and mask-stability geometry study for one screened candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-id", type=str, default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--handoff-jsonl", type=str, default=str(DEFAULT_HANDOFF_JSONL))
    parser.add_argument("--cleaned-map-template", type=str, default=str(DEFAULT_CLEANED_MAP_TEMPLATE))
    parser.add_argument("--common-mask", type=str, default=str(DEFAULT_COMMON_MASK))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--acceptance-min-valid-grid", type=str, default="0.85,0.875,0.885,0.89,0.895,0.9")
    parser.add_argument("--mask-threshold-grid", type=str, default="0.7,0.8,0.85,0.9,0.95")
    parser.add_argument("--strict-mask-threshold", type=float, default=0.9)
    parser.add_argument("--strict-min-valid-fraction", type=float, default=0.9)
    parser.add_argument("--candidate-center-retention-block", type=float, default=0.25)
    parser.add_argument("--center-spread-factor-flag", type=float, default=3.0)
    parser.add_argument("--fits-field", type=int, default=0)
    parser.add_argument("--target-nside", type=int, default=256)
    parser.add_argument("--remove-low-modes", action="store_true")
    parser.add_argument("--radius-window-deg", type=float, default=3.0)
    parser.add_argument("--radius-step-deg", type=float, default=0.5)
    parser.add_argument("--min-radius-deg", type=float, default=5.0)
    parser.add_argument("--max-radius-deg", type=float, default=25.0)
    parser.add_argument("--support-extra-deg", type=float, default=5.0)
    parser.add_argument("--support-factor", type=float, default=1.5)
    parser.add_argument("--edge-sigma-deg", type=float, default=0.0)
    parser.add_argument("--theta-shift-threshold-deg", type=float, default=2.0)
    parser.add_argument("--z0-retention-min", type=float, default=0.5)
    parser.add_argument("--z0-retention-max", type=float, default=2.0)
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def parse_float_grid(text: str, *, label: str, lo: float, hi: float) -> np.ndarray:
    """Return a unique sorted float grid inside the closed interval ``[lo, hi]``."""

    values = np.asarray([float(part.strip()) for part in str(text).split(",") if part.strip()], dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"{label} must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite values.")
    if np.any(values < float(lo)) or np.any(values > float(hi)):
        raise ValueError(f"{label} must lie inside [{lo}, {hi}].")
    return np.unique(np.round(values, 6))


def validate_args(args: argparse.Namespace) -> None:
    """Validate required inputs and non-physical settings."""

    args.acceptance_min_valid_grid = parse_float_grid(
        args.acceptance_min_valid_grid,
        label="--acceptance-min-valid-grid",
        lo=0.0,
        hi=1.0,
    )
    args.mask_threshold_grid = parse_float_grid(
        args.mask_threshold_grid,
        label="--mask-threshold-grid",
        lo=0.0,
        hi=1.0,
    )
    if ":" not in str(args.candidate_id):
        raise ValueError("--candidate-id must have the form map:patch_index.")
    for label, value in (
        ("--strict-mask-threshold", args.strict_mask_threshold),
        ("--strict-min-valid-fraction", args.strict_min_valid_fraction),
        ("--candidate-center-retention-block", args.candidate_center_retention_block),
        ("--center-spread-factor-flag", args.center_spread_factor_flag),
        ("--radius-window-deg", args.radius_window_deg),
        ("--radius-step-deg", args.radius_step_deg),
        ("--min-radius-deg", args.min_radius_deg),
        ("--max-radius-deg", args.max_radius_deg),
        ("--support-extra-deg", args.support_extra_deg),
        ("--support-factor", args.support_factor),
        ("--edge-sigma-deg", args.edge_sigma_deg),
        ("--theta-shift-threshold-deg", args.theta_shift_threshold_deg),
        ("--z0-retention-min", args.z0_retention_min),
        ("--z0-retention-max", args.z0_retention_max),
    ):
        if not np.isfinite(float(value)):
            raise ValueError(f"{label} must be finite.")
    if not (0.0 <= float(args.strict_mask_threshold) <= 1.0):
        raise ValueError("--strict-mask-threshold must lie in [0, 1].")
    if not (0.0 <= float(args.strict_min_valid_fraction) <= 1.0):
        raise ValueError("--strict-min-valid-fraction must lie in [0, 1].")
    if float(args.candidate_center_retention_block) <= 0.0:
        raise ValueError("--candidate-center-retention-block must be positive.")
    if float(args.center_spread_factor_flag) < 1.0:
        raise ValueError("--center-spread-factor-flag must be >= 1.")
    handoff_path = Path(str(args.handoff_jsonl)).expanduser()
    if not handoff_path.exists():
        raise FileNotFoundError(f"Missing handoff JSONL: {handoff_path}")
    if int(args.target_nside) <= 0:
        raise ValueError("--target-nside must be positive.")
    if int(args.plot_dpi) <= 0:
        raise ValueError("--plot-dpi must be positive.")


def separation_deg(coord_a: SkyCoord, coord_b: SkyCoord) -> float:
    """Return the great-circle separation in degrees."""

    return float(coord_a.separation(coord_b).to_value(u.deg))


def coord_from_lonlat(glon_deg: float, glat_deg: float) -> SkyCoord:
    """Return a Galactic sky coordinate."""

    if float(glat_deg) < -90.0 or float(glat_deg) > 90.0:
        raise ValueError(f"Non-physical Galactic latitude {glat_deg:.6f} deg.")
    return SkyCoord(l=float(glon_deg) * u.deg, b=float(glat_deg) * u.deg, frame="galactic")


def load_candidate_row(args: argparse.Namespace) -> dict[str, Any]:
    """Load the requested candidate from the handoff packet."""

    rows = load_jsonl(Path(str(args.handoff_jsonl)).expanduser().resolve())
    indexed = {candidate_id(row): row for row in rows}
    if str(args.candidate_id) not in indexed:
        raise KeyError(f"Candidate {args.candidate_id!r} was not found in {args.handoff_jsonl}.")
    row = indexed[str(args.candidate_id)]
    if str(row["map"]).lower() != "nilc":
        raise ValueError(
            "This study is intended for the NILC survivor geometry case; "
            f"requested candidate has map {row['map']!r}."
        )
    return row


def center_specs(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the center choices tested for the candidate."""

    candidate_coord = coord_from_lonlat(float(row["candidate_glon_deg"]), float(row["candidate_glat_deg"]))
    tile_coord = coord_from_lonlat(float(row["patch_center_glon_deg"]), float(row["patch_center_glat_deg"]))
    seed_glon = finite_float(row.get("template_seed", {}).get("seed_glon_deg"))
    seed_glat = finite_float(row.get("template_seed", {}).get("seed_glat_deg"))
    if seed_glon is None or seed_glat is None:
        raise ValueError(f"{candidate_id(row)} is missing finite template-seed coordinates.")
    seed_coord = coord_from_lonlat(seed_glon, seed_glat)

    specs = [
        {
            "center_mode": "candidate_center",
            "center_label": "screening candidate center",
            "center_glon_deg": float(row["candidate_glon_deg"]),
            "center_glat_deg": float(row["candidate_glat_deg"]),
            "coord": candidate_coord,
        },
        {
            "center_mode": "tile_center",
            "center_label": "original tile center",
            "center_glon_deg": float(row["patch_center_glon_deg"]),
            "center_glat_deg": float(row["patch_center_glat_deg"]),
            "coord": tile_coord,
        },
        {
            "center_mode": "template_seed_center",
            "center_label": "template-seed center",
            "center_glon_deg": seed_glon,
            "center_glat_deg": seed_glat,
            "coord": seed_coord,
        },
    ]
    for spec in specs:
        spec["separation_to_candidate_deg"] = separation_deg(spec["coord"], candidate_coord)
        spec["separation_to_tile_deg"] = separation_deg(spec["coord"], tile_coord)
        spec["separation_to_seed_deg"] = separation_deg(spec["coord"], seed_coord)
    return specs


def projection_pair_record(
    row: dict[str, Any],
    center_spec: dict[str, Any],
    *,
    hp_map: np.ndarray,
    mask_map: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate one center choice on gnomonic and equal-area projections."""

    probe = dict(row)
    probe["candidate_glon_deg"] = float(center_spec["center_glon_deg"])
    probe["candidate_glat_deg"] = float(center_spec["center_glat_deg"])
    gnomonic = build_fit_payload(hp_map, mask_map, probe, projection="gnomonic", args=args)
    equal_area = build_fit_payload(hp_map, mask_map, probe, projection="equal_area", args=args)
    comparison = compare_fits(gnomonic, equal_area, args)

    seed_z0 = abs(float(row["template_seed"]["z0"]))
    seed_delta = float(row["template_seed"]["delta_chi2_vs_plane_null"])
    if seed_z0 <= 0.0 or not np.isfinite(seed_z0):
        raise ValueError(f"{candidate_id(row)} has non-physical template seed z0.")
    if seed_delta <= 0.0 or not np.isfinite(seed_delta):
        raise ValueError(f"{candidate_id(row)} has non-physical template seed delta chi2.")

    def project_summary(proj_row: dict[str, Any]) -> dict[str, Any]:
        z0 = finite_float(proj_row.get("z0_fit"))
        delta_chi2 = finite_float(proj_row.get("delta_chi2_vs_plane_null"))
        return {
            "patch_valid_fraction": float(proj_row["patch_valid_fraction"]),
            "fit_status": str(proj_row.get("fit_status")),
            "reason": proj_row.get("reason"),
            "theta_crit_fit_deg": finite_float(proj_row.get("theta_crit_fit_deg")),
            "z0_fit": z0,
            "zcrit_fit": finite_float(proj_row.get("zcrit_fit")),
            "delta_chi2_vs_plane_null": delta_chi2,
            "z0_retention_to_seed": (
                float(abs(z0) / seed_z0) if z0 is not None else None
            ),
            "delta_chi2_retention_to_seed": (
                float(delta_chi2 / seed_delta) if delta_chi2 is not None else None
            ),
        }

    return {
        "center_mode": str(center_spec["center_mode"]),
        "center_label": str(center_spec["center_label"]),
        "center_glon_deg": float(center_spec["center_glon_deg"]),
        "center_glat_deg": float(center_spec["center_glat_deg"]),
        "separation_to_candidate_deg": float(center_spec["separation_to_candidate_deg"]),
        "separation_to_tile_deg": float(center_spec["separation_to_tile_deg"]),
        "separation_to_seed_deg": float(center_spec["separation_to_seed_deg"]),
        "gnomonic": project_summary(gnomonic),
        "equal_area": project_summary(equal_area),
        "projection_comparison": comparison,
    }


def load_maps_for_mask_threshold(
    args: argparse.Namespace,
    row: dict[str, Any],
    mask_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the analysis mask and cleaned map for one binary mask threshold."""

    threshold_args = argparse.Namespace(**vars(args))
    threshold_args.mask_threshold = float(mask_threshold)
    mask_map = load_analysis_mask(threshold_args)
    hp_map = prepare_cleaned_map(threshold_args, str(row["map"]).lower(), mask_map, {})
    return mask_map, hp_map


def run_acceptance_scan(
    row: dict[str, Any],
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Scan the follow-up acceptance cutoff at fixed binary mask threshold."""

    mask_map, hp_map = load_maps_for_mask_threshold(args, row, float(args.strict_mask_threshold))
    records = []
    for min_valid_fraction in args.acceptance_min_valid_grid:
        trial_args = argparse.Namespace(**vars(args))
        trial_args.mask_threshold = float(args.strict_mask_threshold)
        trial_args.min_valid_fraction = float(min_valid_fraction)
        for spec in specs:
            pair = projection_pair_record(row, spec, hp_map=hp_map, mask_map=mask_map, args=trial_args)
            records.append(
                {
                    "scan_family": "acceptance_min_valid_fraction",
                    "scan_value": float(min_valid_fraction),
                    "mask_threshold": float(args.strict_mask_threshold),
                    "min_valid_fraction": float(min_valid_fraction),
                    **pair,
                }
            )
    return records


def run_mask_threshold_scan(
    row: dict[str, Any],
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Scan the binary analysis-mask threshold at fixed follow-up cutoff."""

    records = []
    for mask_threshold in args.mask_threshold_grid:
        trial_args = argparse.Namespace(**vars(args))
        trial_args.mask_threshold = float(mask_threshold)
        trial_args.min_valid_fraction = float(args.strict_min_valid_fraction)
        mask_map, hp_map = load_maps_for_mask_threshold(trial_args, row, float(mask_threshold))
        for spec in specs:
            pair = projection_pair_record(row, spec, hp_map=hp_map, mask_map=mask_map, args=trial_args)
            records.append(
                {
                    "scan_family": "mask_threshold",
                    "scan_value": float(mask_threshold),
                    "mask_threshold": float(mask_threshold),
                    "min_valid_fraction": float(args.strict_min_valid_fraction),
                    **pair,
                }
            )
    return records


def pair_lookup(
    rows: list[dict[str, Any]],
    *,
    center_mode: str,
    scan_family: str,
    scan_value: float,
) -> dict[str, Any]:
    """Return the unique record matching one center and scan point."""

    for row in rows:
        if (
            str(row["center_mode"]) == str(center_mode)
            and str(row["scan_family"]) == str(scan_family)
            and abs(float(row["scan_value"]) - float(scan_value)) <= 1.0e-9
        ):
            return row
    raise KeyError(
        f"No study row for center_mode={center_mode!r}, "
        f"scan_family={scan_family!r}, scan_value={scan_value!r}."
    )


def latest_candidate_center_success(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the strictest candidate-centered acceptance row with both fits available."""

    candidate_rows = [
        row
        for row in rows
        if str(row["scan_family"]) == "acceptance_min_valid_fraction"
        and str(row["center_mode"]) == "candidate_center"
        and str(row["gnomonic"]["fit_status"]) == "fit"
        and str(row["equal_area"]["fit_status"]) == "fit"
    ]
    if not candidate_rows:
        return None
    return max(candidate_rows, key=lambda row: float(row["min_valid_fraction"]))


def common_center_stability_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the strictest acceptance row where every center has both fits."""

    accepted_values = sorted(
        {
            float(row["min_valid_fraction"])
            for row in rows
            if str(row["scan_family"]) == "acceptance_min_valid_fraction"
        }
    )
    center_modes = sorted({str(row["center_mode"]) for row in rows})
    best_value = None
    for min_valid_fraction in accepted_values:
        ok = True
        for center_mode in center_modes:
            row = pair_lookup(
                rows,
                center_mode=center_mode,
                scan_family="acceptance_min_valid_fraction",
                scan_value=float(min_valid_fraction),
            )
            ok = ok and str(row["gnomonic"]["fit_status"]) == "fit" and str(row["equal_area"]["fit_status"]) == "fit"
        if ok:
            best_value = float(min_valid_fraction)
    if best_value is None:
        return None

    summaries = []
    for center_mode in center_modes:
        row = pair_lookup(
            rows,
            center_mode=center_mode,
            scan_family="acceptance_min_valid_fraction",
            scan_value=best_value,
        )
        median_retention = float(
            np.median(
                [
                    float(row["gnomonic"]["z0_retention_to_seed"]),
                    float(row["equal_area"]["z0_retention_to_seed"]),
                ]
            )
        )
        summaries.append(
            {
                "center_mode": center_mode,
                "center_label": row["center_label"],
                "median_z0_retention_to_seed": median_retention,
            }
        )
    values = np.asarray([item["median_z0_retention_to_seed"] for item in summaries], dtype=np.float64)
    spread_factor = float(np.max(values) / max(np.min(values), 1.0e-30))
    return {
        "min_valid_fraction": best_value,
        "center_retention_rows": summaries,
        "median_z0_retention_spread_factor": spread_factor,
    }


def decision_summary(
    row: dict[str, Any],
    study_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return a paper-facing decision summary for the candidate."""

    strict_candidate = pair_lookup(
        study_rows,
        center_mode="candidate_center",
        scan_family="acceptance_min_valid_fraction",
        scan_value=float(args.strict_min_valid_fraction),
    )
    strict_tile = pair_lookup(
        study_rows,
        center_mode="tile_center",
        scan_family="acceptance_min_valid_fraction",
        scan_value=float(args.strict_min_valid_fraction),
    )
    strict_seed = pair_lookup(
        study_rows,
        center_mode="template_seed_center",
        scan_family="acceptance_min_valid_fraction",
        scan_value=float(args.strict_min_valid_fraction),
    )
    candidate_relaxed = latest_candidate_center_success(study_rows)
    common_stability = common_center_stability_row(study_rows)

    max_candidate_retention = None
    if candidate_relaxed is not None:
        max_candidate_retention = float(
            max(
                candidate_relaxed["gnomonic"]["z0_retention_to_seed"] or 0.0,
                candidate_relaxed["equal_area"]["z0_retention_to_seed"] or 0.0,
            )
        )

    center_spread_flag = (
        common_stability is not None
        and float(common_stability["median_z0_retention_spread_factor"]) >= float(args.center_spread_factor_flag)
    )
    tile_stable = bool(strict_tile["projection_comparison"]["geometry_consistent"])
    seed_stable = bool(strict_seed["projection_comparison"]["geometry_consistent"])
    candidate_strict_failed = bool(strict_candidate["projection_comparison"]["failures"])

    if (
        candidate_strict_failed
        and candidate_relaxed is not None
        and max_candidate_retention is not None
        and max_candidate_retention < float(args.candidate_center_retention_block)
        and tile_stable
        and seed_stable
        and center_spread_flag
    ):
        status = "center_unstable_screening_only"
        reason = (
            "Candidate-centered follow-up fails at the strict 0.90 support cutoff, "
            "and when relaxed enough to fit, the recovered amplitude is only a small "
            "fraction of the screening seed while tile- and seed-centered fits remain "
            "projection-consistent but disagree strongly in amplitude."
        )
        action = (
            "Keep this system out of the downstream likelihood handoff and treat it "
            "as a geometry-limited screening excess until a native-sphere center-marginalized "
            "follow-up exists."
        )
    else:
        status = "needs_manual_review"
        reason = (
            "The automated geometry-resolution logic did not satisfy the full center-instability "
            "signature; inspect the scan tables directly before changing the candidate status."
        )
        action = "Do not promote automatically."

    return {
        "candidate_id": candidate_id(row),
        "status": status,
        "reason": reason,
        "recommended_action": action,
        "strict_candidate_center": strict_candidate,
        "strict_tile_center": strict_tile,
        "strict_template_seed_center": strict_seed,
        "candidate_relaxed_latest_success": candidate_relaxed,
        "common_center_stability": common_stability,
    }


def flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested study rows for CSV export."""

    flat = []
    for row in rows:
        out = {
            "scan_family": row["scan_family"],
            "scan_value": row["scan_value"],
            "mask_threshold": row["mask_threshold"],
            "min_valid_fraction": row["min_valid_fraction"],
            "center_mode": row["center_mode"],
            "center_label": row["center_label"],
            "center_glon_deg": row["center_glon_deg"],
            "center_glat_deg": row["center_glat_deg"],
            "separation_to_candidate_deg": row["separation_to_candidate_deg"],
            "separation_to_tile_deg": row["separation_to_tile_deg"],
            "separation_to_seed_deg": row["separation_to_seed_deg"],
            "compare_status": row["projection_comparison"]["status"],
            "theta_shift_deg": finite_float(row["projection_comparison"].get("theta_shift_deg")),
            "pair_z0_retention": finite_float(row["projection_comparison"].get("z0_retention")),
            "pair_delta_chi2_retention": finite_float(row["projection_comparison"].get("delta_chi2_retention")),
        }
        for projection in ("gnomonic", "equal_area"):
            prefix = "g" if projection == "gnomonic" else "e"
            section = row[projection]
            out[f"{prefix}_patch_valid_fraction"] = section["patch_valid_fraction"]
            out[f"{prefix}_fit_status"] = section["fit_status"]
            out[f"{prefix}_theta_crit_fit_deg"] = section["theta_crit_fit_deg"]
            out[f"{prefix}_z0_fit"] = section["z0_fit"]
            out[f"{prefix}_zcrit_fit"] = section["zcrit_fit"]
            out[f"{prefix}_delta_chi2_vs_plane_null"] = section["delta_chi2_vs_plane_null"]
            out[f"{prefix}_z0_retention_to_seed"] = section["z0_retention_to_seed"]
            out[f"{prefix}_delta_chi2_retention_to_seed"] = section["delta_chi2_retention_to_seed"]
            out[f"{prefix}_reason"] = section["reason"]
        flat.append(out)
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a flat CSV table."""

    flat_rows = flatten_rows(rows)
    if not flat_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def plot_study(
    acceptance_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    path: Path,
    *,
    strict_min_valid_fraction: float,
    plot_dpi: int,
) -> None:
    """Write a compact visual summary of center and mask sensitivity."""

    plt.rcParams.update(
        {
            "figure.dpi": plot_dpi,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    colors = {
        "candidate_center": "#C44900",
        "tile_center": "#1E5AA8",
        "template_seed_center": "#2A7F62",
    }
    linestyles = {"gnomonic": "-", "equal_area": "--"}

    ax = axes[0]
    for center_mode in sorted({row["center_mode"] for row in acceptance_rows}):
        center_rows = sorted(
            [row for row in acceptance_rows if str(row["center_mode"]) == str(center_mode)],
            key=lambda row: float(row["min_valid_fraction"]),
        )
        x = np.asarray([float(row["min_valid_fraction"]) for row in center_rows], dtype=np.float64)
        for projection in ("gnomonic", "equal_area"):
            y = np.asarray(
                [
                    (
                        float(row[projection]["z0_retention_to_seed"])
                        if row[projection]["z0_retention_to_seed"] is not None
                        else np.nan
                    )
                    for row in center_rows
                ],
                dtype=np.float64,
            )
            ax.plot(
                x,
                y,
                color=colors[center_mode],
                linestyle=linestyles[projection],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=f"{center_mode} / {projection}",
            )
    ax.axhline(1.0, color="0.4", linewidth=1.0, linestyle=":")
    ax.axvline(float(strict_min_valid_fraction), color="0.4", linewidth=1.0, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("Follow-up min valid fraction")
    ax.set_ylabel(r"$|z_0| / |z_{0,\mathrm{seed}}|$")
    ax.set_title("Center Dependence at Fixed Mask Threshold")
    ax.legend(fontsize=7, ncol=1, frameon=False, loc="best")

    ax = axes[1]
    for center_mode in sorted({row["center_mode"] for row in mask_rows}):
        center_rows = sorted(
            [row for row in mask_rows if str(row["center_mode"]) == str(center_mode)],
            key=lambda row: float(row["mask_threshold"]),
        )
        x = np.asarray([float(row["mask_threshold"]) for row in center_rows], dtype=np.float64)
        for projection in ("gnomonic", "equal_area"):
            y = np.asarray(
                [float(row[projection]["patch_valid_fraction"]) for row in center_rows],
                dtype=np.float64,
            )
            ax.plot(
                x,
                y,
                color=colors[center_mode],
                linestyle=linestyles[projection],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=f"{center_mode} / {projection}",
            )
    ax.axhline(float(strict_min_valid_fraction), color="0.4", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Binary mask threshold")
    ax.set_ylabel("Projected valid fraction")
    ax.set_title("Mask-Support Stability at Strict Follow-Up Cut")
    ax.legend(fontsize=7, ncol=1, frameon=False, loc="best")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    path: Path,
    row: dict[str, Any],
    specs: list[dict[str, Any]],
    acceptance_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    plot_path: Path,
) -> None:
    """Write a paper-facing markdown summary."""

    lines = [f"# Candidate Geometry Resolution: {candidate_id(row)}", ""]
    lines.append("## Assumptions")
    lines.append("")
    lines.append("- This is a deterministic geometry-resolution study for one screened candidate, not a posterior or evidence calculation.")
    lines.append("- Gnomonic and equal-area fits reuse the same local template model so changes are attributable to geometry and center choice.")
    lines.append("- The target should only be promotable if the candidate-centered follow-up remains stable under the deployed mask guardrails.")
    lines.append("")
    lines.append("## Candidate Context")
    lines.append("")
    lines.append(f"- `screening candidate center`: `({float(row['candidate_glon_deg']):.3f}, {float(row['candidate_glat_deg']):.3f}) deg`")
    lines.append(f"- `original tile center`: `({float(row['patch_center_glon_deg']):.3f}, {float(row['patch_center_glat_deg']):.3f}) deg`")
    lines.append(
        f"- `template seed center`: `({float(row['template_seed']['seed_glon_deg']):.3f}, {float(row['template_seed']['seed_glat_deg']):.3f}) deg`"
    )
    lines.append(f"- `screening seed theta_crit_deg`: `{float(row['template_seed']['theta_crit_deg']):.2f}`")
    lines.append(f"- `screening seed z0`: `{float(row['template_seed']['z0']):.3e}`")
    lines.append(f"- `screening seed delta chi2`: `{float(row['template_seed']['delta_chi2_vs_plane_null']):.3e}`")
    lines.append("")
    lines.append("| center | separation to candidate | separation to tile | separation to seed |")
    lines.append("|---|---:|---:|---:|")
    for spec in specs:
        lines.append(
            f"| {spec['center_label']} | {float(spec['separation_to_candidate_deg']):.2f} | "
            f"{float(spec['separation_to_tile_deg']):.2f} | {float(spec['separation_to_seed_deg']):.2f} |"
        )
    lines.append("")

    lines.append("## Acceptance-Cut Scan")
    lines.append("")
    lines.append("| min valid frac | center | compare | g VF | e VF | g theta | e theta | g z0/seed | e z0/seed |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for row_i in sorted(acceptance_rows, key=lambda item: (float(item["min_valid_fraction"]), str(item["center_mode"]))):
        g_theta = row_i["gnomonic"]["theta_crit_fit_deg"]
        e_theta = row_i["equal_area"]["theta_crit_fit_deg"]
        g_ret = row_i["gnomonic"]["z0_retention_to_seed"]
        e_ret = row_i["equal_area"]["z0_retention_to_seed"]
        lines.append(
            f"| {float(row_i['min_valid_fraction']):.3f} | {row_i['center_label']} | {row_i['projection_comparison']['status']} | "
            f"{float(row_i['gnomonic']['patch_valid_fraction']):.4f} | {float(row_i['equal_area']['patch_valid_fraction']):.4f} | "
            f"{0.0 if g_theta is None else float(g_theta):.2f} | {0.0 if e_theta is None else float(e_theta):.2f} | "
            f"{0.0 if g_ret is None else float(g_ret):.3f} | {0.0 if e_ret is None else float(e_ret):.3f} |"
        )
    lines.append("")

    lines.append("## Mask-Threshold Scan")
    lines.append("")
    lines.append("| mask threshold | center | compare | g VF | e VF | g z0/seed | e z0/seed |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for row_i in sorted(mask_rows, key=lambda item: (float(item["mask_threshold"]), str(item["center_mode"]))):
        g_ret = row_i["gnomonic"]["z0_retention_to_seed"]
        e_ret = row_i["equal_area"]["z0_retention_to_seed"]
        lines.append(
            f"| {float(row_i['mask_threshold']):.2f} | {row_i['center_label']} | {row_i['projection_comparison']['status']} | "
            f"{float(row_i['gnomonic']['patch_valid_fraction']):.4f} | {float(row_i['equal_area']['patch_valid_fraction']):.4f} | "
            f"{0.0 if g_ret is None else float(g_ret):.3f} | {0.0 if e_ret is None else float(e_ret):.3f} |"
        )
    lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(f"- `status`: `{decision['status']}`")
    lines.append(f"- `reason`: {decision['reason']}")
    lines.append(f"- `recommended_action`: {decision['recommended_action']}")
    common = decision.get("common_center_stability")
    if common is not None:
        lines.append(
            f"- `strictest all-center fit cutoff`: `{float(common['min_valid_fraction']):.3f}` with "
            f"`median z0 retention spread factor = {float(common['median_z0_retention_spread_factor']):.2f}`"
        )
    relaxed = decision.get("candidate_relaxed_latest_success")
    if relaxed is not None:
        g_ret = relaxed["gnomonic"]["z0_retention_to_seed"]
        e_ret = relaxed["equal_area"]["z0_retention_to_seed"]
        lines.append(
            f"- `latest candidate-centered success`: `min_valid_fraction={float(relaxed['min_valid_fraction']):.3f}` "
            f"with `g z0/seed={float(g_ret):.3f}` and `e z0/seed={float(e_ret):.3f}`"
        )
    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append(f"- Plot written to `{plot_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the candidate geometry-resolution study."""

    args = parse_args()
    validate_args(args)

    row = load_candidate_row(args)
    specs = center_specs(row)
    acceptance_rows = run_acceptance_scan(row, specs, args)
    mask_rows = run_mask_threshold_scan(row, specs, args)
    all_rows = acceptance_rows + mask_rows
    decision = decision_summary(row, all_rows, args)

    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.candidate_id).replace(":", "_")
    json_path = output_dir / f"{stem}_geometry_resolution.json"
    csv_path = output_dir / f"{stem}_geometry_resolution.csv"
    md_path = output_dir / f"{stem}_geometry_resolution.md"
    plot_path = output_dir / f"{stem}_geometry_resolution.png"

    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "candidate_id": candidate_id(row),
        "candidate_context": {
            "map": str(row["map"]).lower(),
            "patch_index": int(row["patch_index"]),
            "global_cluster_rank": int(row["global_cluster_rank"]),
            "screening_priority_tier": str(row.get("screening_priority_tier", "")),
            "screening_pooled_bh_q": finite_float(row.get("screening_pooled_bh_q")),
            "hm_signflip_p_value": finite_float(row.get("hm_signflip_followup", {}).get("empirical_p_value")),
            "frequency_jackknife_stable": bool(row.get("frequency_jackknife_followup", {}).get("stable")),
            "template_seed": row["template_seed"],
        },
        "center_specs": [
            {
                key: value
                for key, value in spec.items()
                if key != "coord"
            }
            for spec in specs
        ],
        "acceptance_scan": acceptance_rows,
        "mask_threshold_scan": mask_rows,
        "decision": decision,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, all_rows)
    plot_study(
        acceptance_rows,
        mask_rows,
        plot_path,
        strict_min_valid_fraction=float(args.strict_min_valid_fraction),
        plot_dpi=int(args.plot_dpi),
    )
    write_markdown(md_path, row, specs, acceptance_rows, mask_rows, decision, plot_path)

    print(
        json.dumps(
            {
                "candidate_id": candidate_id(row),
                "decision_status": decision["status"],
                "output_json": str(json_path),
                "output_markdown": str(md_path),
                "output_plot": str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
