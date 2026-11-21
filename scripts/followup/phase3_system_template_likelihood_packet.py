"""Build a local template-likelihood packet for projection-robust-ready systems.

Assumptions
-----------
* Input systems have already survived HM sign-flip, frequency jackknife, and a
  projection-robust equal-area re-extraction gate.
* The likelihood proxy here is a local Gaussian-patch template fit with a
  plane-null nuisance model, using deterministic least-squares profiles. It is
  not a masked-sky Bayesian evidence ratio.
* Member maps within one system are not treated as statistically independent
  evidence sources for cosmological inference. Any combined profile is a
  conditional consensus proxy for downstream follow-up prioritization.
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

from phase2_generate_training import projected_unmasked_fraction
from phase5_half_mission_signflip_null import load_analysis_mask
from phase3_projection_robust_survivor_followup import (
    DEFAULT_CLEANED_MAP_TEMPLATE,
    DEFAULT_COMMON_MASK,
    build_fit_payload,
    candidate_id,
    equal_area_project_patch,
    finite_float,
    load_json,
    prepare_cleaned_map,
)
from phase3_template_fit_candidates import (
    build_template_columns,
    fit_least_squares,
    nuisance_design,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECTION_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_projection_robust_survivor_followup"
    / "projection_robust_survivor_followup.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_system_template_likelihood_packet"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build local equal-area template-likelihood packets for projection-robust-ready survivor systems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--projection-json", type=str, default=str(DEFAULT_PROJECTION_JSON))
    parser.add_argument("--cleaned-map-template", type=str, default=str(DEFAULT_CLEANED_MAP_TEMPLATE))
    parser.add_argument("--common-mask", type=str, default=str(DEFAULT_COMMON_MASK))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--theta-window-deg", type=float, default=3.0)
    parser.add_argument("--theta-step-deg", type=float, default=0.25)
    parser.add_argument("--max-systems", type=int, default=0)
    parser.add_argument("--fits-field", type=int, default=0)
    parser.add_argument("--target-nside", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.9)
    parser.add_argument("--min-valid-fraction", type=float, default=0.9)
    parser.add_argument("--remove-low-modes", action="store_true")
    parser.add_argument("--radius-window-deg", type=float, default=3.0)
    parser.add_argument("--radius-step-deg", type=float, default=0.5)
    parser.add_argument("--min-radius-deg", type=float, default=5.0)
    parser.add_argument("--max-radius-deg", type=float, default=25.0)
    parser.add_argument("--support-extra-deg", type=float, default=5.0)
    parser.add_argument("--support-factor", type=float, default=1.5)
    parser.add_argument("--edge-sigma-deg", type=float, default=0.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate non-physical parameters."""

    projection_path = Path(args.projection_json).expanduser()
    if not projection_path.exists():
        raise FileNotFoundError(f"Missing projection follow-up JSON: {projection_path}")
    for label, value in (
        ("--theta-window-deg", args.theta_window_deg),
        ("--theta-step-deg", args.theta_step_deg),
        ("--mask-threshold", args.mask_threshold),
        ("--min-valid-fraction", args.min_valid_fraction),
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
    if float(args.theta_window_deg) <= 0.0:
        raise ValueError("--theta-window-deg must be positive.")
    if float(args.theta_step_deg) <= 0.0:
        raise ValueError("--theta-step-deg must be positive.")
    if not (0.0 <= float(args.mask_threshold) <= 1.0):
        raise ValueError("--mask-threshold must lie in [0, 1].")
    if not (0.0 <= float(args.min_valid_fraction) <= 1.0):
        raise ValueError("--min-valid-fraction must lie in [0, 1].")
    if int(args.max_systems) < 0:
        raise ValueError("--max-systems must be non-negative.")


def theta_profile_grid(seed_theta_deg: float, args: argparse.Namespace) -> np.ndarray:
    """Return the theta profile grid around one seed."""

    lo = max(float(args.min_radius_deg), float(seed_theta_deg) - float(args.theta_window_deg))
    hi = min(float(args.max_radius_deg), float(seed_theta_deg) + float(args.theta_window_deg))
    grid = np.arange(lo, hi + 0.5 * float(args.theta_step_deg), float(args.theta_step_deg))
    grid = np.clip(grid, float(args.min_radius_deg), float(args.max_radius_deg))
    return np.unique(np.round(grid, 6))


def extract_equal_area_member_payload(
    member_row: dict[str, Any],
    *,
    hp_map: np.ndarray,
    mask_map: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Extract the equal-area patch, theta grid, and support mask for one member."""

    patch_glon = float(member_row["candidate_glon_deg"])
    patch_glat = float(member_row["candidate_glat_deg"])
    mask_patch, theta_grid = equal_area_project_patch(mask_map, patch_glon, patch_glat)
    valid_patch = np.isfinite(mask_patch) & (mask_patch > 0.5)
    valid_fraction = float(projected_unmasked_fraction(mask_patch))
    if valid_fraction < float(args.min_valid_fraction):
        raise ValueError(
            f"{candidate_id(member_row)} valid fraction {valid_fraction:.4f} is below "
            f"--min-valid-fraction {float(args.min_valid_fraction):.4f}."
        )
    patch, theta_grid = equal_area_project_patch(hp_map, patch_glon, patch_glat)
    patch = np.asarray(patch, dtype=np.float64)
    patch[~valid_patch] = 0.0
    return {
        "patch": patch,
        "theta_grid": np.asarray(theta_grid, dtype=np.float64),
        "valid_patch": valid_patch,
        "valid_fraction": valid_fraction,
    }


def profile_one_theta(
    patch: np.ndarray,
    theta_grid: np.ndarray,
    support: np.ndarray,
    theta_deg: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate one fixed-theta profile row and return fit metadata."""

    y = np.asarray(patch[support], dtype=np.float64)
    nuisance = nuisance_design(theta_grid, support)
    nuisance_coeff, null_sse, null_dof, _, _ = fit_least_squares(y, nuisance)
    basis_z0, basis_zcrit = build_template_columns(
        theta_grid=theta_grid,
        theta_crit_deg=float(theta_deg),
        support=support,
        edge_sigma_deg=float(args.edge_sigma_deg),
    )
    design = np.column_stack([nuisance, basis_z0, basis_zcrit])
    coeff, template_sse, template_dof, rank, singular_values = fit_least_squares(y, design)
    sigma2 = float(template_sse / max(template_dof, 1))
    xtx = np.asarray(design.T @ design, dtype=np.float64)
    cov = sigma2 * np.linalg.pinv(xtx)
    cov_signal = cov[-2:, -2:]
    se_z0 = float(np.sqrt(max(cov_signal[0, 0], 0.0)))
    se_zcrit = float(np.sqrt(max(cov_signal[1, 1], 0.0)))
    corr = float(
        cov_signal[0, 1]
        / max(np.sqrt(max(cov_signal[0, 0], 0.0) * max(cov_signal[1, 1], 0.0)), 1.0e-30)
    )
    return {
        "theta_crit_deg": float(theta_deg),
        "z0_fit": float(coeff[-2]),
        "zcrit_fit": float(coeff[-1]),
        "z0_se": se_z0,
        "zcrit_se": se_zcrit,
        "z0_zcrit_corr": corr,
        "template_sse": float(template_sse),
        "template_dof": int(template_dof),
        "template_reduced_sse": sigma2,
        "plane_null_sse": float(null_sse),
        "plane_null_dof": int(null_dof),
        "delta_chi2_vs_plane_null": float(null_sse - template_sse),
        "nuisance_offset_k": float(coeff[0]),
        "nuisance_x_slope_k": float(coeff[1]),
        "nuisance_y_slope_k": float(coeff[2]),
        "rank": int(rank),
        "condition_proxy": float(np.max(singular_values) / max(np.min(singular_values), 1.0e-30)),
        "support_pixels": int(np.count_nonzero(support)),
        "nuisance_plane_offset_k": float(nuisance_coeff[0]),
    }


def profile_member(
    member_row: dict[str, Any],
    *,
    hp_map: np.ndarray,
    mask_map: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return the equal-area profile packet for one surviving member."""

    payload = extract_equal_area_member_payload(member_row, hp_map=hp_map, mask_map=mask_map, args=args)
    theta_seed = finite_float(member_row["candidate_centered_equal_area_fit"].get("theta_crit_fit_deg"))
    if theta_seed is None or theta_seed <= 0.0:
        raise ValueError(f"{member_row['candidate_id']} is missing a positive equal-area theta seed.")
    theta_grid_deg = theta_profile_grid(theta_seed, args)

    support_radius = min(
        float(args.max_radius_deg) + float(args.support_extra_deg),
        max(float(theta_seed) + float(args.support_extra_deg), float(theta_seed) * float(args.support_factor)),
    )
    support = np.asarray(payload["valid_patch"], dtype=bool) & (
        np.degrees(payload["theta_grid"]) <= float(support_radius)
    )
    if int(np.count_nonzero(support)) < 32:
        raise ValueError(f"{member_row['candidate_id']} has insufficient support pixels for profiling.")

    profile_rows = [
        profile_one_theta(
            patch=payload["patch"],
            theta_grid=payload["theta_grid"],
            support=support,
            theta_deg=float(theta_deg),
            args=args,
        )
        for theta_deg in theta_grid_deg
    ]
    best_row = max(profile_rows, key=lambda row: float(row["delta_chi2_vs_plane_null"]))
    delta_values = np.asarray(
        [float(best_row["delta_chi2_vs_plane_null"]) - float(row["delta_chi2_vs_plane_null"]) for row in profile_rows],
        dtype=np.float64,
    )
    theta_arr = np.asarray([float(row["theta_crit_deg"]) for row in profile_rows], dtype=np.float64)
    theta_1sigma = theta_arr[delta_values <= 1.0]
    return {
        "candidate_id": str(member_row["candidate_id"]),
        "map": str(member_row["map"]),
        "patch_index": int(member_row["patch_index"]),
        "candidate_glon_deg": float(member_row["candidate_glon_deg"]),
        "candidate_glat_deg": float(member_row["candidate_glat_deg"]),
        "valid_fraction": float(payload["valid_fraction"]),
        "support_radius_deg": float(support_radius),
        "profile_rows": profile_rows,
        "best_profile_row": best_row,
        "theta_crit_1sigma_interval_deg": (
            [float(np.min(theta_1sigma)), float(np.max(theta_1sigma))]
            if theta_1sigma.size
            else None
        ),
    }


def system_consensus(member_packets: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a conservative consensus seed across surviving members."""

    theta_values = np.asarray(
        [float(packet["best_profile_row"]["theta_crit_deg"]) for packet in member_packets],
        dtype=np.float64,
    )
    z0_values = np.asarray(
        [float(packet["best_profile_row"]["z0_fit"]) for packet in member_packets],
        dtype=np.float64,
    )
    zcrit_values = np.asarray(
        [float(packet["best_profile_row"]["zcrit_fit"]) for packet in member_packets],
        dtype=np.float64,
    )
    delta_values = np.asarray(
        [float(packet["best_profile_row"]["delta_chi2_vs_plane_null"]) for packet in member_packets],
        dtype=np.float64,
    )
    return {
        "theta_crit_deg_median": float(np.median(theta_values)),
        "theta_crit_deg_min": float(np.min(theta_values)),
        "theta_crit_deg_max": float(np.max(theta_values)),
        "z0_median": float(np.median(z0_values)),
        "z0_min": float(np.min(z0_values)),
        "z0_max": float(np.max(z0_values)),
        "zcrit_median": float(np.median(zcrit_values)),
        "zcrit_min": float(np.min(zcrit_values)),
        "zcrit_max": float(np.max(zcrit_values)),
        "member_delta_chi2_median": float(np.median(delta_values)),
        "member_delta_chi2_min": float(np.min(delta_values)),
    }


def combine_theta_profiles(member_packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a conditional consensus profile by summing per-member delta chi2."""

    theta_sets = [
        tuple(float(row["theta_crit_deg"]) for row in packet["profile_rows"])
        for packet in member_packets
    ]
    if len(set(theta_sets)) != 1:
        raise ValueError("Member theta profile grids do not match.")
    combined_rows = []
    for row_idx in range(len(member_packets[0]["profile_rows"])):
        theta_value = float(member_packets[0]["profile_rows"][row_idx]["theta_crit_deg"])
        combined_rows.append(
            {
                "theta_crit_deg": theta_value,
                "sum_delta_chi2_proxy": float(
                    sum(
                        float(packet["profile_rows"][row_idx]["delta_chi2_vs_plane_null"])
                        for packet in member_packets
                    )
                ),
            }
        )
    return combined_rows


def write_profile_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a flat profile CSV."""

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, packet: dict[str, Any]) -> None:
    """Write a human-readable system packet summary."""

    lines = [f"# System Template-Likelihood Packet: {packet['system_id']}", ""]
    lines.append("## Assumptions")
    lines.append("")
    for note in packet["assumption_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Consensus Seed")
    lines.append("")
    consensus = packet["consensus_seed"]
    lines.append(f"- `theta_crit_deg_median`: `{consensus['theta_crit_deg_median']:.2f}`")
    lines.append(f"- `z0_median`: `{consensus['z0_median']:.3e}`")
    lines.append(f"- `zcrit_median`: `{consensus['zcrit_median']:.3e}`")
    lines.append("")
    lines.append("## Members")
    lines.append("")
    lines.append("| member | theta_best | theta_1sigma | z0 | z0_se | zcrit | zcrit_se | delta chi2 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for member in packet["members"]:
        best = member["best_profile_row"]
        interval = member.get("theta_crit_1sigma_interval_deg")
        interval_text = (
            f"[{interval[0]:.2f}, {interval[1]:.2f}]"
            if interval is not None
            else "n/a"
        )
        lines.append(
            f"| {member['candidate_id']} | {best['theta_crit_deg']:.2f} | {interval_text} | "
            f"{best['z0_fit']:.3e} | {best['z0_se']:.3e} | "
            f"{best['zcrit_fit']:.3e} | {best['zcrit_se']:.3e} | {best['delta_chi2_vs_plane_null']:.3e} |"
        )
    lines.append("")
    lines.append("## Conditional Consensus Theta Profile")
    lines.append("")
    lines.append("| theta_deg | sum delta chi2 proxy |")
    lines.append("|---:|---:|")
    for row in packet["combined_theta_profile"]:
        lines.append(f"| {row['theta_crit_deg']:.2f} | {row['sum_delta_chi2_proxy']:.3e} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the system packet builder."""

    args = parse_args()
    validate_args(args)
    projection_path = Path(args.projection_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = load_json(projection_path)
    ready_systems = [
        row for row in report.get("systems", [])
        if str(row.get("projection_robust_status")) == "projection_robust_ready"
    ]
    if int(args.max_systems) > 0:
        ready_systems = ready_systems[: int(args.max_systems)]
    if not ready_systems:
        raise ValueError("No projection_robust_ready systems found.")

    member_index = {row["candidate_id"]: row for row in report.get("members", [])}
    mask_map = load_analysis_mask(args)
    map_cache: dict[str, np.ndarray] = {}
    summary_rows = []

    for system_row in ready_systems:
        surviving_ids = [str(item) for item in system_row.get("surviving_member_ids", [])]
        member_packets = []
        for member_id in surviving_ids:
            member_row = member_index[member_id]
            hp_map = prepare_cleaned_map(args, str(member_row["map"]), mask_map, map_cache)
            member_packets.append(profile_member(member_row, hp_map=hp_map, mask_map=mask_map, args=args))
        combined_theta_profile = combine_theta_profiles(member_packets)
        packet = {
            "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "system_id": f"system_{int(system_row['system_rank']):02d}_{str(system_row['representative_candidate_id']).replace(':', '_')}",
            "system_rank": int(system_row["system_rank"]),
            "representative_candidate_id": str(system_row["representative_candidate_id"]),
            "projection_robust_status": str(system_row["projection_robust_status"]),
            "consensus_seed": system_consensus(member_packets),
            "combined_theta_profile": combined_theta_profile,
            "members": member_packets,
            "assumption_notes": [
                "This is a local equal-area template-likelihood proxy with a plane-null nuisance model.",
                "Member maps are used as a conditional consensus check, not as independent cosmological evidence channels.",
                "Use the consensus seed to initialize downstream likelihood or sampler work on the surviving system.",
            ],
        }
        system_dir = output_dir / packet["system_id"]
        system_dir.mkdir(parents=True, exist_ok=True)
        json_path = system_dir / "system_template_likelihood_packet.json"
        md_path = system_dir / "system_template_likelihood_packet.md"
        json_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
        write_markdown(md_path, packet)
        combined_csv = system_dir / "combined_theta_profile.csv"
        write_profile_csv(combined_csv, combined_theta_profile)
        for member in member_packets:
            member_csv = system_dir / f"{member['candidate_id'].replace(':', '_')}_theta_profile.csv"
            write_profile_csv(member_csv, member["profile_rows"])
        summary_rows.append(
            {
                "system_rank": int(system_row["system_rank"]),
                "system_id": packet["system_id"],
                "representative_candidate_id": packet["representative_candidate_id"],
                "theta_crit_deg_median": float(packet["consensus_seed"]["theta_crit_deg_median"]),
                "z0_median": float(packet["consensus_seed"]["z0_median"]),
                "zcrit_median": float(packet["consensus_seed"]["zcrit_median"]),
                "num_members": int(len(member_packets)),
                "output_json": str(json_path),
                "output_markdown": str(md_path),
            }
        )

    summary_path = output_dir / "system_template_likelihood_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "projection_json": str(projection_path),
                "num_systems": int(len(summary_rows)),
                "rows": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_summary": str(summary_path),
                "num_systems": int(len(summary_rows)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
