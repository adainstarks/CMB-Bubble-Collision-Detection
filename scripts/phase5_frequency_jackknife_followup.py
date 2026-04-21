"""Common-resolution frequency jackknife follow-up for frozen candidates.

Assumptions
-----------
* The frozen Phase 3 candidates are screening outputs, not detections.
* The frequency follow-up uses official Planck PR3 foreground-reduced SEVEM
  temperature maps at 100/143/217 GHz in ``K_CMB`` units as a practical
  real-sky jackknife proxy for shared-foreground failure modes.
* All frequency maps are smoothed to a common beam before projection and
  fitting so that raw score changes are not dominated by native beam
  differences. This is a common-resolution analysis, not a literal rerun of
  SMICA/NILC/Commander/SEVEM with one channel removed.
* Template-fit amplitudes are normalized by the local plane-null residual RMS
  from the same common-resolution patch. This is an SNR-like stability proxy,
  not a Bayesian evidence ratio.
* The fitted template family follows Feeney et al. Phys. Rev. D 84, 043507
  (2011), arXiv:1012.3667. Frequency jackknife stability is a candidate
  follow-up guardrail, not a cosmological claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
from astropy import units as u

from phase_dataset_utils import pixel_to_patch_offsets_deg
from phase2_generate_training import projected_unmasked_fraction
from phase2_observing_model import project_patch, remove_real_map_low_modes
from phase3_template_fit_candidates import fit_one_candidate
from phase5_half_mission_signflip_null import degrade_map_if_needed, load_analysis_mask, load_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_tile_constrained_candidates"
    / "cluster_representatives_15deg.jsonl"
)
DEFAULT_BASELINE_TEMPLATE_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_template_fit_handoff"
    / "template_fit_records.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "phase5_frequency_jackknife_followup"
)
DEFAULT_COMMON_MASK = PROJECT_ROOT / "data" / "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
DEFAULT_FREQ_MAP_TEMPLATE = (
    PROJECT_ROOT
    / "data"
    / "planck_pr3_freq"
    / "COM_CMB_IQU-{freq}-fgsub-sevem_2048_R3.00_full.fits"
)
DEFAULT_CHANNEL_BEAMS = "100:9.68,143:7.30,217:5.02"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run a common-resolution 100/143/217 GHz frequency jackknife on frozen candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-jsonl", type=str, default=str(DEFAULT_CANDIDATE_JSONL))
    parser.add_argument(
        "--baseline-template-jsonl",
        type=str,
        default=str(DEFAULT_BASELINE_TEMPLATE_JSONL),
        help="Optional cleaned-map template-fit JSONL used as the screening-era reference.",
    )
    parser.add_argument("--freq-map-template", type=str, default=str(DEFAULT_FREQ_MAP_TEMPLATE))
    parser.add_argument("--freqs-ghz", type=str, default="100,143,217")
    parser.add_argument("--channel-beams-arcmin", type=str, default=DEFAULT_CHANNEL_BEAMS)
    parser.add_argument(
        "--common-beam-fwhm-arcmin",
        type=float,
        default=10.0,
        help="Target common beam after smoothing all frequency maps.",
    )
    parser.add_argument("--common-mask", type=str, default=str(DEFAULT_COMMON_MASK))
    parser.add_argument("--fits-field", type=int, default=0)
    parser.add_argument("--target-nside", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.9)
    parser.add_argument("--min-valid-fraction", type=float, default=0.9)
    parser.add_argument("--skip-low-mode-removal", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--snr-retention-threshold",
        type=float,
        default=0.5,
        help="Minimum leave-one-out |z0|/sigma retention relative to the all-channel composite.",
    )
    parser.add_argument(
        "--theta-shift-threshold-deg",
        type=float,
        default=3.0,
        help="Maximum allowed leave-one-out shift in fitted theta_crit.",
    )
    parser.add_argument("--radius-window-deg", type=float, default=3.0)
    parser.add_argument("--radius-step-deg", type=float, default=0.5)
    parser.add_argument("--min-radius-deg", type=float, default=5.0)
    parser.add_argument("--max-radius-deg", type=float, default=25.0)
    parser.add_argument("--support-extra-deg", type=float, default=5.0)
    parser.add_argument("--support-factor", type=float, default=1.5)
    parser.add_argument("--edge-sigma-deg", type=float, default=0.0)
    return parser.parse_args()


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse a comma-separated float list."""

    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_beam_map(text: str) -> dict[int, float]:
    """Parse ``freq:beam`` pairs in arcmin."""

    out: dict[int, float] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        freq_text, beam_text = item.split(":", maxsplit=1)
        freq = int(freq_text.strip())
        beam = float(beam_text.strip())
        if freq <= 0 or beam <= 0.0 or not np.isfinite(beam):
            raise ValueError(f"Invalid frequency/beam pair: {item!r}")
        out[freq] = beam
    if not out:
        raise ValueError("Expected at least one frequency beam pair.")
    return out


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments and non-physical settings."""

    args.freqs_ghz = tuple(int(item.strip()) for item in str(args.freqs_ghz).split(",") if item.strip())
    if len(args.freqs_ghz) < 2:
        raise ValueError("--freqs-ghz must contain at least two channels.")
    args.channel_beams_arcmin = parse_beam_map(args.channel_beams_arcmin)
    missing = sorted(set(args.freqs_ghz) - set(args.channel_beams_arcmin))
    if missing:
        raise ValueError(f"Missing beam entries for channels: {missing}")
    if args.common_beam_fwhm_arcmin <= 0.0:
        raise ValueError("--common-beam-fwhm-arcmin must be positive.")
    if float(args.common_beam_fwhm_arcmin) + 1.0e-9 < max(
        float(args.channel_beams_arcmin[freq]) for freq in args.freqs_ghz
    ):
        raise ValueError("Common beam must be at least as broad as the broadest input channel beam.")
    if int(args.target_nside) <= 0 or not hp.isnsideok(int(args.target_nside)):
        raise ValueError("--target-nside must be a valid HEALPix Nside.")
    if not (0.0 < float(args.mask_threshold) <= 1.0):
        raise ValueError("--mask-threshold must lie in (0, 1].")
    if not (0.0 <= float(args.min_valid_fraction) <= 1.0):
        raise ValueError("--min-valid-fraction must lie in [0, 1].")
    if float(args.snr_retention_threshold) <= 0.0:
        raise ValueError("--snr-retention-threshold must be positive.")
    if float(args.theta_shift_threshold_deg) < 0.0:
        raise ValueError("--theta-shift-threshold-deg must be non-negative.")
    for label, value in (
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
    if float(args.radius_window_deg) < 0.0:
        raise ValueError("--radius-window-deg must be non-negative.")
    if float(args.radius_step_deg) <= 0.0:
        raise ValueError("--radius-step-deg must be positive.")
    if float(args.min_radius_deg) <= 0.0:
        raise ValueError("--min-radius-deg must be positive.")
    if float(args.max_radius_deg) <= float(args.min_radius_deg):
        raise ValueError("--max-radius-deg must exceed --min-radius-deg.")
    if float(args.support_extra_deg) < 0.0:
        raise ValueError("--support-extra-deg must be non-negative.")
    if float(args.support_factor) < 1.0:
        raise ValueError("--support-factor must be >= 1.")
    if float(args.edge_sigma_deg) < 0.0:
        raise ValueError("--edge-sigma-deg must be non-negative.")
    for path_text in (args.candidate_jsonl, args.common_mask):
        if not Path(path_text).expanduser().exists():
            raise FileNotFoundError(f"Missing required input: {path_text}")
    if args.baseline_template_jsonl and not Path(args.baseline_template_jsonl).expanduser().exists():
        raise FileNotFoundError(f"Missing baseline template JSONL: {args.baseline_template_jsonl}")
    for freq in args.freqs_ghz:
        map_path = Path(str(args.freq_map_template).format(freq=f"{int(freq):03d}")).expanduser()
        if not map_path.exists():
            raise FileNotFoundError(f"Missing frequency map for {freq} GHz: {map_path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def candidate_key(row: dict[str, Any]) -> tuple[str, int]:
    """Return a stable candidate key from candidate or template-fit rows."""

    patch_index = row.get("patch_index", row.get("sample_index"))
    if patch_index is None:
        raise KeyError("Row is missing patch_index/sample_index.")
    return str(row["map"]).lower(), int(patch_index)


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Index rows by map and patch index."""

    return {candidate_key(row): row for row in rows}


def adapt_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a frozen tile candidate into a template-fit record."""

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


def sign_label(value: float | None) -> int:
    """Return the sign label in ``{-1, 0, +1}``."""

    if value is None or not np.isfinite(float(value)):
        return 0
    return int(np.sign(float(value)))


def smooth_to_common_beam(
    hp_map: np.ndarray,
    *,
    native_beam_arcmin: float,
    target_beam_arcmin: float,
) -> np.ndarray:
    """Smooth a HEALPix map to the requested common beam."""

    native = (float(native_beam_arcmin) * u.arcmin).to_value(u.rad)
    target = (float(target_beam_arcmin) * u.arcmin).to_value(u.rad)
    extra_sq = target * target - native * native
    if extra_sq < -1.0e-12:
        raise ValueError("Target beam is narrower than the native beam.")
    if extra_sq <= 0.0:
        return np.asarray(hp_map, dtype=np.float64)
    return np.asarray(hp.smoothing(np.asarray(hp_map, dtype=np.float64), fwhm=float(np.sqrt(extra_sq))), dtype=np.float64)


def prepare_frequency_map(args: argparse.Namespace, *, freq_ghz: int, mask_map: np.ndarray) -> np.ndarray:
    """Load, smooth, degrade, and low-mode clean one frequency map."""

    path = Path(str(args.freq_map_template).format(freq=f"{int(freq_ghz):03d}")).expanduser().resolve()
    hp_map = load_map(path, int(args.fits_field))
    hp_map = smooth_to_common_beam(
        hp_map,
        native_beam_arcmin=float(args.channel_beams_arcmin[int(freq_ghz)]),
        target_beam_arcmin=float(args.common_beam_fwhm_arcmin),
    )
    hp_map = degrade_map_if_needed(hp_map, int(args.target_nside))
    if hp.get_nside(hp_map) != int(args.target_nside):
        raise ValueError(f"Prepared frequency map has wrong Nside: {hp.get_nside(hp_map)}")
    if not args.skip_low_mode_removal:
        hp_map = remove_real_map_low_modes(hp_map, mask=mask_map)
    bad = (mask_map <= 0.0) | ~np.isfinite(hp_map)
    hp_map = np.asarray(hp_map, dtype=np.float64)
    hp_map[bad] = 0.0
    max_abs = float(np.max(np.abs(hp_map)))
    if not np.isfinite(max_abs) or max_abs > 1.0:
        raise ValueError(
            f"Prepared {freq_ghz} GHz map exceeds anisotropy scale: max |T|={max_abs:.3g} K."
        )
    return hp_map


def combine_maps(maps: list[np.ndarray]) -> np.ndarray:
    """Return the arithmetic mean of already common-resolution maps."""

    if not maps:
        raise ValueError("Need at least one map to build a composite.")
    stack = np.stack([np.asarray(hp_map, dtype=np.float64) for hp_map in maps], axis=0)
    out = np.mean(stack, axis=0, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError("Composite frequency map contains non-finite values.")
    return out


def fit_candidate_on_map(
    hp_map: np.ndarray,
    candidate_row: dict[str, Any],
    *,
    mask_map: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Project one map to the candidate patch and run the deterministic template fit."""

    adapted = adapt_record(candidate_row)
    patch_glon = float(adapted["patch_center_glon_deg"])
    patch_glat = float(adapted["patch_center_glat_deg"])
    mask_patch = np.asarray(project_patch(mask_map, patch_glon, patch_glat), dtype=np.float64)
    valid_patch = np.isfinite(mask_patch) & (mask_patch > 0.5)
    valid_fraction = float(projected_unmasked_fraction(mask_patch))
    result: dict[str, Any] = {
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

    patch = np.asarray(project_patch(hp_map, patch_glon, patch_glat), dtype=np.float64)
    patch[~valid_patch] = 0.0
    fit_row = fit_one_candidate(patch, adapted, args)
    result.update(fit_row)
    if str(result.get("fit_status")) == "fit":
        variance = float(result.get("plane_null_reduced_sse", np.nan))
        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("Template fit returned a non-physical plane-null variance.")
        sigma = float(np.sqrt(variance))
        result["z0_snr_proxy"] = float(result["z0_fit"]) / sigma
        result["zcrit_snr_proxy"] = float(result["zcrit_fit"]) / sigma
        result["delta_chi2_per_pixel"] = float(result["delta_chi2_vs_plane_null"]) / max(
            int(result.get("support_pixels", 0)),
            1,
        )
    return result


def summarize_quantiles(values: list[float]) -> dict[str, float]:
    """Return fixed quantiles from a non-empty list."""

    arr = np.asarray(values, dtype=np.float64)
    q = np.quantile(arr, [0.05, 0.5, 0.95])
    return {"q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2])}


def build_candidate_result(
    candidate_row: dict[str, Any],
    *,
    baseline_row: dict[str, Any] | None,
    channel_rows: dict[str, dict[str, Any]],
    composite_rows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assemble one frequency-jackknife result row."""

    all_key = "all"
    all_row = composite_rows[all_key]
    all_status = str(all_row.get("fit_status"))
    failures: list[str] = []
    if all_status != "fit":
        failures.append(f"all_channels_status={all_status}")

    baseline_sign = sign_label(None if baseline_row is None else baseline_row.get("z0_fit"))
    all_sign = sign_label(all_row.get("z0_fit"))
    if baseline_sign != 0 and all_sign != baseline_sign:
        failures.append("baseline_to_all_channel_sign_flip")

    all_abs_snr = abs(float(all_row.get("z0_snr_proxy", 0.0))) if all_status == "fit" else 0.0
    drop_retentions: dict[str, float] = {}
    drop_theta_shifts: dict[str, float] = {}
    drop_sign_consistency: dict[str, bool] = {}
    for freq in args.freqs_ghz:
        key = f"drop_{int(freq)}"
        row = composite_rows[key]
        status = str(row.get("fit_status"))
        if status != "fit":
            failures.append(f"{key}_status={status}")
            continue
        retention = abs(float(row["z0_snr_proxy"])) / max(all_abs_snr, 1.0e-12)
        theta_shift = abs(float(row["theta_crit_fit_deg"]) - float(all_row["theta_crit_fit_deg"]))
        sign_ok = sign_label(row.get("z0_fit")) == all_sign
        drop_retentions[key] = float(retention)
        drop_theta_shifts[key] = float(theta_shift)
        drop_sign_consistency[key] = bool(sign_ok)
        if not sign_ok:
            failures.append(f"{key}_sign_flip")
        if retention < float(args.snr_retention_threshold):
            failures.append(f"{key}_snr_retention")
        if theta_shift > float(args.theta_shift_threshold_deg):
            failures.append(f"{key}_theta_shift")

    individual_signs = {
        str(int(freq)): sign_label(channel_rows[str(int(freq))].get("z0_fit"))
        for freq in args.freqs_ghz
    }
    individual_consistency_count = int(sum(int(sign == all_sign) for sign in individual_signs.values()))
    stable = len(failures) == 0
    return {
        "source_candidate": {
            key: candidate_row.get(key)
            for key in (
                "map",
                "patch_index",
                "global_cluster_rank",
                "cluster_id",
                "policy_slug",
                "patch_center_glon_deg",
                "patch_center_glat_deg",
                "candidate_glon_deg",
                "candidate_glat_deg",
            )
            if key in candidate_row
        },
        "baseline_cleaned_template": None
        if baseline_row is None
        else {
            "fit_status": baseline_row.get("fit_status"),
            "theta_crit_fit_deg": baseline_row.get("theta_crit_fit_deg"),
            "z0_fit": baseline_row.get("z0_fit"),
            "zcrit_fit": baseline_row.get("zcrit_fit"),
            "delta_chi2_vs_plane_null": baseline_row.get("delta_chi2_vs_plane_null"),
        },
        "common_resolution_channels": channel_rows,
        "leave_one_out_composites": composite_rows,
        "baseline_to_all_channel_sign_consistent": bool(baseline_sign == 0 or all_sign == baseline_sign),
        "all_channel_sign": int(all_sign),
        "all_channel_abs_z0_snr_proxy": float(all_abs_snr),
        "leave_one_out_snr_retention": drop_retentions,
        "leave_one_out_theta_shift_deg": drop_theta_shifts,
        "leave_one_out_sign_consistency": drop_sign_consistency,
        "individual_channel_signs": individual_signs,
        "individual_channel_sign_consistency_count": int(individual_consistency_count),
        "frequency_jackknife_stable": bool(stable),
        "frequency_jackknife_failures": failures,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact summary CSV."""

    columns = [
        "map",
        "patch_index",
        "global_cluster_rank",
        "frequency_jackknife_stable",
        "all_channel_abs_z0_snr_proxy",
        "min_loo_snr_retention",
        "max_loo_theta_shift_deg",
        "individual_channel_sign_consistency_count",
        "failures",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            src = row["source_candidate"]
            retentions = list(row["leave_one_out_snr_retention"].values())
            shifts = list(row["leave_one_out_theta_shift_deg"].values())
            writer.writerow(
                {
                    "map": src.get("map"),
                    "patch_index": src.get("patch_index"),
                    "global_cluster_rank": src.get("global_cluster_rank"),
                    "frequency_jackknife_stable": row["frequency_jackknife_stable"],
                    "all_channel_abs_z0_snr_proxy": row["all_channel_abs_z0_snr_proxy"],
                    "min_loo_snr_retention": min(retentions) if retentions else np.nan,
                    "max_loo_theta_shift_deg": max(shifts) if shifts else np.nan,
                    "individual_channel_sign_consistency_count": row["individual_channel_sign_consistency_count"],
                    "failures": ";".join(row["frequency_jackknife_failures"]),
                }
            )


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a human-readable summary."""

    lines = ["# Frequency Jackknife Follow-Up", ""]
    lines.append("This is a common-resolution real-sky follow-up check on frozen candidates.")
    lines.append("It is not a Bayesian evidence calculation.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in (
        "num_candidates",
        "num_stable",
        "num_unstable",
        "common_beam_fwhm_arcmin",
        "freqs_ghz",
    ):
        lines.append(f"- `{key}`: `{report[key]}`")
    if report.get("stable_all_channel_abs_z0_snr_proxy"):
        lines.append(
            f"- `stable_all_channel_abs_z0_snr_proxy`: "
            f"`{report['stable_all_channel_abs_z0_snr_proxy']}`"
        )
    lines.append("")
    lines.append("| rank | map | patch | stable | all-channel | min LOO retention | max theta shift | failures |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for row in report["top_rows"]:
        lines.append(
            f"| {row['global_cluster_rank']} | {row['map']} | {row['patch_index']} | "
            f"{row['frequency_jackknife_stable']} | {row['all_channel_abs_z0_snr_proxy']:.3f} | "
            f"{row['min_loo_snr_retention']:.3f} | {row['max_loo_theta_shift_deg']:.3f} | "
            f"{'; '.join(row['failures'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)

    candidate_rows = load_jsonl(Path(args.candidate_jsonl).expanduser().resolve())
    baseline_rows = (
        load_jsonl(Path(args.baseline_template_jsonl).expanduser().resolve())
        if args.baseline_template_jsonl
        else []
    )
    baseline_index = index_rows(baseline_rows) if baseline_rows else {}

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_map = load_analysis_mask(args)
    prepared_maps = {
        str(int(freq)): prepare_frequency_map(args, freq_ghz=int(freq), mask_map=mask_map)
        for freq in args.freqs_ghz
    }
    all_map = combine_maps([prepared_maps[str(int(freq))] for freq in args.freqs_ghz])

    rows: list[dict[str, Any]] = []
    for idx, candidate_row in enumerate(candidate_rows, start=1):
        baseline_row = baseline_index.get(candidate_key(candidate_row))
        channel_rows = {
            str(int(freq)): fit_candidate_on_map(
                prepared_maps[str(int(freq))],
                candidate_row,
                mask_map=mask_map,
                args=args,
            )
            for freq in args.freqs_ghz
        }
        composite_rows: dict[str, dict[str, Any]] = {
            "all": fit_candidate_on_map(
                all_map,
                candidate_row,
                mask_map=mask_map,
                args=args,
            )
        }
        for freq in args.freqs_ghz:
            keep = [
                prepared_maps[str(int(other))]
                for other in args.freqs_ghz
                if int(other) != int(freq)
            ]
            composite_rows[f"drop_{int(freq)}"] = fit_candidate_on_map(
                combine_maps(keep),
                candidate_row,
                mask_map=mask_map,
                args=args,
            )
        rows.append(
            build_candidate_result(
                candidate_row,
                baseline_row=baseline_row,
                channel_rows=channel_rows,
                composite_rows=composite_rows,
                args=args,
            )
        )
        if idx % 8 == 0 or idx == len(candidate_rows):
            print(f"  Frequency jackknife {idx:4d} / {len(candidate_rows)}", flush=True)

    rows.sort(
        key=lambda row: (
            not bool(row["frequency_jackknife_stable"]),
            -float(row["all_channel_abs_z0_snr_proxy"]),
            int(row["source_candidate"].get("global_cluster_rank", 0)),
        )
    )
    stable_rows = [row for row in rows if bool(row["frequency_jackknife_stable"])]
    unstable_rows = [row for row in rows if not bool(row["frequency_jackknife_stable"])]
    stable_snr = [float(row["all_channel_abs_z0_snr_proxy"]) for row in stable_rows]

    report = {
        "metadata": {
            "candidate_jsonl": str(Path(args.candidate_jsonl).expanduser().resolve()),
            "baseline_template_jsonl": (
                str(Path(args.baseline_template_jsonl).expanduser().resolve())
                if args.baseline_template_jsonl
                else ""
            ),
            "freq_map_template": str(args.freq_map_template),
            "channel_beams_arcmin": {str(key): float(value) for key, value in args.channel_beams_arcmin.items()},
            "target_nside": int(args.target_nside),
            "mask_threshold": float(args.mask_threshold),
            "min_valid_fraction": float(args.min_valid_fraction),
            "common_mask": str(Path(args.common_mask).expanduser().resolve()),
            "skip_low_mode_removal": bool(args.skip_low_mode_removal),
            "snr_retention_threshold": float(args.snr_retention_threshold),
            "theta_shift_threshold_deg": float(args.theta_shift_threshold_deg),
        },
        "assumption_notes": [
            "This is a common-resolution frequency-domain follow-up, not a literal rerun of component separation with one band removed.",
            "Official PR3 SEVEM foreground-reduced 100/143/217 GHz maps are smoothed to a shared beam before fitting.",
            "Template-fit amplitudes are normalized by the local plane-null residual RMS from the same common-resolution patch.",
            "A candidate is marked stable only if the leave-one-out composites keep the fitted sign, keep enough normalized amplitude, and avoid large theta shifts.",
        ],
        "num_candidates": int(len(rows)),
        "num_stable": int(len(stable_rows)),
        "num_unstable": int(len(unstable_rows)),
        "freqs_ghz": [int(freq) for freq in args.freqs_ghz],
        "common_beam_fwhm_arcmin": float(args.common_beam_fwhm_arcmin),
        "stable_all_channel_abs_z0_snr_proxy": (
            summarize_quantiles(stable_snr) if stable_snr else {}
        ),
        "top_rows": [
            {
                "global_cluster_rank": int(row["source_candidate"].get("global_cluster_rank", 0)),
                "map": str(row["source_candidate"].get("map")),
                "patch_index": int(row["source_candidate"].get("patch_index", -1)),
                "frequency_jackknife_stable": bool(row["frequency_jackknife_stable"]),
                "all_channel_abs_z0_snr_proxy": float(row["all_channel_abs_z0_snr_proxy"]),
                "min_loo_snr_retention": (
                    float(min(row["leave_one_out_snr_retention"].values()))
                    if row["leave_one_out_snr_retention"]
                    else float("nan")
                ),
                "max_loo_theta_shift_deg": (
                    float(max(row["leave_one_out_theta_shift_deg"].values()))
                    if row["leave_one_out_theta_shift_deg"]
                    else float("nan")
                ),
                "failures": list(row["frequency_jackknife_failures"]),
            }
            for row in rows[:10]
        ],
        "candidates": rows,
    }

    json_path = output_dir / "frequency_jackknife_report.json"
    jsonl_path = output_dir / "frequency_jackknife_candidates.jsonl"
    csv_path = output_dir / "frequency_jackknife_summary.csv"
    md_path = output_dir / "frequency_jackknife_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(csv_path, rows)
    write_markdown(md_path, report)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "jsonl": str(jsonl_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
                "num_candidates": report["num_candidates"],
                "num_stable": report["num_stable"],
                "num_unstable": report["num_unstable"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
