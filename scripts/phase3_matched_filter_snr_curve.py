"""Compute ideal Feeney matched-filter SNR curves for Phase 3 cells.

Assumptions
-----------
* This is a harmonic-space, isotropic-Gaussian CMB calculation for a known
  Feeney-template profile. It is an idealized sensitivity ceiling, not a
  Bayesian evidence calculation and not a masked-sky OSS likelihood.
* Signal amplitudes are dimensionless fractional ``Delta T / T`` values. The
  spherical template is converted to Kelvin with ``T_CMB``.
* The filter template uses the McEwen et al. first-order additive approximation
  ``deltaT = f(n) * T_CMB``. The production injection convention remains the
  Feeney et al. full-temperature modulation; the cross term is tested in
  ``phase2_physics_checks.py``.
* CMB covariance is ``B_l^2 C_l + N_l``. When the sensitivity artifact records
  ``synfast_pixwin_true``, the effective beam includes the HEALPix pixel window.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.stats import norm

from phase2_observing_model import camb_tt_cls
from phase3_classical_filters import (
    effective_beam_l as classical_effective_beam_l,
    feeney_template_l0,
    load_cl,
    matched_filter_transfer,
    white_noise_cl,
)
from phase_config import (
    DEFAULTS,
    INJECTION_CONVENTION_FEENEY2011,
    INJECTION_CONVENTION_MCEWEN2012,
    INJECTION_CONVENTION_NOTES,
    NSIDE_WORKING,
    T_CMB,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENSITIVITY_H5 = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_data.h5"
)
DEFAULT_SENSITIVITY_REPORT = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_report.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_matched_filter_snr"
SIGN_QUADRANTS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compute ideal harmonic-space matched-filter SNR curves for the remediated sensitivity grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sensitivity-h5", type=str, default=str(DEFAULT_SENSITIVITY_H5))
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_SENSITIVITY_REPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cmb-cl", type=str, default="", help="Optional 1D TT C_l file in K^2.")
    parser.add_argument("--lmax", type=int, default=0, help="Harmonic lmax. Defaults to 3*nside-1.")
    parser.add_argument("--quadrature-order", type=int, default=2048)
    parser.add_argument("--beam-fwhm-arcmin", type=float, default=-1.0)
    parser.add_argument("--noise-sigma-uk-arcmin", type=float, default=-1.0)
    parser.add_argument(
        "--pixel-window-policy",
        type=str,
        default="auto",
        choices=("auto", "none", "synfast_pixwin_true"),
        help="Effective beam pixel-window policy for the ideal calculation.",
    )
    parser.add_argument("--fpr-target", type=float, default=-1.0)
    return parser.parse_args()


def parse_json_attr(value: Any) -> Any:
    """Parse a JSON-encoded HDF5 attribute when needed."""

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def load_grid_from_h5(path: Path) -> dict[str, Any]:
    """Load grid and observing assumptions from a sensitivity HDF5 artifact."""

    if not path.exists():
        raise FileNotFoundError(f"Missing sensitivity HDF5: {path}")
    with h5py.File(path, "r") as h5:
        if "summary" not in h5:
            raise ValueError("Sensitivity HDF5 is missing summary attrs.")
        summary = dict(h5["summary"].attrs)
    out = {
        "amplitude_grid": [float(x) for x in parse_json_attr(summary["amplitude_grid"])],
        "theta_grid_deg": [float(x) for x in parse_json_attr(summary["theta_grid_deg"])],
        "zcrit_ratio_grid": [float(x) for x in parse_json_attr(summary["zcrit_ratio_grid"])],
        "nside": int(summary.get("nside", NSIDE_WORKING)),
        "beam_fwhm_arcmin": float(summary.get("beam_fwhm_arcmin", DEFAULTS.beam_fwhm_arcmin)),
        "noise_sigma_uk_arcmin": float(summary.get("noise_sigma_uk_arcmin", 0.0)),
        "fpr_target": float(summary.get("fpr_target", 0.05)),
        "sky_fraction": float(summary.get("sky_fraction", 1.0)),
        "pixel_window_policy": str(summary.get("pixel_window_policy", DEFAULTS.pixel_window_policy)),
        "injection_convention": summary.get("injection_convention"),
        "matched_filter_approximation_convention": summary.get("matched_filter_approximation_convention"),
    }
    return out


def validate_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides and validate physical ranges."""

    config = dict(config)
    if args.lmax:
        config["lmax"] = int(args.lmax)
    else:
        config["lmax"] = 3 * int(config["nside"]) - 1
    if args.beam_fwhm_arcmin >= 0.0:
        config["beam_fwhm_arcmin"] = float(args.beam_fwhm_arcmin)
    if args.noise_sigma_uk_arcmin >= 0.0:
        config["noise_sigma_uk_arcmin"] = float(args.noise_sigma_uk_arcmin)
    if args.fpr_target > 0.0:
        config["fpr_target"] = float(args.fpr_target)
    if args.pixel_window_policy == "auto":
        policy = str(config.get("pixel_window_policy", "none"))
    else:
        policy = args.pixel_window_policy
    config["effective_pixel_window_policy"] = policy

    if config["lmax"] < 2:
        raise ValueError("lmax must be at least 2.")
    if int(args.quadrature_order) <= config["lmax"]:
        raise ValueError("--quadrature-order must exceed lmax for stable Legendre quadrature.")
    if any(value <= 0.0 for value in config["amplitude_grid"]):
        raise ValueError("All amplitudes must be positive.")
    if any(value <= 0.0 or value >= 180.0 for value in config["theta_grid_deg"]):
        raise ValueError("All theta_crit values must lie in (0, 180) deg.")
    if any(value < 0.0 for value in config["zcrit_ratio_grid"]):
        raise ValueError("zcrit ratios must be non-negative.")
    if config["beam_fwhm_arcmin"] < 0.0:
        raise ValueError("Beam FWHM must be non-negative.")
    if config["noise_sigma_uk_arcmin"] < 0.0:
        raise ValueError("Noise depth must be non-negative.")
    if not (0.0 < config["fpr_target"] < 1.0):
        raise ValueError("FPR target must lie in (0, 1).")
    if not (0.0 < config["sky_fraction"] <= 1.0):
        raise ValueError("Sky fraction must lie in (0, 1].")
    return config


def effective_beam_l(config: dict[str, Any]) -> np.ndarray:
    """Return the effective harmonic beam for the covariance calculation."""

    return classical_effective_beam_l(
        nside=int(config["nside"]),
        lmax=int(config["lmax"]),
        beam_fwhm_arcmin=float(config["beam_fwhm_arcmin"]),
        pixel_window_policy=str(config["effective_pixel_window_policy"]),
    )


def load_cmb_cl(args: argparse.Namespace, lmax: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or generate the TT spectrum used by the ideal filter."""

    if args.cmb_cl:
        path = Path(args.cmb_cl).expanduser().resolve()
        return load_cl(path, int(lmax)), {"source": str(path), "spectrum": "user_supplied_TT_K2"}
    cl, provenance = camb_tt_cls(lmax=int(lmax))
    return cl, {"source": "phase2_observing_model.camb_tt_cls", **provenance}


def ideal_recall_from_snr(snr_value: float, fpr: float) -> float:
    """Return Gaussian ideal detection probability at fixed FPR and known sign."""

    threshold = float(norm.isf(float(fpr)))
    return float(norm.sf(threshold - float(snr_value)))


def load_recall_report(path: Path) -> dict[tuple[str, float, float], float]:
    """Load observed method recall by method/amplitude/theta cell."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    lookup: dict[tuple[str, float, float], float] = {}
    for row in report.get("rows", []):
        lookup[(str(row["method"]), float(row["amplitude"]), float(row["theta_crit_deg"]))] = float(row["p_det"])
    return lookup


def summarize(values: list[float]) -> dict[str, float]:
    """Return min/median/max summary statistics."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Cannot summarize empty or non-finite values.")
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def build_rows(
    *,
    config: dict[str, Any],
    cmb_cl: np.ndarray,
    beam_l: np.ndarray,
    recall_lookup: dict[tuple[str, float, float], float],
    quadrature_order: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute profile-level and cell-level SNR rows."""

    lmax = int(config["lmax"])
    noise_cl = white_noise_cl(lmax, float(config["noise_sigma_uk_arcmin"]))
    fpr = float(config["fpr_target"])
    fsky_scale = float(np.sqrt(config["sky_fraction"]))
    profile_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    template_basis: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for theta_deg in config["theta_grid_deg"]:
        theta = float(theta_deg)
        template_basis[theta] = (
            feeney_template_l0(
                theta_crit_deg=theta,
                z0=1.0,
                zcrit=0.0,
                lmax=lmax,
                quadrature_order=int(quadrature_order),
            ),
            feeney_template_l0(
                theta_crit_deg=theta,
                z0=0.0,
                zcrit=1.0,
                lmax=lmax,
                quadrature_order=int(quadrature_order),
            ),
        )

    for theta_deg in config["theta_grid_deg"]:
        theta = float(theta_deg)
        z0_basis, zcrit_basis = template_basis[theta]
        unit_rows = []
        for ratio in config["zcrit_ratio_grid"]:
            for z0_sign, zcrit_sign in SIGN_QUADRANTS:
                unit_template = float(z0_sign) * z0_basis + float(zcrit_sign) * float(ratio) * zcrit_basis
                _, unit_norm = matched_filter_transfer(
                    template_l0=unit_template,
                    cmb_cl=cmb_cl,
                    beam_l=beam_l,
                    noise_cl=noise_cl,
                )
                unit_snr_per_a1 = float(np.sqrt(unit_norm))
                unit_rows.append(
                    {
                        "theta_crit_deg": theta,
                        "zcrit_ratio": float(ratio),
                        "z0_sign": float(z0_sign),
                        "zcrit_sign": float(zcrit_sign),
                        "unit_snr_per_abs_z0_1": unit_snr_per_a1,
                        "matched_filter_norm_per_abs_z0_1": float(unit_norm),
                    }
                )

        for amp in config["amplitude_grid"]:
            amp = float(amp)
            snr_values = []
            fsky_snr_values = []
            recall_values = []
            fsky_recall_values = []
            for unit_row in unit_rows:
                snr = amp * float(unit_row["unit_snr_per_abs_z0_1"])
                fsky_snr = snr * fsky_scale
                ideal_recall = ideal_recall_from_snr(snr, fpr)
                fsky_recall = ideal_recall_from_snr(fsky_snr, fpr)
                profile_rows.append(
                    {
                        "amplitude": amp,
                        **unit_row,
                        "snr_full_sky": float(snr),
                        "snr_fsky_scaled": float(fsky_snr),
                        "ideal_recall_full_sky": float(ideal_recall),
                        "ideal_recall_fsky_scaled": float(fsky_recall),
                    }
                )
                snr_values.append(float(snr))
                fsky_snr_values.append(float(fsky_snr))
                recall_values.append(float(ideal_recall))
                fsky_recall_values.append(float(fsky_recall))

            snr_summary = summarize(snr_values)
            fsky_snr_summary = summarize(fsky_snr_values)
            recall_summary = summarize(recall_values)
            fsky_recall_summary = summarize(fsky_recall_values)
            cell_row: dict[str, Any] = {
                "amplitude": amp,
                "theta_crit_deg": theta,
                "num_profile_variants": int(len(unit_rows)),
                "snr_full_sky_min": snr_summary["min"],
                "snr_full_sky_median": snr_summary["median"],
                "snr_full_sky_max": snr_summary["max"],
                "snr_fsky_scaled_min": fsky_snr_summary["min"],
                "snr_fsky_scaled_median": fsky_snr_summary["median"],
                "snr_fsky_scaled_max": fsky_snr_summary["max"],
                "ideal_recall_full_sky_min": recall_summary["min"],
                "ideal_recall_full_sky_median": recall_summary["median"],
                "ideal_recall_full_sky_max": recall_summary["max"],
                "ideal_recall_fsky_scaled_min": fsky_recall_summary["min"],
                "ideal_recall_fsky_scaled_median": fsky_recall_summary["median"],
                "ideal_recall_fsky_scaled_max": fsky_recall_summary["max"],
            }
            for (method, method_amp, method_theta), p_det in recall_lookup.items():
                if np.isclose(method_amp, amp, rtol=1.0e-8, atol=0.0) and np.isclose(
                    method_theta,
                    theta,
                    rtol=1.0e-8,
                    atol=0.0,
                ):
                    cell_row[f"{method}_p_det"] = float(p_det)
            cell_rows.append(cell_row)
    return profile_rows, cell_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with a stable header."""

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a compact paper-facing Markdown summary."""

    config = report["config"]
    rows = report["cell_rows"]
    lines = ["# Matched-Filter SNR Curves", ""]
    lines.append("This is an ideal harmonic-space SNR calculation for known Feeney linear-cap templates.")
    lines.append("It is not a masked-sky Bayesian evidence calculation and does not close the same-grid ML-vs-SMHW benchmark by itself.")
    lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    lines.append(f"- injection convention in generated products: `{config['injection_convention_effective']}`")
    lines.append(f"- matched-filter template convention: `{INJECTION_CONVENTION_MCEWEN2012}`")
    lines.append(f"- `lmax`: `{config['lmax']}`")
    lines.append(f"- beam FWHM: `{config['beam_fwhm_arcmin']:.3g} arcmin`")
    lines.append(f"- noise depth: `{config['noise_sigma_uk_arcmin']:.3g} uK arcmin`")
    lines.append(f"- sky fraction used for rough fsky scaling: `{config['sky_fraction']:.4f}`")
    lines.append(f"- effective pixel-window policy: `{config['effective_pixel_window_policy']}`")
    lines.append("")
    lines.append("## Cell Summary")
    lines.append("")
    lines.append("| A | theta_deg | median SNR | fsky SNR | ideal recall | ImageNet recall | circular recall |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['amplitude']:.0e} | {row['theta_crit_deg']:.1f} | "
            f"{row['snr_full_sky_median']:.3f} | "
            f"{row['snr_fsky_scaled_median']:.3f} | "
            f"{row['ideal_recall_fsky_scaled_median']:.3f} | "
            f"{row.get('imagenet_b64_aux_p_det', float('nan')):.3f} | "
            f"{row.get('circular_template_screen_p_det', float('nan')):.3f} |"
        )
    lines.append("")
    weak = [
        row
        for row in rows
        if row["ideal_recall_fsky_scaled_median"] < 0.2
        and row.get("imagenet_b64_aux_p_det", 0.0) < 0.2
    ]
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Cells with both low ideal matched-filter recall and low ML recall are CMB-confusion-limited under this idealized Gaussian calculation."
    )
    lines.append(
        "- Cells where ML recall is far below the ideal matched-filter recall are algorithmic targets for architecture, score fusion, or training-distribution work."
    )
    lines.append(
        "- The fsky-scaled column is a rough masked-sky proxy; a true masked-sky optimum still needs the OSS-style covariance treatment."
    )
    if weak:
        lines.append("")
        lines.append("Low-SNR cells under the fsky proxy:")
        for row in weak[:10]:
            lines.append(
                f"- `A={row['amplitude']:.0e}`, `theta={row['theta_crit_deg']:.1f} deg`: "
                f"median fsky SNR `{row['snr_fsky_scaled_median']:.3f}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_report(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot median SNR and observed recalls by angular radius."""

    plt.style.use("seaborn-v0_8-whitegrid")
    theta_values = sorted({float(row["theta_crit_deg"]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharex=True)
    for theta in theta_values:
        theta_rows = sorted(
            [row for row in rows if float(row["theta_crit_deg"]) == theta],
            key=lambda item: float(item["amplitude"]),
        )
        x = np.asarray([row["amplitude"] for row in theta_rows], dtype=np.float64)
        axes[0].plot(x, [row["snr_fsky_scaled_median"] for row in theta_rows], marker="o", label=f"{theta:g} deg")
        axes[1].plot(
            x,
            [row["ideal_recall_fsky_scaled_median"] for row in theta_rows],
            marker="o",
            label=f"{theta:g} deg",
        )
        axes[2].plot(
            x,
            [row.get("imagenet_b64_aux_p_det", np.nan) for row in theta_rows],
            marker="s",
            label=f"{theta:g} deg",
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("A = |z0|")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("median SNR, fsky scaled")
    axes[1].set_ylabel("ideal recall at fixed FPR")
    axes[2].set_ylabel("observed ImageNet recall")
    axes[0].set_title("Matched-filter SNR")
    axes[1].set_title("Gaussian Ideal")
    axes[2].set_title("Current ML")
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    sensitivity_h5 = Path(args.sensitivity_h5).expanduser().resolve()
    sensitivity_report = Path(args.sensitivity_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = validate_config(load_grid_from_h5(sensitivity_h5), args)
    cmb_cl, cl_provenance = load_cmb_cl(args, int(config["lmax"]))
    beam_l = effective_beam_l(config)
    recall_lookup = load_recall_report(sensitivity_report)
    profile_rows, cell_rows = build_rows(
        config=config,
        cmb_cl=cmb_cl,
        beam_l=beam_l,
        recall_lookup=recall_lookup,
        quadrature_order=int(args.quadrature_order),
    )

    config_out = dict(config)
    config_out["injection_convention_effective"] = (
        config_out.get("injection_convention") or INJECTION_CONVENTION_FEENEY2011
    )
    config_out["injection_convention_note"] = INJECTION_CONVENTION_NOTES[INJECTION_CONVENTION_FEENEY2011]
    config_out["matched_filter_approximation_convention_effective"] = (
        config_out.get("matched_filter_approximation_convention") or INJECTION_CONVENTION_MCEWEN2012
    )
    config_out["matched_filter_approximation_note"] = INJECTION_CONVENTION_NOTES[INJECTION_CONVENTION_MCEWEN2012]
    config_out["t_cmb_k"] = float(T_CMB.to_value(u.K))
    config_out["quadrature_order"] = int(args.quadrature_order)

    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "sensitivity_h5": str(sensitivity_h5),
        "sensitivity_report": str(sensitivity_report),
        "config": config_out,
        "cmb_cl": cl_provenance,
        "assumption_warnings": [
            "This is an ideal isotropic-Gaussian full-sky SNR calculation, not a masked-sky Bayesian evidence.",
            "The fsky-scaled SNR is a rough proxy and not a replacement for OSS-style mask covariance.",
            "Ideal matched-filter recall assumes the correct template radius and sign family are in the bank.",
        ],
        "profile_rows": profile_rows,
        "cell_rows": cell_rows,
    }

    json_path = output_dir / "matched_filter_snr_report.json"
    cell_csv = output_dir / "matched_filter_snr_cells.csv"
    profile_csv = output_dir / "matched_filter_snr_profiles.csv"
    md_path = output_dir / "matched_filter_snr_report.md"
    png_path = output_dir / "matched_filter_snr_curves.png"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(cell_csv, cell_rows)
    write_csv(profile_csv, profile_rows)
    write_markdown(md_path, report)
    plot_report(png_path, cell_rows)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "cell_csv": str(cell_csv),
                "profile_csv": str(profile_csv),
                "markdown": str(md_path),
                "plot": str(png_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
