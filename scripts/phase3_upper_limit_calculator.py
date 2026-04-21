"""Compute post-processing upper limits from remediated sensitivity curves.

Assumptions
-----------
* This is a post-processing calculator for candidate-screening sensitivity. It
  does not replace the Feeney-style Bayesian evidence calculation.
* The number of detectable collisions on the full sky, ``Nbar_s``, is treated as
  a Poisson-rate parameter following the approximation in Feeney et al.
  Phys. Rev. D 84, Appendix A, Eq. A17/A20 for no identified blobs.
* The exposure entering that Poisson process is ``f_sky * <epsilon>``, where
  ``<epsilon>`` is the sensitivity-grid detection efficiency averaged over an
  explicit amplitude/radius prior. Change the prior before quoting a result.
* The source-backed eternal-inflation conversion uses Feeney et al. Eq. 1:
  ``Nbar = (16*pi/3) * lambda*H_F^-4 * (H_F/H_I)^2 * sqrt(Omega_k)``.
* Any ``lambda/B`` mapping beyond Eq. 1 is model-specific. This script only
  computes it when the user supplies an explicit exposure factor ``B`` such that
  ``Nbar = B * (lambda/B_parameter)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.stats import gamma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENSITIVITY_REPORT = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_sensitivity_curve"
    / "sensitivity_report.json"
)
DEFAULT_DATASET_SUMMARY = PROJECT_ROOT / "data" / "remediated_v1" / "summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_upper_limits"
PRIOR_CHOICES = ("log_uniform", "linear_uniform", "grid_uniform")
THETA_PRIOR_CHOICES = ("sin_theta", "grid_uniform")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute efficiency-weighted Nbar_s upper limits from "
            "remediated sensitivity curves."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sensitivity-report", type=str, default=str(DEFAULT_SENSITIVITY_REPORT))
    parser.add_argument("--dataset-summary", type=str, default=str(DEFAULT_DATASET_SUMMARY))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method to include. Default: all methods.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--num-detections", type=int, default=0)
    parser.add_argument(
        "--f-sky",
        type=float,
        default=0.0,
        help="Override sky fraction. Default reads dataset summary.",
    )
    parser.add_argument(
        "--amplitude-prior",
        type=str,
        default="log_uniform",
        choices=PRIOR_CHOICES,
    )
    parser.add_argument(
        "--theta-prior",
        type=str,
        default="sin_theta",
        choices=THETA_PRIOR_CHOICES,
    )
    parser.add_argument("--min-amplitude", type=float, default=0.0)
    parser.add_argument("--max-amplitude", type=float, default=0.0)
    parser.add_argument("--min-theta-deg", type=float, default=0.0)
    parser.add_argument("--max-theta-deg", type=float, default=0.0)
    parser.add_argument(
        "--omega-k",
        type=float,
        default=0.0,
        help="Optional curvature density for Feeney Eq. 1 conversion to lambda*H_F^-4.",
    )
    parser.add_argument(
        "--hf-over-hi",
        type=float,
        default=0.0,
        help="Optional H_F/H_I ratio for Feeney Eq. 1 conversion.",
    )
    parser.add_argument(
        "--lambda-over-b-exposure",
        type=float,
        default=0.0,
        help=(
            "Optional model-specific exposure B for a lambda/B-style parameter. "
            "If supplied, lambda_over_b_95 = Nbar95 / B."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    finite_fields = (
        "confidence",
        "f_sky",
        "min_amplitude",
        "max_amplitude",
        "min_theta_deg",
        "max_theta_deg",
        "omega_k",
        "hf_over_hi",
        "lambda_over_b_exposure",
    )
    for field in finite_fields:
        if not math.isfinite(float(getattr(args, field))):
            raise ValueError(f"--{field.replace('_', '-')} must be finite.")
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must lie in (0, 1).")
    if args.num_detections < 0:
        raise ValueError("--num-detections must be non-negative.")
    if args.f_sky < 0.0 or args.f_sky > 1.0:
        raise ValueError("--f-sky must lie in [0, 1].")
    if args.min_amplitude < 0.0 or args.max_amplitude < 0.0:
        raise ValueError("Amplitude bounds must be non-negative.")
    if args.max_amplitude and args.min_amplitude and args.min_amplitude > args.max_amplitude:
        raise ValueError("--min-amplitude must be <= --max-amplitude.")
    if args.max_theta_deg and args.min_theta_deg and args.min_theta_deg > args.max_theta_deg:
        raise ValueError("--min-theta-deg must be <= --max-theta-deg.")
    if args.omega_k < 0.0:
        raise ValueError("--omega-k must be non-negative.")
    if args.hf_over_hi < 0.0:
        raise ValueError("--hf-over-hi must be non-negative.")
    if args.lambda_over_b_exposure < 0.0:
        raise ValueError("--lambda-over-b-exposure must be non-negative.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_sky_fraction(args: argparse.Namespace) -> float:
    if args.f_sky > 0.0:
        return float(args.f_sky)
    summary_path = Path(args.dataset_summary)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Dataset summary missing and --f-sky not supplied: {summary_path}"
        )
    summary = load_json(summary_path)
    sky_fraction = float(summary.get("sky_fraction", 0.0))
    if not (0.0 < sky_fraction <= 1.0):
        raise ValueError("Could not determine a physical sky fraction.")
    return sky_fraction


def selected_rows(report: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    methods = set(args.method) if args.method else None
    for row in report.get("rows", []):
        amp = float(row["amplitude"])
        theta = float(row["theta_crit_deg"])
        p_det = float(row["p_det"])
        if not math.isfinite(amp) or amp <= 0.0:
            raise ValueError(f"Non-physical amplitude in sensitivity row: {amp}")
        if not math.isfinite(theta) or theta <= 0.0 or theta >= 180.0:
            raise ValueError(f"Non-physical theta_crit_deg in sensitivity row: {theta}")
        if not math.isfinite(p_det) or p_det < 0.0 or p_det > 1.0:
            raise ValueError(f"Non-physical detection efficiency in sensitivity row: {p_det}")
        if methods is not None and str(row["method"]) not in methods:
            continue
        if args.min_amplitude and amp < args.min_amplitude * (1.0 - 1.0e-12):
            continue
        if args.max_amplitude and amp > args.max_amplitude * (1.0 + 1.0e-12):
            continue
        if args.min_theta_deg and theta < args.min_theta_deg * (1.0 - 1.0e-12):
            continue
        if args.max_theta_deg and theta > args.max_theta_deg * (1.0 + 1.0e-12):
            continue
        rows.append(row)
    if not rows:
        raise ValueError("No sensitivity rows remain after method/range selection.")
    return rows


def discrete_bin_widths(values: list[float], *, log_space: bool) -> dict[float, float]:
    unique = np.asarray(sorted({float(value) for value in values}), dtype=np.float64)
    if not np.all(np.isfinite(unique)):
        raise ValueError("Prior-grid values must be finite.")
    if unique.size == 1:
        return {float(unique[0]): 1.0}
    if log_space and np.any(unique <= 0.0):
        raise ValueError("Log-uniform prior requires positive amplitudes.")
    work = np.log(unique) if log_space else unique
    edges = np.empty(unique.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (work[:-1] + work[1:])
    edges[0] = work[0] - 0.5 * (work[1] - work[0])
    edges[-1] = work[-1] + 0.5 * (work[-1] - work[-2])
    widths = np.diff(edges)
    return {float(value): float(width) for value, width in zip(unique, widths)}


def row_weights(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[tuple[float, float], float]:
    amplitudes = [float(row["amplitude"]) for row in rows]
    thetas = [float(row["theta_crit_deg"]) for row in rows]
    amp_width = discrete_bin_widths(amplitudes, log_space=args.amplitude_prior == "log_uniform")
    theta_width = discrete_bin_widths(thetas, log_space=False)
    weights: dict[tuple[float, float], float] = {}
    for amp in sorted(set(amplitudes)):
        for theta in sorted(set(thetas)):
            if args.amplitude_prior == "grid_uniform":
                amp_w = 1.0
            elif args.amplitude_prior == "linear_uniform":
                amp_w = amp_width[float(amp)]
            else:
                amp_w = amp_width[float(amp)]
            if args.theta_prior == "grid_uniform":
                theta_w = 1.0
            else:
                theta_w = theta_width[float(theta)] * math.sin(
                    (float(theta) * u.deg).to_value(u.rad)
                )
            weights[(float(amp), float(theta))] = max(0.0, float(amp_w) * float(theta_w))
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("Prior weights sum to zero.")
    return {key: value / total for key, value in weights.items()}


def poisson_mean_upper(num_detections: int, confidence: float) -> float:
    """Flat-prior Poisson-rate upper limit for observed count n."""

    return float(gamma.ppf(float(confidence), int(num_detections) + 1, scale=1.0))


def feeney_lambda_hf_minus4_limit(
    nbar_limit: float,
    omega_k: float,
    hf_over_hi: float,
) -> float | None:
    if omega_k <= 0.0 or hf_over_hi <= 0.0:
        return None
    denominator = 16.0 * math.pi * (float(hf_over_hi) ** 2) * math.sqrt(float(omega_k))
    return float((3.0 * float(nbar_limit)) / denominator)


def compute_limits(
    report: dict[str, Any],
    args: argparse.Namespace,
    f_sky: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = selected_rows(report, args)
    methods = sorted({str(row["method"]) for row in rows})
    weights = row_weights(rows, args)
    mu_upper = poisson_mean_upper(int(args.num_detections), float(args.confidence))
    limit_rows = []
    cell_rows = []
    for method in methods:
        method_rows = [row for row in rows if str(row["method"]) == method]
        seen_cells = {
            (float(row["amplitude"]), float(row["theta_crit_deg"]))
            for row in method_rows
        }
        if set(weights) - seen_cells:
            missing = sorted(set(weights) - seen_cells)
            raise ValueError(f"Method {method} is missing sensitivity cells: {missing[:5]}")
        efficiency = 0.0
        for row in method_rows:
            amp = float(row["amplitude"])
            theta = float(row["theta_crit_deg"])
            weight = weights[(amp, theta)]
            p_det = float(row["p_det"])
            efficiency += weight * p_det
            cell_rows.append(
                {
                    "method": method,
                    "amplitude": amp,
                    "theta_crit_deg": theta,
                    "prior_weight": weight,
                    "p_det": p_det,
                    "weighted_efficiency_contribution": weight * p_det,
                }
            )
        if (
            not math.isfinite(efficiency)
            or efficiency < 0.0
            or efficiency > 1.0 + 1.0e-10
        ):
            raise ValueError(f"Computed non-physical mean efficiency for {method}: {efficiency}")
        exposure = f_sky * efficiency
        if not math.isfinite(exposure) or exposure < 0.0:
            raise ValueError(f"Computed non-physical exposure for {method}: {exposure}")
        if exposure <= 0.0:
            nbar95 = float("inf")
        else:
            nbar95 = mu_upper / exposure
        lambda_hf = feeney_lambda_hf_minus4_limit(
            nbar95,
            float(args.omega_k),
            float(args.hf_over_hi),
        )
        lambda_over_b = None
        if args.lambda_over_b_exposure > 0.0:
            lambda_over_b = nbar95 / float(args.lambda_over_b_exposure)
        limit_rows.append(
            {
                "method": method,
                "confidence": float(args.confidence),
                "num_detections": int(args.num_detections),
                "poisson_mean_upper": mu_upper,
                "f_sky": float(f_sky),
                "mean_efficiency": float(efficiency),
                "effective_exposure": float(exposure),
                "nbar_s_upper": float(nbar95),
                "lambda_hf_minus4_upper": lambda_hf,
                "lambda_over_b_upper": lambda_over_b,
            }
        )
    metadata = {
        "sensitivity_report": str(Path(args.sensitivity_report).resolve()),
        "dataset_summary": str(Path(args.dataset_summary).resolve()),
        "confidence": float(args.confidence),
        "num_detections": int(args.num_detections),
        "f_sky": float(f_sky),
        "amplitude_prior": args.amplitude_prior,
        "theta_prior": args.theta_prior,
        "amplitude_range": [
            float(args.min_amplitude) if args.min_amplitude else None,
            float(args.max_amplitude) if args.max_amplitude else None,
        ],
        "theta_range_deg": [
            float(args.min_theta_deg) if args.min_theta_deg else None,
            float(args.max_theta_deg) if args.max_theta_deg else None,
        ],
        "omega_k": float(args.omega_k) if args.omega_k else None,
        "hf_over_hi": float(args.hf_over_hi) if args.hf_over_hi else None,
        "lambda_over_b_exposure": (
            float(args.lambda_over_b_exposure)
            if args.lambda_over_b_exposure
            else None
        ),
        "formula_notes": [
            "Nbar_s upper limit = Poisson mean upper / (f_sky * prior-averaged efficiency).",
            (
                "lambda_hf_minus4_upper uses Feeney et al. PRD 84 Eq. 1 "
                "when omega_k and H_F/H_I are supplied."
            ),
            (
                "lambda_over_b_upper is reported only for a user-supplied "
                "model-specific exposure factor B."
            ),
        ],
    }
    return limit_rows, {"metadata": metadata, "cell_rows": cell_rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    rows = report["limits"]
    lines = ["# Remediated v1 Upper-Limit Calculator", ""]
    lines.append(
        "This is an efficiency-weighted Poisson post-processing result, "
        "not a Bayesian evidence calculation."
    )
    lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    lines.append(f"- Confidence: `{metadata['confidence']}`")
    lines.append(f"- Observed credible detections: `{metadata['num_detections']}`")
    lines.append(f"- Sky fraction: `{metadata['f_sky']:.6f}`")
    lines.append(f"- Amplitude prior: `{metadata['amplitude_prior']}`")
    lines.append(f"- Radius prior: `{metadata['theta_prior']}`")
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append(
        "| method | mean efficiency | exposure | Nbar_s upper | "
        "lambda H_F^-4 upper | lambda/B upper |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lambda_hf = row["lambda_hf_minus4_upper"]
        lambda_b = row["lambda_over_b_upper"]
        lambda_hf_text = f"{lambda_hf:.4e}" if lambda_hf is not None else "n/a"
        lambda_b_text = f"{lambda_b:.4e}" if lambda_b is not None else "n/a"
        lines.append(
            f"| `{row['method']}` | {row['mean_efficiency']:.4f} | "
            f"{row['effective_exposure']:.4f} | {row['nbar_s_upper']:.4f} | "
            f"{lambda_hf_text} | {lambda_b_text} |"
        )
    lines.append("")
    lines.append("## Formula Notes")
    lines.append("")
    for note in metadata["formula_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        "Do not quote `lambda/B` unless the supplied exposure factor is "
        "derived from a specific model."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_limits(path: Path, rows: list[dict[str, Any]]) -> None:
    finite_rows = [row for row in rows if math.isfinite(float(row["nbar_s_upper"]))]
    if not finite_rows:
        return
    methods = [row["method"] for row in finite_rows]
    nbar = [row["nbar_s_upper"] for row in finite_rows]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(methods)), 4.0))
    ax.bar(methods, nbar, color="#355C7D")
    ax.set_ylabel("Nbar_s upper limit")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Efficiency-weighted detectable-collision upper limits")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    report = load_json(Path(args.sensitivity_report).resolve())
    f_sky = read_sky_fraction(args)
    limits, extra = compute_limits(report, args, f_sky)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {**extra, "limits": limits}
    json_path = output_dir / "upper_limits.json"
    md_path = output_dir / "upper_limits.md"
    limit_csv = output_dir / "upper_limits.csv"
    cell_csv = output_dir / "upper_limit_cell_weights.csv"
    plot_path = output_dir / "upper_limits.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(limit_csv, limits)
    write_csv(cell_csv, extra["cell_rows"])
    write_markdown(md_path, payload)
    plot_limits(plot_path, limits)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "limits_csv": str(limit_csv),
                "cell_weights_csv": str(cell_csv),
                "plot": str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
