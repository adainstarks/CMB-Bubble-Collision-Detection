"""Audit whether the true Wiener/SMHW benchmark is same-grid complete.

Assumptions
-----------
* The true ``wiener_feeney_matched_filter`` and ``smhw_screen`` operators are
  full-sky spherical filters. They are nonlocal after inverse-CMB-covariance
  weighting and cannot be reproduced by filtering cropped patch tensors.
* The remediated sensitivity HDF5 contains patch-space injections and ML scores,
  not full-sky injected maps.
* A valid same-grid ML-vs-Wiener/SMHW benchmark requires positives and
  negatives scored by every method on the same sky realizations, coordinates,
  masks, amplitude/radius cells, and split policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENSITIVITY_H5 = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_sensitivity_curve"
    / "sensitivity_data.h5"
)
DEFAULT_ML_SCORES = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_sensitivity_curve"
    / "sensitivity_scores.npz"
)
DEFAULT_CLASSICAL_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_classical_fullsky"
    / "smica_mask090_wiener_smhw_scores.json"
)
DEFAULT_CLASSICAL_NPZ = DEFAULT_CLASSICAL_JSON.with_suffix(".npz")
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_classical_same_grid_status"
)
DEFAULT_BENCHMARK_REPORT = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_same_grid_fullsky_manifest" / "same_grid_fullsky_report.json"
)
REQUIRED_CLASSICAL_METHODS = ("wiener_feeney_matched_filter", "smhw_screen")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Report whether same-grid true Wiener/SMHW benchmarking is scientifically complete.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sensitivity-h5", type=str, default=str(DEFAULT_SENSITIVITY_H5))
    parser.add_argument("--ml-scores", type=str, default=str(DEFAULT_ML_SCORES))
    parser.add_argument("--classical-json", type=str, default=str(DEFAULT_CLASSICAL_JSON))
    parser.add_argument("--classical-npz", type=str, default=str(DEFAULT_CLASSICAL_NPZ))
    parser.add_argument("--benchmark-report", type=str, default=str(DEFAULT_BENCHMARK_REPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def file_info(path: Path) -> dict[str, Any]:
    """Return compact file-existence metadata."""

    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        info["size_bytes"] = int(stat.st_size)
    return info


def h5_summary(path: Path) -> dict[str, Any]:
    """Summarize the sensitivity HDF5 without loading large arrays."""

    if not path.exists():
        return {"exists": False}
    with h5py.File(path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
        summary = {
            "exists": True,
            "num_rows": int(labels.size),
            "num_positive": int(np.count_nonzero(labels)),
            "num_negative": int(np.count_nonzero(labels == 0)),
            "patch_shape": list(h5["patches"].shape),
            "has_full_sky_maps": any(
                key in h5
                for key in (
                    "full_sky_maps",
                    "injected_maps",
                    "healpix_maps",
                    "cmb_maps",
                )
            ),
            "has_coordinates": "metadata/glon_deg" in h5 and "metadata/glat_deg" in h5,
            "has_truth_grid": "truth/amplitude" in h5 and "truth/theta_crit_deg" in h5,
        }
    return summary


def score_npz_summary(path: Path) -> dict[str, Any]:
    """Summarize score arrays in an NPZ file."""

    if not path.exists():
        return {"exists": False}
    with np.load(path) as data:
        keys = list(data.files)
        return {
            "exists": True,
            "keys": keys,
            "has_labels": "labels" in keys,
            "classical_keys": [
                key
                for key in keys
                if key.startswith("score__wiener")
                or key.startswith("score__smhw")
                or "wiener_feeney_matched_filter" in key
                or "smhw_screen" in key
            ],
        }


def load_json_if_present(path: Path) -> dict[str, Any]:
    """Load JSON if present, otherwise return an empty object."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_status(
    sensitivity: dict[str, Any],
    ml_scores: dict[str, Any],
    classical_json: dict[str, Any],
    classical_npz: dict[str, Any],
    benchmark_report: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    """Return status, blockers, and warnings."""

    blockers = []
    warnings = []
    benchmark_methods = benchmark_report.get("method_metadata", {})
    benchmark_status = str(benchmark_report.get("benchmark_status", ""))
    if benchmark_report and all(method in benchmark_methods for method in REQUIRED_CLASSICAL_METHODS):
        if benchmark_status in ("stratified_same_grid_complete", "full_same_grid_complete"):
            warnings.append(
                f"Same-grid closure is satisfied by merged benchmark report status `{benchmark_status}`."
            )
            return "complete", blockers, warnings
    if not sensitivity.get("exists"):
        blockers.append("Missing remediated sensitivity HDF5.")
    elif not sensitivity.get("has_full_sky_maps"):
        blockers.append(
            "Sensitivity artifact is patch-space only; full-sky spherical filters cannot be applied post-crop."
        )
    if not ml_scores.get("exists") or not ml_scores.get("has_labels"):
        blockers.append("Missing ML sensitivity score file with labels.")
    methods = classical_json.get("methods", {})
    for method in REQUIRED_CLASSICAL_METHODS:
        if method not in methods:
            blockers.append(f"Classical full-sky metadata missing {method}.")
    if not classical_npz.get("exists"):
        blockers.append("Missing full-sky classical score-map NPZ.")
    if classical_npz.get("classical_keys"):
        warnings.append(
            "Full-sky classical score maps exist for real-sky screening, but they are not scores on injected full-sky simulations."
        )
    if blockers:
        return "blocked", blockers, warnings
    return "complete", blockers, warnings


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write Markdown status report."""

    lines = ["# True Classical Same-Grid Benchmark Status", ""]
    lines.append(f"Status: `{report['status']}`")
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if report["blockers"]:
        for item in report["blockers"]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Required To Close")
    lines.append("")
    for item in report["required_to_close"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Available Artifacts")
    lines.append("")
    lines.append(f"- sensitivity rows: `{report['sensitivity'].get('num_rows')}`")
    lines.append(f"- sensitivity patch shape: `{report['sensitivity'].get('patch_shape')}`")
    lines.append(f"- sensitivity has full-sky maps: `{report['sensitivity'].get('has_full_sky_maps')}`")
    lines.append(f"- ML score keys include labels: `{report['ml_scores'].get('has_labels')}`")
    lines.append(f"- full-sky classical score maps exist: `{report['classical_npz'].get('exists')}`")
    lines.append("")
    lines.append("## Non-Claim")
    lines.append("")
    lines.append(
        "Do not claim ML superiority over `wiener_feeney_matched_filter` or "
        "`smhw_screen` until this status is `complete`."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_h5 = Path(args.sensitivity_h5).expanduser().resolve()
    ml_scores_path = Path(args.ml_scores).expanduser().resolve()
    classical_json_path = Path(args.classical_json).expanduser().resolve()
    classical_npz_path = Path(args.classical_npz).expanduser().resolve()
    benchmark_report_path = Path(args.benchmark_report).expanduser().resolve()

    sensitivity = h5_summary(sensitivity_h5)
    ml_scores = score_npz_summary(ml_scores_path)
    classical_json = load_json_if_present(classical_json_path)
    classical_npz = score_npz_summary(classical_npz_path)
    benchmark_report = load_json_if_present(benchmark_report_path)
    status, blockers, warnings = evaluate_status(
        sensitivity,
        ml_scores,
        classical_json,
        classical_npz,
        benchmark_report,
    )
    report = {
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "inputs": {
            "sensitivity_h5": file_info(sensitivity_h5),
            "ml_scores": file_info(ml_scores_path),
            "classical_json": file_info(classical_json_path),
            "classical_npz": file_info(classical_npz_path),
            "benchmark_report": file_info(benchmark_report_path),
        },
        "sensitivity": sensitivity,
        "ml_scores": ml_scores,
        "classical_methods": classical_json.get("methods", {}),
        "classical_npz": classical_npz,
        "benchmark_report": benchmark_report,
        "required_to_close": [
            (
                "Generate or retain full-sky positive and negative maps for the same "
                "amplitude/radius/zcrit grid used by the ML sensitivity curve; use "
                "`--injection-convention mcewen2012_first_order_additive` for an "
                "explicit McEwen/OSS additive benchmark, or document the "
                "Feeney-vs-additive cross term if scoring Feeney-modulated maps."
            ),
            (
                "Use `scripts/phase3_same_grid_fullsky_benchmark.py --max-rows 0` "
                "as the guarded full-sky driver, or an equivalent audited run that "
                "writes patches and classical scores from the same injected maps."
            ),
            (
                "Run `scripts/phase3_classical_filters.py` on each full-sky "
                "realization using the same beam, pixel-window policy, mask, "
                "lmax, and C_l/noise model."
            ),
            "Extract per-sample classical scores at the same candidate coordinates or tile centers used for ML scoring.",
            "Calibrate thresholds on the same negative split and report recall/FPR with the same cell stratification and multiple-testing correction.",
        ],
    }
    json_path = output_dir / "classical_same_grid_status.json"
    md_path = output_dir / "classical_same_grid_status.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": status}, indent=2))


if __name__ == "__main__":
    main()
