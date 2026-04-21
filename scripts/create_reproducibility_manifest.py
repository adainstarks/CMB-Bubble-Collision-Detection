"""Create a compact reproducibility manifest for the active remediated flow.

Assumptions
-----------
* The active product is the remediated-v1 candidate-screening pipeline.
* Large HDF5 datasets and checkpoints are local artifacts; this manifest records
  their paths, sizes, timestamps, and embedded provenance rather than hashing
  multi-GB arrays by default.
* The manifest is a reproducibility aid, not a scientific result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_reproducibility"
)
ACTIVE_SOURCE_FILES = (
    "scripts/phase_config.py",
    "scripts/phase2_signal_model.py",
    "scripts/phase2_observing_model.py",
    "scripts/phase2_generate_training.py",
    "scripts/phase2_extract_smica_null_controls.py",
    "scripts/phase2_audit_dataset.py",
    "scripts/phase3_train_unet.py",
    "scripts/phase3_evaluate_run.py",
    "scripts/phase3_template_baseline.py",
    "scripts/phase3_classical_filters.py",
    "scripts/phase3_sensitivity_curve.py",
    "scripts/phase3_ml_gain_heatmap.py",
    "scripts/phase3_circular_template_features.py",
    "scripts/phase3_noise_floor_analysis.py",
    "scripts/phase3_matched_filter_snr_curve.py",
    "scripts/phase3_upper_limit_calculator.py",
    "scripts/phase3_deployment_burden_table.py",
    "scripts/phase3_policy_pareto_search.py",
    "scripts/phase3_deployment_policy_decision.py",
    "scripts/phase3_tile_constrained_policy_search.py",
    "scripts/phase3_emit_tile_constrained_candidates.py",
    "scripts/phase3_calibrate_candidate_scores.py",
    "scripts/phase3_classical_same_grid_status.py",
    "scripts/phase3_same_grid_fullsky_benchmark.py",
    "scripts/phase3_remediated_policy_tile_audit.py",
    "scripts/phase3_remediated_null_policy_audit.py",
    "scripts/phase3_mf_channel_tile_audit.py",
    "scripts/phase3_fullsky_tile.py",
    "scripts/phase3_geometry_router.py",
    "scripts/phase5_half_mission_signflip_null.py",
    "scripts/batch6_overnight_analysis.py",
    "scripts/audit_remediated_flow.py",
    "scripts/run_quality_gates.py",
)
KEY_ARTIFACTS = (
    "data/remediated_v1/summary.json",
    "data/remediated_v1/audit_report.json",
    "data/remediated_v1/training_data.h5",
    "data/remediated_v1/calibration_data.h5",
    "data/remediated_v1/test_data.h5",
    "runs/phase3_unet/remediated_v1_sensitivity_curve/sensitivity_report.json",
    (
        "runs/phase3_unet/remediated_v1_sensitivity_curve/"
        "ml_gain_heatmap_imagenet_preselected/ml_gain_heatmap.json"
    ),
    "runs/phase3_unet/remediated_v1_noise_floor/noise_floor_report.json",
    "runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_report.json",
    "runs/phase3_unet/remediated_v1_upper_limits/upper_limits.json",
    "runs/phase3_unet/remediated_v1_deployment_burden/deployment_burden.json",
    "runs/phase3_unet/remediated_v1_policy_pareto/policy_pareto.json",
    "runs/phase3_unet/remediated_v1_deployment_policy_decision/deployment_policy_decision.json",
    (
        "runs/phase3_unet/remediated_v1_tile_constrained_policy_search/"
        "tile_constrained_policy_search.json"
    ),
    (
        "runs/phase3_unet/remediated_v1_tile_constrained_candidates/"
        "candidate_emission_summary.json"
    ),
    (
        "runs/phase3_unet/remediated_v1_candidate_score_calibration/"
        "candidate_score_calibration.json"
    ),
    "runs/phase3_unet/phase5_half_mission_signflip_null/hm_signflip_preflight_report.json",
    "runs/phase3_unet/remediated_v1_classical_same_grid_status/classical_same_grid_status.json",
    "runs/phase3_unet/remediated_v1_same_grid_fullsky_pilot/same_grid_fullsky_report.json",
    "runs/phase3_unet/remediated_v1_policy_tile_audit/policy_tile_audit.json",
    "runs/phase3_unet/remediated_v1_null_policy_audit/null_policy_audit.json",
    "runs/phase3_unet/remediated_v1_mf_channel_tile_audit/mf_channel_tile_audit.json",
    "runs/phase3_unet/remediated_v1_classical_fullsky/smica_mask090_wiener_smhw_scores.json",
    "runs/phase3_unet/batch6_fullsky_nside32_smica/crossmap_recalibration_nside32.json",
    "runs/phase3_unet/phase5_half_mission_signflip_null/hm_signflip_null_report.json",
)
DEPENDENCIES = (
    "astropy",
    "camb",
    "h5py",
    "healpy",
    "matplotlib",
    "numpy",
    "scikit-learn",
    "scipy",
    "torch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a remediated-v1 reproducibility manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hash-large-files", action="store_true")
    parser.add_argument("--max-hash-bytes", type=int, default=8_000_000)
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        return ""
    return completed.stdout.strip()


def sha256_file(path: Path, max_bytes: int | None) -> str | None:
    if max_bytes is not None and path.stat().st_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path_text: str, *, hash_large: bool, max_hash_bytes: int) -> dict[str, Any]:
    path = PROJECT_ROOT / path_text
    record: dict[str, Any] = {"path": path_text, "exists": path.exists()}
    if not path.exists():
        return record
    stat = path.stat()
    record.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime_utc": (
                dt.datetime.utcfromtimestamp(stat.st_mtime)
                .replace(microsecond=0)
                .isoformat()
                + "Z"
            ),
        }
    )
    max_bytes = None if hash_large else int(max_hash_bytes)
    digest = sha256_file(path, max_bytes)
    if digest is None:
        record["sha256"] = None
        record["sha256_note"] = "Skipped because file exceeds max hash size."
    else:
        record["sha256"] = digest
    return record


def load_json_if_present(path_text: str) -> dict[str, Any] | None:
    path = PROJECT_ROOT / path_text
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dependency_versions() -> dict[str, str]:
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def summarize_science_state() -> dict[str, Any]:
    dataset_summary = load_json_if_present("data/remediated_v1/summary.json") or {}
    dataset_audit = load_json_if_present("data/remediated_v1/audit_report.json") or {}
    sensitivity = load_json_if_present(
        "runs/phase3_unet/remediated_v1_sensitivity_curve/sensitivity_report.json"
    ) or {}
    heatmap = load_json_if_present(
        (
            "runs/phase3_unet/remediated_v1_sensitivity_curve/"
            "ml_gain_heatmap_imagenet_preselected/ml_gain_heatmap.json"
        )
    ) or {}
    batch6 = load_json_if_present(
        "runs/phase3_unet/batch6_fullsky_nside32_smica/crossmap_recalibration_nside32.json"
    ) or {}
    upper_limits = load_json_if_present(
        "runs/phase3_unet/remediated_v1_upper_limits/upper_limits.json"
    ) or {}
    matched_filter_snr = load_json_if_present(
        "runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_report.json"
    ) or {}
    deployment_burden = load_json_if_present(
        "runs/phase3_unet/remediated_v1_deployment_burden/deployment_burden.json"
    ) or {}
    policy_pareto = load_json_if_present(
        "runs/phase3_unet/remediated_v1_policy_pareto/policy_pareto.json"
    ) or {}
    deployment_policy_decision = load_json_if_present(
        "runs/phase3_unet/remediated_v1_deployment_policy_decision/deployment_policy_decision.json"
    ) or {}
    tile_constrained_policy_search = load_json_if_present(
        (
            "runs/phase3_unet/remediated_v1_tile_constrained_policy_search/"
            "tile_constrained_policy_search.json"
        )
    ) or {}
    tile_constrained_candidates = load_json_if_present(
        (
            "runs/phase3_unet/remediated_v1_tile_constrained_candidates/"
            "candidate_emission_summary.json"
        )
    ) or {}
    candidate_score_calibration = load_json_if_present(
        (
            "runs/phase3_unet/remediated_v1_candidate_score_calibration/"
            "candidate_score_calibration.json"
        )
    ) or {}
    phase5_hm_preflight = load_json_if_present(
        "runs/phase3_unet/phase5_half_mission_signflip_null/hm_signflip_preflight_report.json"
    ) or {}
    classical_same_grid_status = load_json_if_present(
        "runs/phase3_unet/remediated_v1_classical_same_grid_status/classical_same_grid_status.json"
    ) or {}
    policy_tile_audit = load_json_if_present(
        "runs/phase3_unet/remediated_v1_policy_tile_audit/policy_tile_audit.json"
    ) or {}
    null_policy_audit = load_json_if_present(
        "runs/phase3_unet/remediated_v1_null_policy_audit/null_policy_audit.json"
    ) or {}
    mf_channel_tile_audit = load_json_if_present(
        "runs/phase3_unet/remediated_v1_mf_channel_tile_audit/mf_channel_tile_audit.json"
    ) or {}
    thresholds = sensitivity.get("thresholds", {})
    multiple_testing = heatmap.get("multiple_testing", {})
    return {
        "dataset": {
            "num_samples": dataset_summary.get("num_samples"),
            "geometry_mode": dataset_summary.get("geometry_mode"),
            "mask_threshold": dataset_summary.get("mask_threshold"),
            "beam_fwhm_arcmin": dataset_summary.get("beam_fwhm_arcmin"),
            "pixel_window_policy": dataset_summary.get("pixel_window_policy"),
            "audit_status": dataset_audit.get("status"),
            "dataset_sha256": dataset_summary.get("dataset_sha256"),
        },
        "sensitivity": {
            "fpr_target": sensitivity.get("fpr_target"),
            "num_positive": sensitivity.get("num_positive"),
            "num_negative": sensitivity.get("num_negative"),
            "threshold_methods": sorted(thresholds),
            "imagenet_threshold": thresholds.get("imagenet_b64_aux", {}).get("threshold"),
            "circular_template_threshold": thresholds.get(
                "circular_template_screen",
                {},
            ).get("threshold"),
        },
        "heatmap": {
            "analysis_mode": heatmap.get("analysis_mode"),
            "primary_method": heatmap.get("primary_method"),
            "winner_counts": heatmap.get("winner_counts"),
            "significant_winner_counts": heatmap.get("significant_winner_counts"),
            "holm_alpha_005_count": multiple_testing.get("holm_alpha_005_count"),
            "bh_fdr_005_count": multiple_testing.get("bh_fdr_005_count"),
        },
        "batch6": {
            "summary_keys": sorted(batch6.keys())[:20],
            "has_crossmap_recalibration": bool(batch6),
        },
        "upper_limits": {
            "confidence": upper_limits.get("metadata", {}).get("confidence"),
            "amplitude_prior": upper_limits.get("metadata", {}).get("amplitude_prior"),
            "theta_prior": upper_limits.get("metadata", {}).get("theta_prior"),
            "limits": [
                {
                    "method": row.get("method"),
                    "mean_efficiency": row.get("mean_efficiency"),
                    "nbar_s_upper": row.get("nbar_s_upper"),
                }
                for row in upper_limits.get("limits", [])
            ],
        },
        "matched_filter_snr": {
            "lmax": matched_filter_snr.get("config", {}).get("lmax"),
            "beam_fwhm_arcmin": matched_filter_snr.get("config", {}).get(
                "beam_fwhm_arcmin"
            ),
            "noise_sigma_uk_arcmin": matched_filter_snr.get("config", {}).get(
                "noise_sigma_uk_arcmin"
            ),
            "low_ideal_recall_cell_count": sum(
                1
                for row in matched_filter_snr.get("cell_rows", [])
                if float(row.get("ideal_recall_fsky_scaled_median", 0.0)) < 0.2
            ),
            "high_ideal_recall_cell_count": sum(
                1
                for row in matched_filter_snr.get("cell_rows", [])
                if float(row.get("ideal_recall_fsky_scaled_median", 0.0)) > 0.8
            ),
            "max_median_fsky_snr": max(
                [
                    float(row.get("snr_fsky_scaled_median", 0.0))
                    for row in matched_filter_snr.get("cell_rows", [])
                ]
                or [None]
            ),
        },
        "deployment_burden": {
            "confidence": deployment_burden.get("metadata", {}).get("confidence"),
            "bootstrap_block_nside": deployment_burden.get("metadata", {}).get(
                "bootstrap_block_nside"
            ),
            "patch_summary": deployment_burden.get("summary", {}).get("patch", []),
            "cluster_summary": [
                row
                for row in deployment_burden.get("summary", {}).get("cluster", [])
                if row.get("threshold_mode") == "tile_recalibrated"
                and row.get("cluster_radius_deg") == 15.0
            ],
        },
        "policy_pareto": {
            "num_policy_rows_searched": policy_pareto.get("metadata", {}).get(
                "num_policy_rows_searched"
            ),
            "top_rows": [
                {
                    "constraint_camb_fpr_max": row.get("constraint_camb_fpr_max"),
                    "constraint_real_fpr_max": row.get("constraint_real_fpr_max"),
                    "policy": row.get("policy"),
                    "family": row.get("family"),
                    "camb_fpr": row.get("camb_fpr"),
                    "real_fpr": row.get("real_fpr"),
                    "camb_recall": row.get("camb_recall"),
                    "real_recall": row.get("real_recall"),
                    "best_single_policy": row.get("best_single_policy"),
                    "best_single_real_recall": row.get("best_single_real_recall"),
                    "real_recall_gain_vs_best_single": row.get(
                        "real_recall_gain_vs_best_single"
                    ),
                }
                for row in policy_pareto.get("top_rows", [])
                if row.get("rank") == 1
            ],
        },
        "deployment_policy_decision": {
            "num_promotable": deployment_policy_decision.get("num_promotable"),
            "constraints": deployment_policy_decision.get("constraints", {}),
            "top_rows": [
                {
                    "policy_key": row.get("policy_key"),
                    "diagnostic_real_recall": row.get("diagnostic_real_recall"),
                    "gain_vs_exact_single": row.get("gain_vs_exact_single"),
                    "pooled_null_fpr_ci_high": row.get("pooled_null_fpr_ci_high"),
                    "max_clusters": row.get("max_clusters"),
                    "max_clusters_map": row.get("max_clusters_map"),
                    "max_trigger_fraction": row.get("max_trigger_fraction"),
                    "max_trigger_fraction_map": row.get("max_trigger_fraction_map"),
                    "promotable_under_defaults": row.get("promotable_under_defaults"),
                    "failure_reasons": row.get("failure_reasons"),
                }
                for row in deployment_policy_decision.get("rows", [])[:5]
            ],
        },
        "tile_constrained_policy_search": {
            "constraints": tile_constrained_policy_search.get("constraints", {}),
            "diagnostics": tile_constrained_policy_search.get("diagnostics", {}),
            "best_feasible_single": {
                "policy": (
                    tile_constrained_policy_search.get("best_feasible_single_row")
                    or {}
                ).get("policy"),
                "real_recall": (
                    tile_constrained_policy_search.get("best_feasible_single_row")
                    or {}
                ).get("real_recall"),
                "real_fpr": (
                    tile_constrained_policy_search.get("best_feasible_single_row")
                    or {}
                ).get("real_fpr"),
                "max_clusters": (
                    tile_constrained_policy_search.get("best_feasible_single_row")
                    or {}
                ).get("max_clusters"),
            },
            "top_rows": [
                {
                    "policy": row.get("policy"),
                    "family": row.get("family"),
                    "real_recall": row.get("real_recall"),
                    "real_fpr": row.get("real_fpr"),
                    "pooled_null_fpr_ci_high": row.get("pooled_null_fpr_ci_high"),
                    "real_recall_gain_vs_best_feasible_single": row.get(
                        "real_recall_gain_vs_best_feasible_single"
                    ),
                    "max_clusters": row.get("max_clusters"),
                    "max_clusters_map": row.get("max_clusters_map"),
                    "max_trigger_fraction": row.get("max_trigger_fraction"),
                    "max_trigger_fraction_map": row.get("max_trigger_fraction_map"),
                }
                for row in tile_constrained_policy_search.get("top_rows", [])[:5]
            ],
        },
        "tile_constrained_candidates": {
            "policy_slug": tile_constrained_candidates.get("policy_slug"),
            "min_mask_fraction": tile_constrained_candidates.get("min_mask_fraction"),
            "cluster_radius_deg": tile_constrained_candidates.get("cluster_radius_deg"),
            "totals": tile_constrained_candidates.get("totals", {}),
            "map_rows": [
                {
                    "map": row.get("map"),
                    "num_tiles_passing_mask_fraction": row.get(
                        "num_tiles_passing_mask_fraction"
                    ),
                    "num_tile_candidates": row.get("num_tile_candidates"),
                    "eligible_trigger_fraction": row.get("eligible_trigger_fraction"),
                    "num_cluster_representatives": row.get(
                        "num_cluster_representatives"
                    ),
                    "max_cluster_size": row.get("max_cluster_size"),
                }
                for row in tile_constrained_candidates.get("map_rows", [])
            ],
            "artifacts": tile_constrained_candidates.get("artifacts", {}),
        },
        "candidate_score_calibration": {
            "policy_slug": candidate_score_calibration.get("policy_slug"),
            "num_candidates": candidate_score_calibration.get("num_candidates"),
            "pooled_null_count": candidate_score_calibration.get("pooled_null_count"),
            "map_rows": candidate_score_calibration.get("map_rows", []),
            "top_candidates": [
                {
                    "map": row.get("map"),
                    "cluster_id": row.get("cluster_id"),
                    "policy_margin": row.get("policy_margin"),
                    "calibration_pooled_survival_p": row.get(
                        "calibration_pooled_survival_p"
                    ),
                    "calibration_pooled_bh_q": row.get(
                        "calibration_pooled_bh_q"
                    ),
                }
                for row in candidate_score_calibration.get("top_candidates", [])[:5]
            ],
            "artifacts": candidate_score_calibration.get("artifacts", {}),
        },
        "phase5_hm_preflight": {
            "status": phase5_hm_preflight.get("status"),
            "num_failures": phase5_hm_preflight.get("num_failures"),
            "num_warnings": phase5_hm_preflight.get("num_warnings"),
            "run_settings": phase5_hm_preflight.get("run_settings", {}),
            "candidate_summary": phase5_hm_preflight.get("candidate_summary", {}),
            "map_info": phase5_hm_preflight.get("map_info", {}),
            "issues": phase5_hm_preflight.get("issues", []),
        },
        "classical_same_grid_status": {
            "status": classical_same_grid_status.get("status"),
            "blockers": classical_same_grid_status.get("blockers", []),
            "sensitivity_has_full_sky_maps": classical_same_grid_status.get(
                "sensitivity",
                {},
            ).get("has_full_sky_maps"),
        },
        "policy_tile_audit": {
            "tile_nside": policy_tile_audit.get("tile_nside"),
            "maps": policy_tile_audit.get("maps", []),
            "rows": [
                {
                    "map": row.get("map"),
                    "policy_slug": row.get("policy_slug"),
                    "diagnostic_real_recall": row.get("diagnostic_real_recall"),
                    "diagnostic_real_fpr": row.get("diagnostic_real_fpr"),
                    "num_triggered_tiles": row.get("num_triggered_tiles"),
                    "num_tiles": row.get("num_tiles"),
                    "trigger_fraction": row.get("trigger_fraction"),
                    "clusters_15deg": row.get("cluster_summary", {})
                    .get("15.0", {})
                    .get("n_clusters"),
                }
                for row in policy_tile_audit.get("rows", [])
            ],
        },
        "null_policy_audit": {
            "split": null_policy_audit.get("split"),
            "mask_tag": null_policy_audit.get("mask_tag"),
            "pooled_rows": [
                {
                    "constraint_camb_fpr_max": row.get("constraint_camb_fpr_max"),
                    "constraint_real_fpr_max": row.get("constraint_real_fpr_max"),
                    "diagnostic_real_recall": row.get("diagnostic_real_recall"),
                    "diagnostic_real_fpr_200": row.get("diagnostic_real_fpr_200"),
                    "num_null": row.get("num_null"),
                    "false_positive_count": row.get("false_positive_count"),
                    "false_positive_rate": row.get("false_positive_rate"),
                    "fpr_ci_low": row.get("fpr_ci_low"),
                    "fpr_ci_high": row.get("fpr_ci_high"),
                }
                for row in null_policy_audit.get("pooled_rows", [])
            ],
        },
        "mf_channel_tile_audit": {
            "model": mf_channel_tile_audit.get("model", {}),
            "threshold": mf_channel_tile_audit.get("threshold", {}),
            "settings": mf_channel_tile_audit.get("settings", {}),
            "diagnostic_global_rows": (
                mf_channel_tile_audit.get("diagnostic_context", {})
                .get("global_model_only_rows", [])
            ),
            "diagnostic_regime_rows": (
                mf_channel_tile_audit.get("diagnostic_context", {})
                .get("real_smica_model_only_regime_rows", [])
            ),
            "map_rows": [
                {
                    "map": row.get("map"),
                    "num_tiles": row.get("num_tiles"),
                    "num_triggered_tiles": row.get("num_triggered_tiles"),
                    "trigger_fraction": row.get("trigger_fraction"),
                    "num_tiles_passing_mask_fraction": row.get(
                        "num_tiles_passing_mask_fraction"
                    ),
                    "num_triggered_eligible_tiles": row.get(
                        "num_triggered_eligible_tiles"
                    ),
                    "eligible_trigger_fraction": row.get(
                        "eligible_trigger_fraction"
                    ),
                    "num_eligible_clusters": row.get("num_eligible_clusters"),
                }
                for row in mf_channel_tile_audit.get("map_rows", [])
            ],
            "assumption_notes": mf_channel_tile_audit.get("assumption_notes", []),
        },
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    status_short = run_git(["status", "--short"]).splitlines()
    manifest = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "project_root": str(PROJECT_ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "dependencies": dependency_versions(),
        "git": {
            "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": run_git(["rev-parse", "HEAD"]),
            "status_short": status_short,
            "is_dirty": bool(status_short),
        },
        "active_source_files": [
            file_record(path, hash_large=False, max_hash_bytes=int(args.max_hash_bytes))
            for path in ACTIVE_SOURCE_FILES
        ],
        "key_artifacts": [
            file_record(
                path,
                hash_large=bool(args.hash_large_files),
                max_hash_bytes=int(args.max_hash_bytes),
            )
            for path in KEY_ARTIFACTS
        ],
        "science_state": summarize_science_state(),
        "notes": [
            "Generated HDF5 datasets and checkpoints are local artifacts.",
            (
                "Use scripts/audit_remediated_flow.py and "
                "scripts/run_quality_gates.py before relying on results."
            ),
            "Dirty git status is recorded for transparency; it is not automatically a failure.",
        ],
    }
    return manifest


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    state = manifest["science_state"]
    lines = ["# Remediated v1 Reproducibility Manifest", ""]
    lines.append(f"Created UTC: `{manifest['created_utc']}`")
    lines.append(f"Git branch: `{manifest['git']['branch']}`")
    lines.append(f"Git head: `{manifest['git']['head']}`")
    lines.append(f"Dirty working tree: `{manifest['git']['is_dirty']}`")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    dataset = state["dataset"]
    for key in (
        "num_samples",
        "geometry_mode",
        "mask_threshold",
        "beam_fwhm_arcmin",
        "pixel_window_policy",
        "audit_status",
        "dataset_sha256",
    ):
        lines.append(f"- `{key}`: `{dataset.get(key)}`")
    lines.append("")
    lines.append("## Sensitivity")
    lines.append("")
    sensitivity = state["sensitivity"]
    for key in (
        "fpr_target",
        "num_positive",
        "num_negative",
        "threshold_methods",
        "imagenet_threshold",
        "circular_template_threshold",
    ):
        lines.append(f"- `{key}`: `{sensitivity.get(key)}`")
    lines.append("")
    lines.append("## Heatmap")
    lines.append("")
    heatmap = state["heatmap"]
    for key in (
        "analysis_mode",
        "primary_method",
        "winner_counts",
        "holm_alpha_005_count",
        "bh_fdr_005_count",
    ):
        lines.append(f"- `{key}`: `{heatmap.get(key)}`")
    lines.append("")
    lines.append("## Matched-Filter SNR")
    lines.append("")
    matched_snr = state["matched_filter_snr"]
    for key in (
        "lmax",
        "beam_fwhm_arcmin",
        "noise_sigma_uk_arcmin",
        "low_ideal_recall_cell_count",
        "high_ideal_recall_cell_count",
        "max_median_fsky_snr",
    ):
        lines.append(f"- `{key}`: `{matched_snr.get(key)}`")
    lines.append("")
    lines.append("## Upper Limits")
    lines.append("")
    upper_limits = state["upper_limits"]
    lines.append(f"- `confidence`: `{upper_limits.get('confidence')}`")
    lines.append(f"- `amplitude_prior`: `{upper_limits.get('amplitude_prior')}`")
    lines.append(f"- `theta_prior`: `{upper_limits.get('theta_prior')}`")
    for row in upper_limits.get("limits", []):
        lines.append(
            f"- `{row.get('method')}`: mean_efficiency=`{row.get('mean_efficiency')}`, "
            f"Nbar_s_upper=`{row.get('nbar_s_upper')}`"
        )
    lines.append("")
    lines.append("## Deployment Burden")
    lines.append("")
    burden = state["deployment_burden"]
    lines.append(f"- `confidence`: `{burden.get('confidence')}`")
    lines.append(f"- `bootstrap_block_nside`: `{burden.get('bootstrap_block_nside')}`")
    for row in burden.get("patch_summary", []):
        if row.get("threshold_mode") != "tile_recalibrated":
            continue
        lines.append(
            f"- `{row.get('method')}` tile-recalibrated: "
            f"mean_patch_candidates=`{row.get('mean_patch_candidates')}`, "
            f"mean_recall=`{row.get('mean_recall')}`"
        )
    for row in burden.get("cluster_summary", []):
        lines.append(
            f"- `{row.get('method')}` tile-recalibrated radius "
            f"`{row.get('cluster_radius_deg')}` deg: "
            f"mean_clusters=`{row.get('mean_clusters')}`"
        )
    lines.append("")
    lines.append("## Policy Pareto")
    lines.append("")
    pareto = state["policy_pareto"]
    lines.append(f"- `num_policy_rows_searched`: `{pareto.get('num_policy_rows_searched')}`")
    for row in pareto.get("top_rows", []):
        lines.append(
            f"- CAMB<=`{row.get('constraint_camb_fpr_max')}`, "
            f"real<=`{row.get('constraint_real_fpr_max')}`: "
            f"`{row.get('policy')}` "
            f"real_recall=`{row.get('real_recall')}`, real_fpr=`{row.get('real_fpr')}`, "
            "gain_vs_best_single="
            f"`{row.get('real_recall_gain_vs_best_single')}`"
        )
    lines.append("")
    lines.append("## Deployment Policy Decision")
    lines.append("")
    decision = state["deployment_policy_decision"]
    lines.append(f"- `num_promotable`: `{decision.get('num_promotable')}`")
    lines.append(f"- `constraints`: `{decision.get('constraints')}`")
    for row in decision.get("top_rows", []):
        lines.append(
            f"- `{row.get('policy_key')}`: recall=`{row.get('diagnostic_real_recall')}`, "
            f"gain=`{row.get('gain_vs_exact_single')}`, "
            f"max_clusters=`{row.get('max_clusters')}` on `{row.get('max_clusters_map')}`, "
            f"max_trigger_fraction=`{row.get('max_trigger_fraction')}` on "
            f"`{row.get('max_trigger_fraction_map')}`, "
            f"promotable=`{row.get('promotable_under_defaults')}`, "
            f"blockers=`{row.get('failure_reasons')}`"
        )
    lines.append("")
    lines.append("## Tile-Constrained Policy Search")
    lines.append("")
    tile_search = state["tile_constrained_policy_search"]
    lines.append(f"- `constraints`: `{tile_search.get('constraints')}`")
    lines.append(f"- `diagnostics`: `{tile_search.get('diagnostics')}`")
    baseline = tile_search.get("best_feasible_single", {})
    lines.append(
        f"- best feasible single: `{baseline.get('policy')}` "
        f"real_recall=`{baseline.get('real_recall')}`, "
        f"real_fpr=`{baseline.get('real_fpr')}`, "
        f"max_clusters=`{baseline.get('max_clusters')}`"
    )
    for row in tile_search.get("top_rows", []):
        lines.append(
            f"- `{row.get('policy')}`: recall=`{row.get('real_recall')}`, "
            f"gain_vs_single=`{row.get('real_recall_gain_vs_best_feasible_single')}`, "
            f"real_fpr=`{row.get('real_fpr')}`, "
            f"null_ci_high=`{row.get('pooled_null_fpr_ci_high')}`, "
            f"max_clusters=`{row.get('max_clusters')}` on `{row.get('max_clusters_map')}`, "
            f"max_trigger_fraction=`{row.get('max_trigger_fraction')}` on "
            f"`{row.get('max_trigger_fraction_map')}`"
        )
    lines.append("")
    lines.append("## Tile-Constrained Candidates")
    lines.append("")
    candidates = state["tile_constrained_candidates"]
    lines.append(f"- `policy_slug`: `{candidates.get('policy_slug')}`")
    lines.append(f"- `min_mask_fraction`: `{candidates.get('min_mask_fraction')}`")
    lines.append(f"- `cluster_radius_deg`: `{candidates.get('cluster_radius_deg')}`")
    lines.append(f"- `totals`: `{candidates.get('totals')}`")
    for row in candidates.get("map_rows", []):
        lines.append(
            f"- `{row.get('map')}`: eligible_tiles="
            f"`{row.get('num_tiles_passing_mask_fraction')}`, "
            f"tile_candidates=`{row.get('num_tile_candidates')}`, "
            f"eligible_trigger_fraction=`{row.get('eligible_trigger_fraction')}`, "
            f"cluster_representatives=`{row.get('num_cluster_representatives')}`, "
            f"max_cluster_size=`{row.get('max_cluster_size')}`"
        )
    artifacts = candidates.get("artifacts", {})
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Candidate Score Calibration")
    lines.append("")
    calibration = state["candidate_score_calibration"]
    lines.append(f"- `policy_slug`: `{calibration.get('policy_slug')}`")
    lines.append(f"- `num_candidates`: `{calibration.get('num_candidates')}`")
    lines.append(f"- `pooled_null_count`: `{calibration.get('pooled_null_count')}`")
    for row in calibration.get("map_rows", []):
        lines.append(
            f"- `{row.get('map')}`: null_count=`{row.get('null_count')}`, "
            f"policy_pass_fraction=`{row.get('policy_pass_fraction')}`, "
            f"candidates=`{row.get('candidate_count')}`, "
            f"min_pooled_p=`{row.get('min_pooled_survival_p')}`, "
            f"min_q=`{row.get('min_pooled_bh_q')}`"
        )
    for row in calibration.get("top_candidates", []):
        lines.append(
            f"- top `{row.get('map')}` cluster `{row.get('cluster_id')}`: "
            f"margin=`{row.get('policy_margin')}`, "
            f"pooled_p=`{row.get('calibration_pooled_survival_p')}`, "
            f"q=`{row.get('calibration_pooled_bh_q')}`"
        )
    for key, value in calibration.get("artifacts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Phase 5 HM Preflight")
    lines.append("")
    hm_preflight = state["phase5_hm_preflight"]
    lines.append(f"- `status`: `{hm_preflight.get('status')}`")
    lines.append(f"- `num_failures`: `{hm_preflight.get('num_failures')}`")
    lines.append(f"- `num_warnings`: `{hm_preflight.get('num_warnings')}`")
    run_settings = hm_preflight.get("run_settings", {})
    lines.append(
        f"- `smallest_possible_empirical_p`: "
        f"`{run_settings.get('smallest_possible_empirical_p')}`"
    )
    candidate_summary = hm_preflight.get("candidate_summary", {})
    lines.append(f"- `candidate_summary`: `{candidate_summary}`")
    for issue in hm_preflight.get("issues", []):
        lines.append(
            f"- `{issue.get('severity')}` `{issue.get('check')}`: "
            f"{issue.get('message')}"
        )
    lines.append("")
    lines.append("## Classical Same-Grid Status")
    lines.append("")
    classical_status = state["classical_same_grid_status"]
    lines.append(f"- `status`: `{classical_status.get('status')}`")
    lines.append(
        f"- `sensitivity_has_full_sky_maps`: "
        f"`{classical_status.get('sensitivity_has_full_sky_maps')}`"
    )
    for blocker in classical_status.get("blockers", []):
        lines.append(f"- blocker: {blocker}")
    lines.append("")
    lines.append("## Policy Tile Audit")
    lines.append("")
    tile = state["policy_tile_audit"]
    lines.append(f"- `tile_nside`: `{tile.get('tile_nside')}`")
    lines.append(f"- `maps`: `{tile.get('maps')}`")
    for row in tile.get("rows", []):
        lines.append(
            f"- `{row.get('map')}` `{row.get('policy_slug')}`: "
            f"tiles=`{row.get('num_triggered_tiles')}/{row.get('num_tiles')}`, "
            f"trigger_fraction=`{row.get('trigger_fraction')}`, "
            f"clusters_15deg=`{row.get('clusters_15deg')}`, "
            f"diagnostic_real_recall=`{row.get('diagnostic_real_recall')}`"
        )
    lines.append("")
    lines.append("## MF-Channel Tile Audit")
    lines.append("")
    mf_audit = state["mf_channel_tile_audit"]
    mf_model = mf_audit.get("model", {})
    mf_threshold = mf_audit.get("threshold", {})
    lines.append(f"- `model`: `{mf_model.get('name')}`")
    lines.append(f"- `checkpoint`: `{mf_model.get('checkpoint_label')}`")
    lines.append(f"- `selected_threshold`: `{mf_threshold.get('selected_threshold')}`")
    for row in mf_audit.get("diagnostic_global_rows", []):
        lines.append(
            f"- diagnostic `{row.get('domain')}`: recall=`{row.get('recall')}`, "
            f"FPR=`{row.get('fpr')}`, precision=`{row.get('precision')}`"
        )
    for row in mf_audit.get("map_rows", []):
        lines.append(
            f"- `{row.get('map')}`: eligible_tiles="
            f"`{row.get('num_tiles_passing_mask_fraction')}`, "
            f"eligible_triggers=`{row.get('num_triggered_eligible_tiles')}`, "
            f"eligible_trigger_fraction=`{row.get('eligible_trigger_fraction')}`, "
            f"clusters_15deg=`{row.get('num_eligible_clusters')}`"
        )
    for note in mf_audit.get("assumption_notes", []):
        lines.append(f"- note: {note}")
    lines.append("")
    lines.append("## Null Policy Audit")
    lines.append("")
    null_policy = state["null_policy_audit"]
    lines.append(f"- `split`: `{null_policy.get('split')}`")
    lines.append(f"- `mask_tag`: `{null_policy.get('mask_tag')}`")
    for row in null_policy.get("pooled_rows", []):
        lines.append(
            f"- CAMB<=`{row.get('constraint_camb_fpr_max')}`, "
            f"real<=`{row.get('constraint_real_fpr_max')}`: "
            f"FP=`{row.get('false_positive_count')}/{row.get('num_null')}`, "
            f"FPR=`{row.get('false_positive_rate')}`, "
            f"95% CI=`[{row.get('fpr_ci_low')}, {row.get('fpr_ci_high')}]`"
        )
    lines.append("")
    lines.append("## Dependencies")
    lines.append("")
    for name, version in manifest["dependencies"].items():
        lines.append(f"- `{name}`: `{version}`")
    lines.append("")
    lines.append("## Dirty Files")
    lines.append("")
    if manifest["git"]["status_short"]:
        for row in manifest["git"]["status_short"]:
            lines.append(f"- `{row}`")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    json_path = output_dir / "reproducibility_manifest.json"
    md_path = output_dir / "reproducibility_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(md_path, manifest)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
