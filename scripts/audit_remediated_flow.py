"""Audit the remediated-v1 upstream/downstream artifact flow.

Assumptions
-----------
* The active scientific product is a candidate-screening pipeline, not a
  cosmological detection or Bayesian-evidence calculation.
* Remediated-v1 artifacts use Feeney-style full-temperature modulation, Planck
  5 arcmin beam handling, ``synfast(pixwin=True)``, and cluster-aware splits.
* This audit is intentionally lightweight: it verifies schema, provenance,
  split accounting, and report consistency without regenerating maps or
  rerunning neural-network inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from phase2_audit_dataset import injection_metadata_required
from phase_config import PROVENANCE_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "remediated_v1"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "phase3_unet"
MAPS = ("smica", "nilc", "sevem", "commander")
MASK_THRESHOLDS = {"mask090": 0.90, "mask050": 0.50}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate remediated-v1 data, null controls, and key reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--runs-dir", type=str, default=str(DEFAULT_RUNS_DIR))
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_RUNS_DIR / "remediated_v1_flow_audit.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_close(value: Any, expected: float, tolerance: float = 1.0e-8) -> bool:
    try:
        return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


class Audit:
    """Accumulate audit failures, warnings, and metrics."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.metrics: dict[str, Any] = {}

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def require_path(self, path: Path, label: str) -> bool:
        if not path.exists():
            self.fail(f"Missing {label}: {path}")
            return False
        return True

    def require_equal(self, actual: Any, expected: Any, label: str) -> None:
        if actual != expected:
            self.fail(f"{label}: expected {expected!r}, found {actual!r}.")

    def require_close(self, actual: Any, expected: float, label: str, tolerance: float = 1.0e-8) -> None:
        if not is_close(actual, expected, tolerance=tolerance):
            self.fail(f"{label}: expected {expected}, found {actual!r}.")

    def require_true(self, condition: bool, label: str) -> None:
        if not condition:
            self.fail(label)


def audit_dataset(data_dir: Path, audit: Audit) -> None:
    summary_path = data_dir / "summary.json"
    report_path = data_dir / "audit_report.json"
    if not audit.require_path(summary_path, "remediated summary"):
        return
    if not audit.require_path(report_path, "remediated dataset audit"):
        return

    summary = load_json(summary_path)
    report = load_json(report_path)
    metrics = report.get("metrics", {})
    audit.metrics["dataset"] = {
        "num_samples": summary.get("num_samples"),
        "mask_threshold": summary.get("mask_threshold"),
        "geometry_mode": summary.get("geometry_mode"),
        "beam_fwhm_arcmin": summary.get("beam_fwhm_arcmin"),
        "pixel_window_policy": summary.get("pixel_window_policy"),
        "injection_convention": summary.get("injection_convention"),
    }

    audit.require_equal(report.get("status"), "pass", "data/remediated_v1 audit status")
    audit.require_equal(summary.get("num_samples"), 20000, "remediated num_samples")
    audit.require_equal(summary.get("num_train_samples"), 16000, "remediated train samples")
    audit.require_equal(summary.get("num_calibration_samples"), 2000, "remediated calibration samples")
    audit.require_equal(summary.get("num_test_samples"), 2000, "remediated test samples")
    audit.require_equal(summary.get("geometry_mode"), "mixed", "remediated geometry mode")
    audit.require_close(summary.get("mask_threshold"), 0.90, "remediated mask threshold")
    audit.require_close(summary.get("beam_fwhm_arcmin"), 5.0, "remediated beam FWHM")
    audit.require_equal(summary.get("beam_domain"), "harmonic_sphere", "remediated beam domain")
    audit.require_equal(summary.get("pixel_window_policy"), "synfast_pixwin_true", "remediated pixel window")
    if summary.get("injection_convention") is None:
        message = (
            "remediated_v1 summary predates explicit injection_convention metadata; "
            "regenerated products must record Feeney full-temperature modulation."
        )
        if injection_metadata_required(summary):
            audit.fail(message)
        else:
            audit.warn(message)
    elif injection_metadata_required(summary):
        audit.require_equal(
            summary.get("provenance_schema_version"),
            PROVENANCE_SCHEMA_VERSION,
            "remediated provenance schema version",
        )
    audit.require_true(
        int(summary.get("num_positive_touching_edge", 0)) > 0,
        "remediated data must include edge-touching positives.",
    )
    audit.require_true(
        float(summary.get("min_positive_visible_target_fraction", 0.0)) >= 0.15 - 1.0e-6,
        "visible target fraction dropped below configured lower bound.",
    )

    observing_model_text = summary.get("observing_model", "{}")
    try:
        observing_model = json.loads(observing_model_text)
    except json.JSONDecodeError:
        audit.fail("summary.observing_model is not valid JSON.")
        observing_model = {}
    audit.require_equal(observing_model.get("pixwin"), True, "observing model pixwin")
    audit.require_close(observing_model.get("beam_fwhm_arcmin"), 5.0, "observing model beam FWHM")
    camb = observing_model.get("camb", {})
    audit.require_equal(camb.get("CMB_unit"), "K", "CAMB CMB_unit")

    for key in (
        "shared_coord_pool_idx_count",
        "shared_cmb_realization_idx_count",
        "shared_background_id_count",
        "shared_split_group_id_count",
        "shared_event_id_count",
    ):
        audit.require_equal(metrics.get(key), 0, f"dataset leakage metric {key}")
    audit.require_close(metrics.get("patch_finite_fraction"), 1.0, "patch finite fraction")
    audit.require_true(
        float(metrics.get("coordinate_pool_mask_fraction_min", 0.0)) >= 0.90 - 1.0e-6,
        "coordinate pool contains patches below canonical mask threshold.",
    )

    for filename in ("training_data.h5", "calibration_data.h5", "test_data.h5"):
        audit.require_path(data_dir / filename, filename)


def audit_null_controls(data_dir: Path, audit: Audit) -> None:
    null_metrics: dict[str, Any] = {}
    for map_name in MAPS:
        for mask_tag, expected_threshold in MASK_THRESHOLDS.items():
            path = data_dir / f"null_controls_{map_name}_{mask_tag}_summary.json"
            if not audit.require_path(path, f"{map_name} {mask_tag} null summary"):
                continue
            summary = load_json(path)
            label = f"{map_name} {mask_tag}"
            audit.require_equal(summary.get("num_samples"), 16000, f"{label} null sample count")
            audit.require_close(summary.get("mask_threshold"), expected_threshold, f"{label} mask threshold")
            audit.require_equal(summary.get("mask_tag"), mask_tag, f"{label} mask tag")
            split_method = str(summary.get("split_method", ""))
            audit.require_true(
                "coordinate_cluster" in split_method,
                f"{label} null split must be coordinate-cluster based.",
            )
            audit.require_true(
                int(summary.get("num_calibration_samples", 0)) > 0
                and int(summary.get("num_test_samples", 0)) > 0,
                f"{label} null controls need calibration and test splits.",
            )
            null_metrics[f"{map_name}_{mask_tag}"] = {
                "num_samples": summary.get("num_samples"),
                "mask_threshold": summary.get("mask_threshold"),
                "num_calibration_samples": summary.get("num_calibration_samples"),
                "num_test_samples": summary.get("num_test_samples"),
            }
    audit.metrics["null_controls"] = null_metrics


def audit_ml_reports(runs_dir: Path, audit: Audit) -> None:
    eval_path = (
        runs_dir
        / "remediated_v1_unet_imagenet_b64_aux"
        / "eval_test_component_score_fpr005_fixed"
        / "evaluation_summary.json"
    )
    if not audit.require_path(eval_path, "ImageNet U-Net held-out test evaluation"):
        return
    report = load_json(eval_path)
    metrics = report.get("selected_threshold_metrics", {})
    audit.metrics["imagenet_component_score"] = {
        "threshold": report.get("selected_threshold"),
        "recall": metrics.get("image_recall"),
        "fpr": metrics.get("image_false_positive_rate"),
        "f1": metrics.get("image_f1"),
    }
    audit.require_equal(report.get("score_mode"), "component_score", "ImageNet score mode")
    audit.require_equal(report.get("image_rule"), "connected_component", "ImageNet image rule")
    audit.require_true(
        int(report.get("image_min_positive_pixels", 0)) >= 1,
        "ImageNet component score must enforce a positive area floor.",
    )
    audit.require_equal(report.get("split"), "test", "ImageNet evaluation split")
    audit.require_equal(report.get("num_samples"), 2000, "ImageNet evaluation sample count")
    audit.require_close(report.get("selected_threshold"), 0.96, "ImageNet selected threshold", tolerance=1.0e-6)
    audit.require_true(
        0.0 <= float(metrics.get("image_false_positive_rate", -1.0)) <= 0.10,
        "ImageNet held-out FPR is outside expected remediated diagnostic range.",
    )


def audit_classical_reports(runs_dir: Path, audit: Audit) -> None:
    summary_path = runs_dir / "remediated_v1_classical_baselines" / "test_fixed_from_calibration" / "summary.json"
    if not audit.require_path(summary_path, "remediated classical test summary"):
        return
    summary = load_json(summary_path)
    audit.require_equal(summary.get("split"), "test", "classical fixed-threshold split")
    audit.require_equal(summary.get("num_samples"), 2000, "classical fixed-threshold sample count")
    methods = summary.get("methods", {})
    circular = methods.get("circular_template_screen")
    if not circular:
        audit.fail("classical summary missing circular_template_screen.")
        return
    metadata = circular.get("method_metadata", {})
    audit.require_equal(metadata.get("is_wiener_matched_filter"), False, "circular template method metadata")
    circular_metrics = circular.get("selected_threshold_metrics", {})
    audit.metrics["circular_template_screen"] = {
        "threshold": circular.get("selected_threshold"),
        "recall": circular_metrics.get("image_recall"),
        "fpr": circular_metrics.get("image_false_positive_rate"),
        "f1": circular_metrics.get("image_f1"),
    }

    fullsky_path = runs_dir / "remediated_v1_classical_fullsky" / "smica_mask090_wiener_smhw_scores.json"
    if audit.require_path(fullsky_path, "SMICA full-sky Wiener/SMHW classical scores"):
        fullsky = load_json(fullsky_path)
        methods = fullsky.get("methods", {})
        audit.require_true(
            bool(methods.get("wiener_feeney_matched_filter", {}).get("is_wiener_matched_filter")),
            "wiener_feeney_matched_filter metadata must mark it as Wiener matched.",
        )
        audit.require_equal(
            methods.get("smhw_screen", {}).get("method_family"),
            "classical_wavelet_screen",
            "SMHW method family",
        )


def audit_sensitivity_reports(runs_dir: Path, audit: Audit) -> None:
    sensitivity_dir = runs_dir / "remediated_v1_sensitivity_curve"
    report_path = sensitivity_dir / "sensitivity_report.json"
    heatmap_path = sensitivity_dir / "ml_gain_heatmap_imagenet_preselected" / "ml_gain_heatmap.json"
    if audit.require_path(report_path, "remediated sensitivity report"):
        report = load_json(report_path)
        audit.require_close(report.get("fpr_target"), 0.05, "sensitivity FPR target")
        audit.require_equal(report.get("zcrit_ratio_grid"), [0.0, 0.5, 1.0, 2.0], "sensitivity zcrit-ratio grid")
        audit.require_true(int(report.get("num_negative", 0)) >= 5000, "sensitivity needs >=5000 negatives")
        circular_meta = report.get("method_metadata", {}).get("circular_template_screen", {})
        audit.require_equal(
            circular_meta.get("is_wiener_matched_filter"),
            False,
            "sensitivity circular-template metadata",
        )
    if audit.require_path(heatmap_path, "preselected ImageNet heatmap"):
        heatmap = load_json(heatmap_path)
        audit.require_equal(heatmap.get("analysis_mode"), "preselected", "heatmap analysis mode")
        audit.require_equal(heatmap.get("primary_method"), "imagenet_b64_aux", "heatmap primary method")
        multiple = heatmap.get("multiple_testing", {})
        audit.require_true(
            int(multiple.get("holm_alpha_005_count", 0)) >= 0
            and int(multiple.get("bh_fdr_005_count", 0)) >= 0,
            "heatmap multiple-testing counts missing.",
        )
        audit.metrics["heatmap"] = {
            "holm_alpha_005_count": multiple.get("holm_alpha_005_count"),
            "bh_fdr_005_count": multiple.get("bh_fdr_005_count"),
            "winner_counts": heatmap.get("winner_counts"),
        }


def audit_deployment_reports(runs_dir: Path, audit: Audit) -> None:
    crossmap_path = runs_dir / "batch6_fullsky_nside32_smica" / "crossmap_recalibration_nside32.json"
    if not audit.require_path(crossmap_path, "Batch 6 Nside=32 cross-map recalibration"):
        return
    report = load_json(crossmap_path)
    audit.require_close(report.get("fpr_target"), 0.08, "Batch 6 FPR target")
    per_map = report.get("per_map", [])
    audit.require_equal(len(per_map), 4, "Batch 6 per-map row count")
    maps = {row.get("map") for row in per_map}
    audit.require_equal(maps, set(MAPS), "Batch 6 map coverage")
    for row in per_map:
        audit.require_true(int(row.get("n_tile", 0)) >= 10000, f"{row.get('map')} tile count")
    mean = report.get("cross_map_mean_tile_recalibrated", {})
    audit.metrics["batch6_cross_map"] = {
        "v6_only": mean.get("v6_only"),
        "gbt_6": mean.get("gbt_6"),
        "gbt_14": mean.get("gbt_14"),
        "delta_g6_minus_v6": mean.get("delta_g6_minus_v6"),
        "delta_g14_minus_g6": mean.get("delta_g14_minus_g6"),
    }
    audit.require_true(
        float(mean.get("delta_g14_minus_g6", 1.0)) < 0.0,
        "Batch 6 should preserve the gbt_14 retraction under deployment calibration.",
    )


def audit_policy_reports(runs_dir: Path, audit: Audit) -> None:
    """Audit remediated composite-policy diagnostics."""

    pareto_path = runs_dir / "remediated_v1_policy_pareto" / "policy_pareto.json"
    decision_path = runs_dir / "remediated_v1_deployment_policy_decision" / "deployment_policy_decision.json"
    tile_search_path = (
        runs_dir
        / "remediated_v1_tile_constrained_policy_search"
        / "tile_constrained_policy_search.json"
    )
    candidate_emission_path = (
        runs_dir
        / "remediated_v1_tile_constrained_candidates"
        / "candidate_emission_summary.json"
    )
    candidate_calibration_path = (
        runs_dir
        / "remediated_v1_candidate_score_calibration"
        / "candidate_score_calibration.json"
    )
    hm_preflight_path = (
        runs_dir
        / "phase5_half_mission_signflip_null"
        / "hm_signflip_preflight_report.json"
    )
    null_path = runs_dir / "remediated_v1_null_policy_audit" / "null_policy_audit.json"
    tile_path = runs_dir / "remediated_v1_policy_tile_audit" / "policy_tile_audit.json"
    classical_status_path = (
        runs_dir
        / "remediated_v1_classical_same_grid_status"
        / "classical_same_grid_status.json"
    )
    mf_channel_path = (
        runs_dir
        / "remediated_v1_mf_channel_tile_audit"
        / "mf_channel_tile_audit.json"
    )

    if audit.require_path(pareto_path, "remediated policy-Pareto report"):
        pareto = load_json(pareto_path)
        top_rows = [row for row in pareto.get("top_rows", []) if row.get("rank") == 1]
        audit.require_true(len(top_rows) >= 3, "policy-Pareto report must contain rank-1 rows.")
        audit.metrics["policy_pareto"] = {
            "num_policy_rows_searched": pareto.get("metadata", {}).get("num_policy_rows_searched"),
            "rank1_count": len(top_rows),
            "max_rank1_real_recall": max((row.get("real_recall", 0.0) for row in top_rows), default=None),
        }

    if audit.require_path(null_path, "remediated null-policy audit"):
        null_report = load_json(null_path)
        pooled = null_report.get("pooled_rows", [])
        audit.require_true(len(pooled) >= 3, "null-policy audit must contain pooled policy rows.")
        for row in pooled:
            audit.require_true(
                int(row.get("num_null", 0)) >= 5000,
                "null-policy audit pooled rows should cover the four-map held-out null split.",
            )
            audit.require_true(
                0.0 <= float(row.get("false_positive_rate", -1.0)) <= 0.10,
                "null-policy pooled FPR outside expected diagnostic range.",
            )
        audit.metrics["null_policy_audit"] = {
            f"camb{row.get('constraint_camb_fpr_max')}_real{row.get('constraint_real_fpr_max')}": {
                "num_null": row.get("num_null"),
                "false_positive_rate": row.get("false_positive_rate"),
                "fpr_ci_low": row.get("fpr_ci_low"),
                "fpr_ci_high": row.get("fpr_ci_high"),
            }
            for row in pooled
        }

    if audit.require_path(tile_path, "remediated policy tile audit"):
        tile_report = load_json(tile_path)
        rows = tile_report.get("rows", [])
        audit.require_equal(set(tile_report.get("maps", [])), set(MAPS), "policy tile audit map coverage")
        audit.require_true(len(rows) >= 4, "policy tile audit must contain per-map rows.")
        cluster_rows = []
        for row in rows:
            clusters_15 = row.get("cluster_summary", {}).get("15.0", {}).get("n_clusters")
            audit.require_true(clusters_15 is not None, "policy tile audit missing 15 deg cluster count.")
            cluster_rows.append(
                {
                    "map": row.get("map"),
                    "policy_slug": row.get("policy_slug"),
                    "trigger_fraction": row.get("trigger_fraction"),
                    "clusters_15deg": clusters_15,
                }
            )
            if row.get("map") in {"sevem", "commander"} and float(row.get("trigger_fraction", 0.0)) > 0.15:
                audit.warn(
                    "Composite policy has high SEVEM/Commander tile burden: "
                    f"{row.get('map')} {row.get('policy_slug')} "
                    f"trigger_fraction={row.get('trigger_fraction'):.4f}."
                )
        audit.metrics["policy_tile_audit"] = cluster_rows

    if audit.require_path(decision_path, "remediated deployment policy decision"):
        decision = load_json(decision_path)
        audit.require_true(
            int(decision.get("num_promotable", -1)) >= 0,
            "deployment policy decision missing promotable count.",
        )
        if int(decision.get("num_promotable", 0)) == 0:
            audit.warn(
                "No original Policy-Pareto rank-1 policy is promotable under the "
                "default cross-map burden constraints."
            )
        audit.metrics["deployment_policy_decision"] = {
            "num_promotable": decision.get("num_promotable"),
            "top_policy": decision.get("rows", [{}])[0].get("policy_key"),
            "top_policy_blockers": decision.get("rows", [{}])[0].get("failure_reasons"),
        }

    if audit.require_path(tile_search_path, "tile-constrained policy search"):
        tile_search = load_json(tile_search_path)
        constraints = tile_search.get("constraints", {})
        diagnostics = tile_search.get("diagnostics", {})
        top_rows = tile_search.get("top_rows", [])
        audit.require_true(len(top_rows) >= 1, "tile-constrained search must find feasible rows.")
        audit.require_true(
            not bool(diagnostics.get("cluster_eval_limit_hit")),
            "current tile-constrained search artifact must be exhaustive, not capped.",
        )
        audit.require_true(
            bool(diagnostics.get("search_exhaustive")),
            "tile-constrained search artifact must declare search_exhaustive=true.",
        )
        if diagnostics.get("cluster_eval_limit_hit"):
            audit.require_true(
                bool(diagnostics.get("best_feasible_real_recall_certified_by_frontier")),
                "capped tile-constrained search must certify best feasible recall by frontier.",
            )
        for row in top_rows[:5]:
            audit.require_true(
                float(row.get("pooled_null_fpr_ci_high", 1.0))
                <= float(constraints.get("max_pooled_null_fpr_ci_high", 0.0)) + 1.0e-12,
                "tile-constrained search top row exceeds pooled null FPR CI constraint.",
            )
            audit.require_true(
                float(row.get("max_trigger_fraction", 1.0))
                <= float(constraints.get("max_trigger_fraction_any_map", 0.0)) + 1.0e-12,
                "tile-constrained search top row exceeds trigger-fraction constraint.",
            )
            audit.require_true(
                int(row.get("max_clusters", 10**9))
                <= int(constraints.get("max_clusters_any_map", -1)),
                "tile-constrained search top row exceeds cluster constraint.",
            )
        audit.metrics["tile_constrained_policy_search"] = {
            "num_feasible": diagnostics.get("num_feasible"),
            "num_cluster_evaluated": diagnostics.get("num_cluster_evaluated"),
            "cluster_eval_limit_hit": diagnostics.get("cluster_eval_limit_hit"),
            "search_exhaustive": diagnostics.get("search_exhaustive"),
            "best_policy": top_rows[0].get("policy") if top_rows else None,
            "best_real_recall": top_rows[0].get("real_recall") if top_rows else None,
            "best_gain_vs_single": top_rows[0].get(
                "real_recall_gain_vs_best_feasible_single"
            )
            if top_rows
            else None,
            "best_real_fpr": top_rows[0].get("real_fpr") if top_rows else None,
            "best_null_ci_high": top_rows[0].get("pooled_null_fpr_ci_high")
            if top_rows
            else None,
            "best_max_clusters": top_rows[0].get("max_clusters") if top_rows else None,
            "best_max_trigger_fraction": top_rows[0].get("max_trigger_fraction")
            if top_rows
            else None,
        }

    if audit.require_path(candidate_emission_path, "tile-constrained candidate emission"):
        emission = load_json(candidate_emission_path)
        totals = emission.get("totals", {})
        map_rows = emission.get("map_rows", [])
        artifacts = emission.get("artifacts", {})
        audit.require_close(
            emission.get("min_mask_fraction"),
            0.9,
            "tile-constrained candidate emission mask fraction",
        )
        audit.require_close(
            emission.get("cluster_radius_deg"),
            15.0,
            "tile-constrained candidate emission cluster radius",
        )
        audit.require_equal(set(totals.get("maps", [])), set(MAPS), "candidate emission map coverage")
        audit.require_true(
            int(totals.get("num_cluster_representatives", 0)) > 0,
            "candidate emission should contain cluster representatives.",
        )
        for key in ("candidate_records_jsonl", "cluster_representatives_jsonl"):
            value = artifacts.get(key)
            audit.require_true(bool(value), f"candidate emission missing artifact {key}.")
            if value:
                audit.require_path(Path(value), f"candidate emission artifact {key}")
        for row in map_rows:
            audit.require_true(
                int(row.get("num_tiles_passing_mask_fraction", 0)) > 0,
                "candidate emission map row has no canonical-mask eligible tiles.",
            )
            audit.require_true(
                float(row.get("eligible_trigger_fraction", 1.0)) <= 0.02,
                "candidate emission eligible trigger fraction is unexpectedly high.",
            )
        audit.metrics["tile_constrained_candidate_emission"] = {
            "policy_slug": emission.get("policy_slug"),
            "min_mask_fraction": emission.get("min_mask_fraction"),
            "cluster_radius_deg": emission.get("cluster_radius_deg"),
            "num_tile_candidates": totals.get("num_tile_candidates"),
            "num_cluster_representatives": totals.get("num_cluster_representatives"),
            "map_rows": [
                {
                    "map": row.get("map"),
                    "num_tile_candidates": row.get("num_tile_candidates"),
                    "eligible_trigger_fraction": row.get("eligible_trigger_fraction"),
                    "num_cluster_representatives": row.get(
                        "num_cluster_representatives"
                    ),
                }
                for row in map_rows
            ],
        }

    if audit.require_path(candidate_calibration_path, "candidate score calibration"):
        calibration = load_json(candidate_calibration_path)
        artifacts = calibration.get("artifacts", {})
        audit.require_equal(
            calibration.get("policy_slug"),
            "tile_constrained_rank1_2_of_3",
            "candidate score calibration policy slug",
        )
        audit.require_true(
            int(calibration.get("pooled_null_count", 0)) >= 5000,
            "candidate score calibration should use pooled calibration null scores.",
        )
        audit.require_true(
            int(calibration.get("num_candidates", 0)) > 0,
            "candidate score calibration has no candidates.",
        )
        audit.require_true(
            "remediated_v1_null_policy_calibration" in str(calibration.get("null_score_template", "")),
            "candidate calibration must use calibration split score caches.",
        )
        audit.require_true(
            "remediated_v1_null_policy_audit" not in str(calibration.get("null_score_template", "")),
            "candidate calibration must not use the held-out test null-policy audit caches.",
        )
        for key in ("calibrated_candidates_jsonl", "calibrated_candidates_csv"):
            value = artifacts.get(key)
            audit.require_true(bool(value), f"candidate score calibration missing artifact {key}.")
            if value:
                audit.require_path(Path(value), f"candidate score calibration artifact {key}")
        top_candidates = calibration.get("top_candidates", [])
        if top_candidates:
            audit.require_true(
                0.0 <= float(top_candidates[0].get("calibration_pooled_survival_p", -1.0)) <= 1.0,
                "candidate calibration top pooled survival p-value outside [0, 1].",
            )
            audit.require_true(
                0.0 <= float(top_candidates[0].get("calibration_pooled_bh_q", -1.0)) <= 1.0,
                "candidate calibration top BH q-value outside [0, 1].",
            )
        audit.metrics["candidate_score_calibration"] = {
            "num_candidates": calibration.get("num_candidates"),
            "pooled_null_count": calibration.get("pooled_null_count"),
            "best_pooled_survival_p": (
                top_candidates[0].get("calibration_pooled_survival_p")
                if top_candidates
                else None
            ),
            "best_bh_q": (
                top_candidates[0].get("calibration_pooled_bh_q")
                if top_candidates
                else None
            ),
        }

    if audit.require_path(hm_preflight_path, "Phase 5 HM sign-flip preflight report"):
        hm_preflight = load_json(hm_preflight_path)
        status = hm_preflight.get("status")
        issues = hm_preflight.get("issues", [])
        failure_checks = {
            issue.get("check")
            for issue in issues
            if issue.get("severity") == "fail"
        }
        audit.require_true(
            status in {"pass", "blocked"},
            "Phase 5 HM preflight status must be pass or blocked.",
        )
        audit.require_true(
            int(hm_preflight.get("candidate_summary", {}).get("num_candidates", 0)) == 24,
            "Phase 5 HM preflight should target the frozen 24 cluster representatives.",
        )
        audit.require_equal(
            hm_preflight.get("candidate_summary", {})
            .get("policy_slug_counts", {})
            .get("tile_constrained_rank1_2_of_3"),
            24,
            "Phase 5 HM preflight policy slug count",
        )
        if status == "blocked":
            audit.require_true(
                failure_checks <= {"hm1_present", "hm2_present", "hm1_exists", "hm2_exists"},
                "Phase 5 HM preflight is blocked by failures beyond missing HM maps.",
            )
        audit.metrics["phase5_hm_preflight"] = {
            "status": status,
            "num_failures": hm_preflight.get("num_failures"),
            "num_warnings": hm_preflight.get("num_warnings"),
            "candidate_summary": hm_preflight.get("candidate_summary", {}),
            "failure_checks": sorted(failure_checks),
        }

    if audit.require_path(classical_status_path, "classical same-grid status report"):
        classical_status = load_json(classical_status_path)
        audit.require_equal(
            classical_status.get("status"),
            "blocked",
            "classical same-grid status",
        )
        audit.require_true(
            bool(classical_status.get("blockers")),
            "blocked classical same-grid status must list blockers.",
        )
        audit.metrics["classical_same_grid_status"] = {
            "status": classical_status.get("status"),
            "blockers": classical_status.get("blockers"),
        }

    if audit.require_path(mf_channel_path, "matched-filter-channel tile audit"):
        mf_audit = load_json(mf_channel_path)
        rows = mf_audit.get("map_rows", [])
        settings = mf_audit.get("settings", {})
        model = mf_audit.get("model", {})
        global_rows = (
            mf_audit.get("diagnostic_context", {})
            .get("global_model_only_rows", [])
        )
        real_row = next(
            (
                row
                for row in global_rows
                if row.get("domain") == "real_smica_recalibrated"
            ),
            {},
        )
        audit.require_equal(set(settings.get("maps", [])), set(MAPS), "MF-channel map coverage")
        audit.require_close(settings.get("mask_threshold"), 0.9, "MF-channel mask threshold")
        audit.require_equal(settings.get("score_limit"), 0, "MF-channel audit must be full-map")
        audit.require_equal(model.get("input_channels"), 2, "MF-channel input channel count")
        audit.require_equal(len(rows), 4, "MF-channel per-map row count")
        audit.require_true(
            float(real_row.get("recall", 0.0)) >= 0.30,
            (
                "MF-channel diagnostic real-SMICA recall should exceed the "
                "current constrained composite."
            ),
        )
        cluster_rows = []
        for row in rows:
            audit.require_true(
                int(row.get("num_tiles", 0)) >= 10000,
                f"MF-channel {row.get('map')} full-sky tile count",
            )
            audit.require_true(
                int(row.get("num_tiles_passing_mask_fraction", 0)) > 0,
                f"MF-channel {row.get('map')} canonical-mask eligible tiles",
            )
            audit.require_true(
                float(row.get("eligible_trigger_fraction", 1.0)) <= 0.10,
                f"MF-channel {row.get('map')} eligible trigger fraction is too high.",
            )
            if float(row.get("trigger_fraction", 0.0)) > 0.25:
                audit.warn(
                    "Legacy MF-channel checkpoint has high unmasked tile trigger fraction: "
                    f"{row.get('map')} trigger_fraction={row.get('trigger_fraction'):.4f}."
                )
            cluster_rows.append(
                {
                    "map": row.get("map"),
                    "trigger_fraction": row.get("trigger_fraction"),
                    "eligible_trigger_fraction": row.get("eligible_trigger_fraction"),
                    "num_eligible_clusters": row.get("num_eligible_clusters"),
                }
            )
        audit.metrics["mf_channel_tile_audit"] = {
            "model": model.get("name"),
            "checkpoint_label": model.get("checkpoint_label"),
            "selected_threshold": mf_audit.get("threshold", {}).get("selected_threshold"),
            "diagnostic_real_smica_recall": real_row.get("recall"),
            "diagnostic_real_smica_fpr": real_row.get("fpr"),
            "map_rows": cluster_rows,
            "assumption_notes": mf_audit.get("assumption_notes", []),
        }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    audit = Audit()

    audit_dataset(data_dir, audit)
    audit_null_controls(data_dir, audit)
    audit_ml_reports(runs_dir, audit)
    audit_classical_reports(runs_dir, audit)
    audit_sensitivity_reports(runs_dir, audit)
    audit_deployment_reports(runs_dir, audit)
    audit_policy_reports(runs_dir, audit)

    report = {
        "status": "pass" if not audit.failures else "fail",
        "num_failures": len(audit.failures),
        "num_warnings": len(audit.warnings),
        "failures": audit.failures,
        "warnings": audit.warnings,
        "metrics": audit.metrics,
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
