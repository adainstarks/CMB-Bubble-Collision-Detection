"""
Run project quality gates from one entrypoint.

This script is deliberately operational: it compiles the active scripts and runs
the physics, dataset, training-dry-run, and evaluation-output checks that guard
against the major scientific and ML failure modes.

Assumptions
-----------
* The active product is the remediated-v1 candidate-screening flow, not the
  historical training_v4 development path.
* Quality gates should be fast enough for routine use and therefore validate
  artifact consistency rather than regenerate large CMB products.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = (
    "astropy",
    "camb",
    "h5py",
    "healpy",
    "matplotlib",
    "numpy",
    "scipy",
    "sklearn",
    "torch",
)
DEFAULT_SCRIPTS = [
    "scripts/audit_remediated_flow.py",
    "scripts/phase_config.py",
    "scripts/phase2_signal_model.py",
    "scripts/phase2_physics_checks.py",
    "scripts/phase2_audit_dataset.py",
    "scripts/phase2_generate_training.py",
    "scripts/phase2_generate_stratified_validation.py",
    "scripts/phase2_observing_model.py",
    "scripts/phase2_extract_smica_null_controls.py",
    "scripts/phase_dataset_utils.py",
    "scripts/phase3_method_registry.py",
    "scripts/phase3_train_unet.py",
    "scripts/phase3_evaluate_run.py",
    "scripts/phase3_audit_outputs.py",
    "scripts/phase3_template_baseline.py",
    "scripts/phase3_classical_filters.py",
    "scripts/phase3_compare_screeners.py",
    "scripts/phase3_boundary_analysis.py",
    "scripts/phase3_template_fit_candidates.py",
    "scripts/phase3_score_null_controls.py",
    "scripts/phase3_score_classical_null_controls.py",
    "scripts/phase3_sensitivity_curve.py",
    "scripts/phase3_ml_gain_heatmap.py",
    "scripts/phase3_eval_stratified_external.py",
    "scripts/phase3_eval_classical_stratified_external.py",
    "scripts/phase3_ensemble_evaluate.py",
    "scripts/phase3_null_burden_matched_fpr.py",
    "scripts/phase3_real_sky_injection.py",
    "scripts/phase3_real_sky_recalibration.py",
    "scripts/phase3_eval_single_model_recalibrated.py",
    "scripts/phase3_postprocess_ablation.py",
    "scripts/phase3_real_sky_v7_gate.py",
    "scripts/phase3_circular_template_features.py",
    "scripts/phase3_cache_matched_filter_channel.py",
    "scripts/phase3_per_radius_threshold.py",
    "scripts/phase3_noise_floor_analysis.py",
    "scripts/phase3_upper_limit_calculator.py",
    "scripts/phase3_deployment_burden_table.py",
    "scripts/phase3_policy_pareto_search.py",
    "scripts/phase3_deployment_policy_decision.py",
    "scripts/phase3_tile_constrained_policy_search.py",
    "scripts/phase3_emit_tile_constrained_candidates.py",
    "scripts/phase3_calibrate_candidate_scores.py",
    "scripts/phase3_classical_same_grid_status.py",
    "scripts/phase3_same_grid_fullsky_benchmark.py",
    "scripts/phase3_matched_filter_snr_curve.py",
    "scripts/phase3_remediated_policy_tile_audit.py",
    "scripts/phase3_remediated_null_policy_audit.py",
    "scripts/phase3_mf_channel_tile_audit.py",
    "scripts/phase5_half_mission_signflip_null.py",
    "scripts/create_reproducibility_manifest.py",
    "scripts/phase3_fullsky_tile.py",
    "scripts/phase3_geometry_router.py",
    "scripts/batch6_overnight_analysis.py",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run compile, physics, dataset, training, and evaluation quality gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-h5", type=str, default="")
    parser.add_argument("--eval-dir", type=str, default="")
    parser.add_argument("--skip-train-dry-run", action="store_true")
    parser.add_argument("--skip-remediated-flow", action="store_true")
    return parser.parse_args()


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def check_required_modules():
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Missing required Python modules for the CMB pipeline: "
            f"{joined}. Activate the repo environment first, e.g. "
            "`conda activate cmb`, or run through `conda run -n cmb python ...`."
        )


def main():
    args = parse_args()
    py = sys.executable

    run([py, "-m", "py_compile", *DEFAULT_SCRIPTS])
    check_required_modules()
    run(
        [
            py,
            "scripts/phase2_physics_checks.py",
            "--output-json",
            "runs/phase2_physics_checks.json",
        ]
    )
    if not args.skip_remediated_flow:
        run([py, "scripts/audit_remediated_flow.py"])

    if args.data_h5:
        run([py, "scripts/phase2_audit_dataset.py", "--data-h5", args.data_h5])
        if not args.skip_train_dry_run:
            run(
                [
                    py,
                    "scripts/phase3_train_unet.py",
                    "--data-h5",
                    args.data_h5,
                    "--run-name",
                    "quality_gate_dry_run",
                    "--dry-run",
                    "--num-workers",
                    "0",
                ]
            )

    if args.eval_dir:
        run([py, "scripts/phase3_audit_outputs.py", "--eval-dir", args.eval_dir])

    print("\nQuality gates passed.")


if __name__ == "__main__":
    main()
