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
import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    "scripts/data/phase0_planck_unit_audit.py",
    "scripts/ops/audit_remediated_flow.py",
    "scripts/core/phase_config.py",
    "scripts/core/phase2_signal_model.py",
    "scripts/data/phase2_physics_checks.py",
    "scripts/data/phase2_audit_dataset.py",
    "scripts/data/phase2_generate_training.py",
    "scripts/data/phase2_generate_stratified_validation.py",
    "scripts/core/phase2_observing_model.py",
    "scripts/data/phase2_extract_smica_null_controls.py",
    "scripts/core/phase_dataset_utils.py",
    "scripts/core/phase3_method_registry.py",
    "scripts/ml/phase3_train_unet.py",
    "scripts/ml/phase3_evaluate_run.py",
    "scripts/analysis/phase3_audit_outputs.py",
    "scripts/analysis/phase3_template_baseline.py",
    "scripts/analysis/phase3_classical_filters.py",
    "scripts/analysis/phase3_compare_screeners.py",
    "scripts/analysis/phase3_boundary_analysis.py",
    "scripts/followup/phase3_template_fit_candidates.py",
    "scripts/analysis/phase3_score_null_controls.py",
    "scripts/analysis/phase3_score_classical_null_controls.py",
    "scripts/analysis/phase3_sensitivity_curve.py",
    "scripts/analysis/phase3_ml_gain_heatmap.py",
    "scripts/ml/phase3_eval_stratified_external.py",
    "scripts/ml/phase3_eval_classical_stratified_external.py",
    "scripts/ml/phase3_ensemble_evaluate.py",
    "scripts/analysis/phase3_null_burden_matched_fpr.py",
    "scripts/analysis/phase3_real_sky_injection.py",
    "scripts/analysis/phase3_real_sky_recalibration.py",
    "scripts/ml/phase3_eval_single_model_recalibrated.py",
    "scripts/analysis/phase3_postprocess_ablation.py",
    "scripts/analysis/phase3_real_sky_v7_gate.py",
    "scripts/core/phase3_circular_template_features.py",
    "scripts/analysis/phase3_cache_matched_filter_channel.py",
    "scripts/analysis/phase3_per_radius_threshold.py",
    "scripts/analysis/phase3_noise_floor_analysis.py",
    "scripts/analysis/phase3_upper_limit_calculator.py",
    "scripts/analysis/phase3_deployment_burden_table.py",
    "scripts/analysis/phase3_policy_pareto_search.py",
    "scripts/analysis/phase3_deployment_policy_decision.py",
    "scripts/analysis/phase3_tile_constrained_policy_search.py",
    "scripts/followup/phase3_emit_tile_constrained_candidates.py",
    "scripts/followup/phase3_calibrate_candidate_scores.py",
    "scripts/analysis/phase3_classical_same_grid_status.py",
    "scripts/analysis/phase3_same_grid_fullsky_benchmark.py",
    "scripts/analysis/phase3_matched_filter_snr_curve.py",
    "scripts/analysis/phase3_remediated_policy_tile_audit.py",
    "scripts/analysis/phase3_remediated_null_policy_audit.py",
    "scripts/analysis/phase3_mf_channel_tile_audit.py",
    "scripts/followup/phase5_half_mission_signflip_null.py",
    "scripts/ops/create_reproducibility_manifest.py",
    "scripts/analysis/phase3_fullsky_tile.py",
    "scripts/analysis/phase3_geometry_router.py",
    "scripts/ops/batch6_overnight_analysis.py",
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
            "scripts/data/phase2_physics_checks.py",
            "--output-json",
            "runs/phase2_physics_checks.json",
        ]
    )
    if not args.skip_remediated_flow:
        run([py, "scripts/ops/audit_remediated_flow.py"])

    if args.data_h5:
        run([py, "scripts/data/phase2_audit_dataset.py", "--data-h5", args.data_h5])
        if not args.skip_train_dry_run:
            run(
                [
                    py,
                    "scripts/ml/phase3_train_unet.py",
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
        run([py, "scripts/analysis/phase3_audit_outputs.py", "--eval-dir", args.eval_dir])

    print("\nQuality gates passed.")


if __name__ == "__main__":
    main()
