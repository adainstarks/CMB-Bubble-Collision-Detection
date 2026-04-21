# Project Structure And Artifact Policy

This repository is a research pipeline. Keep source, documentation, compact
metadata, and decision records in git. Keep generated HDF5 datasets, Planck
FITS maps, checkpoints, score caches, and large run products local unless a
specific compact artifact is promoted.

## Active Source

- `scripts/phase_config.py`: shared physical constants and remediated defaults.
- `scripts/phase2_signal_model.py`: Feeney-style signal template,
  full-temperature modulation, and first-order additive benchmark utilities.
- `scripts/phase2_observing_model.py`: CAMB/synfast observing-model utilities.
- `scripts/phase2_generate_training.py`: current remediated training-data
  generator.
- `scripts/phase2_extract_smica_null_controls.py`: per-map real-null extraction.
- `scripts/phase2_audit_dataset.py`: split/provenance/data sanity checks.
- `scripts/phase3_train_unet.py`: U-Net training harness.
- `scripts/phase3_evaluate_run.py`: primary held-out evaluator.
- `scripts/phase3_template_baseline.py`: circular-template patch baseline.
- `scripts/phase3_classical_filters.py`: full-sky Wiener Feeney matched filter
  and SMHW screen maps.
- `scripts/phase3_sensitivity_curve.py`: remediated sensitivity-grid builder.
- `scripts/phase3_ml_gain_heatmap.py`: paired ML-vs-circular-template
  sensitivity analysis.
- `scripts/phase3_circular_template_features.py`: shared circular-template
  response feature-map utilities for two-channel U-Net experiments.
- `scripts/phase3_noise_floor_analysis.py`: empirical signal-scale and CMB
  confusion diagnostics for the remediated sensitivity grid.
- `scripts/phase3_matched_filter_snr_curve.py`: ideal harmonic-space Feeney
  matched-filter SNR curves under repo beam/noise/CAMB assumptions.
- `scripts/phase3_upper_limit_calculator.py`: efficiency-weighted
  detectable-collision upper-limit post-processing.
- `scripts/phase3_deployment_burden_table.py`: full-sky patch and clustered
  candidate-volume accounting from Batch 6 tile caches.
- `scripts/phase3_policy_pareto_search.py`: composite-policy recall-vs-FP
  search over remediated score caches.
- `scripts/phase3_deployment_policy_decision.py`: promotion/rejection decision
  report for composite policies under explicit cross-map burden constraints.
- `scripts/phase3_tile_constrained_policy_search.py`: exhaustive
  tile-constrained policy search after cheap FPR, pooled-null, and
  trigger-fraction filters.
- `scripts/phase3_emit_tile_constrained_candidates.py`: canonical-mask
  candidate and cluster-representative emitter for the constrained policy.
- `scripts/phase3_calibrate_candidate_scores.py`: calibration-split empirical
  null-survival scores for emitted candidate representatives.
- `scripts/phase3_classical_same_grid_status.py`: guard report for true
  Wiener/SMHW same-grid benchmark claims.
- `scripts/phase3_same_grid_fullsky_benchmark.py`: guarded full-sky
  same-grid pilot/production driver for Wiener/SMHW and optional ML scoring.
- `scripts/phase3_remediated_null_policy_audit.py`: larger held-out real-null
  audit for remediated composite policies.
- `scripts/phase3_remediated_policy_tile_audit.py`: full-sky overlapping-tile
  burden audit for remediated composite policies.
- `scripts/phase3_mf_channel_tile_audit.py`: recall-development tile-burden
  audit for the legacy two-channel circular-template-response U-Net.
- `scripts/phase3_fullsky_tile.py`: full-sky tile calibration audit.
- `scripts/phase3_geometry_router.py`: score-router experiments.
- `scripts/phase5_half_mission_signflip_null.py`: downstream HM1/HM2
  sign-flip null calibration and preflight checks for emitted candidates.
- `scripts/batch6_overnight_orchestrator.sh`: current Nside=32 tile audit
  launcher.
- `scripts/batch6_overnight_analysis.py`: current cross-map deployment summary.
- `scripts/audit_remediated_flow.py`: lightweight artifact graph audit.
- `scripts/create_reproducibility_manifest.py`: compact environment/source/
  artifact manifest for researchers.
- `scripts/run_quality_gates.py`: compile/physics/artifact quality gate.
- `docs/evaluation_inventory.md`: active/deferred/historical evaluation map.
- `docs/phase5_half_mission_signflip_null.md`: real-data half-mission
  sign-flip null design and CLI usage notes.

## Current Artifacts

- `data/remediated_v1/`: current synthetic training/calibration/test products
  and per-map null-control summaries.
- `runs/phase3_unet/remediated_v1_*`: current remediated ML/classical reports.
- `runs/phase3_unet/remediated_v1_noise_floor/`: current empirical CMB
  noise-floor diagnostic report.
- `runs/phase3_unet/remediated_v1_matched_filter_snr/`: current ideal
  matched-filter SNR diagnostic report.
- `runs/phase3_unet/remediated_v1_upper_limits/`: current efficiency-weighted
  detectable-collision upper-limit report.
- `runs/phase3_unet/remediated_v1_deployment_burden/`: current patch and
  clustered candidate-volume report.
- `runs/phase3_unet/remediated_v1_policy_pareto/`: current composite-policy
  recall-vs-FP diagnostic report.
- `runs/phase3_unet/remediated_v1_deployment_policy_decision/`: current
  composite-policy promotion/rejection decision report.
- `runs/phase3_unet/remediated_v1_tile_constrained_policy_search/`: current
  recall-vs-tile-burden search result for deployable composite policies.
- `runs/phase3_unet/remediated_v1_tile_constrained_candidates/`: current
  canonical-mask candidate and cluster-representative artifacts.
- `runs/phase3_unet/remediated_v1_candidate_score_calibration/`: current
  empirical null-survival and BH-q calibration for candidate representatives.
- `runs/phase3_unet/remediated_v1_null_policy_calibration/`: calibration-split
  real-null score caches used by candidate score calibration.
- `runs/phase3_unet/remediated_v1_classical_same_grid_status/`: current status
  report for true Wiener/SMHW same-grid benchmark closure.
- `runs/phase3_unet/remediated_v1_same_grid_fullsky_pilot/`: optional
  same-grid full-sky pilot output; subset runs validate the path but do not
  close paper-facing comparator claims.
- `runs/phase3_unet/remediated_v1_null_policy_audit/`: held-out real-null
  audit of composite policies.
- `runs/phase3_unet/remediated_v1_policy_tile_audit/`: full-sky tile burden
  audit of composite policies.
- `runs/phase3_unet/remediated_v1_mf_channel_tile_audit/`: diagnostic
  tile-burden audit for the legacy matched-filter-channel recall candidate.
- `runs/phase3_unet/phase5_half_mission_signflip_null/`: HM sign-flip
  preflight report now; candidate p-value reports when HM inputs are available.
- `runs/phase3_unet/remediated_v1_reproducibility/`: current reproducibility
  manifest.
- `runs/phase3_unet/batch6_fullsky_nside32_*`: current deployment tile audits.
- `work/batch6_review_response.md`: current deployment-calibration narrative.

## Historical Artifacts

These are retained for provenance, not as current instructions:

- `data/training_v4/`
- `data/training_v5_mixed_geometry/`
- `data/validation_stratified_v1/`
- `runs/phase3_unet/*_v1` legacy reports outside the `remediated_v1_*` family
- `docs/phase3_matched_fpr_sensitivity_decision.md`
- `docs/nside512_probe_decision.md`
- older `work/batch2_*` through `work/batch5_*` notes

When a historical artifact is cited, label it as historical and state the
current remediated replacement.

## Cleanup Rule

Ad hoc scratch files such as `stuff`, `stuff2`, `stuff3`, `stuff4`, and
`largestuff.md` are not repo artifacts. Consolidate useful findings into
`README.md`, `PROJECT_HANDOFF.md`, `docs/`, or `work/`, then remove the scratch
file.
