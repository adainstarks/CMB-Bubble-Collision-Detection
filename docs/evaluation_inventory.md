# Evaluation Inventory

## Assumptions

- `remediated_v1` is the current science family.
- Historical scripts and reports are retained only when they preserve
  provenance or can reproduce a prior decision record.
- Deleting code is appropriate only when the code is scratch, duplicated, or
  unreferenced by any current or historical result.

## Current Science Path

These scripts are part of the active reproducible path and are included in the
quality gate or manifest:

- `scripts/audit_remediated_flow.py`
- `scripts/create_reproducibility_manifest.py`
- `scripts/run_quality_gates.py`
- `scripts/phase2_generate_training.py`
- `scripts/phase2_extract_smica_null_controls.py`
- `scripts/phase2_audit_dataset.py`
- `scripts/phase3_train_unet.py`
- `scripts/phase3_evaluate_run.py`
- `scripts/phase3_template_baseline.py`
- `scripts/phase3_classical_filters.py`
- `scripts/phase3_sensitivity_curve.py`
- `scripts/phase3_ml_gain_heatmap.py`
- `scripts/phase3_circular_template_features.py`
- `scripts/phase3_noise_floor_analysis.py`
- `scripts/phase3_upper_limit_calculator.py`
- `scripts/phase3_deployment_burden_table.py`
- `scripts/phase3_policy_pareto_search.py`
- `scripts/phase3_deployment_policy_decision.py`
- `scripts/phase3_tile_constrained_policy_search.py`
- `scripts/phase3_emit_tile_constrained_candidates.py`
- `scripts/phase3_calibrate_candidate_scores.py`
- `scripts/phase3_classical_same_grid_status.py`
- `scripts/phase3_remediated_null_policy_audit.py`
- `scripts/phase3_remediated_policy_tile_audit.py`
- `scripts/phase3_mf_channel_tile_audit.py`
- `scripts/phase3_fullsky_tile.py`
- `scripts/phase5_half_mission_signflip_null.py`
- `scripts/batch6_overnight_analysis.py`

## Current Post-Processing Products

- Sensitivity-grid results are the source for efficiency statements and
  upper-limit post-processing.
- The upper-limit calculator reports `Nbar_s` under an explicit amplitude and
  radius prior. It reports `lambda H_F^-4` only when `Omega_k` and `H_F/H_I`
  are supplied, and reports `lambda/B` only when an explicit model exposure
  factor is supplied.
- The noise-floor diagnostic is a physical scale diagnostic. It is not an
  impossibility proof.
- The deployment-burden table reports patch candidates and clustered candidates
  per full-sky tile audit under explicit threshold and clustering policies. It
  is candidate-volume accounting, not a detection statistic.
- The policy-Pareto search identifies recall-vs-FP composite-policy candidates
  from existing score caches. Its winners require full-sky tile calibration
  before deployment.
- The deployment-policy decision report is the current promotion gate for
  composite policies. Under default cross-map burden constraints it promotes
  `0 / 5` current Policy-Pareto winners.
- The tile-constrained policy search broadens beyond the original
  Policy-Pareto winners and exhaustively clusters every policy that passes the
  cheap FPR, pooled-null, and trigger-fraction filters under the same cross-map
  tile-burden constraints. Its current best feasible policy reaches
  real-injection recall `0.2620` with real FPR `0.0650`, pooled null FPR CI
  high `0.0290`, and at most `62` clusters on any cleaned map.
- The tile-constrained candidate emitter freezes that policy on the canonical
  `mask_fraction >= 0.9` science footprint. Current output contains `102`
  overlapping tile candidates and `24` clustered representatives across the
  four Planck cleaned maps.
- Candidate score calibration assigns empirical null-survival p-values and
  Benjamini-Hochberg q-values to those `24` cluster representatives using only
  the real-map null-control calibration split. The held-out test null split is
  not used for this calibration.
- The true Wiener/SMHW same-grid status report is now `complete` on the
  stratified full-sky manifest built for the remediated-v1 closure pass.
- On that fixed manifest, the true Wiener/Feeney matched filter is the
  strongest average screener, with the ImageNet U-Net retaining localized
  wins rather than uniform superiority.
- The remediated null-policy audit expands the real-null stress test from the
  `200`-negative injection diagnostic to held-out null controls across all four
  Planck cleaned maps.
- The remediated policy tile audit is the current composite-policy deployment
  stress test. Its overlapping full-sky tile burden is stricter than the
  held-out null-control patch FPR and currently blocks promotion of the
  high-recall composite policies.
- The Phase 5 HM sign-flip script is downstream candidate calibration. It
  produces conditional noise-robustness p-values when HM1/HM2 maps and frozen
  candidate JSONL files are supplied. Its preflight mode currently validates
  the frozen candidates, policy slug, model checkpoints, and common mask; the
  local status is `blocked` only because HM1/HM2 maps are not staged in the
  artifact tree.
- The matched-filter-channel tile audit is a recall-development diagnostic for
  the legacy v7 two-channel checkpoint. It reports diagnostic real-SMICA recall
  `0.3526` at FPR `0.0440` and canonical-mask clustered burden of `20-23`
  clusters per cleaned map, but is not a promoted science result because the
  checkpoint predates the `remediated_v1` beam/mask/split contract.

## Deferred But Scientifically Useful

- Full posterior/evidence calibration remains downstream work; current
  candidate score calibration is empirical null-tail screening metadata.
- A full masked-sky Bayesian evidence comparison still remains downstream
  work; the same-grid closure is a screening benchmark, not the Feeney/OSS
  optimal-likelihood ceiling.
- The remediated-v1 true-Wiener two-stream retrain is now complete and should
  be treated as a candidate development branch until it also passes the
  real-null, tile-burden, candidate-calibration, and HM follow-up gates.

## Historical Or Provenance Only

These scripts should not drive current claims unless their outputs are clearly
labelled historical:

- `scripts/phase3_operating_table.py`: legacy mixed operating score for branch
  triage, not a discovery statistic.
- `scripts/phase3_final_policy_eval.py`: older policy-evaluation surface before
  the current remediated-v1 and Batch 6 framing.
- `scripts/phase3_eval_tta_ensemble.py`: legacy ensemble/TTA evaluation path.
- `scripts/phase3_nside512_probe.py`: old probe tied to
  `docs/nside512_probe_decision.md`.
- `scripts/phase3_threshold_volume_sweep.py`: useful diagnostic, but not the
  current deployment-calibration source.
- `scripts/phase3_two_pass_policy.py`: policy experiment retained for
  provenance.
- `scripts/phase3_visualize_smoothed_examples.py`: visualization utility, not a
  quantitative result.

## Removed Scratch

The root scratch files `stuff`, `stuff2`, `stuff3`, `stuff4`, and
`largestuff.md` were consolidated into maintained documentation and removed.
The old README backup was removed because the current README and handoff now
contain the active directive.
