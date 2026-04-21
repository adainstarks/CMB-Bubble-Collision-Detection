# Phase 3 Candidate Artifact Schema

Phase 3 emits candidate tables from `scripts/phase3_screen_and_verify.py` for
historical two-branch validation and from
`scripts/phase3_emit_tile_constrained_candidates.py` for the current
remediated full-sky policy. These tables are handoff products, not discovery
statistics.

## Operating Modes

- `union`: emit a candidate when either the broad proposal branch or the boundary-aware verifier passes. This is proposal mode. It intentionally favors recall and must be filtered downstream.
- `both`: emit a candidate only when both branches pass. This is high-confidence shortlist mode. It favors purity and null control at the cost of recall.
- `proposal`: emit only original proposal-branch candidates.
- `verifier`: emit only boundary-aware verifier candidates.

The current remediated full-sky emitter freezes the tile-constrained composite
policy and emits two products:

- `candidate_records.jsonl`: all overlapping full-sky tile triggers on the
  canonical `mask_fraction >= 0.9` footprint.
- `cluster_representatives_15deg.jsonl`: one highest-margin representative per
  greedy `15 deg` cluster for first-pass HM sign-flip or template-fit follow-up.

The current downstream follow-up stack adds two more products:

- `runs/phase3_unet/remediated_v1_template_fit_handoff/template_fit_records.jsonl`:
  deterministic Feeney-template fit seeds for frozen representatives.
- `runs/phase3_unet/remediated_v1_bayesian_template_handoff/bayesian_template_handoff.jsonl`:
  merged screening calibration, template seeds, and projection/clustering
  guardrails for downstream likelihood or Bayesian tooling.

## Core Fields

- `rank`: one-indexed rank after sorting by `rank_score` descending.
- `sample_index`: row index in the source HDF5 dataset.
- `rank_score`: operational ranking score combining proposal score, verifier score, template-fit improvement, and consensus bonus. It is not calibrated probability.
- `proposal_pass`: whether the broad original V4 branch emitted a candidate at its frozen threshold.
- `verifier_pass`: whether the boundary-aware branch emitted a candidate at its frozen threshold.
- `risk_tag`: categorical routing label. Current values are `priority_consensus`, `boundary_only_low_null`, `proposal_only_template_supported`, and `proposal_only_high_null_risk`.
- `truth_label`: synthetic validation truth label when available. Real-map screening outputs should treat this as unavailable or zero depending on source.

For tile-constrained full-sky candidates, `global_candidate_rank`,
`global_cluster_rank`, `policy_slug`, `policy_margin`, `policy_thresholds`,
`cluster_id`, `cluster_radius_deg`, and `cluster_n_members` replace the
historical proposal/verifier fields.

## Sky And Patch Provenance

- `patch_center_glon_deg`, `patch_center_glat_deg`: Galactic coordinates of the extracted patch center.
- `candidate_glon_deg`, `candidate_glat_deg`: estimated Galactic coordinates of the candidate center.
- `candidate_dx_deg`, `candidate_dy_deg`: candidate offset from patch center in local patch coordinates.
- `radius_est_deg`: candidate angular-size estimate from the selected branch mask.
- `coord_pool_idx`: coordinate-pool index used to reproduce extraction.
- `cmb_realization_idx`: synthetic CMB realization index when applicable.
- `background_id`: stable background/provenance identifier.

For tile-constrained full-sky candidates, `patch_glon_deg` and `patch_glat_deg`
are aliases for the tile center. `peak_glon_deg` and `peak_glat_deg` record the
ML peak or tile center used for clustering.

## Branch Scores

- `proposal_score_max`, `proposal_score_mean`, `proposal_positive_fraction`: original V4 branch mask statistics.
- `proposal_threshold`: frozen threshold used by the proposal branch.
- `verifier_score_max`, `verifier_score_mean`, `verifier_positive_fraction`: boundary-aware branch mask statistics.
- `verifier_threshold`: frozen threshold used by the verifier branch.

## Template-Fit Handoff

- `template_fit_status`: template-fit status for emitted candidates when a template-fit JSONL is provided.
- `template_delta_chi2_vs_plane_null`: improvement of the local Feeney-style template fit over a plane/null nuisance model. This is a handoff diagnostic, not Bayesian evidence.
- `template_theta_crit_fit_deg`, `template_z0_fit`, `template_zcrit_fit`: fitted template parameters for downstream inspection.

## Bayesian / Likelihood Handoff

- `screening_priority_tier`: descriptive follow-up tier derived from pooled BH
  q-values. It is a routing label, not a significance class.
- `screening_pooled_survival_p`, `screening_pooled_bh_q`: empirical
  calibration-split ranking metadata for the frozen candidate.
- `template_seed`: deterministic local template-fit seed for downstream
  initialization.
- `template_refinement_window`: local theta/ROI refinement window for the next
  follow-up stage. This is not a model prior.
- `projection_systematics_caution`: whether the projection/clustering audit
  attaches a geometry caution to the follow-up seed.
- `followup_route`: recommended downstream routing string based on screening
  rank, template-seed validity, and geometry caution.

## Truth Fields For Synthetic Evaluation

- `truth_theta_crit_deg`, `truth_z0`, `truth_zcrit`, `truth_edge_sigma_deg`: injected signal parameters when available.

## Guardrails

- Do not compare `union` mode directly against FPR-capped single-branch results as if it were the same operating point.
- Do not interpret `rank_score` or branch scores as posterior probabilities.
- Do not treat a candidate row as a detection claim. It is input to classical/statistical follow-up.
- Do not recycle screening thresholds or screening envelopes as cosmological
  priors in the downstream Bayesian stage.
