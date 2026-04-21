# Project Handoff: CMB Bubble Collision Screening

Last updated: 2026-04-21
Repo path: `/data/william/CMB-Collision-Bubbles`

## Assumptions

- This is an astrophysical **candidate-screening** repository.
- It is not a standalone bubble-collision discovery claim.
- It is not yet a Feeney-style Bayesian evidence or posterior pipeline.
- Signal amplitudes are dimensionless fractional `Delta T / T`.
- Patch tensors are CMB anisotropies in Kelvin, not microkelvin and not full
  thermodynamic CMB temperature.
- Current remediated products use `float64` for geometry/harmonic work and
  `float32` for stored tensors after finite/range checks.

## Current Directive

Use `remediated_v1` as the active baseline. The working claim is:

```text
A reproducible Planck-era ML/classical candidate-screening front end for
localized bubble-collision signatures, intended to accelerate or supplement
classical/Bayesian follow-up.
```

Framing rule: treat current outputs as candidate screening, synthetic
sensitivity calibration, candidate-volume accounting, or template-fit handoff
products. Do not write them up as standalone cosmological detection evidence.
Paper-facing wording guardrails live in
`docs/manuscript_framing_candidate_screening.md`.

Do not use historical `training_v4`, `validation_stratified_v1`, old
`matched_template`, or post-hoc best-branch heatmap numbers as current science
claims. They are development history only.

Run this before relying on the current artifact graph:

```bash
python scripts/audit_remediated_flow.py
```

For a compact environment/source/artifact manifest:

```bash
python scripts/create_reproducibility_manifest.py
```

## Active Artifact Graph

Core remediated products:

| product | status | purpose |
|---|---|---|
| `data/remediated_v1/training_data.h5` | current | `20000` synthetic rows, split `16000/2000/2000` |
| `data/remediated_v1/calibration_data.h5` | current | threshold calibration copy |
| `data/remediated_v1/test_data.h5` | current | held-out synthetic test copy |
| `data/remediated_v1/null_controls_{map}_{mask}.h5` | current | real-map null controls for four Planck cleaned maps and `mask090/mask050` |
| `runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux/` | current | best single remediated U-Net branch |
| `runs/phase3_unet/remediated_v1_unet_random_b64_aux/` | current | encoder-initialization ablation |
| `runs/phase3_unet/remediated_v1_classical_baselines/` | current | held-out circular-template comparisons |
| `runs/phase3_unet/remediated_v1_sensitivity_curve/` | current | amplitude/radius/zcrit-ratio sensitivity grid |
| `runs/phase3_unet/remediated_v1_upper_limits/` | current | efficiency-weighted detectable-collision upper limits |
| `runs/phase3_unet/remediated_v1_deployment_burden/` | current | full-sky patch and clustered candidate-volume accounting |
| `runs/phase3_unet/remediated_v1_tile_constrained_policy_search/` | current | deployable composite-policy recall-vs-burden search |
| `runs/phase3_unet/remediated_v1_tile_constrained_candidates/` | current | canonical-mask emitted candidates and cluster representatives |
| `runs/phase3_unet/remediated_v1_candidate_score_calibration/` | current | calibration-split null-survival scores for candidate representatives |
| `runs/phase3_unet/remediated_v1_mf_channel_tile_audit/` | current diagnostic | full-sky burden audit for legacy two-channel recall candidate |
| `runs/phase3_unet/phase5_half_mission_signflip_null/` | current | HM sign-flip preflight; p-value reports once HM maps are supplied |
| `runs/phase3_unet/remediated_v1_classical_fullsky/` | current | SMICA full-sky Wiener Feeney and SMHW maps |
| `runs/phase3_unet/batch6_fullsky_nside32_*/` | current | deployment tile calibration |

Physics/observing-model contract:

- `Nside=256`, `PATCH_PIX=256`, `RESO_ARCMIN=13`.
- Planck cleaned maps modeled with `5 arcmin` beam.
- `synfast(pixwin=True)` for current synthetic skies.
- Canonical mask threshold is `0.9`.
- Stress-test null controls at `0.5` are present and separate.
- Mixed geometry includes contained and edge-crossing positives.
- Minimum visible target fraction is `0.15`.

## Current Results

Held-out remediated synthetic test:

| method | threshold source | threshold | precision | recall | FPR | F1 |
|---|---|---:|---:|---:|---:|---:|
| ImageNet U-Net `component_score` | calibration split | `0.96` | `0.888` | `0.396` | `0.050` | `0.548` |
| random-init U-Net `component_score` | calibration split | `0.99` | `0.819` | `0.362` | `0.080` | `0.502` |
| `circular_template_screen` | calibration split | `65.0217` | `0.921` | `0.267` | `0.023` | `0.414` |

Preselected ImageNet-vs-`circular_template_screen` heatmap:

- ImageNet wins `30 / 35` cells.
- `14` cells are Holm-significant at family-wise alpha `0.05`.
- `17` cells are Benjamini-Hochberg significant at FDR `0.05`.
- This supersedes the historical post-hoc best-branch `8 / 35` count.

Real-SMICA injection diagnostic:

- `imagenet_b64_aux_only`: real recall/FPR `0.372/0.185`.
- CAMB comparison point: recall/FPR `0.349/0.050`.
- Interpretation: useful transfer diagnostic, not a deployable threshold.

Noise-floor diagnostic:

- Report: `runs/phase3_unet/remediated_v1_noise_floor/noise_floor_report.md`.
- Median negative-patch RMS is about `99.8 uK`.
- Instrument white-noise scale is about `2.3 uK/pixel`, so CMB confusion and
  domain transfer dominate over the configured white-noise term.
- `A = 1e-6` support RMS is only about `1.6-2.1 uK`; `A = 5e-6` is about
  `7.8-10.4 uK`. Treat this as a patch-RMS diagnostic, not a proof that the
  whole weak-family problem is physically impossible.
- This is not an impossibility proof. A real bound needs the full covariance,
  foreground residuals, mask coupling, and a Bayesian/template-fit likelihood.

Matched-filter SNR diagnostic:

- Script: `scripts/phase3_matched_filter_snr_curve.py`.
- Report:
  `runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_report.md`.
- Under the repo CAMB/beam/noise assumptions with rough `f_sky` scaling,
  `A <= 2e-6` cells have low ideal recall, while several `A >= 1e-5` cells have
  high ideal matched-filter recall but low current ML recall.
- Interpretation: the lowest amplitudes are CMB-confusion dominated for this
  idealized template calculation; higher-SNR cells are algorithmic and
  same-grid-benchmark targets.

Upper-limit post-processing:

- Script: `scripts/phase3_upper_limit_calculator.py`.
- Report: `runs/phase3_unet/remediated_v1_upper_limits/upper_limits.md`.
- Default prior: log-uniform amplitude and `sin(theta)` radius weighting.
- At 95% confidence with zero credible detections, `Nbar_s^95` is `10.2674`
  for `imagenet_b64_aux`, `11.4544` for `random_b64_aux`, `12.1677` for
  `circular_template_screen`, and `17.6292` for `centered_disc`.
- `lambda H_F^-4` is computed only when `Omega_k` and `H_F/H_I` are supplied.
  `lambda/B` is computed only for an explicit model-specific exposure factor.
- Treat this as a screening-derived detectable-collision upper-limit proxy,
  not as a masked-sky Bayesian evidence result or a competitive cosmological
  constraint.

## Deployment Calibration

Batch 6 is the current deployment-calibration reference. It uses Nside=32
full-sky tile audits with about `11200` tile patches per cleaned map.

Tile-recalibrated mixed recall at FPR `0.08`:

| map | `v6_only` | `gbt_6` | `gbt_14` | `gbt_6 - v6` |
|---|---:|---:|---:|---:|
| SMICA | `0.347` | `0.337` | `0.329` | `-0.010` |
| NILC | `0.296` | `0.307` | `0.320` | `+0.011` |
| SEVEM | `0.173` | `0.237` | `0.192` | `+0.065` |
| Commander | `0.161` | `0.230` | `0.193` | `+0.070` |
| mean | `0.244` | `0.278` | `0.258` | `+0.034` |

Policy interpretation:

- `gbt_6` survives as a cross-map score-ensemble policy.
- `gbt_6` is not an SMICA-specific improvement; on SMICA it is slightly worse
  than `v6_only`.
- `gbt_14` is archived. The geometry features overfit clean-null mask
  structure and do not survive deployment calibration.
- Quote per-map numbers and bootstrap intervals. Do not hide map dependence in
  a single cross-map mean.

Deployment candidate burden:

- Script: `scripts/phase3_deployment_burden_table.py`.
- Report: `runs/phase3_unet/remediated_v1_deployment_burden/deployment_burden.md`.
- At tile-recalibrated FPR `0.08`, each map has about `896` patch candidates by
  construction.
- At `15 deg` greedy peak clustering, mean clustered burden is `55.00` for
  `v6_only`, `69.25` for `gbt_6`, and `68.50` for `gbt_14`.
- `gbt_6` has the best cross-map mean recall (`0.2779`) but costs more
  clustered follow-up than `v6_only` (`0.2440` recall).
- Shipped clean-null thresholds are now drift diagnostics only; they produce
  thousands of patch candidates on SEVEM and Commander.

Policy Pareto search:

- Script: `scripts/phase3_policy_pareto_search.py`.
- Report: `runs/phase3_unet/remediated_v1_policy_pareto/policy_pareto.md`.
- Under CAMB FPR `<=0.05` and real-SMICA FPR `<=0.02`, the best diagnostic
  policy is `2-of-3(random>=0.987007, imagenet>=0.947194, circular>=68.79)`,
  with real recall `0.2520`, a `+0.0289` gain over the best exact-threshold
  single-score baseline.
- Under CAMB FPR `<=0.05` and real-SMICA FPR `<=0.05`, the best diagnostic
  policy is `random>=0.994006 OR imagenet>=0.998799`, with real recall
  `0.2671`, a `+0.0441` gain over the best exact-threshold single-score
  baseline.
- Under CAMB FPR `<=0.08` and real-SMICA FPR `<=0.08`, the best diagnostic
  policy is `2-of-3(random>=0.946233, imagenet>=0.947194, circular>=75.0896)`,
  with real recall `0.3028`, a `+0.0490` gain over the best exact-threshold
  single-score baseline.
- These policies are recall-boost candidates only. The follow-up audits below
  are now the limiting evidence.

Composite-policy audits:

- Scripts: `scripts/phase3_remediated_null_policy_audit.py` and
  `scripts/phase3_remediated_policy_tile_audit.py`; the broader constrained
  search is `scripts/phase3_tile_constrained_policy_search.py`.
- Reports:
  `runs/phase3_unet/remediated_v1_null_policy_audit/null_policy_audit.md` and
  `runs/phase3_unet/remediated_v1_policy_tile_audit/policy_tile_audit.md`.
- On `5976` held-out real-null patches, pooled policy FPRs are a few percent:
  `0.0281` for the `0.05/0.02` budget, `0.0246` for `0.05/0.05`, `0.0131`
  for `0.08/0.05`, and `0.0343` for `0.08/0.08`.
- Full-sky overlapping tiles are stricter. At `15 deg` clustering, the
  `0.05/0.05` policy gives `41` SMICA clusters and `28` NILC clusters, but
  `68` SEVEM clusters and `101` Commander clusters.
- Deployment decision report:
  `runs/phase3_unet/remediated_v1_deployment_policy_decision/deployment_policy_decision.md`.
- Default decision constraints are `15 deg` clustering, max `70` clusters on
  every cleaned map, max tile-trigger fraction `0.15` on every cleaned map, and
  pooled held-out null-control FPR upper CI `<=0.04`.
- Under those constraints, `0 / 5` composite policies are promotable. The
  highest-recall `0.08/0.08` policy reaches recall `0.3028`, but has `83`
  Commander clusters and Commander trigger fraction `0.1772`.
- The broader tile-constrained search finds a deployment-safe composite under
  the same rule: `2-of-3(random>=0.999983, imagenet>=0.983177,
  circular>=51.0957)`, with real-injection recall `0.2620`, real FPR `0.0650`,
  pooled null FPR CI high `0.0290`, max `62` clusters, and max trigger fraction
  `0.0899`. Its recall gain over the best feasible single-score baseline is
  `+0.0403`.
- Candidate emission for that policy is now frozen on the canonical
  `mask_fraction >= 0.9` science footprint. It emits `102` overlapping tile
  candidates and `24` clustered representatives across SMICA, NILC, SEVEM, and
  Commander. Use
  `runs/phase3_unet/remediated_v1_tile_constrained_candidates/cluster_representatives_15deg.jsonl`
  as the first-pass HM sign-flip/template-fit input.
- Projection/systematics context now lives in
  `runs/phase3_unet/remediated_v1_projection_clustering_audit/projection_clustering_audit.md`.
- The template-fit handoff for the frozen `24` cluster representatives now
  lives in
  `runs/phase3_unet/remediated_v1_template_fit_handoff/template_fit_summary.json`.
- The merged Bayesian/template-likelihood handoff for those same `24`
  representatives now lives in
  `runs/phase3_unet/remediated_v1_bayesian_template_handoff/bayesian_template_handoff_summary.json`.
- Candidate score calibration now uses only the null-control calibration split
  (`5768` pooled null margins) and writes empirical survival p-values plus BH
  q-values to
  `runs/phase3_unet/remediated_v1_candidate_score_calibration/calibrated_candidates.jsonl`.
- Phase 5 HM sign-flip preflight validates the frozen `24` candidate
  representatives, policy slug, model checkpoints, and common mask. It is
  currently `blocked` only because HM1/HM2 cleaned-map paths are not available
  in the local artifact tree.
  The best current representative has plus-one pooled null-survival
  `p=0.000173` and BH q-value `0.000693`. These are follow-up prioritization
  scores, not global detection probabilities.
- Treat that as the current deployment-safe composite candidate, but do not
  promote any diagnostic recall result to a cosmological claim.

Matched-filter-channel recall candidate:

- Script: `scripts/phase3_mf_channel_tile_audit.py`.
- Report:
  `runs/phase3_unet/remediated_v1_mf_channel_tile_audit/mf_channel_tile_audit.md`.
- Shared feature implementation:
  `scripts/phase3_circular_template_features.py`.
- This audits the legacy v7 two-channel checkpoint without modifying the
  current tile HDF5 files. The second channel is reconstructed in memory from
  the legacy circular-template response bank.
- Diagnostic real-SMICA recall is `0.3526` at FPR `0.0440`, above the current
  deployment-safe composite recall `0.2620`.
- Hard-regime recall remains limited: `0.0466` for `A <= 2e-6`, `0.1724` for
  `5e-6 <= A <= 2e-5`, and `0.9288` for `A >= 5e-5`.
- Canonical-mask full-sky burden is `285/307/290/422` eligible triggered tiles
  for SMICA/NILC/SEVEM/Commander and `20/21/22/23` clusters at `15 deg`.
- Do not promote this checkpoint as current science. It was trained on
  historical `training_v4` with a legacy `15 arcmin` feature channel. The next
  model-development step is a remediated-v1 two-channel retrain with the
  current `5 arcmin` beam, canonical masks, disjoint splits, and real-null
  deployment gates.
- The old duplicated response-map code path was consolidated so the HDF5
  feature cacher and tile audit now use the same circular-template feature
  implementation. Torch and scipy response maps agree to about `1.9e-5` max
  absolute difference on sample remediated patches.

## Classical Naming Contract

Use these names exactly:

- `circular_template_screen`: historical patch-space circular disc/ring
  correlation screen.
- `wiener_feeney_matched_filter`: harmonic-space Feeney template with beam,
  HEALPix pixel-window, and CMB/noise inverse-covariance weighting.
- `smhw_screen`: spherical Mexican-hat / scale-space context screen.

Never describe `circular_template_screen` as an optimal matched filter. Any
paper-facing ML-vs-classical claim must say whether it is comparing against the
simple circular screen or the true Wiener Feeney matched filter.

Same-grid classical status:

- Script: `scripts/phase3_classical_same_grid_status.py`.
- Report:
  `runs/phase3_unet/remediated_v1_classical_same_grid_status/classical_same_grid_status.md`.
- Current status is now `complete` for the stratified same-grid manifest in
  `runs/phase3_unet/remediated_v1_same_grid_fullsky_manifest/`.
- Mean recall on that fixed manifest is `0.3841` for
  `wiener_feeney_matched_filter`, `0.3517` for `imagenet_b64_aux`, `0.2994`
  for `random_b64_aux`, and `0.2734` for `smhw_screen`.
- ImageNet beats Wiener in `17 / 35` raw cells and loses in `16 / 35`; under
  non-overlapping exact 95% CIs it is better in `7` cells and worse in `8`.
- Scientific consequence: the true Wiener/Feeney filter is the strongest
  average classical screener on the same-grid benchmark, while the ML branch
  remains locally competitive rather than uniformly superior.
- Pilot driver: `scripts/phase3_same_grid_fullsky_benchmark.py` streams
  full-sky injected maps, writes patches projected from those maps, scores
  Wiener/SMHW locally, and can optionally score ML checkpoints on the same
  patches. Subset output is a path-validation artifact, not a superiority
  claim.

## Remaining Blockers

1. **Weak-signal recall.** At real-SMICA FPR `0.08`, models remain below `0.10`
   recall for `A <= 5e-6`. Treat `A <= 2e-6` as CMB-confusion dominated under
   the current ideal SNR diagnostic; treat higher-SNR cells as algorithmic
   targets under the now-closed same-grid benchmark.
2. **Candidate-volume convention.** Candidate-volume accounting exists, but the
   paper narrative still needs a final cluster radius and score-calibration
   convention.
3. **Posterior/evidence calibration.** Empirical null-survival scores exist,
   but top candidates still need template likelihood or Bayesian follow-up.
4. **Projection-robust follow-up.** The deterministic template/Bayesian
   handoff exists, but projection cautions remain attached to a majority of the
   frozen candidates and need native-sphere or projection-robust follow-up
   before any parameter statement.
5. **True-Wiener two-stream branch.** The corrected full-cache rerun is now
   complete. Relative to `imagenet_b64_aux`, `true_wiener_ft` raises mean
   synthetic sensitivity from `0.34875` to `0.36839` overall, helps
   moderate/high-amplitude cells (`+0.03881` mean delta) and large-radius
   cells (`+0.02226`), but remains worse on the hardest low-amplitude subset
   (`-0.00592`). Keep it as a candidate secondary branch or fusion input, not
   an automatic baseline replacement.
6. **Half-mission null report.** The Phase 5 HM sign-flip preflight exists and
   validates candidate/policy/model readiness, but no HM1/HM2-backed candidate
   p-value report has been produced because the HM maps are not local.
7. **Remediated matched-filter-channel retrain.** The legacy v7 tile audit is
   promising for recall, but it must be retrained and recalibrated under the
   current artifact contract before it can replace the deployment-safe
   composite.

## Current Next Steps

1. Retrain and audit a remediated-v1 matched-filter-channel branch using the
   current `5 arcmin` beam, canonical masks, disjoint splits, and real-map null
   deployment gates.
2. Cache `features/circular_template_response` with
   `scripts/phase3_cache_matched_filter_channel.py` for the remediated train,
   calibration, test, sensitivity, real-injection, and null-control products
   needed by that retrain. Do not use the historical `15 arcmin` feature
   contract for new claims.
3. Close the true Wiener/SMHW classical comparison on the same remediated
   injection/null products.
4. Select/provide Planck HM1/HM2 component-separated map paths, rerun the Phase
   5 preflight, then run sign-flip calibration on the frozen cluster
   representatives.
5. Add posterior/template-likelihood calibration for the top HM-vetted
   candidates.
6. Choose and justify the paper-facing cluster radius for candidate burden.
7. Train against real-map null backgrounds rather than another CAMB-only
   distribution.
8. Build the template-fit/Bayesian handoff path for emitted candidates.
9. Run Phase 5 HM sign-flip p-values after Planck HM1/HM2 maps are selected.

## Quality Gates

Fast artifact graph audit:

```bash
python scripts/audit_remediated_flow.py
```

Compile + physics + remediated-flow audit:

```bash
python scripts/run_quality_gates.py --skip-train-dry-run
```

Full local gate including dataset audit:

```bash
python scripts/run_quality_gates.py \
  --data-h5 data/remediated_v1/training_data.h5 \
  --skip-train-dry-run
```

Use the training dry run only when the local PyTorch/CUDA environment is known
to be healthy.

Last validated local gate:

```bash
python scripts/run_quality_gates.py \
  --data-h5 data/remediated_v1/training_data.h5 \
  --skip-train-dry-run
```

Status: pass. Expected warnings are the already documented high-burden
diagnostic composite policies, no original Policy-Pareto rank-1 policy being
promotable under default burden constraints, and high unmasked SEVEM/Commander
trigger fractions for the legacy MF-channel checkpoint.

## Historical Notes

Keep historical batch notes under `work/` for provenance, but do not treat them
as current instructions unless they are explicitly referenced above.

Key retained history:

- `work/batch5_fullsky_calibration_gap.md`: identified clean-null deployment
  drift.
- `work/batch6_review_response.md`: current Nside=32 deployment calibration and
  review-driven ablations.
- `work/radius_head_post_mortem.md`: failed radius-head branch hygiene notes.
- `docs/nside512_probe_decision.md`: old probe result; not a final statement
  about Nside=512 physics.
- `docs/evaluation_inventory.md`: current map of active, deferred, and
  historical evaluation surfaces.
- `docs/phase5_half_mission_signflip_null.md`: real-data half-mission null
  design and CLI usage notes.
