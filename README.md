# CMB Bubble Collision Screener

Planck-era machine-learning and classical candidate screening for localized
bubble-collision signatures in the Cosmic Microwave Background.

## Assumptions

- This repository implements a **candidate-screening front end**, not a
  cosmological detection claim.
- It does not compute a Feeney-style Bayesian evidence ratio, a posterior over
  collision parameters, or a constraint on the expected number of detectable
  collisions.
- Paper-facing wording guardrails live in
  `docs/manuscript_framing_candidate_screening.md`.
- Signal amplitudes are dimensionless fractional temperature modulations
  (`Delta T / T`); stored patch tensors are CMB anisotropies in Kelvin.
- Current science artifacts use `Nside=256`, `256 x 256` gnomonic patches,
  Planck `5 arcmin` beam handling, `synfast(pixwin=True)`, and mixed
  contained/truncated signal geometry.
- All operating thresholds are screening thresholds. They are valid only for
  the calibration distribution named with the result.

## Current Directive

The active claim is deliberately narrow:

> A reproducible Planck-era candidate-screening method for localized
> bubble-collision signatures, intended to accelerate or supplement
> classical/Bayesian follow-up.

Framing rule for any manuscript or talk built from this repo: describe outputs
as candidate screening, sensitivity calibration, candidate-volume accounting,
or screening-derived handoff products. Do not describe the current artifacts as
standalone cosmological detection evidence.

Use `remediated_v1` as the current artifact family. Treat `training_v4`,
`validation_stratified_v1`, old `matched_template` reports, and the old
best-branch `8 / 35` heatmap as historical development artifacts.

## Remediated v1

Primary products:

| product | rows | role |
|---|---:|---|
| `data/remediated_v1/training_data.h5` | `20000` | train/calibration/test source, split `16000/2000/2000` |
| `data/remediated_v1/calibration_data.h5` | `2000` | threshold calibration copy |
| `data/remediated_v1/test_data.h5` | `2000` | held-out final synthetic test copy |
| `data/remediated_v1/null_controls_{map}_{mask}.h5` | `16000` each | real-map null controls for `smica,nilc,sevem,commander` and `mask090,mask050` |

The current generator/audit contract is:

- Feeney-style linear-cap disc template with full-temperature modulation
  (`feeney2011_full_temperature_modulation`). McEwen/OSS matched-filter
  benchmarks use the first-order additive approximation and must record that
  convention separately.
- `mask_threshold = 0.9` for canonical science products.
- `mask_threshold = 0.5` null controls retained for deployment stress tests.
- Planck cleaned-map beam modeled as `5 arcmin`.
- Beam/pixel-window transfer handled in harmonic space for current synthetic
  products.
- Geometry and harmonic work use `float64`; stored tensors use `float32` after
  finite/range checks.
- Splits are train/calibration/test and coordinate-cluster/CAMB-realization
  disjoint.

Run the lightweight artifact-flow audit:

```bash
python scripts/audit_remediated_flow.py
```

Run the broader quality gate:

```bash
python scripts/run_quality_gates.py \
  --data-h5 data/remediated_v1/training_data.h5 \
  --skip-train-dry-run
```

Write a compact reproducibility manifest:

```bash
python scripts/create_reproducibility_manifest.py
```

## Current Results

Held-out remediated synthetic test, thresholds selected on the calibration
split:

| method | score mode | threshold | precision | recall | FPR | F1 |
|---|---|---:|---:|---:|---:|---:|
| ImageNet EfficientNet-B0 U-Net | `component_score` | `0.96` | `0.888` | `0.396` | `0.050` | `0.548` |
| random-init EfficientNet-B0 U-Net | `component_score` | `0.99` | `0.819` | `0.362` | `0.080` | `0.502` |
| `circular_template_screen` | fixed from calibration | `65.0217` | `0.921` | `0.267` | `0.023` | `0.414` |

The preselected ImageNet-vs-`circular_template_screen` sensitivity heatmap uses
paired bootstrap tests with Holm and Benjamini-Hochberg correction. It reports:

- ImageNet wins `30 / 35` amplitude-radius cells.
- `14` cells are Holm-significant at family-wise alpha `0.05`.
- `17` cells are BH-significant at FDR `0.05`.

This replaces the historical post-hoc best-branch `8 / 35` number.

## Noise-Floor Diagnostic

The remediated sensitivity grid now has an empirical signal-scale diagnostic:

```bash
python scripts/phase3_noise_floor_analysis.py
```

Current output:

- Median negative-patch RMS is about `99.8 uK`.
- The configured instrument white-noise term is only about `2.3 uK/pixel`, so
  CMB structure, foreground/domain transfer, and mask effects dominate.
- For `A = 1e-6`, support RMS is only about `1.6-2.1 uK`, or `0.016-0.021`
  of the empirical CMB patch RMS.
- For `A = 5e-6`, support RMS is about `7.8-10.4 uK`, still below `0.11` of
  the empirical CMB patch RMS.

Interpretation: low-amplitude recall is expected to remain poor for patch-level
screeners unless the next stage uses stronger priors, full covariance weighting,
or Bayesian/template-fit likelihoods. This is a diagnostic, not an impossibility
proof; a real bound needs the full CMB covariance and foreground residual model.

## Matched-Filter SNR Curves

The ideal harmonic-space Feeney matched-filter SNR diagnostic is:

```bash
python scripts/phase3_matched_filter_snr_curve.py
```

Default output:

- `runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_report.json`
- `runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_report.md`
- `runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_cells.csv`
- `runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_profiles.csv`
- `runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_curves.png`

The calculation uses the McEwen/OSS first-order additive template convention,
CAMB TT covariance, the repo beam/noise assumptions, and a rough `f_sky` scaling.
It does not close the masked-sky Wiener/SMHW same-grid benchmark by itself.
Current results show `A <= 2e-6` is low-SNR under the fsky proxy, while several
`A >= 1e-5` cells have high ideal matched-filter recall but low current ML
recall. Those higher-SNR cells are algorithmic targets, not physics-impossibility
claims.

## Upper-Limit Calculator

The remediated sensitivity grid now has a source-backed post-processing
calculator for detectable-collision upper limits:

```bash
python scripts/phase3_upper_limit_calculator.py
```

Default output:

- `runs/phase3_unet/remediated_v1_upper_limits/upper_limits.json`
- `runs/phase3_unet/remediated_v1_upper_limits/upper_limits.md`
- `runs/phase3_unet/remediated_v1_upper_limits/upper_limits.csv`
- `runs/phase3_unet/remediated_v1_upper_limits/upper_limit_cell_weights.csv`
- `runs/phase3_unet/remediated_v1_upper_limits/upper_limits.png`

Under the default `log_uniform` amplitude prior and `sin_theta` radius prior,
with zero credible detections at 95% confidence:

| method | mean efficiency | exposure | `Nbar_s^95` |
|---|---:|---:|---:|
| `centered_disc` | `0.2201` | `0.1699` | `17.6292` |
| `circular_template_screen` | `0.3189` | `0.2462` | `12.1677` |
| `imagenet_b64_aux` | `0.3780` | `0.2918` | `10.2674` |
| `random_b64_aux` | `0.3388` | `0.2615` | `11.4544` |

This is an efficiency-weighted Poisson upper limit on the detectable-collision
rate parameter `Nbar_s`. It uses the no-blob form from Feeney et al. Appendix A
and can optionally convert to `lambda H_F^-4` using Feeney et al. Eq. 1 when
`Omega_k` and `H_F/H_I` are supplied. It reports `lambda/B` only when an
explicit model-specific exposure factor is supplied; no universal `lambda/B`
mapping is hard-coded.

Interpretation guardrail: this is a screening-derived detectable-collision
upper-limit proxy built from the synthetic efficiency table. It is not a
masked-sky Bayesian evidence result and should not be framed as a competitive
cosmological constraint.

## Deployment Calibration

Batch 6 reran the full-sky tile audit at `Nside=32`, giving about `11200`
patches per cleaned map and reducing FPR-calibration uncertainty to about
`0.003`.

Tile-recalibrated mixed recall at FPR `0.08`:

| map | `v6_only` | `gbt_6` | `gbt_14` | `gbt_6 - v6` |
|---|---:|---:|---:|---:|
| SMICA | `0.347` | `0.337` | `0.329` | `-0.010` |
| NILC | `0.296` | `0.307` | `0.320` | `+0.011` |
| SEVEM | `0.173` | `0.237` | `0.192` | `+0.065` |
| Commander | `0.161` | `0.230` | `0.193` | `+0.070` |
| mean | `0.244` | `0.278` | `0.258` | `+0.034` |

Interpretation:

- `gbt_6` survives as a **cross-map score-ensemble lever**.
- On SMICA alone, `gbt_6` is marginally worse than `v6_only`.
- `gbt_14` is archived; its geometry features overfit the clean-null
  distribution and do not survive deployment calibration.
- Per-map calibration is mandatory. Do not quote cross-map means as SMICA
  claims.

## Deployment Candidate Burden

The current full-sky candidate-volume table is generated from cached Batch 6
tile features:

```bash
python scripts/phase3_deployment_burden_table.py
```

Default output:

- `runs/phase3_unet/remediated_v1_deployment_burden/deployment_burden.json`
- `runs/phase3_unet/remediated_v1_deployment_burden/deployment_burden.md`
- `runs/phase3_unet/remediated_v1_deployment_burden/deployment_patch_burden.csv`
- `runs/phase3_unet/remediated_v1_deployment_burden/deployment_cluster_burden.csv`

Tile-recalibrated patch burden is about `896` eligible tile candidates per map
at FPR `0.08`, by construction. At `15 deg` greedy peak clustering, mean
clustered burden across the four cleaned maps is:

| method | mean patch candidates | mean recall | mean clusters at `15 deg` |
|---|---:|---:|---:|
| `v6_only` | `896.25` | `0.2440` | `55.00` |
| `gbt_6` | `897.75` | `0.2779` | `69.25` |
| `gbt_14` | `915.50` | `0.2583` | `68.50` |

Interpretation: `gbt_6` remains the best cross-map recall policy after
deployment calibration, but it carries a higher clustered follow-up burden than
`v6_only`. The shipped clean-null thresholds are retained only as drift
diagnostics; on SEVEM and Commander they produce thousands of patch candidates
and must not be used as deployment thresholds.

## Policy Pareto Search

The current recall-vs-FP policy search over existing remediated scores is:

```bash
python scripts/phase3_policy_pareto_search.py
```

Default output:

- `runs/phase3_unet/remediated_v1_policy_pareto/policy_pareto.json`
- `runs/phase3_unet/remediated_v1_policy_pareto/policy_pareto.md`
- `runs/phase3_unet/remediated_v1_policy_pareto/policy_pareto_top.csv`

Best diagnostic policies under selected constraints:

| constraints | best policy | CAMB FPR | real FPR | CAMB recall | real recall | gain vs exact single |
|---|---|---:|---:|---:|---:|---:|
| CAMB `<=0.05`, real `<=0.02` | `2-of-3(random>=0.987007, imagenet>=0.947194, circular>=68.79)` | `0.0498` | `0.0200` | `0.3288` | `0.2520` | `+0.0289` |
| CAMB `<=0.05`, real `<=0.05` | `random>=0.994006 OR imagenet>=0.998799` | `0.0398` | `0.0500` | `0.3163` | `0.2671` | `+0.0441` |
| CAMB `<=0.08`, real `<=0.08` | `2-of-3(random>=0.946233, imagenet>=0.947194, circular>=75.0896)` | `0.0706` | `0.0750` | `0.3596` | `0.3028` | `+0.0490` |

Interpretation: there are composite-score policies that recover more recall
than exact-threshold single-score choices under the same diagnostic FPR
budgets. They are not deployment thresholds by themselves.

## Composite-Policy Audits

The Policy-Pareto winners have now been stress-tested two ways:

```bash
python scripts/phase3_remediated_null_policy_audit.py
python scripts/phase3_remediated_policy_tile_audit.py
python scripts/phase3_tile_constrained_policy_search.py
python scripts/phase3_emit_tile_constrained_candidates.py
python scripts/phase3_calibrate_candidate_scores.py
```

Default outputs:

- `runs/phase3_unet/remediated_v1_null_policy_audit/null_policy_audit.json`
- `runs/phase3_unet/remediated_v1_null_policy_audit/null_policy_audit.md`
- `runs/phase3_unet/remediated_v1_policy_tile_audit/policy_tile_audit.json`
- `runs/phase3_unet/remediated_v1_policy_tile_audit/policy_tile_audit.md`
- `runs/phase3_unet/remediated_v1_deployment_policy_decision/deployment_policy_decision.json`
- `runs/phase3_unet/remediated_v1_deployment_policy_decision/deployment_policy_decision.md`
- `runs/phase3_unet/remediated_v1_tile_constrained_policy_search/tile_constrained_policy_search.json`
- `runs/phase3_unet/remediated_v1_tile_constrained_policy_search/tile_constrained_policy_search.md`
- `runs/phase3_unet/remediated_v1_tile_constrained_candidates/candidate_emission_summary.json`
- `runs/phase3_unet/remediated_v1_tile_constrained_candidates/candidate_records.jsonl`
- `runs/phase3_unet/remediated_v1_tile_constrained_candidates/cluster_representatives_15deg.jsonl`
- `runs/phase3_unet/remediated_v1_projection_clustering_audit/projection_clustering_audit.md`
- `runs/phase3_unet/remediated_v1_template_fit_handoff/template_fit_summary.json`
- `runs/phase3_unet/remediated_v1_candidate_score_calibration/candidate_score_calibration.json`
- `runs/phase3_unet/remediated_v1_candidate_score_calibration/calibrated_candidates.jsonl`

On `5976` held-out real-map null-control patches, pooled FPRs remain at a few
percent:

| policy budget | recall diag | null FP | null FPR | 95% CI |
|---|---:|---:|---:|---:|
| CAMB `<=0.05`, real `<=0.02` | `0.2520` | `168 / 5976` | `0.0281` | `[0.0241, 0.0326]` |
| CAMB `<=0.05`, real `<=0.05` | `0.2671` | `147 / 5976` | `0.0246` | `[0.0208, 0.0288]` |
| CAMB `<=0.08`, real `<=0.05` | `0.2778` | `78 / 5976` | `0.0131` | `[0.0103, 0.0163]` |
| CAMB `<=0.08`, real `<=0.08` | `0.3028` | `205 / 5976` | `0.0343` | `[0.0298, 0.0392]` |

The full-sky overlapping-tile audit is more restrictive. At `15 deg` greedy
peak clustering, selected cluster burdens are:

| map | `0.05/0.02` | `0.05/0.05` | `0.08/0.05` | `0.08/0.08` |
|---|---:|---:|---:|---:|
| SMICA | `41` | `41` | `46` | `53` |
| NILC | `34` | `28` | `33` | `39` |
| SEVEM | `64` | `68` | `64` | `72` |
| Commander | `82` | `101` | `71` | `83` |

Interpretation: the larger null-control split does not reproduce the extreme
Commander/SEVEM full-sky tile burden. Do not promote the high-recall
Policy-Pareto winners until candidate-volume criteria are chosen jointly with
cross-map tile burden. In particular, the `0.05/0.05` pair-OR policy has low
SMICA/NILC burden but too many Commander clusters.

The default deployment-decision rule uses `15 deg` clustering, at most `70`
clusters on every cleaned map, at most `0.15` trigger fraction on every map,
and pooled held-out null-control FPR upper CI at most `0.04`. Under that rule,
`0 / 5` composite policies are promotable. The highest-recall row
(`0.08/0.08`) reaches recall `0.3028`, but has `83` Commander clusters and
Commander trigger fraction `0.1772`, so it remains a recall-boost candidate
only.

The broader tile-constrained search uses the same default burden rule but
searches beyond the original Policy-Pareto winners. Current best feasible row:

| policy | real recall | gain vs feasible single | real FPR | pooled null FPR CI high | max clusters | max trigger fraction |
|---|---:|---:|---:|---:|---:|---:|
| `2-of-3(random>=0.999983, imagenet>=0.983177, circular>=51.0957)` | `0.2620` | `+0.0403` | `0.0650` | `0.0290` | `62` Commander | `0.0899` SEVEM |

The best feasible single-score baseline in that same constrained search is
`circular_template_screen >= 73.3128` with recall `0.2216`. The exhaustive
search evaluated all `9582` policies that passed the cheap FPR, pooled-null,
and trigger-fraction filters; no cluster-evaluation cap is active in the
current report. This is the current deployment-safe composite candidate, not a
cosmological detection statistic.

The candidate emitter applies that frozen policy to cached full-sky tile scores
on the canonical science footprint `mask_fraction >= 0.9`. Current emitted
volume:

| map | eligible tiles | tile candidates | eligible trigger frac | cluster reps |
|---|---:|---:|---:|---:|
| SMICA | `5298` | `41` | `0.0077` | `9` |
| NILC | `5298` | `19` | `0.0036` | `5` |
| SEVEM | `5298` | `17` | `0.0032` | `5` |
| Commander | `5298` | `25` | `0.0047` | `5` |

Use `cluster_representatives_15deg.jsonl` for first-pass HM sign-flip or
template-fit follow-up; `candidate_records.jsonl` preserves all overlapping
tile triggers for provenance.
Projection/clustering systematics for that frozen candidate set live in
`runs/phase3_unet/remediated_v1_projection_clustering_audit/`.

Candidate score calibration uses the real-map null-control calibration split,
not the held-out test split:

```bash
python scripts/phase3_remediated_null_policy_audit.py \
  --split calibration \
  --output-dir runs/phase3_unet/remediated_v1_null_policy_calibration
python scripts/phase3_calibrate_candidate_scores.py
```

Current calibrated candidate set: `24` cluster representatives scored against
`5768` pooled calibration-null margins. The most extreme representatives have
plus-one empirical pooled null-survival `p = 0.000173` and BH q-value
`0.000693`. These are screening-tail scores for follow-up prioritization, not
global detection probabilities.

Corrected template-fit and Bayesian handoff artifacts now live in:

- `runs/phase3_unet/remediated_v1_template_fit_handoff/`
- `runs/phase3_unet/remediated_v1_bayesian_template_handoff/`

These package deterministic local Feeney-template seeds plus projection-aware
follow-up guardrails. They are handoff artifacts, not Bayesian evidence.

## Matched-Filter-Channel Recall Candidate

The legacy two-channel v7 U-Net branch now has a full-sky tile-burden audit:

```bash
python scripts/phase3_mf_channel_tile_audit.py
```

Default output:

- `runs/phase3_unet/remediated_v1_mf_channel_tile_audit/mf_channel_tile_audit.json`
- `runs/phase3_unet/remediated_v1_mf_channel_tile_audit/mf_channel_tile_audit.md`
- `runs/phase3_unet/remediated_v1_mf_channel_tile_audit/mf_channel_tile_audit.csv`

At the SMICA-recalibrated model threshold for nominal FPR `0.05`, the legacy
checkpoint reaches diagnostic real-SMICA recall `0.3526` at FPR `0.0440`.
The hard regimes remain difficult: recall is `0.0466` for `A <= 2e-6`,
`0.1724` for `5e-6 <= A <= 2e-5`, and `0.9288` for `A >= 5e-5`.

Canonical-footprint full-sky burden is manageable but larger than the frozen
tile-constrained composite:

| map | eligible tiles | eligible triggers | eligible trigger frac | clusters |
|---|---:|---:|---:|---:|
| SMICA | `5298` | `285` | `0.0538` | `20` |
| NILC | `5298` | `307` | `0.0579` | `21` |
| SEVEM | `5298` | `290` | `0.0547` | `22` |
| Commander | `5298` | `422` | `0.0797` | `23` |

Do not promote this checkpoint as current science. It was trained on the older
`training_v4` feature contract with a legacy `15 arcmin` circular-template
response channel. The result is a strong engineering target for the next
remediated-v1 retrain: cache `features/circular_template_response` with the
current `5 arcmin` beam contract, retrain a two-channel U-Net, then rerun the
same real-null, tile-burden, candidate-calibration, and HM preflight gates.

## Classical Comparators

Historical `matched_template` reports are not true Wiener matched filters. They
are now named `circular_template_screen`.

Current naming contract:

- `circular_template_screen`: patch-space circular disc/ring correlation
  screen; useful as a simple classical comparator and candidate sanity check.
- `wiener_feeney_matched_filter`: harmonic-space Feeney template with beam,
  HEALPix pixel-window, and CMB/noise inverse-covariance weighting.
- `smhw_screen`: spherical Mexican-hat / scale-space context screen.

The true Wiener/SMHW full-sky score-map implementation lives in
`scripts/phase3_classical_filters.py`. Any headline ML-vs-classical claim must
state exactly which comparator was used.

The same-grid true-classical status report is:

```bash
python scripts/phase3_classical_same_grid_status.py
```

A small, non-claim pilot that exercises the full-sky benchmark path is:

```bash
python scripts/phase3_same_grid_fullsky_benchmark.py \
  --max-rows 64 \
  --skip-ml
```

Current output is `complete` for the stratified same-grid manifest in
`runs/phase3_unet/remediated_v1_same_grid_fullsky_manifest/`.

Current same-grid summary on that fixed manifest:

- mean recall: Wiener `0.3841`, ImageNet U-Net `0.3517`, random-init U-Net
  `0.2994`, SMHW `0.2734`
- ImageNet beats Wiener in `17 / 35` raw cells, loses in `16 / 35`, ties in `2`
- under non-overlapping exact 95% CIs, ImageNet is better in `7` cells and
  worse in `8`

Interpretation: the true Wiener/Feeney matched filter is the strongest average
screener on the closed same-grid benchmark, but the ImageNet U-Net keeps
localized wins in selected amplitude/radius cells. That supports a
complementary-screening claim, not uniform ML superiority over the true
Wiener/SMHW comparators. For remediated synthetic products, keep
`--pixel-window-policy synfast_pixwin_true` aligned with
`synfast(pixwin=True)`.

For the benchmark-design fork, generators expose `--injection-convention`.
Use `feeney2011_full_temperature_modulation` for remediated products and
`mcewen2012_first_order_additive` for explicit additive full-sky benchmark
products. If Feeney-modulated products are scored with additive filters instead,
quote the `phase2_physics_checks.py` cross-term report as a direction-specific
approximation check; do not treat it as a universal bound.

## Not Solved

- Weak-family recall remains unsolved. The matched-filter SNR report supports a
  CMB-confusion explanation for `A <= 2e-6`, but it also exposes algorithmic
  headroom in higher-SNR cells where ideal recall is high and ML recall is low.
- The noise-floor diagnostic is still useful as a patch-RMS scale check, but it
  should not be phrased as the central physical limit without the matched-filter
  and same-grid classical results.
- The real-SMICA injection diagnostic still has material FPR inflation for the
  ImageNet branch (`0.372` recall / `0.185` FPR), so it is a transfer diagnostic,
  not a deployment result.
- The same-grid Wiener/SMHW benchmark is now closed on the stratified manifest
  and shows Wiener as the strongest average screener. What remains open is how
  to use that result in deployment: score fusion, the now-completed
  true-Wiener two-stream branch, and projection-robust follow-up still need
  final policy decisions.
- The legacy matched-filter-channel checkpoint improves diagnostic recall, but
  is not promoted because it was trained before `remediated_v1`. It now has a
  tile-burden audit and should be replaced by a remediated two-channel retrain
  before paper-facing claims.
- Half-mission sign-flip calibration is implemented as a Phase 5 downstream
  CLI. The preflight report validates the frozen `24` cluster representatives,
  policy slug, model checkpoints, and common mask, but is currently `blocked`
  because HM1/HM2 cleaned-map paths have not been provided.
- There is no Bayesian parameter inference or model comparison against
  LambdaCDM.
- Candidate-volume accounting exists, but the paper-facing deployment section
  still needs a final cluster radius and score-calibration convention.

## Next Work

1. Decide how to use the completed remediated true-Wiener two-stream branch in
   the active comparison set.
   Status update:
   the corrected full-cache rerun is complete. `true_wiener_ft` raises mean
   sensitivity from `0.34875` to `0.36839` overall (`+0.01964`), helps
   moderate/high-amplitude cells (`+0.03881` mean delta) and large-radius cells
   (`+0.02226`), but still hurts the hardest low-amplitude subset
   (`-0.00592`). Treat it as a promising secondary branch, not a default
   replacement, until deployment burden or score-fusion tests are closed.
2. Select/provide Planck HM1/HM2 component-separated map paths, rerun the Phase
   5 preflight, then run HM sign-flip calibration on the frozen cluster
   representatives.
3. Choose and justify the paper-facing cluster radius for deployment burden
   using the completed projection/clustering audit, rather than the old fixed
   `15 deg` convention by default.
4. Promote the deterministic template-fit and Bayesian handoff into the
   follow-up narrative for the frozen `24` cluster representatives, then add
   native-sphere or projection-robust follow-up for the `17` projection-caution
   candidates.
5. Prioritize real-map null backgrounds over another CAMB-only branch.
6. Feed screened candidates into template fitting, Bayesian follow-up, and
   `scripts/phase5_half_mission_signflip_null.py` when HM inputs are available.

## Key Commands

Generate current training data:

```bash
python scripts/phase2_generate_training.py \
  --num-samples 20000 \
  --pool-size 16000 \
  --num-cmb-realizations 384 \
  --geometry-mode mixed \
  --truncated-positive-fraction 0.35 \
  --mask-threshold 0.9 \
  --beam-fwhm-arcmin 5.0 \
  --output-dir data/remediated_v1
```

Audit the dataset:

```bash
python scripts/phase2_audit_dataset.py \
  --data-h5 data/remediated_v1/training_data.h5 \
  --output-json data/remediated_v1/audit_report.json
```

Evaluate a trained branch on the held-out test split:

```bash
python scripts/phase3_evaluate_run.py \
  --run-dir runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux \
  --checkpoint best \
  --split test \
  --score-mode component_score \
  --image-rule connected_component
```

Rebuild the preselected heatmap from existing remediated scores:

```bash
python scripts/phase3_ml_gain_heatmap.py \
  --analysis-mode preselected \
  --primary-method imagenet_b64_aux \
  --bootstrap-resamples 2000
```

Update reproducibility and noise-floor reports:

```bash
python scripts/create_reproducibility_manifest.py
python scripts/phase3_noise_floor_analysis.py
python scripts/phase3_matched_filter_snr_curve.py
python scripts/phase3_upper_limit_calculator.py
python scripts/phase3_deployment_burden_table.py
python scripts/phase3_policy_pareto_search.py
python scripts/phase3_remediated_null_policy_audit.py
python scripts/phase3_remediated_policy_tile_audit.py
python scripts/phase3_deployment_policy_decision.py
python scripts/phase3_tile_constrained_policy_search.py
python scripts/phase3_emit_tile_constrained_candidates.py
python scripts/phase3_calibrate_candidate_scores.py
python scripts/phase3_classical_same_grid_status.py
python scripts/phase5_half_mission_signflip_null.py \
  --preflight-only \
  --candidate-jsonl runs/phase3_unet/remediated_v1_tile_constrained_candidates/cluster_representatives_15deg.jsonl
```

Run the Nside=32 deployment audit:

```bash
bash scripts/batch6_overnight_orchestrator.sh
python scripts/batch6_overnight_analysis.py
```

## Repository Layout

- `scripts/phase_config.py`: canonical physical constants and remediated
  defaults.
- `scripts/phase2_signal_model.py`: Feeney-style signal template,
  full-temperature modulation, and first-order additive benchmark utilities.
- `scripts/phase2_observing_model.py`: CAMB/synfast observing-model utilities.
- `scripts/phase2_generate_training.py`: current synthetic generator.
- `scripts/phase2_extract_smica_null_controls.py`: per-map real-null extraction.
- `scripts/phase2_audit_dataset.py`: split/provenance/data sanity checks.
- `scripts/phase3_train_unet.py`: U-Net training harness.
- `scripts/phase3_evaluate_run.py`: primary held-out evaluator.
- `scripts/phase3_template_baseline.py`: circular-template patch baseline.
- `scripts/phase3_classical_filters.py`: full-sky Wiener Feeney and SMHW
  classical filters.
- `scripts/phase3_sensitivity_curve.py`: remediated sensitivity grid.
- `scripts/phase3_ml_gain_heatmap.py`: paired ML-vs-circular-template heatmap.
- `scripts/phase3_noise_floor_analysis.py`: empirical CMB-confusion diagnostic.
- `scripts/phase3_matched_filter_snr_curve.py`: ideal harmonic-space Feeney
  matched-filter SNR diagnostic under repo beam/noise/CAMB assumptions.
- `scripts/phase3_upper_limit_calculator.py`: efficiency-weighted
  detectable-collision upper limits.
- `scripts/phase3_deployment_burden_table.py`: full-sky patch and clustered
  candidate-volume accounting from Batch 6 tile caches.
- `scripts/phase3_policy_pareto_search.py`: composite-policy recall-vs-FP
  search over existing remediated score caches.
- `scripts/phase3_deployment_policy_decision.py`: reproducible promotion/
  rejection rule for composite policies.
- `scripts/phase3_tile_constrained_policy_search.py`: exhaustive constrained
  search for deployable composite policies.
- `scripts/phase3_emit_tile_constrained_candidates.py`: frozen full-sky
  candidate and cluster-representative emitter.
- `scripts/phase3_calibrate_candidate_scores.py`: calibration-split
  empirical null-survival scores for emitted representatives.
- `scripts/phase3_classical_same_grid_status.py`: guard report for true
  Wiener/SMHW same-grid benchmark claims.
- `scripts/phase3_same_grid_fullsky_benchmark.py`: guarded full-sky
  same-grid pilot/production driver for Wiener/SMHW and optional ML scoring.
- `scripts/phase3_remediated_null_policy_audit.py`: larger held-out real-null
  audit for composite policies.
- `scripts/phase3_remediated_policy_tile_audit.py`: full-sky overlapping-tile
  burden audit for composite policies.
- `scripts/phase3_fullsky_tile.py`: full-sky tile audit.
- `scripts/phase5_half_mission_signflip_null.py`: downstream HM1/HM2
  sign-flip null calibration and preflight checks for emitted candidates.
- `scripts/audit_remediated_flow.py`: lightweight artifact graph audit.
- `scripts/create_reproducibility_manifest.py`: compact source/artifact
  manifest.
- `work/`: historical decision notes and batch reports.
- `docs/evaluation_inventory.md`: active/deferred/historical evaluation map.
- `docs/phase5_half_mission_signflip_null.md`: HM sign-flip null design and
  usage notes.

Generated datasets, Planck FITS maps, checkpoints, score caches, and large run
artifacts are local products and should remain out of git unless explicitly
promoted as compact metadata.

## References

- Feeney, Johnson, Mortlock, and Peiris. *First Observational Tests of Eternal
  Inflation.* arXiv:1012.1995.
- Feeney, Johnson, Mortlock, and Peiris. *First Observational Tests of Eternal
  Inflation: Analysis Methods and WMAP 7-Year Results.* arXiv:1012.3667.
- McEwen, Feeney, Johnson, and Peiris. *Optimal filters for detecting cosmic
  bubble collisions.* arXiv:1202.2861.
- HEALPix: Gorski et al., Astrophysical Journal, 2005.
- CAMB: Lewis, Challinor, and Lasenby, Astrophysical Journal, 2000.
- Planck Collaboration. *Planck 2018 results. IV. Diffuse component
  separation.* Astronomy & Astrophysics, 2020.

## License

MIT

## Authors

William Starks

Gus Marcum
