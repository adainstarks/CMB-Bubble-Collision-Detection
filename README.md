# CMB Bubble Collision Screener

Planck-era machine-learning and classical candidate screening for localized
bubble-collision signatures in the Cosmic Microwave Background.

## Status note, 21 November 2026

Work continued in November 2026 on a new cluster. The April results below reproduce exactly from the saved
checkpoints and data (identical test-split confusion counts; sensitivity scores equal to numerical noise).
A code review then found four defects that change what those results mean. The sections below this note
are the April text and have not been rewritten; read them with these in mind. Measurements and reasoning
are in `work/notebook.md`.

1. **Signal geometry.** Injected discs were drawn on a pixel grid that is not the gnomonic projection used
   for the backgrounds. Radius labels are up to 6% (centred) or 13% (off-centre) too large.
2. **Sphere placement.** `offset_signal_vector` is mirrored left-right.
3. **Candidate coordinates.** `peak_sky_coord` has both axes reversed, so every candidate position in the
   April reports is reflected through its tile centre. Corrected positions are in
   `work/results_2026-11-21/candidates_corrected.json`: the 102 frozen tile candidates form about five sky
   locations, two of which coincide with published WMAP candidates (the Cold Spot at 3 degrees and
   McEwen et al. 2012 candidate 10 at 5.5 to 9 degrees). The clustering, grouping and survivor tables below
   were built on the reflected coordinates.
4. **Classical comparator.** The Wiener filter bank contains only templates with `|z_crit| = |z0|`. On
   edge-free signals it is at chance (0.06 against 0.30 for the U-Net between 1e-5 and 5e-5), so "Wiener is
   the strongest average screener" is an average over a mixture of edge ratios, not a test against the
   matched filter for the signal theory predicts.

The sharp temperature jump `z_crit` itself is not supported by theory after 2011.
`scripts/core/collision_template_transfer.py` builds the template from the primordial curvature through
CAMB transfer functions instead. With that template an ideal matched filter needs an observed central
amplitude near 5e-5, and most of the April sensitivity between 1e-5 and 5e-5 came from the sharp edge.

First benchmark on the new template (`scripts/analysis/benchmark_transfer_template.py`, notebook entries
(l) to (n)): 12,000 patches of Gaussian sky, signals quoted by observed peak amplitude, 5% false positives.
A filter bank built from the same templates and searching within 6 degrees of the patch centre detects
0.32, 0.59 and 0.98 at 2e-5, 3e-5 and 5e-5. The April ImageNet U-Net, never trained on these templates,
detects 0.24, 0.45 and 0.90, and behaves like the filter with a 12 degree search. Fine-tuned for 12 epochs
on the new templates (`scripts/ml/finetune_on_transfer_template.py`, entries (o), (p)) it detects 0.28, 0.57
and 0.97: about equal to the filter and not above it, as detection theory predicts for Gaussian skies.

Real-sky test (`scripts/analysis/benchmark_real_sky.py`, entries (q) to (s); numbers in
`work/results_2026-11-21/benchmark_real_sky_report.json`): the four PR3 maps, 1,287 tiles at least 90%
unmasked, thresholds from 300 simulated skies prepared with the same mask.

- The matched filter needs no correction on the real sky. 3.5 to 4.0% of real tiles exceed a threshold set
  for 5% on simulated skies, and it detects 0.30, 0.66 and 0.99 of injected templates at 2e-5, 3e-5 and 5e-5.
- Four of its five strongest real-sky responses lie within 4.1 degrees of WMAP 7-year candidates of
  McEwen et al. 2012 (numbers 7, 10, 11 and 8; number 7 is the Cold Spot region), with the same sign and
  size class. The largest is 4.7 sigma and 32 of 300 simulated skies hold a larger one. This is a
  consistency check of the pipeline and a null result, not a detection.
- The real-sky false-positive inflation reported below (0.185 for a nominal 0.05) is reproduced on
  simulated skies and comes from masked pixels. The networks treat zeroed or filled pixels as signal; the
  April real-sky patches had them and the simulated patches did not. With the same preparation on both
  sides the rate is 5 to 6%.
- Given the map's own values under the mask, the fine-tuned network is level with the filter on the real
  sky within the uncertainty (0.50 to 0.70 against 0.66 to 0.68 at 3e-5, depending on the map) and the April
  network is 0.16 to 0.28 behind. With masked pixels zeroed or filled, both networks lose most of their
  sensitivity.

Open as of 23 November 2026 (notebook entries (t) to (ap)):

- The template (entries (u), (v), (y), (z)). It has been rebuilt by two routes that share nothing with the
  CAMB calculation: the Boltzmann code CLASS in real space, and a fluid model written from the textbook
  equations with no Boltzmann code (`scripts/analysis/realspace_ramp_check.py`). For a wall 279 Mpc inside
  the last-scattering surface all three agree to 3%: the centre is 2.5 times the Sachs-Wolfe value and the
  edge is at 0.49 of the centre. The published transfer function of Feeney et al. (arXiv:1506.01716, Fig. 6)
  is reproduced as well. Fig. 3 of Osborne, Senatore and Smith (arXiv:1305.1970) is not reproduced as
  labelled (11.39 degrees, 279 Mpc deep): its ramp is close to the Sachs-Wolfe one (edge at 0.16 of the
  centre) and has the shape this calculation gives for a wall 110 to 160 Mpc deep. The distance behind the
  label, 14,165.5 Mpc, is the last-scattering distance of the WMAP 5-year maximum-likelihood parameters
  (to 3 Mpc); the paper says it uses WMAP7+BAO+H0, for which the same wall is 185 Mpc deep. Which
  parameters went into the figure is a question for its authors. The template is unchanged.
- Full-sky filter search after McEwen et al. (arXiv:1202.2861), entries (w), (x), (ah), (ai)
  (`scripts/analysis/fullsky_filter_search.py`): every unmasked position, 33 sizes from 1 to 90 degrees,
  the thresholds of Feeney et al. (arXiv:1210.2725). On the four Planck maps (77% of the sky) the number of
  candidates and the largest significance are ordinary among Gaussian skies with the same mask: 31 to 34
  candidates per map where simulated skies give 23 to 106, largest significance 4.8 with p = 0.73 to 0.76,
  32 of 33 candidates common to all four maps, none at 45 degrees or more. Eight of the 16 WMAP candidates
  of McEwen et al. are found again. The cosine template of this search is the one their text describes
  (z_crit = 0). With it, the check on their own data (WMAP 7-year W band) missed both criteria fixed
  beforehand: the significances came out lower by about 0.3 and their two largest candidates did not
  appear. The cause is the template, see the next item; with the template of their code the same check
  meets both criteria. One step of the algorithm was corrected on the way (detections on neighbouring
  sizes are now compared in z0, as in their text); the Planck candidate lists did not change. Open: the
  Planck end-to-end simulations.
- The same search on Planck with the template of their code, entries (al), (am): a like-for-like repeat of
  the published search (its template is described in the next item). No excess. Per map 50 to 54
  candidates where simulated skies give 85 on average (33 to 171), p = 0.75 to 0.80; largest significance
  4.7, p = 0.82 to 0.87; 4 or 5 candidates of 45 degrees or more where simulated skies have a median of 22,
  p = 0.93 to 0.94. With multipoles up to 767 the same (94 to 100 candidates against 144, largest
  significance 5.1 to 5.2 with p = 0.40 to 0.47). Thirteen of the 16 WMAP candidates of McEwen et al. are
  candidates here (14 with multipoles up to 767), their two largest included, with amplitudes within
  0.03 mK of the published ones except one 2 degree feature that is weaker on Planck. So their candidates
  are features of the sky, and a Gaussian sky run through the same search has more of them, not fewer. In
  the central amplitude z0 this search has the same reach as the z_crit = 0 one: 8e-5 to 10e-5 from 5 to
  60 degrees for the 95% level.
- The template behind the published filter search, entries (aj), (ak) (figure
  `plots/11_mcewen_template_edge.png`, `scripts/analysis/check_mcewen_template.py`). The text of McEwen et
  al. sets the temperature at the edge of the disc to z_crit ~ 0. The authors' public code (COMB,
  `comb_tmplmap_bubble`) has z_crit = 0.2 z0 as its default and in its object generator, with a smooth
  cut-off between 1.0 and 1.1 times the size. Their own figures follow the code: the drawn profile reads 19
  to 20 uK at the edge of a 100 uK, 10 degree disc; the drawn spectrum of that template matches to 2 to 5%;
  and their signal-to-noise curve of the matched filter is reproduced to 0.4% (1.7% at most) with the code's
  template and their own spectrum file, against 3 to 10% too low with z_crit = 0. On the WMAP W band, with
  the code's template, the significance is within 0.5 of the published value for 14 of their 16 candidates
  (10 with z_crit = 0; 12 were required), 9 of the 11 eligible ones are found (6 before; 7 required), all 16
  amplitudes agree to 0.02 mK, and their two largest candidates are there. One of those and one other are
  centred on masked pixels, which this search leaves out and theirs did not. Whether "~ 0" was meant
  loosely or a default stayed in is a question for the authors; the parameter files of their run are not
  public. Theory after 2011 gives no sharp jump at the edge and no single edge value for every size: in the
  ramp model the temperature at the causal edge is about 0.5 of the centre at 10 degrees, 0.2 at 20
  degrees and under 0.1 above 40 degrees, and it falls smoothly to zero about one sound horizon outside
  (entries (f), (g), (an)). Measured (entries (ao), (ap), figure `plots/19_code_template_vs_transfer.png`,
  `scripts/analysis/mcewen_template_vs_transfer.py`): as a matched filter sees them, the template of their
  code overlaps the transfer template by 0.98 to 0.99 at 20 to 22 degrees, 0.70 at 10 degrees and 0.87 at
  40 degrees, and its 20 degree default matches a 20.0 degree transfer disc best. So their defaults are the
  physical template of a disc of about 20 to 24 degrees and of no other size. The z_crit = 0 template
  reaches 0.56, 0.76 and 0.92 at 10, 20 and 40 degrees at the same nominal size. With an edge value and
  a cut-off that depend on the size (0.53 and 1.39 at 10 degrees, 0.25 and 1.10 at 20, 0.09 and 1.03 at
  40) their two-parameter form reproduces the transfer template to better than 0.99 from 7 degrees up.
  No paper or code comment gives a reason for 0.2 and 1.1; the first bubble commit of their repository
  (10 November 2011) already has them, applied to 10, 15 and 20 degree discs alike. This project keeps
  z_crit = 0 for the cosine template, as the published papers describe it, and carries the size-dependent
  edge in the transfer template; `--cosine-edge` exists for comparisons with their paper. For a signal that did carry such an edge, limits
  on z0 from the z_crit = 0 filters would be up to a third weaker below 60 degrees and a factor of two
  weaker at 90 degrees.
- What that search could have seen, entries (ae) to (ag) (figure `plots/10_fullsky_search_sensitivity.png`,
  numbers in `work/results_2026-11-22/fullsky_inject_summary_planck_v1.md`): templates of known size and
  amplitude were added at 150 positions on 100 simulated skies, and on the real maps. A collision whose
  peak in the map is above about 8e-5 in dT/T (220 uK) below 30 degrees, falling to 4e-5 at 90 degrees,
  would have been the strongest feature of the whole map in 95% of cases; the real maps hold no such
  feature. Half of the collisions reach the candidate threshold at 3e-5 to 4e-5 up to 14 degrees and at
  1e-5 to 2e-5 from 50 degrees up. In the usual parameter z0 the 95% level is 8e-5 to 10e-5 from 5 to 60
  degrees for the cosine template. This holds for collisions centred in the unmasked 77% of the sky, for
  Gaussian simulated skies and for multipoles up to 256. The measurement uses a linear shortcut that was
  checked against the full search on 160 cases (identical). On WMAP, at the two positions McEwen et al.
  used, the same measurement agrees with their published exclusion and sensitivity regions within one step
  of their grid in 7 of 8 comparisons between 5 and 60 degrees with z_crit = 0, and in all 8 with the
  template of their code. At 50 degrees their six detections out of six still have a probability of only
  5% under the detection rate measured here. The 3 to 10% by which their filter signal-to-noise curve was
  missed is the difference between the two templates (previous item).
- The April scripts and data sets are unchanged, so the April results still reproduce. Defects 1 to 4 are
  still in them; the corrected helpers (`scripts/core/patch_geometry.py`) are used by new work only.
- Not redone: the candidate-centred re-extraction at the corrected positions, and the project's full
  quality-gate script in the new environment.
- Not started: a network trained with masked pixels in its patches, discs of 25 degrees and wider (the
  patch is the limit there), and the ramp+step and quadratic templates. The fine-tuned network is one run
  with one seed, and the simulated skies are Gaussian with white noise.

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
python scripts/ops/audit_remediated_flow.py
```

Audit the raw local Planck FITS headers, units, and release provenance:

```bash
python scripts/data/phase0_planck_unit_audit.py
```

Run the broader quality gate:

```bash
python scripts/ops/run_quality_gates.py \
  --data-h5 data/remediated_v1/training_data.h5 \
  --skip-train-dry-run
```

Write a compact reproducibility manifest:

```bash
python scripts/ops/create_reproducibility_manifest.py
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
python scripts/analysis/phase3_noise_floor_analysis.py
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
python scripts/analysis/phase3_matched_filter_snr_curve.py
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
python scripts/analysis/phase3_upper_limit_calculator.py
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
python scripts/analysis/phase3_deployment_burden_table.py
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
python scripts/analysis/phase3_policy_pareto_search.py
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
python scripts/analysis/phase3_remediated_null_policy_audit.py
python scripts/analysis/phase3_remediated_policy_tile_audit.py
python scripts/analysis/phase3_tile_constrained_policy_search.py
python scripts/followup/phase3_emit_tile_constrained_candidates.py
python scripts/followup/phase3_calibrate_candidate_scores.py
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
python scripts/analysis/phase3_remediated_null_policy_audit.py \
  --split calibration \
  --output-dir runs/phase3_unet/remediated_v1_null_policy_calibration
python scripts/followup/phase3_calibrate_candidate_scores.py
```

Current calibrated candidate set: `24` cluster representatives scored against
`5768` pooled calibration-null margins. The most extreme representatives have
plus-one empirical pooled null-survival `p = 0.000173` and BH q-value
`0.000693`. These are screening-tail scores for follow-up prioritization, not
global detection probabilities.

Corrected template-fit and Bayesian handoff artifacts now live in:

- `runs/phase3_unet/remediated_v1_template_fit_handoff/`
- `runs/phase3_unet/remediated_v1_bayesian_template_handoff/`
- `runs/phase3_unet/phase5_half_mission_signflip_null/`
- `runs/phase3_unet/phase5_frequency_jackknife_followup/`
- `runs/phase3_unet/remediated_v1_scale_dependent_followup_table/`

These package deterministic local Feeney-template seeds plus projection-aware
follow-up guardrails. They are handoff artifacts, not Bayesian evidence.

Current real-sky follow-up outcome on the frozen `24` representatives:

- Common-resolution `100/143/217 GHz` frequency jackknife: `23 / 24` stable;
  the lone failure is a mask-coverage skip (`nilc`, patch `9034`), not a broad
  spectral collapse.
- HM sign-flip calibration: `3 / 24` candidates satisfy the default
  `p <= 0.05` stability gate; `1 / 24` is skipped because the projected HM
  valid-mask fraction falls below `0.9`.
- Combined handoff: only `3 / 24` representatives survive both HM and
  frequency follow-up, so the screening story must be framed as aggressive
  candidate triage rather than a near-detection list.
- Scale-dependent follow-up grouping with
  `r_assoc = clamp(theta_crit_fit / 2, 5 deg, 15 deg)` collapses those `24`
  representatives into `10` sky systems; only `2 / 10` systems contain any
  HM-plus-frequency survivor, and `8 / 10` systems remain geometry-limited
  until projection-robust re-extraction.
- Projection-robust survivor follow-up now exists for those `2` adaptive
  systems in
  `runs/phase3_unet/remediated_v1_projection_robust_survivor_followup/`.
  Current outcome: one surviving system (`nilc/sevem:9629`) remains stable on
  candidate-centered equal-area re-extraction, while the other survivor
  (`nilc:9628`) fails candidate-centered mask support and should not be treated
  as projection-robust yet.
- A candidate-specific geometry-resolution study now lives in
  `runs/phase3_unet/remediated_v1_candidate_geometry_resolution/`.
  Current result: `nilc:9628` is `center_unstable_screening_only`. Its
  candidate-centered follow-up fails at the strict `0.90` support cut, and the
  strictest relaxed fit (`0.885`) recovers only about `6-8%` of the screening
  seed amplitude while tile- and seed-centered fits remain projection-consistent
  but disagree by about a factor of `15` in median amplitude retention.
- The downstream local template-likelihood packet for the only
  `projection_robust_ready` system now lives in
  `runs/phase3_unet/remediated_v1_system_template_likelihood_packet/`.
- The paper-facing ranked survivor table now lives in
  `runs/phase3_unet/remediated_v1_paper_survivor_table/`.

## Matched-Filter-Channel Recall Candidate

The legacy two-channel v7 U-Net branch now has a full-sky tile-burden audit:

```bash
python scripts/analysis/phase3_mf_channel_tile_audit.py
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
`scripts/analysis/phase3_classical_filters.py`. Any headline ML-vs-classical claim must
state exactly which comparator was used.

The same-grid true-classical status report is:

```bash
python scripts/analysis/phase3_classical_same_grid_status.py
```

A small, non-claim pilot that exercises the full-sky benchmark path is:

```bash
python scripts/analysis/phase3_same_grid_fullsky_benchmark.py \
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

An exploratory out-of-fold Wiener-plus-ImageNet logistic fusion now lives in
`runs/phase3_unet/remediated_v1_same_grid_wiener_ml_fusion/`. On the fixed
same-grid manifest it reaches mean recall `0.4046` at the same realized
negative `FPR = 0.0498`, above Wiener alone (`0.3841`) and ImageNet alone
(`0.3517`). Keep the interpretation narrow: this is a synthetic same-grid
screening gain, not yet a promoted deployment branch.

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
  to use that result in deployment: the calibrated score-fusion test now exists
  and improves mean same-grid recall, but both it and the true-Wiener
  two-stream branch remain unpromoted because they still lack Batch-6
  deployment-burden, candidate-calibration, and frozen candidate follow-up
  evidence.
- The follow-up story is now narrower than the original frozen-candidate list:
  only `nilc/sevem:9629` is downstream-hand-off-ready, while `nilc:9628`
  remains screening-only because the recovered fit depends too strongly on
  center choice.
- The legacy matched-filter-channel checkpoint improves diagnostic recall, but
  is not promoted because it was trained before `remediated_v1`. It now has a
  tile-burden audit and should be replaced by a remediated two-channel retrain
  before paper-facing claims.
- Half-mission sign-flip calibration is now fully run on the PR3 HM maps.
  Only `3 / 24` frozen representatives survive the default HM `p <= 0.05`
  gate, so cross-map agreement alone is not sufficient evidence that the
  screened candidates are robust real-sky follow-up targets.
- There is no Bayesian parameter inference or model comparison against
  LambdaCDM.
- Candidate-volume accounting exists, but the paper-facing follow-up section
  should now use the scale-dependent system table for scientific identity and
  reserve the fixed `15 deg` clustering for workload accounting only.

## Next Work

1. Keep the corrected true-Wiener two-stream branch as a secondary synthetic
   result, not a promoted deployment branch. The calibrated Wiener-plus-ML
   score-fusion test now exists on the same-grid manifest and is the leading
   synthetic-only model-improvement result, but it still needs deployment,
   calibration, and real-sky follow-up closure before promotion.
2. Treat the new scale-dependent follow-up table as the paper-facing candidate
   identity convention and reserve the fixed `15 deg` clusters for workload
   accounting only.
3. Push the projection-robust-ready survivor system (`nilc/sevem:9629`) into
   the downstream template-likelihood or Bayesian stage.
4. Treat the other surviving system (`nilc:9628`) as screening-only under the
   current pipeline. Do not promote it into the downstream likelihood handoff
   without a native-sphere center-marginalized follow-up path.
5. Prioritize real-map null backgrounds and later domain adaptation only after
   the current screening-plus-follow-up narrative is scientifically clean.

## Key Commands

Generate current training data:

```bash
python scripts/data/phase2_generate_training.py \
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
python scripts/data/phase2_audit_dataset.py \
  --data-h5 data/remediated_v1/training_data.h5 \
  --output-json data/remediated_v1/audit_report.json
```

Evaluate a trained branch on the held-out test split:

```bash
python scripts/ml/phase3_evaluate_run.py \
  --run-dir runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux \
  --checkpoint best \
  --split test \
  --score-mode component_score \
  --image-rule connected_component
```

Rebuild the preselected heatmap from existing remediated scores:

```bash
python scripts/analysis/phase3_ml_gain_heatmap.py \
  --analysis-mode preselected \
  --primary-method imagenet_b64_aux \
  --bootstrap-resamples 2000
```

Update reproducibility and noise-floor reports:

```bash
python scripts/ops/create_reproducibility_manifest.py
python scripts/analysis/phase3_noise_floor_analysis.py
python scripts/analysis/phase3_matched_filter_snr_curve.py
python scripts/analysis/phase3_upper_limit_calculator.py
python scripts/analysis/phase3_deployment_burden_table.py
python scripts/analysis/phase3_policy_pareto_search.py
python scripts/analysis/phase3_remediated_null_policy_audit.py
python scripts/analysis/phase3_remediated_policy_tile_audit.py
python scripts/analysis/phase3_deployment_policy_decision.py
python scripts/analysis/phase3_tile_constrained_policy_search.py
python scripts/followup/phase3_emit_tile_constrained_candidates.py
python scripts/followup/phase3_calibrate_candidate_scores.py
python scripts/analysis/phase3_classical_same_grid_status.py
python scripts/followup/phase5_half_mission_signflip_null.py \
  --preflight-only \
  --candidate-jsonl runs/phase3_unet/remediated_v1_tile_constrained_candidates/cluster_representatives_15deg.jsonl
```

Run the Nside=32 deployment audit:

```bash
bash scripts/ops/batch6_overnight_orchestrator.sh
python scripts/ops/batch6_overnight_analysis.py
```

## Repository Layout

- `scripts/core/phase_config.py`: canonical physical constants and remediated
  defaults.
- `scripts/core/phase2_signal_model.py`: Feeney-style signal template,
  full-temperature modulation, and first-order additive benchmark utilities.
- `scripts/core/phase2_observing_model.py`: CAMB/synfast observing-model utilities.
- `scripts/data/phase2_generate_training.py`: current synthetic generator.
- `scripts/data/phase2_extract_smica_null_controls.py`: per-map real-null extraction.
- `scripts/data/phase2_audit_dataset.py`: split/provenance/data sanity checks.
- `scripts/ml/phase3_train_unet.py`: U-Net training harness.
- `scripts/ml/phase3_evaluate_run.py`: primary held-out evaluator.
- `scripts/analysis/phase3_template_baseline.py`: circular-template patch baseline.
- `scripts/analysis/phase3_classical_filters.py`: full-sky Wiener Feeney and SMHW
  classical filters.
- `scripts/analysis/phase3_sensitivity_curve.py`: remediated sensitivity grid.
- `scripts/analysis/phase3_ml_gain_heatmap.py`: paired ML-vs-circular-template heatmap.
- `scripts/analysis/phase3_noise_floor_analysis.py`: empirical CMB-confusion diagnostic.
- `scripts/analysis/phase3_matched_filter_snr_curve.py`: ideal harmonic-space Feeney
  matched-filter SNR diagnostic under repo beam/noise/CAMB assumptions.
- `scripts/analysis/phase3_upper_limit_calculator.py`: efficiency-weighted
  detectable-collision upper limits.
- `scripts/analysis/phase3_deployment_burden_table.py`: full-sky patch and clustered
  candidate-volume accounting from Batch 6 tile caches.
- `scripts/analysis/phase3_policy_pareto_search.py`: composite-policy recall-vs-FP
  search over existing remediated score caches.
- `scripts/analysis/phase3_deployment_policy_decision.py`: reproducible promotion/
  rejection rule for composite policies.
- `scripts/analysis/phase3_tile_constrained_policy_search.py`: exhaustive constrained
  search for deployable composite policies.
- `scripts/followup/phase3_emit_tile_constrained_candidates.py`: frozen full-sky
  candidate and cluster-representative emitter.
- `scripts/followup/phase3_calibrate_candidate_scores.py`: calibration-split
  empirical null-survival scores for emitted representatives.
- `scripts/analysis/phase3_classical_same_grid_status.py`: guard report for true
  Wiener/SMHW same-grid benchmark claims.
- `scripts/analysis/phase3_same_grid_fullsky_benchmark.py`: guarded full-sky
  same-grid pilot/production driver for Wiener/SMHW and optional ML scoring.
- `scripts/analysis/phase3_remediated_null_policy_audit.py`: larger held-out real-null
  audit for composite policies.
- `scripts/analysis/phase3_remediated_policy_tile_audit.py`: full-sky overlapping-tile
  burden audit for composite policies.
- `scripts/analysis/phase3_fullsky_tile.py`: full-sky tile audit.
- `scripts/followup/phase5_half_mission_signflip_null.py`: downstream HM1/HM2
  sign-flip null calibration and preflight checks for emitted candidates.
- `scripts/ops/audit_remediated_flow.py`: lightweight artifact graph audit.
- `scripts/ops/create_reproducibility_manifest.py`: compact source/artifact
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
