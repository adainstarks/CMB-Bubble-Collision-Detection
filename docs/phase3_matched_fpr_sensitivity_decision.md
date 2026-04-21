# Historical Phase 3 Matched-FPR Sensitivity Decision

This note is retained for provenance only. It predates the remediated-v1
artifact family, the method-name cleanup, and the preselected ImageNet heatmap.
Historical `matched_template` below means `circular_template_screen`, not a
Wiener-whitened Feeney matched filter.

Current replacement artifacts:

- `runs/phase3_unet/remediated_v1_sensitivity_curve/sensitivity_report.{json,md}`
- `runs/phase3_unet/remediated_v1_sensitivity_curve/ml_gain_heatmap_imagenet_preselected/`
- `PROJECT_HANDOFF.md`

Generated from the Day 1-4 evaluation stack:

- Sensitivity curves: `runs/phase3_unet/sensitivity_curve_v1/sensitivity_report.json`
- Sensitivity plot: `runs/phase3_unet/sensitivity_curve_v1/sensitivity_curve.png`
- SMICA null burden at sensitivity thresholds: `runs/phase3_unet/null_burden_matched_fpr_v1/null_burden_matched_fpr.json`
- Combined stratified validation table: `runs/phase3_unet/stratified_external_combined_v1/combined_stratified_eval_report.json`

## Decision

Outcome B, but constrained:

At matched synthetic FPR `0.05`, at least one ML branch beats the beam-matched Feeney-template baseline in several `(A, theta_c)` cells with non-overlapping exact binomial confidence intervals. The result is not a universal ML win. It is localized to specific amplitude/radius regimes, mainly moderate amplitudes at larger angular radii and high amplitudes at small angular radii.

This supports the statement:

> U-Net screening provides localized sensitivity gains over a beam-matched circular Feeney-template screen at selected angular scales and amplitudes, while preserving comparable or lower SMICA null burden for the boundary-aware and V6 branches.

It does not support:

> The U-Net dominates the matched filter everywhere.

## Sensitivity Summary

Sensitivity setup:

- Positives: `7000`
- Negatives for threshold calibration: `5000`
- Grid: `A = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]`
- Angular radii: `theta_c = [5, 10, 15, 20, 25] deg`
- Positives per cell: `200`
- Signal: hard-boundary Feeney Eq. 1
- Amplitude definition: `|z0| = |zcrit| = A`
- Sign quadrants: exactly balanced
- Edge smoothing: `0.0 deg`
- Beam/noise forward model: `15 arcmin` beam and `30 uK-arcmin` white noise
- Coordinate exclusions: training and stratified-validation coordinate pools excluded at `0.25 deg`

Threshold calibration:

| method | threshold | synthetic negative FP | synthetic FPR |
|---|---:|---:|---:|
| matched_template | `76.977921` | `249 / 5000` | `0.0498` |
| centered_disc | `0.66798174` | `249 / 5000` | `0.0498` |
| original_v4 | `0.99662161` | `249 / 5000` | `0.0498` |
| boundary_v4 | `0.96436530` | `249 / 5000` | `0.0498` |
| v5_consensus | `0.99094790` | `249 / 5000` | `0.0498` |
| v6_aux_only | `0.99208200` | `249 / 5000` | `0.0498` |
| v6_hard_w15 | `0.99615461` | `249 / 5000` | `0.0498` |

Non-overlapping CI wins versus matched template:

| method | better cells | lower cells |
|---|---:|---:|
| centered_disc | `0` | `9` |
| original_v4 | `3` | `1` |
| boundary_v4 | `5` | `0` |
| v5_consensus | `5` | `0` |
| v6_aux_only | `6` | `0` |
| v6_hard_w15 | `6` | `0` |

Representative ML-favorable cells:

| cell | matched_template | v6_aux_only | v6_hard_w15 |
|---|---:|---:|---:|
| `A=1e-5, theta=25 deg` | `0.065` | `0.170` | `0.175` |
| `A=2e-5, theta=20 deg` | `0.235` | `0.555` | `0.445` |
| `A=2e-5, theta=25 deg` | `0.375` | `0.700` | `0.710` |
| `A=5e-5, theta=5 deg` | `0.065` | `0.250` | `0.295` |
| `A=5e-5, theta=10 deg` | `0.490` | `0.960` | `0.975` |
| `A=1e-4, theta=5 deg` | `0.500` | `0.865` | `0.890` |

## SMICA Null Burden At Sensitivity Thresholds

All methods were applied to the same `5000` SMICA null-control patches using the thresholds above.

| method | false positives | null FPR | exact 95% CI |
|---|---:|---:|---:|
| matched_template | `0 / 5000` | `0.0000` | `[0.0000, 0.0007]` |
| centered_disc | `114 / 5000` | `0.0228` | `[0.0188, 0.0273]` |
| original_v4 | `20 / 5000` | `0.0040` | `[0.0024, 0.0062]` |
| boundary_v4 | `0 / 5000` | `0.0000` | `[0.0000, 0.0007]` |
| v5_consensus | `0 / 5000` | `0.0000` | `[0.0000, 0.0007]` |
| v6_aux_only | `0 / 5000` | `0.0000` | `[0.0000, 0.0007]` |
| v6_hard_w15 | `0 / 5000` | `0.0000` | `[0.0000, 0.0007]` |

Interpretation:

The strongest null-cleanliness claim is not that ML beats the matched template on SMICA nulls. The matched template is also clean at this operating point. The correct claim is that the boundary-aware and V6 branches achieve localized sensitivity gains while matching the matched-template SMICA null burden in this control set.

## Stratified Validation Comparator

On the independent stratified validation set at matched FPR `0.08`, the combined table is:

| method | AUROC | AUPRC | recall | weak recall | Dice+ |
|---|---:|---:|---:|---:|---:|
| matched_template | `0.712 [0.698,0.727]` | `0.881 [0.871,0.889]` | `0.401` | `0.282 [0.263,0.300]` | `0.295 [0.283,0.307]` |
| centered_disc | `0.639 [0.623,0.656]` | `0.833 [0.820,0.845]` | `0.252` | `0.176 [0.160,0.191]` | `0.046 [0.043,0.050]` |
| original_v4 | `0.775 [0.762,0.787]` | `0.914 [0.907,0.921]` | `0.541` | `0.425 [0.405,0.443]` | `0.344 [0.331,0.356]` |
| boundary_v4 | `0.774 [0.761,0.786]` | `0.914 [0.907,0.921]` | `0.540` | `0.420 [0.401,0.439]` | `0.377 [0.363,0.391]` |
| v5_consensus | `0.774 [0.762,0.787]` | `0.913 [0.906,0.920]` | `0.536` | `0.422 [0.403,0.442]` | `0.396 [0.383,0.409]` |
| v6_aux_only | `0.773 [0.761,0.786]` | `0.913 [0.906,0.920]` | `0.539` | `0.422 [0.404,0.442]` | `0.401 [0.388,0.416]` |
| v6_hard_w15 | `0.773 [0.761,0.785]` | `0.913 [0.905,0.919]` | `0.529` | `0.416 [0.395,0.435]` | `0.372 [0.359,0.385]` |

Interpretation:

The ML branches beat the two classical baselines on the stratified synthetic validation table. However, the ML branches remain nearly indistinguishable from one another as detectors. Their main separation is morphology/Dice and null behavior, not pooled detection sensitivity.

## Paper Claim Skeleton

Working claim:

> We present a reproducible Planck-era ML candidate-screening pipeline for localized bubble-collision signatures. On independent hard-boundary Feeney Eq. 1 injections, U-Net screeners show localized sensitivity gains over a beam-matched circular-template baseline at selected amplitude/radius cells, while boundary-aware variants preserve a zero-candidate SMICA null burden at the matched synthetic-FPR operating point. The method is positioned as a candidate generator and morphology-aware pre-filter for downstream classical/Bayesian follow-up, not as a standalone detection claim.

Do not claim:

- Do not claim full Feeney-style detection framework.
- Do not claim universal ML dominance over matched filtering.
- Do not claim real-sky systematics robustness beyond the evaluated SMICA null-control set from this run.
- Do not claim calibrated cosmological evidence or constraints on expected detectable collisions.

## Methods Section Skeleton

1. Data and observable domain:
   Planck-era patch extraction, mask-aware coordinate pools, gnomonic geometry, `Nside=256`, `256 x 256` patches, `13 arcmin` pixels.

2. Signal model:
   Feeney Eq. 1 hard-boundary template, Feeney 2011 full-temperature modulation, balanced sign quadrants, amplitude grid, radius grid.

3. Forward model:
   CAMB CMB realizations, beam smoothing, noise injection, coordinate exclusion from previous train/validation products.

4. Screeners:
   Beam-matched circular-template screen, centered-disc shortcut baseline, original V4 U-Net, boundary-aware V4, V5/V6 branches.

5. Operating point:
   Thresholds fixed by `5000` synthetic negatives at `FPR=0.05`; no threshold changes during real-SMICA null scoring.

6. Sensitivity:
   Report `P_det(A, theta_c)` with exact binomial 95% confidence intervals.

7. Null controls:
   Report SMICA false-candidate burden at sensitivity-calibrated thresholds with exact binomial 95% confidence intervals.

8. Stratified validation:
   Report AUROC, AUPRC, matched-FPR recall, weak-family recall, and Dice with bootstrap confidence intervals.

9. Interpretation:
   ML contribution is localized sensitivity plus mask/morphology output for handoff, not replacement of Bayesian evidence calculation.
