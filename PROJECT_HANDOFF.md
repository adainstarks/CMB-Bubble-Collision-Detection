# Project Handoff: CMB Bubble Collision Screening

Last updated: 2026-04-17 (Batch 5 full-sky audit: retracts Batch 4 shipped lift; all FPR-calibrated thresholds in the repo are suspect until deployment-representative null pools are built)
Repo path: `/data/william/CMB-Collision-Bubbles`
Primary current claim: Planck-era ML/classical candidate-screening front end for localized bubble-collision signatures; not a standalone cosmological detection or Feeney-style Bayesian evidence pipeline.

## 0. Current Working Interpretation

This repo is a research pipeline for candidate screening, not a completed discovery framework.

Best current operational stack (revised 2026-04-17 after Batch 5 full-sky audit retracted the Batch 4 shipped lift):

- **Primary deployment policy: UNCERTAIN, pending deployment-representative recalibration.** The Batch 4 shipped `learned_gbt` with 14 features lifted the gate-cell recall by +0.043 on clean-null calibration, but the Batch 5 full-sky tile audit showed this lift (a) does not survive deployment-representative tile calibration on any map except weakly NILC, (b) actively reverses by −0.045 on SEVEM and −0.036 on Commander. Cross-map mean delta is −0.015, stdev 0.027. See Section 27 and `work/batch5_fullsky_calibration_gap.md`. The 14-feature router is no longer recommended as primary.
- **Falling back to the PR #8 baselines until recalibration finishes**: `v6_aux_only` at per-map-calibrated thresholds, with `learned_gbt` on 6 score features as an optional composite. Both are also subject to clean-null calibration bias (Section 27) — their real-sky-deployment recall has not yet been measured on a deployment-representative null pool. Do not quote the shipped FPR-0.08 recall numbers as deployment expectations until Step 2 finishes.
- `v6_aux_only`: primary backbone ML screener; heavily weighted by the router (~44% feature importance). Also retained as a documented single-model fallback when a reviewer does not want a composite policy.
- `v7_mixed_ft`: retained as an input to the learned router and as a Phase 5 specialist on truncated candidates. Not run as a separate parallel screener.
- `matched_template`: classical Feeney-template reference, fallback, and independent score.
- Thresholds must be calibrated on real-map null controls, not CAMB-only negatives.
- The five-model stack was important for investigation but should not be treated as the default deployment path.

Current blocker:

- Weak-family recall at low amplitude (A <= 5e-6) is still near the Planck SMICA noise floor. This is physical, not engineering.
- `v7_mixed_ft` dominates on CAMB backgrounds but loses to `v6_aux_only` on real SMICA in isolation.
- Batch 2 post-processing ablation (`work/batch2_postprocess_ablation.md`) confirmed Gaussian smoothing is null and matched-filter-on-mask is a geometry specialist, not a general replacement.
- Batch 3 simple ensemble / heuristic-router policies all underperformed `v6_only` at FPR 0.08; the 6-feature learned GBT classifier beat `v6_only` by +3.1pp under clean-null calibration. The real-sky-tile-calibrated comparison has not yet been recomputed (Step 2b pending).
- Batch 4 router feature expansion reported +0.043 gate-cell lift on clean-null calibration, but the Batch 5 full-sky audit retracted that claim for deployment: geometry features specifically overfit to the clean null pool's mask-fraction distribution. See Section 27.
- **Biggest blocker (as of 2026-04-17): the `smica_null_controls_all.h5` calibration pool was drawn at `MASK_THRESHOLD = 0.95` and systematically excludes mask-adjacent sky. Deployment-tile patch-level FPR is 2-6x what the calibration pool says (SMICA 1.96x, NILC 2.01x, SEVEM 3.64x, Commander 5.78x). Fix: rebuild null pools per map at `MASK_THRESHOLD = 0.5` and re-evaluate every FPR-calibrated threshold the repo has shipped.**
- Remaining untouched training-signal lever: `v8` retrain with matched-filter response map as a second input channel on mixed geometry. Earlier MF-channel training (`phase3_v7_mf_channel_aux_w4`) on contained data gave only +0.013 contested recall; Section 25's speculative "+4-10pp" estimate is not calibrated by that prior. Mixed-geometry retrain has not been run and its expected lift is now unclear.

Do not proceed with:

- The current `radius_head_w02` branch. It failed the fixed stratified external gate.
- More small U-Net/loss variants without a specific failure-cell hypothesis.
- Full Nside=512 retraining; the focused probe did not justify it.

## 1. Scientific Background And Required Framing

Relevant local references:

- `references/papers/1012.1995.pdf`: Feeney, Johnson, Mortlock, Peiris, "First Observational Tests of Eternal Inflation", arXiv:1012.1995.
- `references/papers/1012.3667.pdf`: related Feeney et al. WMAP/eternal-inflation observational analysis.

Feeney-style full detection framework includes:

- Candidate localization.
- Causal-boundary/edge testing.
- Bayesian parameter estimation.
- Model comparison against LambdaCDM.
- Interpretation in terms of expected detectable collision count.

This repo currently covers:

- Physically motivated candidate generation.
- ML/classical screening and ranking.
- Null calibration and threshold-volume analysis.
- Structured handoff artifacts for later classical/Bayesian follow-up.

This repo currently does not cover:

- Final Bayesian posterior over bubble-collision parameters.
- Evidence ratio against LambdaCDM.
- Cosmological detection claim.
- Constraint on expected detectable collision count.

Correct public claim:

```text
A reproducible Planck-era ML candidate-screening method for localized bubble-collision signatures, intended to accelerate or supplement classical follow-up.
```

## 2. Signal Model And Injection Physics

Feeney Eq. 1 template used in `scripts/phase2_signal_model.py`:

```math
deltaT/T =
[
(zcrit - z0 cos(theta_crit)) / (1 - cos(theta_crit))
+
(z0 - zcrit) / (1 - cos(theta_crit)) cos(theta)
] H(theta_crit - theta)
```

Variables:

- `z0`: central modulation amplitude.
- `zcrit`: boundary/discontinuity parameter.
- `theta_crit`: angular radius of causal disc.
- `theta`: angular distance from collision center.
- `H`: causal-boundary window.

Implementation:

- `bubble_collision_signal(theta, z0, zcrit, theta_crit, edge_sigma_deg=0.0)`.
- `edge_sigma_deg=0` gives hard Heaviside causal boundary.
- `edge_sigma_deg>0` uses Gaussian-CDF smoothing as a robustness heuristic, not a settled physical theorem.

Injection is multiplicative, not additive:

```math
T_injected = (1 + f(nhat)) * (T_CMB0 + deltaT_CMB) - T_CMB0
```

Code:

- `scripts/phase2_signal_model.py::inject_signal_into_patch`
- `T_CMB_K = 2.7255`

Scientific safeguards:

- Feeney Eq. 1 form is preserved.
- Azimuthally symmetric disc geometry is preserved.
- Causal disc boundary is preserved.
- Multiplicative injection is preserved.
- `sin(theta_crit)` sampling is documented as a training-design choice, not an inference prior.

## 3. Phase 1: Observable Domain

Planck-era patch observable:

- Working map products are Planck cleaned maps, primarily SMICA for current real-map tests.
- Common mask is used to build clean coordinate pools.
- Working resolution: `Nside=256`.
- Patch geometry: gnomonic projection.
- Patch size: `256 x 256`.
- Pixel scale: `13 arcmin/pixel`.
- Patch angular width: about `55.47 deg`.

Key scripts:

- `scripts/phase1_explore.py`
- `scripts/phase2_generate_training.py::project_patch`
- `scripts/phase_dataset_utils.py`

Coordinate handling:

- Galactic longitude/latitude.
- Gnomonic patch centered on selected sky coordinate.
- Patch-center pixel: `(npix - 1) / 2`.
- Angular-distance grid maps tangent-plane pixels back to 3D directions before separation calculation.

## 4. Phase 2: Current Training Data Product

Primary current training dataset:

```text
data/training_v4/training_data.h5
```

Verified run config summary:

- `num_samples = 10000`
- `num_positive = 5000`
- `num_negative = 5000`
- train split: `9000` samples = `4500 pos / 4500 neg`
- validation split: `1000` samples = `500 pos / 500 neg`
- coordinate pool: `5000`
- train coordinates: `4500`
- validation coordinates: `500`
- CAMB realizations: `192`
- train realizations: `173`
- validation realizations: `19`
- `nside = 256`
- `patch_pixels = 256`
- `reso_arcmin = 13.0`
- `beam_fwhm_arcmin = 15.0`
- `noise_sigma_uk_arcmin = 30.0`
- `noise_corr_fwhm_arcmin = 0.0`
- `edge_sigma_min_deg = 0.3`
- `edge_sigma_max_deg = 1.0`
- `geometry_mode = contained` for the existing `training_v4`
- `contained_margin_deg = 0.5`
- `theta_training_distribution = sin(theta_crit)`
- `split_method = coordinate_and_realization_disjoint`

Audit status for existing `training_v4`:

- Pass.
- Shared train/val coordinate count: `0`.
- Shared train/val CAMB realization count: `0`.
- Shared background ID count: `0`.
- Shared event ID count: `0`.
- Positive targets touching edge: `0` because this dataset is contained-only.

Important limitation:

- Existing `training_v4` contains fully contained positive discs only.
- Real full-sky tiling can see partial discs spanning patch boundaries.
- This was identified as scientifically important and has now been addressed in local generator code, but the full mixed/truncated dataset has not yet been generated/evaluated.

## 5. Newly Implemented Local Data Geometry Correction

Status: implemented locally; smoke-tested; not yet full-scale evaluated.

Modified files:

- `scripts/phase2_generate_training.py`
- `scripts/phase2_generate_stratified_validation.py`
- `scripts/phase3_real_sky_injection.py`
- `scripts/phase2_audit_dataset.py`
- `scripts/phase3_eval_stratified_external.py`

New geometry modes:

- `contained`: old/default behavior; full causal disc visible inside patch.
- `truncated`: positive mask must touch patch edge; only visible patch intersection is labeled.
- `mixed`: per-positive draw can be contained or truncated.

New main arguments:

```bash
--geometry-mode {contained,truncated,mixed}
--truncated-positive-fraction FLOAT
--truncated-visible-fraction-min FLOAT
--truncated-visible-fraction-max FLOAT
--truncated-max-center-draws INT
--signal-center-edge-margin-pix FLOAT
```

New truth fields:

- `geometry_mode_code`: `0=contained`, `1=truncated`.
- `fully_contained`: boolean-like uint8.
- `target_touches_edge`: boolean-like uint8.
- `visible_target_fraction`: visible patch pixels / estimated full-disc pixels.
- `visible_target_pixels`: target-mask pixels visible in patch.
- `full_disc_pixels_est`: full disc pixel count estimated from centered disc of same radius.
- `target_edge_contact_pixels`: number of target pixels on patch boundary.
- `disc_edge_margin_pix`: nearest patch-edge distance minus disc radius; negative means disc crosses patch edge.
- `signal_center_in_patch`: whether full physical disc center lies inside current patch.

Mathematical interpretation:

- The physical bubble signal remains the full Feeney disc on the sky.
- The patch label is the intersection `disc_support ∩ patch_window`.
- This is not a shortcut or fake signal. It is the correct patch-level observable for tiled inference.
- Downstream candidate stitching must use full physical metadata, not just the visible patch mask.

Smoke tests already run:

- Mixed training smoke: `10` positives, `5/10` touched edge, audit pass.
- Truncated training smoke: `6/6` positives touched edge, audit pass.
- Contained regression smoke: `0/6` positives touched edge, visible fraction `1.0`, audit pass.
- Mixed stratified validation smoke: `72` positives, `32/72` touched edge, `20` centers outside patch.
- Mixed real-SMICA injection smoke: `4/4` positives touched edge, `2` centers outside patch.
- Temporary smoke artifacts were removed from `work/`.

Next required gate:

```bash
python scripts/phase2_generate_stratified_validation.py \
  --output-dir data/validation_stratified_mixed_geometry_v1 \
  --num-samples 5000 \
  --samples-per-cell 50 \
  --geometry-mode mixed \
  --truncated-positive-fraction 0.30 \
  --truncated-visible-fraction-min 0.15 \
  --truncated-visible-fraction-max 0.95
```

Then evaluate current `v6_aux_only + matched_template` on this mixed validation set before training anything new.

## 6. Phase 3 Model Architecture

Training harness:

- `scripts/phase3_train_unet.py`

Primary architecture family:

- `segmentation_models_pytorch.Unet`
- Encoder: `efficientnet-b0`
- Default encoder weights: ImageNet unless otherwise specified.
- Input: normalized 1-channel CMB patch unless extra-channel experiment is enabled.
- Output: segmentation mask logits.
- Optional image-level auxiliary head for presence/candidateness.
- Recently added optional radius-bin auxiliary head, but current trained branch failed externally.

Loss structure:

- Pixel BCE with positive pixel reweighting.
- Soft Dice loss.
- Boundary-weighted loss option.
- Optional image-level BCE auxiliary loss.
- Optional radius-bin cross-entropy auxiliary loss for positives only.

Important training constants for `v6_aux_only`:

- Dataset: `data/training_v4/training_data.h5`
- Epochs: `8`
- Batch size: `64`
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Encoder: `efficientnet-b0`
- BCE weight: `1.0`
- Dice weight: `1.0`
- Aux head weight: `0.1`
- Aux dropout: `0.2`
- Boundary weight: `4.0`
- Boundary width: `5 pixels`
- Threshold during training previews: `0.99`
- Normalization source: `runs/phase3_unet/phase3_v4_full_2gpu_b64w8_cached/run_config.json`
- Train mean: `-8.208702282003653e-07`
- Train std: `0.00010452365572382741`
- Positive pixel fraction: `0.17111089409722222`
- BCE positive weight: `4.844163256091792`

Current hardware note:

- System has 2 x RTX 3090.
- Use `--gpu-ids 0,1` for hard training/evaluation tasks where supported.
- A full controlled fine-tune with batch size `64` and cache data has run successfully on both GPUs.

## 7. Verified Phase 3 Metrics: Stratified Synthetic Validation

Main report:

```text
runs/phase3_unet/stratified_external_combined_v1/combined_stratified_eval_report.json
```

Matched FPR: `0.08`.

Table:

| method | AUROC | AUPRC | recall | weak recall | Dice+ |
|---|---:|---:|---:|---:|---:|
| matched_template | 0.712 | 0.881 | 0.401 | 0.282 | not primary here |
| centered_disc | 0.639 | 0.833 | 0.252 | 0.176 | not primary here |
| original_v4 | 0.775 | 0.914 | 0.541 | 0.425 | 0.344 |
| boundary_v4 | 0.774 | 0.914 | 0.540 | 0.420 | 0.377 |
| v5_consensus | 0.774 | 0.913 | 0.536 | 0.422 | 0.396 |
| v6_aux_only | 0.773 | 0.913 | 0.539 | 0.422 | 0.401 |
| v6_hard_w15 | 0.773 | 0.913 | 0.529 | 0.416 | not preferred |

Interpretation:

- Learned branches are very close on pooled synthetic metrics.
- `v6_aux_only` has best positive Dice among ML branches in this table.
- Differences among learned branches are not large enough to justify running all five by default.
- Matched template remains scientifically important as a Feeney-style classical comparator and score.

## 8. Failed Radius-Head Branch

Status: failed. Do not proceed with this branch as trained.

Run:

```text
runs/phase3_unet/phase3_scale_radius_head_w02
```

Purpose:

- Add radius-bin auxiliary head with bins `5,10,15,20,25` deg.
- Test whether explicit scale supervision improves weak/small-radius recall without changing signal physics.

Training command concept:

```bash
python scripts/phase3_train_unet.py \
  --data-h5 data/training_v4/training_data.h5 \
  --run-name phase3_scale_radius_head_w02 \
  --epochs 8 \
  --batch-size 64 \
  --learning-rate 5e-5 \
  --scheduler cosine \
  --normalization-config runs/phase3_unet/phase3_v6_aux_only_w4/run_config.json \
  --resume-checkpoint runs/phase3_unet/phase3_v6_aux_only_w4/best_checkpoint.pt \
  --model-only-resume \
  --aux-head-weight 0.1 \
  --boundary-weight 4.0 \
  --boundary-width-pixels 5 \
  --radius-head-weight 0.2 \
  --radius-bin-edges-deg 5,10,15,20,25 \
  --gpu-ids 0,1 \
  --cache-data
```

External gate:

```text
runs/phase3_unet/stratified_external_radius_head_w02/stratified_eval_report.json
```

At matched FPR about `0.08`:

| model | AUROC | AUPRC | recall | weak recall | Dice+ |
|---|---:|---:|---:|---:|---:|
| radius_head_w02 | 0.670 | 0.852 | 0.302 | 0.224 | 0.006 |
| v6_aux_only | 0.773 | 0.913 | 0.539 | 0.422 | 0.401 |

Conclusion:

- Radius-head implementation plumbing works.
- Current trained branch damaged detection and segmentation.
- Do not use or continue this exact branch.
- Scale-aware modeling may still be valid, but only after geometry-correct data evaluation.

## 9. Beam Consistency And Sensitivity

Beam audit:

```text
runs/phase3_unet/beam_consistency_audit/audit.md
```

Verdict:

- Current sensitivity comparison is beam-consistent.
- Injections are hard Eq. 1 at analytic stage, then observed patch is beam-smoothed.
- Matched-template kernel is also hard Eq. 1 then smoothed with same recorded beam.
- Current sensitivity grid uses `beam_fwhm_arcmin = 15.0`.
- This validates internal comparison, not native SMICA `~5 arcmin` readiness.

ML gain heatmap:

```text
runs/phase3_unet/ml_gain_heatmap_v1/ml_gain_heatmap.json
```

At matched synthetic FPR `0.05`:

- Significant ML wins: `8 / 35` amplitude-radius cells.
- Significant ML losses: `0 / 35`.
- Gain is localized, not universal.

Important significant cells:

| A | theta_deg | best_model | matched P_det | best ML P_det | delta |
|---:|---:|---|---:|---:|---:|
| 1e-5 | 20 | boundary_v4 | 0.050 | 0.120 | +0.070 |
| 1e-5 | 25 | v6_hard_w15 | 0.065 | 0.175 | +0.110 |
| 2e-5 | 20 | v6_aux_only | 0.235 | 0.555 | +0.320 |
| 2e-5 | 25 | v6_hard_w15 | 0.375 | 0.710 | +0.335 |
| 5e-5 | 5 | v6_hard_w15 | 0.065 | 0.295 | +0.230 |
| 5e-5 | 10 | v5_consensus | 0.490 | 0.980 | +0.490 |
| 5e-5 | 15 | v6_hard_w15 | 0.960 | 1.000 | +0.040 |
| 1e-4 | 5 | v5_consensus | 0.500 | 0.980 | +0.480 |

## 10. Real-SMICA Recalibration

Main report:

```text
runs/phase3_unet/real_sky_recalibration_v1/real_sky_recalibration_report.json
```

Core result:

- Initial real-SMICA gate failure was threshold miscalibration, not a full domain gap.
- CAMB-calibrated thresholds were too strict on real SMICA negatives.

At FPR target `0.05`:

| method | real recalibrated recall | real FPR | precision | CAMB-threshold real recall |
|---|---:|---:|---:|---:|
| v6_aux_only | 0.353 | 0.040 | 0.997 | 0.262 |
| matched_template | 0.323 | 0.052 | 0.995 | 0.238 |
| either_v6_or_matched | 0.387 | 0.076 | 0.994 | 0.271 |
| both_v6_and_matched | 0.289 | 0.016 | 0.998 | 0.230 |

At FPR target `0.10`:

| method/policy | real recall | real/null FPR | expected FP per 3000 independent patches |
|---|---:|---:|---:|
| v6_aux_only | 0.395 | 0.0998 | 299.4 |
| matched_template | 0.385 | 0.0998 | 299.4 |
| either_v6_or_matched | 0.454 | 0.1638 | 491.4 |
| both_v6_and_matched | 0.326 | 0.0358 | 107.4 |

Thresholds:

| FPR target | method | CAMB threshold | real-SMICA threshold |
|---:|---|---:|---:|
| 0.05 | v6_aux_only | 0.992082 | 0.8884694 |
| 0.05 | matched_template | 76.977921 | 61.829514 |
| 0.08 | v6_aux_only | 0.992082 | 0.87323481 |
| 0.08 | matched_template | 76.977921 | 58.569695 |
| 0.10 | v6_aux_only | 0.992082 | 0.8648743 |
| 0.10 | matched_template | 76.977921 | 56.798725 |

Conclusion:

- Any real-map screening must use map-calibrated null thresholds.
- FFP10 retraining is optional, not mandatory, unless future mixed-geometry/real-map tests expose a new gap.

## 11. Smoothed Real-SMICA Sensitivity

Report:

```text
runs/phase3_unet/real_sky_smoothed_sensitivity_v1/smoothed_sensitivity_report.json
```

Setup:

- Model: `v6_aux_only`
- Edge smoothing: uniform `0.30` to `1.00` deg.
- Real SMICA backgrounds.
- Threshold source: real-SMICA recalibration report.

At FPR target `0.05`:

- Global recall: `0.351`
- Dead `A <= 2e-6`: `0.042`
- Contested `5e-6 to 2e-5`: `0.174`
- Solved `A >= 5e-5`: `0.925`
- Contested+solved `A >= 5e-6`: `0.474`

At FPR target `0.10`:

- Global recall: `0.392`
- Dead `A <= 2e-6`: `0.089`
- Contested `5e-6 to 2e-5`: `0.232`
- Solved `A >= 5e-5`: `0.935`
- Contested+solved `A >= 5e-6`: `0.514`

Interpretation:

- Smoothing mismatch was not the main bottleneck.
- Solved-zone behavior is strong.
- Contested-zone recall remains limited under strict FPR.

## 12. Threshold-Volume Sweep

Report:

```text
runs/phase3_unet/threshold_volume_sweep_v1/threshold_volume_sweep.json
```

`v6_aux_only` on cached smoothed real-SMICA positives + 5000-patch SMICA nulls:

| threshold | dead recall | contested recall | solved recall | global recall | null FPR | expected FP / 3000 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.538 | 0.654 | 0.976 | 0.713 | 0.544 | 1631 |
| 0.80 | 0.272 | 0.442 | 0.954 | 0.540 | 0.298 | 895 |
| 0.85 | 0.124 | 0.278 | 0.940 | 0.423 | 0.134 | 401 |
| 0.90 | 0.028 | 0.150 | 0.922 | 0.336 | 0.035 | 106 |

Operating points:

- Contested recall `>=0.50` needs threshold about `0.784`, expected FP about `1099`.
- Expected FP `<=800` gives contested recall about `0.412`.
- Expected FP `<=500` gives contested recall about `0.317`.
- Expected FP `<=300` gives contested recall about `0.232`.

Conclusion:

- The model is not blind to contested positives.
- Recall versus false-positive volume is now an operational budget question.
- Candidate clustering and downstream fitting are needed to make high-recall mode practical.

## 13. Two-Pass ML Then Matched-Template Policy

Report:

```text
runs/phase3_unet/two_pass_policy_v1/two_pass_policy.json
```

Pass 1:

- `v6_aux_only >= 0.75`
- Positive kept: `12473 / 17500`
- Global recall: `0.713`
- Null kept: `2718 / 5000`
- Expected FP / 3000: `1631`
- Dead recall: `0.538`
- Contested recall: `0.654`
- Solved recall: `0.976`

Pass 2:

- Matched-template verifier calibrated only on ML-kept null subset.

| FP budget | matched threshold | expected FP | global recall | contested recall | solved recall |
|---:|---:|---:|---:|---:|---:|
| 200 | 58.933 | 200 | 0.336 | 0.167 | 0.858 |
| 400 | 53.124 | 400 | 0.408 | 0.263 | 0.904 |
| 600 | 49.418 | 600 | 0.462 | 0.338 | 0.929 |
| 800 | 46.536 | 800 | 0.508 | 0.400 | 0.943 |

Conclusion:

- Matched-template verification reduces false positives but also removes many contested positives.
- The ML false positives often look disc-like to the matched template; they are not trivially rejected.
- Bayesian/template-fit follow-up or clustering is needed for practical high-recall operation.

## 14. Current Candidate Output Direction

Desired candidate object fields:

- patch provenance: source map, map product, sky center, patch index, mask fraction.
- model scores: `v6_aux_only_score`, `matched_template_score`, thresholds, pass/fail flags.
- candidate center estimate in patch pixels and sky coordinates.
- candidate angular radius estimate or matched radius.
- thresholded mask and mask path.
- visible fraction / truncation metadata once mixed geometry is used.
- fit/handoff artifact paths.

Existing candidate-output scripts:

- `scripts/phase3_screen_and_verify.py`
- candidate records should remain machine-readable JSONL/CSV/NPZ style.

## 15. Immediate Next Steps

Do these in order. Do not train before step 2 completes.

1. Generate full mixed-geometry stratified validation:

```bash
python scripts/phase2_generate_stratified_validation.py \
  --output-dir data/validation_stratified_mixed_geometry_v1 \
  --num-samples 5000 \
  --samples-per-cell 50 \
  --geometry-mode mixed \
  --truncated-positive-fraction 0.30 \
  --truncated-visible-fraction-min 0.15 \
  --truncated-visible-fraction-max 0.95
```

2. Evaluate current models on mixed geometry:

```bash
python scripts/phase3_eval_stratified_external.py \
  --data-h5 data/validation_stratified_mixed_geometry_v1/validation_data.h5 \
  --output-dir runs/phase3_unet/stratified_external_mixed_geometry_v1 \
  --model v6_aux_only:runs/phase3_unet/phase3_v6_aux_only_w4:best:mask_max \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda
```

Also score `matched_template` through the classical harness or an equivalent matched-template evaluator.

3. Read geometry group metrics:

- `geometry_contained`
- `geometry_truncated`
- `center_inside_patch`
- `center_outside_patch`
- `visible_fraction_low`
- `visible_fraction_mid`
- `visible_fraction_high`
- `weak_family_union`

4. Decision gate:

- If `v6_aux_only + matched_template` holds contained performance and does not collapse on truncated positives, do not retrain immediately; proceed to real-SMICA mixed injection calibration.
- If truncated recall collapses while contained remains stable, generate mixed training data and fine-tune `v6_aux_only` with a modest truncated fraction.
- If false positives rise under mixed geometry, do not lower thresholds until real-map nulls and clustering are rechecked.

5. If retraining is justified:

Suggested first mixed training product:

```bash
python scripts/phase2_generate_training.py \
  --num-samples 10000 \
  --pool-size 5000 \
  --num-cmb-realizations 192 \
  --output-dir data/training_v5_mixed_geometry \
  --geometry-mode mixed \
  --truncated-positive-fraction 0.30 \
  --truncated-visible-fraction-min 0.15 \
  --truncated-visible-fraction-max 0.95
```

Suggested fine-tune:

```bash
python scripts/phase3_train_unet.py \
  --data-h5 data/training_v5_mixed_geometry/training_data.h5 \
  --output-root runs/phase3_unet \
  --run-name phase3_v8_mixed_geometry_v6_ft \
  --epochs 6 \
  --batch-size 64 \
  --learning-rate 5e-5 \
  --scheduler cosine \
  --normalization-config runs/phase3_unet/phase3_v6_aux_only_w4/run_config.json \
  --resume-checkpoint runs/phase3_unet/phase3_v6_aux_only_w4/best_checkpoint.pt \
  --model-only-resume \
  --aux-head-weight 0.1 \
  --aux-head-dropout 0.2 \
  --boundary-weight 4.0 \
  --boundary-width-pixels 5 \
  --checkpoint-metric image_f1 \
  --gpu-ids 0,1 \
  --num-workers 8 \
  --cache-data
```

Required post-training gates:

- Fixed contained validation should not regress materially.
- Mixed/truncated validation should improve.
- Real-SMICA null burden should stay controlled under map-calibrated thresholds.
- Candidate-volume sweep must be rerun.
- Matched-template comparison must be rerun.

## 16. Low-Risk High-Gain Ideas Still Valid

Prioritized:

1. Mixed/truncated positive geometry correction.
   - Already implemented locally.
   - High scientific validity.
   - Must evaluate before retraining.

2. Real-map threshold calibration.
   - Already shown essential.
   - Keep mandatory for SMICA/NILC/SEVEM/Commander.

3. Candidate clustering/deduplication.
   - High practical gain for false-positive burden.
   - Does not alter signal physics.
   - Needed before full-sky candidate-count claims.

4. Mixed geometry plus real-SMICA injection validation.
   - Needed after synthetic mixed validation.
   - Tests whether patch-edge positives transfer on real backgrounds.

5. Real-null training or FFP10.
   - Plausible but higher leakage/systematics risk.
   - Only use after mixed geometry and calibration gates.

6. Scale-aware model changes.
   - Radius head failed in current form.
   - Revisit only if mixed-geometry evaluation shows scale-specific failure not fixed by data.

## 17. Things Tested And Outcome

Completed and useful:

- Feeney Eq. 1 implementation checks.
- Multiplicative injection checks.
- Provenance-clean split audit.
- Beam/noise forward-model inclusion.
- Classical baselines: matched template and centered disc.
- Matched-FPR sensitivity curves.
- Bootstrap CIs for ML gain heatmap.
- Real-SMICA threshold recalibration.
- Threshold-volume sweep.
- Two-pass ML+matched-template policy.
- Smoothed real-SMICA sensitivity.
- Nside=512 probe.
- Matched-filter input-channel experiment.
- Radius-head auxiliary branch.
- Mixed/truncated geometry implementation smoke tests.

Failed or not worth continuing as-is:

- Radius-head branch `phase3_scale_radius_head_w02`: external metrics collapsed.
- Matched-filter input channel: moved contested recall by only about `0.013`; not main lever.
- Nside=512 probe: not enough improvement without unacceptable null saturation.
- Strict two-pass matched-template verifier: reduces FPs but cuts too much contested recall.

Still needs evaluation:

- Full mixed-geometry stratified validation.
- Current `v6_aux_only + matched_template` on contained vs truncated patches.
- Real-SMICA mixed-geometry injection and recalibration.
- Cross-map threshold calibration for NILC/SEVEM/Commander after final model choice.
- Full-sky candidate clustering/deduplication.

## 18. File/Directory Notes

Root files:

- `README.md`: public project overview.
- `README_main_backup_2026-04-16.md`: backup of previous README before overhaul.
- `PROJECT_HANDOFF.md`: this handoff.
- `environment.yml`: conda environment.

Data:

- `data/training_v4/training_data.h5`: current primary contained training dataset.
- `data/validation_stratified_v1/validation_data.h5`: current contained stratified validation.
- `data/training_v4/smica_null_controls_all.h5`: real SMICA null controls used for recalibration.

Runs:

- `runs/phase3_unet/phase3_v6_aux_only_w4`: current preferred ML branch.
- `runs/phase3_unet/stratified_external_combined_v1`: main contained stratified model comparison.
- `runs/phase3_unet/real_sky_recalibration_v1`: key SMICA threshold recalibration result.
- `runs/phase3_unet/threshold_volume_sweep_v1`: recall vs FP-volume tradeoff.
- `runs/phase3_unet/two_pass_policy_v1`: ML proposal + matched-template verifier test.
- `runs/phase3_unet/phase3_scale_radius_head_w02`: failed radius-head branch; keep as negative result.

Work folder:

- New throwaway scripts/artifacts should go under `work/` unless intentionally promoted.
- Smoke artifacts created during geometry implementation were removed.

## 19. Git State Warning

At the time this handoff was created, local modifications exist in:

- `scripts/phase2_audit_dataset.py`
- `scripts/phase2_generate_stratified_validation.py`
- `scripts/phase2_generate_training.py`
- `scripts/phase3_eval_stratified_external.py`
- `scripts/phase3_real_sky_injection.py`
- `scripts/phase3_train_unet.py`

Meaning:

- Geometry correction is local and not yet necessarily pushed.
- Radius-head plumbing is also local and remains in code, but the trained radius-head branch failed.
- Before publishing, review whether to keep radius-head code as disabled experimental infrastructure or remove it.

## 20. Minimal Mental Model For Next AI

The project is not blocked by lack of model complexity. It is blocked by observable correctness and operating-point discipline.

Current best reasoning:

1. Existing training used contained positives only.
2. Real full-sky patch tiling can observe partial/clipped bubble discs.
3. The generator now supports this physically correct geometry.
4. Evaluate current `v6_aux_only + matched_template` on mixed geometry before retraining.
5. If truncated geometry is a real failure mode, fine-tune on mixed geometry.
6. After any model/data change, recalibrate thresholds on real-map nulls.
7. Judge success by weak/truncated recall at controlled real-map false-positive burden, not pooled F1 alone.

Non-negotiable scientific rule:

Do not simplify the signal into easier blobs. Preserve Feeney Eq. 1, multiplicative injection, causal disc support, azimuthal symmetry, long-wavelength modulation, real-map null calibration, and explicit metadata sufficient for classical/Bayesian follow-up.

## 21. 2026-04-17 Real-SMICA Validation Gate: v7_mixed_ft vs v6_aux_only

Full report: `work/v7_real_sky_gate.md`. Gate harness: `scripts/phase3_real_sky_v7_gate.py`.

Artifacts:

- `runs/phase3_unet/real_sky_v7_gate_v1/contained/v7_vs_v6_contained_report.{md,json}`
- `runs/phase3_unet/real_sky_v7_gate_v1/mixed/v7_vs_v6_mixed_report.{md,json}`

Key numbers at real-SMICA-calibrated FPR 0.08, n_positive = 17500 per geometry:

### Contained geometry

| model | recall @ FPR 0.05 | recall @ FPR 0.08 | recall @ FPR 0.10 |
|---|---:|---:|---:|
| v7_mixed_ft | 0.286 | 0.357 | 0.386 |
| v6_aux_only | 0.348 | 0.372 | 0.389 |

v7 loses to v6 on real-SMICA contained geometry at FPR 0.05 by 6.2 points. The
contained-geometry advantage v7 showed on synthetic CAMB backgrounds does not
survive the foreground-residual domain shift.

### Mixed geometry (30% truncated)

| group | v7 recall | v6 recall | delta |
|---|---:|---:|---:|
| geometry_contained | 0.360 | 0.380 | -0.020 |
| geometry_truncated | 0.246 | 0.205 | +0.041 |
| center_outside_patch | 0.207 | 0.163 | +0.044 |
| visible_fraction_low | 0.196 | 0.145 | +0.051 |

v7 wins on every truncated / edge-crossing subgroup, as expected from its
mixed-geometry fine-tune. v6 remains better on fully contained positives on
real backgrounds.

### Decision (as it stood at the end of this gate; see Section 22 for the final answer)

At the time of the real-SMICA gate, the natural interpretation was a
two-model portfolio at Phase 5:

1. v6_aux_only for contained positives.
2. v7_mixed_ft for truncated / edge-crossing positives.
3. Union of triggers at lowered per-model thresholds.

This reading was correct given only the gate numbers. Section 22 shows that
the simple portfolio policies (OR, AND, rank_max, heuristic routing) all
underperform `v6_only` at deployment FPR because v6 and v7 false positives
are highly correlated. The portfolio is recovered by the learned GBT
router in Section 22, not by raw model ensembling.

### Next step: Batch 2 — completed

See `work/batch2_postprocess_ablation.md` for the full ablation.

- Harness: `scripts/phase3_postprocess_ablation.py`.
- Artifacts: `runs/phase3_unet/batch2_postprocess_ablation_v1/`.

Two transforms evaluated on frozen v6 / v7 probability masks against the
same real-SMICA gate data:

- `smooth_multi` (Gaussian at sigma in {4, 8, 16} pix, max over sigmas):
  null result. The U-Net mask is already smooth at that scale.
- `mf_on_mask` (matched-filter rescoring with positive-disc kernels on the
  smoothed mask): positive for `v7_mixed_ft` specifically, at tight FPR,
  specifically on contained geometry (+0.039 recall @ FPR 0.05). Hurts
  `v6_aux_only` at every setting. Also hurts `v7_mixed_ft` on truncated /
  edge-crossing positives.

Portfolio decision holds. Best `v7 + mf_on_mask` (0.325 at FPR 0.05 on
contained) is still below `v6 baseline` (0.348). Post-processing alone does
not close the gap.

## 22. 2026-04-17 Batch 3 Portfolio Router

Full report: `work/batch3_geometry_router.md`. Harness:
`scripts/phase3_geometry_router.py`. Artifacts:
`runs/phase3_unet/batch3_geometry_router_v1/`.

Eight portfolio policies evaluated against the single-model baselines from
Section 21, using the cached Batch 2 transform scores. The six simple /
heuristic policies all underperform the best single model at FPR 0.08. A
learned GBT classifier on all six transform features beats `v6_only`
cleanly in every cell.

### Learned GBT router result (honest evaluation)

Cross-geometry training (fit on contained to score mixed, and vice versa)
with a disjoint null train/eval split (2500/2500 by fixed seed):

| geometry | FPR | v6_only | learned_gbt | delta |
|---|---:|---:|---:|---:|
| contained | 0.05 | 0.348 | 0.359 | +0.011 |
| contained | 0.08 | 0.372 | **0.403** | **+0.031** |
| contained | 0.10 | 0.389 | **0.431** | **+0.042** |
| mixed | 0.05 | 0.305 | 0.322 | +0.017 |
| mixed | 0.08 | 0.331 | **0.365** | **+0.034** |
| mixed | 0.10 | 0.347 | **0.392** | **+0.045** |

Per-geometry-group recall on mixed at FPR 0.08 — the GBT wins biggest on
exactly the groups where `v6_only` is weakest:

| group | v6_only | learned_gbt | delta |
|---|---:|---:|---:|
| geometry_contained | 0.380 | **0.409** | +0.029 |
| geometry_truncated | 0.205 | **0.253** | +0.048 |
| center_outside_patch | 0.163 | **0.207** | +0.044 |
| visible_fraction_low | 0.145 | **0.191** | +0.046 |

GBT feature importances: `v6_baseline` 0.44, `v6_smooth_multi` 0.16,
`v7_mf_on_mask` 0.12, `v7_baseline` 0.11, `v7_smooth_multi` 0.09,
`v6_mf_on_mask` 0.07. Primary signal is v6, with v7 and matched-filter
acting as complementary sub-scores. Non-linear interactions matter:
`learned_logistic` only gains ~+0.004, `learned_gbt` gains +0.034.

### Honest-evaluation note

An initial pass fit on `{positives, full 5000-null pool}` and calibrated
the threshold on the same full null pool, producing inflated numbers (GBT
FPR 0.08 mixed recall 0.396 in that pass). The numbers above use the
corrected disjoint null split. The Batch 2 transform caches are shared
across geometries (same 5000 null patches), so cross-geometry positive
training alone does not prevent null leakage. The 2500/2500 null split by
fixed seed is the right control.

### Simple policies (all negative at FPR 0.08)

Mixed geometry:

| policy | recall |
|---|---:|
| v6_only | **0.331** |
| v7_only | 0.328 |
| either_OR | 0.325 |
| rank_max | 0.325 |
| both_AND | 0.272 |
| geometry_routed (best quantile) | 0.305 |

Contained geometry:

| policy | recall |
|---|---:|
| v6_only | **0.372** |
| v7_only | 0.357 |
| either_OR | 0.367 |
| rank_max | 0.367 |

Mixed geometry at deployment FPR 0.08:

| policy | recall |
|---|---:|
| v6_only | **0.331** |
| v7_only | 0.328 |
| either_OR | 0.325 |
| rank_max | 0.325 |
| both_AND | 0.272 |
| geometry_routed (best quantile) | 0.305 |

Contained geometry at FPR 0.08:

| policy | recall |
|---|---:|
| v6_only | **0.372** |
| v7_only | 0.357 |
| either_OR | 0.367 |
| rank_max | 0.367 |

Why the two-model ensemble does not help:

- v6 and v7 share an EfficientNet encoder and overlapping training
  distribution, so their false positives are highly correlated. `either_OR`
  buys little independent signal.
- To keep joint null FPR at 0.08, each model's threshold rises above its
  single-model setting. The recall lost to tightening roughly cancels
  recall gained from v7's truncated positives.

Why the simple geometry router does not help:

- The `v7_mf_on_mask - v7_baseline` signal correlates with both signal
  strength and geometry, not cleanly with geometry. Strong truncated
  positives get routed to v6 (which does not detect them). Weak contained
  positives get routed to v7 (which scores them lower than v6 would).
- Full route-quantile sweep in {0.25, 0.50, 0.70, 0.85, 0.95} did not find
  a setting that beat best single model at any FPR target.

Revised deployment advice from this result:

- **Primary operating policy: `learned_gbt` router.** 200-tree gradient-
  boosted classifier on six features derived from the frozen v6 and v7
  probability masks (baseline, smooth_multi, mf_on_mask per model).
  Deployable at inference time with negligible compute.
- **Single-model fallback: `v6_aux_only`** at real-SMICA threshold 0.873
  (FPR 0.08). Clean reference when a reviewer does not want a composite
  score.
- `v7_mixed_ft` is retained as an input to the learned router and as a
  Phase 5 specialist on truncated candidates. Not run as a separate
  parallel screener.
- `matched_template` role unchanged.

This supersedes the "v6 alone, portfolio walked back" note from my earlier
Section 21 draft: the portfolio is recovered via the learned classifier,
not via raw model ensembling.

## 23. 2026-04-17 Batch 2 Ablation v2 Small-Sigma Sweep

Full report: `work/batch2_postprocess_ablation.md` (extended). Artifacts:
`runs/phase3_unet/batch2_postprocess_ablation_v2_smallsigma/`.

Gaussian smoothing sigma sweep at (0.5, 1.0, 2.0) pixels, following up on
the null result in the (4, 8, 16) sweep. Also null at every cell
(delta <= 0.001 in recall).

`mf_on_mask` with base sigma 0.5 gives identical numbers to sigma 4
(v7 contained @ FPR 0.05: 0.326 vs 0.325). Matched-filter-on-mask captures
mask shape coherence, not mask smoothness.

This closes the Gaussian-smoothing branch of post-processing.

## 24. What Does NOT Help (Running Negative-Result Registry)

Closed off by direct measurement so the next AI does not loop on them:

- Naive multi-model probability-mask averaging. Measured negative in
  `work/tta_ensemble_eval.md`.
- D4 test-time augmentation beyond +0.7pp Dice. Same doc. Theoretical
  sqrt(8) argument does not apply to a CNN trained with rotation / flip
  augmentation.
- Gaussian smoothing on the probability mask at any practical sigma.
  Measured null in Batch 2 v1 (sigma 4, 8, 16) and Batch 2 v2 (sigma
  0.5, 1, 2).
- Matched-filter-on-mask as a general screener replacement. Hurts
  `v6_aux_only` at every setting; helps `v7_mixed_ft` only at FPR 0.05 on
  contained. Useful as a measurement of "disc-like coherence", not a
  recall lever.
- Two-model OR / AND / rank_max ensembling at joint FPR 0.08. Measured
  negative in `work/batch3_geometry_router.md`.
- Heuristic geometry routing on the `v7_mf - v7_baseline` contrast.
  Swept quantile, none win. Same doc.
- Linear (logistic-regression) router on the six frozen-mask features.
  +0.004 recall at FPR 0.08, which is within noise. The optimal decision
  boundary is non-linear; linear combination of the six features does not
  find it. Same doc.
- Nside=512 retrain. Measured negative in `docs/nside512_probe_decision.md`.
- Radius-head auxiliary branch at weight 0.2 cold-started. Measured
  negative in `work/radius_head_post_mortem.md`. Retry requires warmup
  + checkpoint metric hygiene + geometry-correct training data.
- **Batch 4 14-feature router as a deployment improvement.** Clean-null
  FPR-0.08 delta over `gbt_6` was +0.043 (shipped claim in PR #9). Under
  deployment-representative Nside=8 tile calibration the cross-map mean
  delta is **−0.015** (SMICA −0.001, NILC +0.023, SEVEM −0.045,
  Commander −0.036). The geometry features specifically overfit the
  clean-pool mask-fraction distribution. See `work/batch5_fullsky_calibration_gap.md`
  and Section 27. Code infrastructure stays; deployment claim retracted.

## 25. What Remains on the Table (Open Recall Levers)

Ordered by expected gain per unit compute:

- **v8 retrain with matched-filter response channel, on
  training_v5_mixed_geometry.** The only remaining training-signal lever.
  Expected to lift truncated recall a further 4-10pp on top of the
  Batch 4 learned-router gain. Wall clock 6-10 hours on 2x3090. Requires
  training hygiene from `work/radius_head_post_mortem.md`. Must run a
  1-epoch 500-sample smoke test first.
- **Isotonic score calibration on real-SMICA nulls.** Not a recall boost;
  required for clean candidate-volume statistics in the paper.
- **Expand the learned-router feature set further.** Batch 4 added 4
  geometry proxies per model. Additional candidates for a Batch 5: a
  matched-filter response channel evaluated as a router feature (rather
  than a model input channel), second-moment / eccentricity of the mask,
  and per-model image-level aux-head score. Each is cheap to implement
  and the router architecture already scales to arbitrary feature counts.

## Closed recall levers (completed)

- **Expand the learned-router feature set with 4 truth-free geometry
  proxies per model.** Executed in Batch 4, measured positive: +0.043
  mixed recall at FPR 0.08 over the PR #8 GBT, robust across 4 null-split
  seeds (min +0.021). See `work/batch4_router_features.md` and Section 26.

## 26. 2026-04-17 Batch 4 Router Feature Expansion

Full report: `work/batch4_router_features.md`. Harness:
`scripts/phase3_postprocess_ablation.py` (extended) +
`scripts/phase3_geometry_router.py` (new `--feature-set` flag).
Artifacts: `runs/phase3_unet/batch4_router_features_v1/`.

Added 4 truth-free geometry proxies per model (8 new features total)
to the PR #8 learned-GBT router. Same architecture (200 trees, depth 3,
LR 0.05), same honest-eval protocol (cross-geometry positive training,
2500/2500 disjoint null split at seed 20260417), same deployment FPR
targets.

New per-patch features, computed from `P = sigmoid(mask_logits)`:

- `mask_area_at_0.5`: fraction of pixels with `P >= 0.5`.
- `centroid_offset_px`: Euclidean distance from probability-weighted
  centroid to patch center.
- `compactness`: perimeter / sqrt(area) on the thresholded binary mask.
- `edge_touching_fraction`: fraction of mask-at-0.5 pixels within
  <= 4 px of the patch boundary.

### Gate result

Pre-registered gate: 14-feature GBT must beat 6-feature GBT by
>= +0.010 recall on mixed geometry at FPR 0.08. Gate **passed**:

| geometry | FPR | `gbt_6` | `gbt_14` | delta | gate |
|---|---:|---:|---:|---:|---|
| contained | 0.05 | 0.359 | **0.388** | **+0.029** | pass |
| contained | 0.08 | 0.403 | **0.434** | **+0.031** | pass |
| contained | 0.10 | 0.431 | **0.466** | **+0.035** | pass |
| mixed | 0.05 | 0.322 | **0.352** | **+0.030** | pass |
| mixed | 0.08 | 0.365 | **0.408** | **+0.043** | **PASS (registered gate)** |
| mixed | 0.10 | 0.392 | **0.438** | **+0.046** | pass |

Every cell improves. Deltas consistent in sign and magnitude.

### Per-geometry-group lift at mixed / FPR 0.08

The geometry proxies do exactly what the hypothesis said they would:
the biggest gains land on the groups where the probability mask has
the most spatial non-triviality and where the 6 scalar scores lose the
most information.

| group | n | `gbt_6` | `gbt_14` | delta |
|---|---:|---:|---:|---:|
| geometry_truncated | 4914 | 0.253 | **0.313** | **+0.060** |
| center_outside_patch | 2625 | 0.207 | **0.265** | **+0.058** |
| visible_fraction_low | 1708 | 0.191 | **0.246** | **+0.056** |
| visible_fraction_mid | 2100 | 0.281 | **0.340** | +0.059 |
| geometry_contained | 12586 | 0.409 | **0.445** | +0.036 |

### Seed sensitivity

Four independent null-split seeds, same 14-feature config, mixed
recall at FPR 0.08:

| seed | `gbt_6` | `gbt_14` | delta |
|---:|---:|---:|---:|
| 20260417 (default) | 0.365 | 0.408 | +0.043 |
| 111 | 0.370 | 0.403 | +0.034 |
| 222 | 0.349 | 0.370 | +0.021 |
| 333 | 0.382 | 0.411 | +0.028 |

Mean delta +0.032, min +0.021, max +0.043, stdev 0.009. Every seed
clears the +0.010 gate. The worst seed gives 2.1x the required delta,
so the result is not an artifact of one null split.

### Feature importances

Default seed, mixed-geometry GBT. New features sum to ~37% of total
importance:

- v6_baseline 0.327 (primary score, same as PR #8).
- v7_mf_on_mask 0.089, v6_smooth_multi 0.074 (PR #8 carryover).
- **v7_centroid_offset 0.067**, **v6_centroid_offset 0.066** (new).
- **v7_edge_touching 0.061**, **v7_mask_area 0.056**, **v6_mask_area
  0.046** (new).
- v7_smooth_multi 0.051, v6_mf_on_mask 0.049, v7_baseline 0.036
  (PR #8 carryover).
- **v6_compactness 0.033**, **v7_compactness 0.031**, **v6_edge_touching
  0.013** (new, weaker but still non-zero).

Centroid offset (both models) and v7 edge_touching are the most
informative new features. Compactness is the weakest new feature.

`learned_logistic` still underperforms `learned_gbt` on the 14
features (+0.013 vs +0.043 on the gate cell), confirming the optimal
boundary is non-linear across the expanded feature set.

### Revised deployment advice (current)

- **Primary policy: `learned_gbt` with `--feature-set all` (14
  features).** Same GBT architecture as PR #8.
- **Fallback: `v6_aux_only` at real-SMICA threshold 0.873 (FPR 0.08).**
- **6-feature `learned_gbt` remains available via `--feature-set
  scores_only`** as the published PR #8 baseline for apples-to-apples
  comparators.
- Roles of `v7_mixed_ft` and `matched_template` unchanged.

**Section 26 note (2026-04-17 post-Batch-5):** The Batch 4 shipped lift
numbers above were measured on the clean null pool and do not survive
deployment-representative calibration. See Section 27. The 14-feature
GBT is no longer recommended as primary. Section 26 is retained as-is
for historical record of what was shipped in PR #9.

## 27. 2026-04-17 Batch 5 Full-sky Calibration Gap (Batch 4 Retraction)

Full report: `work/batch5_fullsky_calibration_gap.md`. Harness:
`scripts/phase3_fullsky_tile.py` (new). Artifacts:
`runs/phase3_unet/batch5_fullsky_fp_audit_*`.

A full-sky gnomonic tiling of all four Planck cleaned maps at HEALPix
Nside=8 (700 patches each after common-mask filter at unmasked fraction
>= 0.5) uncovered two findings:

### Finding 1: Calibration pool mask-fraction bias

The `data/training_v4/smica_null_controls_all.h5` calibration pool was
drawn with `MASK_THRESHOLD = 0.95` (coord_mask_fraction min 0.950,
median 0.984). Deployment tiling at 0.5 threshold covers patches with
50-99% unmasked, median 0.87. False positives fire preferentially on
mask-adjacent patches, which the calibration pool excludes.

Real-sky patch-level FPR at the PR #9 shipped 14-feature-GBT threshold
(calibrated to 0.08 on the clean-null eval half):

| map | patch-level FPR | FPR inflation vs 0.08 target |
|---|---:|---:|
| SMICA | 0.157 | 1.96x |
| NILC | 0.161 | 2.01x |
| SEVEM | 0.291 | 3.64x |
| Commander | 0.463 | **5.78x** |

On SMICA, stratified by patch mask fraction:

| mask fraction band | n | triggers | observed FPR |
|---|---:|---:|---:|
| [0.50, 0.70) | 59 | 14 | 0.237 |
| [0.70, 0.90) | 32 | 5 | 0.156 |
| [0.90, 0.95) | 15 | 2 | 0.133 |
| [0.95, 1.00] | 69 | 5 | 0.073 |

In the clean-pool band (>= 0.95) the tile FPR matches the calibration
target; in the [0.50, 0.70) band it is 3.3x higher. The bias is clean
and monotonic.

**Implication**: all FPR-calibrated thresholds shipped in PR #6, #8,
#9 (and the real-SMICA recalibration numbers in Sections 10-13) are
underestimates of deployment FPR by 2-6x depending on map.

### Finding 2: Batch 4 (14-feature GBT) lift retraction

Under deployment-representative tile calibration (threshold set so
exactly 8% of tile patches trigger per map), the Batch 4 PR #9 gate-cell
delta `gbt_14 - gbt_6` on mixed geometry:

| map | n tiles | rec6_tile | rec14_tile | delta | vs PR #9 shipped +0.043 |
|---|---:|---:|---:|---:|---|
| SMICA | 700 | 0.331 | 0.329 | **−0.001** | evaporates |
| NILC | 700 | 0.292 | 0.315 | **+0.023** | survives weakly |
| SEVEM | 700 | 0.237 | 0.192 | **−0.045** | reverses |
| Commander | 700 | 0.230 | 0.193 | **−0.036** | reverses |

Cross-map mean delta **−0.015**, stdev **0.027**. Stable across 4
null-split seeds on SMICA (seed range: −0.034 to −0.001).

**The 14-feature GBT's apparent +0.043 advantage over the 6-feature
GBT is an artifact of the clean-null calibration pool.** The
`edge_touching_fraction` and `centroid_offset_px` features specifically
learned statistics of the clean-pool mask-fraction distribution; in
deployment they raise the threshold more than they raise recall.

### Clustering side-result (Step 1 of the original session plan)

At realistic Nside=8 deployment density (7.3 deg spacing, ~87% per-
neighbor overlap), greedy angular-distance clustering of triggers does
reduce apparent FP burden meaningfully, consistent with my original
hypothesis:

| tile Nside | cluster @ 15 deg | cluster @ 25 deg |
|---|---:|---:|
| 4 (SMICA) | 1.18x | 1.53x |
| 8 (SMICA) | 2.29x | 4.78x |
| 8 (NILC) | 2.76x | 4.35x |
| 8 (SEVEM) | 3.24x | 7.03x |
| 8 (Commander) | ~ | ~ |

Clustering is therefore a real, cheap FP-reduction lever for deployment
tiling, but it only reduces apparent-FP count. The underlying patch-
level FPR inflation (Finding 1) is the dominant issue and has to be
fixed first.

### Actions taken

- `work/batch5_fullsky_calibration_gap.md` — full writeup.
- `scripts/phase3_fullsky_tile.py` — new, reusable.
- Negative registry entry added to Section 24.
- Deployment-advice header in Section 0 revised.
- Section 26 retained as historical record of what was shipped in PR
  #9; a note at the top of Section 26 points here.

### Corrective work still open (Step 2 and beyond)

1. **Build deployment-representative null pools per map** at
   `MASK_THRESHOLD = 0.5`. Target 5000 patches per map. Likely needs a
   small code change to `phase2_extract_smica_null_controls.py` to
   accept the lower threshold, or a new script that enumerates random
   patch centers uniformly on the sky with loose mask filtering.
2. **Re-run `phase3_postprocess_ablation.py`** on the new null pools.
3. **Re-run `phase3_geometry_router.py`** on the new transform caches
   for all three policies (`v6_only`, `gbt_6`, `gbt_14`) to get honest
   deployment-representative recall at FPR 0.08, per map.
4. **Ship the corrected deployment advice** based on the outcome.
   Likely `v6_aux_only` single-model with per-map calibrated thresholds
   remains the primary; any router claim needs to pass a cross-map
   gate on the new pools.
5. **Revisit the `phase3_v7_mf_channel_aux_w4` model** (the existing
   MF-channel training). Hypothesis: the MF-channel input may suppress
   mask-adjacent FPs because it only responds to disc-shaped kernel
   correlations, which mask-edge residuals lack. If true, its
   tile/clean-null FPR-inflation ratio would be smaller than the vanilla
   models' 2-6x. That would be a deployment-relevant advantage even
   if its clean-pool recall gain is small. This is the revised Step 3.
