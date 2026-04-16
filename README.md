# CMB Bubble Collision Detection

Planck-era machine-learning candidate screening for localized bubble-collision signatures in the Cosmic Microwave Background.

This repository is not claiming a cosmological detection. It implements and audits a reproducible screening front end: generate physically motivated candidates, rank them, emit structured outputs, and hand promising regions to classical or Bayesian follow-up. That framing is deliberate. Feeney et al.'s WMAP pipeline did candidate localization, edge checks, Bayesian parameter estimation, model comparison against LambdaCDM, and interpretation in terms of the expected detectable collision count. This project currently covers the candidate-screening and handoff layer.

Working claim:

> A reproducible Planck-era ML candidate-screening method for localized bubble-collision signatures, intended to accelerate or supplement classical follow-up.

## Current State

- Phase 1 defines the observable domain: Planck 2018 cleaned maps, mask-aware sky coordinates, and gnomonic patch geometry.
- Phase 2 builds the current synthetic generator: Feeney Eq. 1 disc templates, multiplicative injection, CAMB backgrounds, Planck mask geometry, beam/noise realism, provenance-clean splits, and real-map null controls.
- Phase 3 contains the current screening stack: U-Net branches, boundary-aware variants, matched-template and centered-disc baselines, sensitivity curves, real-SMICA recalibration, threshold-volume analysis, and machine-readable candidate outputs.
- The best current interpretation is operational, not triumphant: ML improves screening and morphology relative to simple classical screens in several regimes, but recall remains limited for low-amplitude, small-radius, weak-edge cases.
- A focused Nside=512 probe did not justify a full 512 retrain. Positive-only recall improved only by saturating on real-SMICA nulls. The current practical baseline remains Nside=256 with calibrated operating points.

## Signal Model

Bubble-collision signatures are modeled as localized, azimuthally symmetric CMB temperature modulations confined to a causal disc. Following Feeney, Johnson, Mortlock, and Peiris, the leading-order template is

```math
\frac{\delta T}{T} =
\left[
\frac{z_{\rm crit} - z_0 \cos\theta_{\rm crit}}{1 - \cos\theta_{\rm crit}}
+
\frac{z_0 - z_{\rm crit}}{1 - \cos\theta_{\rm crit}}\cos\theta
\right]\Theta(\theta_{\rm crit}-\theta).
```

Key parameters:

- `z0`: central modulation amplitude.
- `zcrit`: boundary amplitude/discontinuity term.
- `theta_crit`: angular radius of the affected disc, currently sampled over `5 deg` to `25 deg`.
- `theta0, phi0`: sky position of the disc center.

The generator uses the multiplicative injection form, not an additive shortcut:

```math
T_{\rm injected} = (1 + f)(T_{\rm CMB}) .
```

The `sin(theta_crit)` sampling law in this repository is a training-design choice motivated by the Feeney geometry discussion. It is not a Bayesian inference prior.

## Data And Geometry

The observable domain is Planck-era, patch-based, and mask-aware.

![Planck 2018 SMICA full-sky CMB map](plots/01_smica_fullsky.png)

**Figure 1.** Planck 2018 SMICA CMB map. SMICA is the primary real-map target for the current screening and null-control tests.

![SMICA with galactic mask](plots/03_smica_masked.png)

**Figure 2.** Planck common mask applied to the cleaned CMB map. The coordinate pool is drawn from clean unmasked regions with mask-fraction checks, not only center-pixel checks.

![Gnomonic patch near Cold Spot](plots/04_gnomonic_cold_spot.png)

**Figure 3.** Example gnomonic patch. The current working product uses `256 x 256` patches at `13 arcmin/pixel`, covering roughly `55 deg` across.

## Phase 2 Generator

The current generator is intentionally stricter than an early ML demo:

- It uses independent CAMB CMB realizations rather than repeatedly training on one real SMICA sky.
- It injects Feeney-style disc templates multiplicatively.
- It balances sign quadrants for `z0` and `zcrit`.
- It randomizes signal centers within patches to remove center-bias shortcut learning.
- It enforces provenance-clean train/validation splits by coordinate and CMB realization.
- It supports beam smoothing, instrumental noise, real-map null controls, and stratified validation products.
- It stores metadata needed to trace patches back to sky coordinates and generation settings.

![Phase 2 signal profiles](plots/06_phase2_signal_profiles.png)

**Figure 4.** Representative Feeney-template profiles. The generator preserves the long-wavelength disc modulation and causal-boundary structure instead of using generic circular blobs.

![Phase 2 validation](plots/07_phase2_validation.png)

**Figure 5.** Phase 2 distribution checks. These checks exist to prevent the model from learning hidden generator shortcuts such as one sign quadrant, one radius band, or a narrow amplitude slice.

<img src="plots/08_phase2_examples.png" alt="Phase 2 training examples" width="760">

**Figure 6.** Example synthetic training patches. The lower visual salience of many examples is intentional: the data include weak amplitudes, mixed signs, off-center discs, beam/noise effects, and smoothed boundaries, all of which are closer to the Feeney search problem than centered high-SNR toy discs.

## Phase 3 Screening Results

Phase 3 is now evaluated against classical baselines on the same audited splits. The five learned/model-policy variants below were important investigative controls: they tested whether boundary-aware losses, consensus policies, auxiliary heads, and branch averaging produced materially different behavior. The current working path is narrower. We are focusing on the two-model pair `v6_aux_only + matched_template` because it keeps the best morphology-aligned ML branch while retaining an interpretable Feeney-template classical score, without carrying the calibration and maintenance burden of several near-tied learned branches.

At matched synthetic FPR `0.08` on the independent stratified validation set:

| method | AUROC | AUPRC | recall | weak recall | positive Dice |
|---|---:|---:|---:|---:|---:|
| matched template | `0.712` | `0.881` | `0.401` | `0.282` | `0.295` |
| centered disc | `0.639` | `0.833` | `0.252` | `0.176` | `0.046` |
| original V4 | `0.775` | `0.914` | `0.541` | `0.425` | `0.344` |
| boundary V4 | `0.774` | `0.914` | `0.540` | `0.420` | `0.377` |
| V5 consensus | `0.774` | `0.913` | `0.536` | `0.422` | `0.396` |
| V6 aux only | `0.773` | `0.913` | `0.539` | `0.422` | `0.401` |

![Phase 3 method comparison](plots/09_phase3_method_comparison.png)

**Figure 7.** Current Phase 3 comparison at matched FPR. The metrics are lower than early high-SNR baselines because the current data intentionally removes shortcuts: off-center injection, provenance-clean splits, weak amplitudes, smoothed causal boundaries, beam/noise effects, and real-map null calibration. That is the right direction scientifically even when it makes headline recall less flattering.

## Sensitivity Versus Matched Template

At matched synthetic FPR `0.05`, the best ML branch significantly beats the beam-matched Feeney-template screen in `8 / 35` amplitude-radius cells and shows no significant losses. This is a localized gain, not universal dominance.

Representative significant cells:

| amplitude `A` | `theta_crit` | matched template `P_det` | best ML `P_det` |
|---:|---:|---:|---:|
| `1e-5` | `25 deg` | `0.065` | `0.175` |
| `2e-5` | `20 deg` | `0.235` | `0.555` |
| `2e-5` | `25 deg` | `0.375` | `0.710` |
| `5e-5` | `5 deg` | `0.065` | `0.295` |
| `5e-5` | `10 deg` | `0.490` | `0.980` |
| `1e-4` | `5 deg` | `0.500` | `0.980` |

![Phase 3 ML gain heatmap](plots/10_phase3_ml_gain_heatmap.png)

**Figure 8.** Significant gain over a beam-matched template screen. Gray cells are not ML failures; they are cells where the confidence interval overlaps parity. The lower-left region remains hard because low amplitude and small angular radius give limited integrated signal-to-noise, especially after Feeney-faithful smoothing and realistic background structure.

## Real-SMICA Calibration

A real-SMICA injection gate initially looked like a domain-gap failure. Recalibration showed the dominant issue was threshold mismatch:

| method | CAMB threshold | SMICA-null threshold | real recall at CAMB threshold | real recall after SMICA recalibration |
|---|---:|---:|---:|---:|
| `v6_aux_only` | `0.992082` | `0.888469` | `0.262` | `0.353` |
| `matched_template` | `76.977921` | `61.829514` | `0.238` | `0.323` |

Interpretation: thresholds calibrated on CAMB negatives were too strict on real SMICA. The model transfers better than the first gate suggested, but recall is still a candidate-volume tradeoff rather than a solved detection problem.

## Threshold And Candidate Volume

The current detector can recover more contested positives by lowering the threshold, but false-positive volume rises quickly.

| threshold | contested recall | solved recall | expected FP over 3000 independent patches |
|---:|---:|---:|---:|
| `0.75` | `0.654` | `0.976` | `1631` |
| `0.80` | `0.442` | `0.954` | `895` |
| `0.85` | `0.278` | `0.940` | `401` |
| `0.90` | `0.150` | `0.922` | `106` |

![Phase 3 threshold tradeoff](plots/11_phase3_threshold_tradeoff.png)

**Figure 9.** Recall versus expected candidate burden. The numbers are lower than a toy segmentation benchmark because the operating point is being treated in the Feeney spirit: thresholds must be fixed against null controls before real-map use, rather than tuned after looking at candidates. High recall is available, but it costs hundreds to thousands of follow-up candidates.

<img src="plots/12_phase3_smoothed_examples.png" alt="Phase 3 smoothed positive examples" width="760">

**Figure 10.** Diagnostic real-SMICA injection examples. The failures are informative: small-radius or low-amplitude discs often produce diffuse probability maps just below threshold, while larger or stronger discs are recovered cleanly. This is consistent with an integrated-SNR limitation, not simple model blindness.

## Current Operating Direction

Current Phase 3 should be treated as a two-score screening system, not a five-model deployment stack.

- Run `v6_aux_only` as the current ML morphology and candidate-score branch.
- Run `matched_template` as the classical Feeney-template reference, fallback, and independent ranking score.
- Calibrate both thresholds on real-map null controls, not CAMB negatives alone.
- Preserve the ML score, matched-template score, threshold decisions, mask, estimated radius, sky metadata, and template-fit artifacts in each candidate record.
- Keep the older learned branches as documented ablation evidence and regression checks, not as the preferred operating stack.

The earlier multi-branch policies remain useful as investigative benchmarks. They showed that consensus-style verification can be very clean, but they did not justify the added complexity as the default deployment path:

| policy | precision | recall | FPR | F1 |
|---|---:|---:|---:|---:|
| `v5_only` | `0.969` | `0.501` | `0.041` | `0.660` |
| `score_avg_only` | `0.975` | `0.495` | `0.032` | `0.656` |
| `matched_template_only` | `0.948` | `0.348` | `0.049` | `0.509` |
| `normal_candidate` | `0.980` | `0.474` | `0.025` | `0.639` |

On the `5000`-patch SMICA null-control set, `matched_template`, `v5_consensus`, `score_avg`, `normal_candidate`, and `all_candidates` each produced `0 / 5000` null candidates at their frozen thresholds. That result is retained because it verifies the investigation, not because all five branches should be run by default.

## What Is Not Solved

- This is not yet a Feeney-style Bayesian detection framework.
- It does not estimate a posterior over bubble-collision parameters.
- It does not perform model selection against LambdaCDM.
- It does not constrain the expected detectable collision count.
- Weak-family recall remains the central blocker.
- Nside=512 is not justified by the current focused probe.

Current engineering targets:

- Run full-sky Planck screening with the `v6_aux_only + matched_template` pair, map-calibrated thresholds, and candidate clustering.
- Calibrate thresholds separately for SMICA, NILC, SEVEM, and Commander.
- Keep matched-template scores in every candidate record as a classical sanity check.
- Add a scale-aware score or radius head only if it improves the weak small-radius cells without increasing real-map null burden.
- Feed candidate records into a classical template-fit or Bayesian follow-up stage.

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate cmb
```

Generate or inspect Phase 1 products:

```bash
python scripts/phase1_explore.py
```

Run Phase 2 checks and generate the current training set:

```bash
python scripts/phase2_physics_checks.py
python scripts/phase2_generate_training.py \
  --num-samples 10000 \
  --pool-size 5000 \
  --num-cmb-realizations 192 \
  --output-dir data/training_v4
python scripts/phase2_audit_dataset.py \
  --data-h5 data/training_v4/training_data.h5 \
  --output-json data/training_v4/audit_report.json
```

Train and evaluate a Phase 3 branch:

```bash
python scripts/phase3_train_unet.py \
  --data-h5 data/training_v4/training_data.h5 \
  --epochs 20 \
  --batch-size 16 \
  --threshold 0.92

python scripts/phase3_evaluate_run.py \
  --run-dir runs/phase3_unet/phase3_v6_aux_only_w4 \
  --checkpoint best \
  --split val
```

Run key evaluation harnesses:

```bash
python scripts/phase3_sensitivity_curve.py
python scripts/phase3_eval_stratified_external.py
python scripts/phase3_eval_classical_stratified_external.py
python scripts/phase3_real_sky_recalibration.py
python scripts/phase3_threshold_volume_sweep.py
python scripts/phase3_screen_and_verify.py
```

## Repository Layout

- `scripts/phase1_explore.py`: Planck map loading, masking, and patch geometry.
- `scripts/phase2_signal_model.py`: Feeney-template implementation and signal checks.
- `scripts/phase2_generate_training.py`: current CAMB/Planck-mask training generator.
- `scripts/phase2_audit_dataset.py`: provenance, split, and leakage audit.
- `scripts/phase3_train_unet.py`: U-Net training harness.
- `scripts/phase3_evaluate_run.py`: validation/evaluation harness.
- `scripts/phase3_template_baseline.py`: matched-template and classical baseline support.
- `scripts/phase3_screen_and_verify.py`: candidate-output CLI.
- `docs/project_structure.md`: artifact policy and source organization.
- `docs/nside512_probe_decision.md`: focused Nside=512 probe verdict.

Generated datasets, Planck FITS maps, checkpoints, and run artifacts are intentionally ignored by git. The repository tracks source, documentation, compact figures, and reproducibility metadata, not multi-GB intermediates.

## Hardware

The current local workstation has 2 NVIDIA RTX 3090 GPUs. The code supports multi-GPU training through PyTorch `DataParallel`.

## Key References

- Feeney, Johnson, Mortlock, and Peiris. *First Observational Tests of Eternal Inflation.* arXiv:1012.1995.
- Feeney, Johnson, Mortlock, and Peiris. *First Observational Tests of Eternal Inflation: Analysis Methods and WMAP 7-Year Results.* arXiv:1012.3667.
- Gorski et al. *HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere.* Astrophysical Journal, 2005.
- Lewis, Challinor, and Lasenby. *Efficient Computation of Cosmic Microwave Background Anisotropies in Closed Friedmann-Robertson-Walker Models.* Astrophysical Journal, 2000. CAMB.
- Planck Collaboration. *Planck 2018 results. IV. Diffuse component separation.* Astronomy & Astrophysics, 2020.

## License

MIT

## Authors

William Starks

Gus Marcum
