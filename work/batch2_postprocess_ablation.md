# Batch 2: Post-Processing Ablation on Real-SMICA

## TL;DR

Two post-processing transforms applied to frozen v6_aux_only and v7_mixed_ft
probability masks, evaluated on the same real-SMICA gate data used in
Batch 1 (500 backgrounds x 7 amplitudes x 5 theta x 4 sign-quadrants,
thresholds calibrated on 5000 real-SMICA null patches).

Result: mixed success.

- `smooth_multi` (Gaussian smoothing with sigma in {4, 8, 16} pix, max over sigmas): null result.
- `mf_on_mask` (matched-filter rescoring on the smoothed probability mask): **positive for v7 at tight FPR on contained geometry** (+0.039 recall at FPR 0.05), **negative elsewhere**. Not a general replacement. Acts as a contained-geometry specialist.
- **Portfolio decision stands.** Post-processing alone does not close the
  v7-vs-v6 gap on real-SMICA contained geometry.

Harness: `scripts/phase3_postprocess_ablation.py` (new).
Artifacts: `runs/phase3_unet/batch2_postprocess_ablation_v1/`.

## Setup

Post-processing transforms applied to the frozen U-Net probability mask
`P = sigmoid(mask_logits)`:

1. **`baseline`** — current operating score `max(P)`.
2. **`smooth_multi`** — `max over sigma in {4, 8, 16} pix of max(gaussian_filter(P, sigma))`. Multi-scale smoothing maxed over sigmas. Suppresses 1-pixel noise while preserving coherent regions.
3. **`mf_on_mask`** — `max over theta in {5, 10, 15, 20, 25} deg of max(fftcorr(gaussian_filter(P, sigma=4), disc_kernel(theta)))`. Matched filtering on the smoothed probability mask using disc-shaped kernels. The kernels are zero-mean L2-normalized positive discs matched to Feeney's theta_crit grid.

The disc kernel is the right matched filter for a probability mask, not a raw CMB patch: the mask output by the U-Net already encodes sign-agnostic "this region is disc-like." A positive disc kernel correlates maximally with a spatially coherent disc-shaped probability elevation.

Thresholds for each transform are independently calibrated on the same 5000-patch real-SMICA null distribution at FPR targets 0.05, 0.08, 0.10.

## Global recall

| model | geometry | transform | FPR 0.05 | FPR 0.08 | FPR 0.10 |
|---|---|---|---:|---:|---:|
| v6_aux_only | contained | baseline | 0.348 | 0.372 | 0.389 |
| v6_aux_only | contained | smooth_multi | 0.346 (-0.003) | 0.371 (-0.001) | 0.389 (+0.000) |
| v6_aux_only | contained | mf_on_mask | 0.306 (-0.042) | 0.339 (-0.032) | 0.361 (-0.028) |
| v6_aux_only | mixed | baseline | 0.305 | 0.331 | 0.347 |
| v6_aux_only | mixed | smooth_multi | 0.303 (-0.003) | 0.327 (-0.004) | 0.344 (-0.003) |
| v6_aux_only | mixed | mf_on_mask | 0.257 (-0.048) | 0.288 (-0.043) | 0.308 (-0.040) |
| v7_mixed_ft | contained | baseline | 0.286 | 0.357 | 0.386 |
| v7_mixed_ft | contained | smooth_multi | 0.293 (+0.007) | 0.356 (-0.001) | 0.383 (-0.002) |
| v7_mixed_ft | contained | mf_on_mask | **0.325 (+0.039)** | 0.363 (+0.006) | 0.379 (-0.007) |
| v7_mixed_ft | mixed | baseline | 0.248 | 0.328 | 0.355 |
| v7_mixed_ft | mixed | smooth_multi | 0.259 (+0.011) | 0.325 (-0.003) | 0.352 (-0.003) |
| v7_mixed_ft | mixed | mf_on_mask | **0.280 (+0.032)** | 0.320 (-0.008) | 0.335 (-0.020) |

Delta vs. baseline is recall - baseline_recall at the same FPR target.

## Per-geometry breakdown at FPR 0.08 (mixed real-SMICA)

| model | transform | contained | truncated | center_out | vis_low |
|---|---|---:|---:|---:|---:|
| v6_aux_only | baseline | 0.380 | 0.205 | 0.163 | 0.145 |
| v6_aux_only | smooth_multi | 0.378 | 0.196 | 0.147 | 0.128 |
| v6_aux_only | mf_on_mask | 0.345 | 0.143 | 0.087 | 0.083 |
| v7_mixed_ft | baseline | 0.360 | 0.246 | 0.207 | 0.196 |
| v7_mixed_ft | smooth_multi | 0.358 | 0.240 | 0.200 | 0.186 |
| v7_mixed_ft | mf_on_mask | **0.370** | 0.192 | 0.137 | 0.118 |

## Interpretation

### smooth_multi: null result

`smooth_multi` moves global recall by at most 0.011 in either direction.
The U-Net already produces a smoothed mask; an additional 4-16 pixel Gaussian
is operating below the natural correlation length of the mask noise, so it
doesn't suppress noise pixels any more than the raw max already does.
Variants to try later: small-sigma (1-2 pix) noise-only suppression, or
theta-matched sigma per radius. Those are tiny experiments and worth one
more ablation before fully writing smoothing off.

### mf_on_mask: contained-geometry specialist for v7

At FPR 0.05, matched-filter-on-mask delivers real gains for v7:

- v7 contained: 0.286 -> 0.325 (+0.039, +14% relative).
  This closes 63% of the v7-vs-v6 gap at FPR 0.05 (0.062 -> 0.023).
- v7 mixed: 0.248 -> 0.280 (+0.032).

But the transform hurts elsewhere. v6 loses ~0.03-0.05 recall at every FPR
and geometry. And the per-group breakdown at FPR 0.08 shows why: on v7,
`mf_on_mask` lifts contained recall (+0.010) while collapsing truncated
(-0.054), center_out (-0.070), and vis_low (-0.078).

This is the expected behavior of a positive disc matched filter: it
assumes a full disc. Partial-overlap responses from truncated signals look
like weaker fully-contained signals, blending into the null distribution.

### The portfolio decision holds

Even the best post-processing (v7 + mf_on_mask at FPR 0.05) does not beat
v6 baseline at the same FPR on contained geometry (0.325 vs 0.348).
Post-processing narrows the v7 gap but does not close it. Two-model
portfolio remains the correct deployment path.

## What this unlocks

The most interesting artifact of this ablation is not a direct recall gain
but a **geometry-routing feature**. `mf_on_mask - baseline` per candidate
should separate contained-disc-like candidates from truncated ones:

- High delta: `mf_on_mask` score is much higher than `baseline`. The
  probability mask is disc-shaped and coherent. Score with v6_aux_only.
- Low or negative delta: the mask is spiky or edge-constrained. Score with
  v7_mixed_ft.

That is a concrete Batch 3 candidate and is a more principled geometry
router than asking the model to self-classify.

## Pre-registered Batch 2 decision criteria

Checking the pre-registered rubric from the plan:

| criterion | pre-registered | actual | verdict |
|---|---|---|---|
| v7+smoothing closes contained gap on real SMICA | matches/beats v6 @ FPR 0.05 | v7+mf_on_mask 0.325 vs v6 0.348 | NOT CLOSED |
| Both models improve but v6 contained still wins | at least some positive delta | smooth_multi null; mf_on_mask specialist | PARTIAL |
| No material gain from either knob | document clean negative | partial positive on v7 contained | N/A |

Pre-registered path forward: **portfolio persists**. Next candidate lever is
a retrain (v8 with matched-filter input channel on mixed geometry) or the
Batch 3 geometry-routing feature built from the `mf_on_mask - baseline`
contrast derived here.

## What will NOT help right now (stop the next AI from trying these)

- Bigger encoder / new architecture: not the bottleneck. The model is
  already confident on its contained-centered-disc prior; what it lacks is
  geometry flexibility and that was resolved by fine-tuning, not scale.
- Naive ensemble averaging: documented negative in
  `work/tta_ensemble_eval.md`.
- D4 test-time augmentation: documented +0.7pp Dice only, also in
  `work/tta_ensemble_eval.md`.
- Gaussian-smoothing-before-max: documented null here.
- Matched-filter-on-mask as a general rescore: hurts v6, mixed effect on
  v7. Only useful as a geometry-routing signal or as a tight-FPR contained-
  only rescore.

## What might help, ordered by expected gain

1. **Batch 3 geometry router** using `mf_on_mask - baseline`: EXECUTED, see
   `work/batch3_geometry_router.md`. Simple heuristic routing did not beat
   best single model at any FPR target. Learned-classifier geometry routing
   is still on the table but out of scope for this pass.
2. **Small-sigma Gaussian (sigma 0.5-2 pix)**: EXECUTED in ablation v2 below.
   Confirmed null at all sigmas tried.
3. **Isotonic score calibration on real-SMICA nulls**: threshold-selection
   robustness, not a recall boost. Required for clean candidate-volume
   reports.
4. **v8 retrain with matched-filter input channel on mixed geometry**: the
   only proper training-signal lever left. Expected to move truncated
   recall 4-10pp with proper training hygiene (see
   `work/radius_head_post_mortem.md` for required hygiene).

## Small-sigma follow-up (ablation v2)

Sigmas swept: (0.5, 1.0, 2.0) pixels.

Artifacts: `runs/phase3_unet/batch2_postprocess_ablation_v2_smallsigma/`.

Result: still null. Delta vs baseline is <= 0.001 absolute in every
(model, geometry, FPR) cell. Full table in the report there.

This closes the loop on Gaussian smoothing as a post-processing lever at
any practical sigma. The U-Net probability mask's dominant noise is not
at any sub-disc spatial scale that Gaussian smoothing can suppress without
also eroding signal.

The `mf_on_mask` numbers are essentially identical across the two sigma
settings (sigma=4 from ablation v1 vs sigma=0.5 from v2): v7 contained at
FPR 0.05 is 0.325 vs 0.326. The matched-filter-on-mask transform is
robust to the choice of base smoothing sigma, confirming that what it
captures is mask *shape coherence*, not mask smoothing.
