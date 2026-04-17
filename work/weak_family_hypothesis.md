# Weak-Family Failure Analysis And Hypothesis

Purpose: Before the next training run, isolate *one* falsifiable hypothesis about where
and why the current screening stack loses weak positives. Data comes from
`runs/phase3_unet/ml_gain_heatmap_v1/ml_gain_heatmap.json` (contained-only, matched
synthetic FPR 0.05, hard Feeney Eq. 1 injections, 200 positives per cell, 15 arcmin beam,
30 uK-arcmin noise).

## Failure surface of the current stack

Per-cell P_det for the best ML branch vs matched-template baseline on the 7x5
amplitude x theta grid:

| | theta 5 deg | theta 10 deg | theta 15 deg | theta 20 deg | theta 25 deg |
|---|---:|---:|---:|---:|---:|
| A = 1e-6 | 0.04 / 0.06 | 0.05 / 0.07 | 0.07 / 0.06 | 0.04 / 0.07 | 0.06 / 0.07 |
| A = 2e-6 | 0.04 / 0.06 | 0.06 / 0.06 | 0.06 / 0.05 | 0.06 / 0.07 | 0.05 / 0.05 |
| A = 5e-6 | 0.06 / 0.07 | 0.06 / 0.07 | 0.05 / 0.07 | 0.09 / 0.07 | 0.09 / 0.11 |
| A = 1e-5 | 0.05 / 0.05 | 0.04 / 0.07 | 0.04 / 0.08 | 0.05 / **0.12** | 0.07 / **0.18** |
| A = 2e-5 | 0.07 / 0.08 | 0.08 / 0.11 | 0.18 / 0.23 | 0.24 / **0.56** | 0.38 / **0.71** |
| A = 5e-5 | 0.07 / **0.30** | 0.49 / **0.98** | 0.96 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 |
| A = 1e-4 | 0.50 / **0.98** | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

Legend: `matched / bestML`. **Bold** marks cells where bootstrap CI says ML wins over
matched. Null FPR is calibrated to `0.05`, so any P_det near `0.05` is at the noise
floor and cannot be improved by a better classifier.

## Three distinct failure regions

1. **Dead zone** (A <= 2e-6, all theta): both matched-template and ML sit at the 0.04-0.07
   P_det floor, which is indistinguishable from the 0.05 target FPR. This is an
   information-theoretic limit on the front-end; Feeney 1012.1995 explicitly flags this
   regime as unreachable without the Bayesian step. No ML change will recover this.

2. **Small-disc failure** (theta = 5 deg, all A < 1e-4): at theta = 5 deg the disc has
   radius ~23 pixels and area ~2.6% of the patch. Integrated signal-to-noise is limited
   by the pixel count, not the amplitude. Even at A = 5e-5, matched-template gets only
   P_det = 0.065 while ML gets 0.295; only A = 1e-4 recovers.

3. **Contested-amplitude-large-disc** (A in [5e-6, 2e-5], theta in [15, 25] deg): this is
   the one cell family where ML produces significant gains (2x - 2.7x) over matched
   template. The current stack sees signal but not at confident thresholds.

## Is this a U-Net problem or a physics problem?

Feeney 1012.1995 reports needlet + edge front-end sensitivity requires
`|z0| >~ 3e-5` or `|zcrit| >~ 3e-5` with `theta_crit >~ 5 deg`. Our contested zone
lies below that limit. The irreducible SNR ceiling is real.

This means the correct scientific framing of weak-family recall is not "make the U-Net
better" but "understand which subset of weak positives are actually shortcut-limited
rather than SNR-limited". Only the shortcut-limited subset can be rescued by model or
data changes. Everything else needs the Bayesian evidence stage that this repo hands off
to.

## Falsifiable hypothesis to test on mixed-geometry validation

**Hypothesis H1 (shortcut-from-contained-training):**

Small-theta contested-amplitude positives with targets that touch the patch edge
(truncated geometry) will have *significantly lower* recall than their contained
counterparts at the same (A, theta, edge_sigma) bin, because training data was
contained-only and the model learned an implicit center-framed prior.

Concretely: compute per-cell recall of `v6_aux_only` on the upcoming
`data/validation_stratified_mixed_geometry_v1`, restricted to
`theta_bin == 0` (theta in [5, 10) deg) and `z0_amp_bin == 0` (A in [1e-6, 1e-5)) and
edge bins, split by `fully_contained` truth. If recall drops by >= 30% absolute
on truncated vs contained in that cell while overall contained recall is stable,
H1 is supported. Retraining on mixed data becomes justified.

**Null H0:** truncated and contained recall are within CI bands of each other in the
weak cell. The model is not using a centering shortcut; weak-family recall is
SNR-limited. Retraining on mixed data will not yield meaningful weak gains.

## Decision table (to be executed after mixed-geometry eval)

| Outcome | Interpretation | Next action |
|---|---|---|
| Truncated weak recall >> contained weak recall (unlikely) | Model learned an inverse shortcut | Investigate as bug |
| Truncated weak recall << contained weak recall (H1 supported) | Center-framed shortcut | Fine-tune on mixed training set (`training_v5_mixed_geometry`), ~6 epochs, LR 5e-5, resume from `v6_aux_only` best |
| Truncated ~ contained weak recall (H0) | SNR-limited; no shortcut | Stop. Do not retrain. Move to candidate clustering + a lightweight Feeney-template-fit verifier per candidate |

## Why this is the right order

Retraining before running the mixed-geometry eval would either burn hours on a
hypothesis (`v5_mixed_geometry`) that isn't needed (H0 case), or get a model that may
look better on truncated metrics but has never been held to the existing baseline on
the proper null. The eval is cheap compared to the training; it runs first.

## 2026-04-16 update — evaluation outcome

H1 supported on the mixed-geometry validation. Detail in
`work/mixed_geometry_eval_decision.md`. Headline:

- `v6_aux_only` recall: contained 0.562, truncated 0.206 (-0.356 absolute drop).
- `matched_template` recall on the same split: contained 0.410, truncated 0.207
  (-0.203 absolute drop).
- The U-Net's +0.152 advantage over matched_template on contained samples *vanishes*
  on truncated samples (0.206 vs 0.207 = statistical tie).
- Dice+ collapses from 0.414 (contained) to 0.057 (truncated) for v6_aux_only,
  while matched_template Dice+ stays at 0.103 in the truncated regime — the U-Net
  mask actively misfires when the disc is half off-frame.

This is consistent with a learned center-framed prior, not with an SNR limit. The
pure-SNR fingerprint (visible_fraction_low recall ~0.13 for both methods) is also
present, but it accounts for only 373 of the 1067 truncated positives. The remaining
694 are recoverable in principle by retraining on geometrically representative data.

Decision (per `mixed_geometry_eval_decision.md`): generate
`data/training_v5_mixed_geometry`, fine-tune `v6_aux_only` for ~6 epochs, gate on
both the new mixed validation and the original contained validation before promotion.
