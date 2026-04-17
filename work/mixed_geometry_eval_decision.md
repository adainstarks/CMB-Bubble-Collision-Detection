# Mixed-Geometry Evaluation Outcome And Decision

Date: 2026-04-16
Validation set: `data/validation_stratified_mixed_geometry_v1` (5000 samples, 3600 positives,
1067 truncated = 29.6%, mean visible target fraction = 0.85, min = 0.15).
Match-FPR calibration target: 0.08.

## Headline numbers (recall at matched FPR=0.08)

| Group | n | v6_aux_only (UNet, contained-only training) | matched_template | centered_disc |
|---|---:|---:|---:|---:|
| All positive | 3600 | 0.456 | 0.350 | 0.222 |
| `geometry_contained` | 2533 | **0.562** | 0.410 | 0.280 |
| `geometry_truncated` | 1067 | **0.206** | 0.207 | 0.083 |
| `center_inside_patch` | 3021 | 0.515 | 0.389 | 0.247 |
| `center_outside_patch` | 579 | **0.152** | 0.145 | 0.090 |
| `visible_fraction_low` (<~0.4) | 373 | 0.134 | 0.131 | 0.086 |
| `visible_fraction_mid` | 456 | 0.200 | 0.206 | 0.083 |
| `visible_fraction_high` (>~0.7) | 2771 | 0.542 | 0.403 | 0.263 |
| `weak_family_union` | 2400 | 0.360 | 0.252 | 0.165 |

Pixel-level Dice+ at the matched threshold:

| Group | v6_aux_only | matched_template | centered_disc |
|---|---:|---:|---:|
| `geometry_contained` | 0.414 | 0.293 | 0.052 |
| `geometry_truncated` | **0.057** | 0.103 | 0.001 |
| `center_outside_patch` | **0.016** | 0.047 | ~0 |

AUROC: v6_aux_only 0.727 [0.712, 0.741], matched_template 0.691 [0.676, 0.705],
centered_disc 0.609 [0.592, 0.625].

## Hypothesis verdict

H1 (center-framed shortcut from contained-only training) is **supported**:

- v6_aux_only beats matched_template on contained samples by +0.152 absolute recall
  (0.562 vs 0.410) and on visible_high samples by +0.139.
- That advantage **disappears** on truncated samples (0.206 vs 0.207, statistically
  tied) and on center-outside samples (0.152 vs 0.145).
- Pixel-level localisation actually flips: matched_template outperforms v6_aux_only
  on truncated Dice (0.103 vs 0.057) and on center-outside Dice (0.047 vs 0.016).
  The ML mask cannot localise a half-disc that crosses the patch edge.

The U-Net learned a "find a roughly centered, fully bounded disc" feature, not the
underlying boundary discontinuity. The `weak_family_union` cell now reads 0.360 for ML
vs 0.252 for matched_template; that gap is real but is bounded by the contained subset
inside `weak_family_union`.

H2 (irreducible SNR floor) is also partly supported: visible_fraction_low recall is
~0.13 for both ML and matched_template. No method recovers a 15-40% disc fraction at
FPR 0.08 with the current beam and noise budget. This is consistent with Feeney
1012.1995 sensitivity limits.

## Decision

**Retrain (fine-tune) on mixed-geometry training data.** Path:

1. Generate `data/training_v5_mixed_geometry` from `phase2_generate_training.py` with
   `--geometry-mode mixed --truncated-positive-fraction 0.30
   --truncated-visible-fraction-min 0.15 --truncated-visible-fraction-max 0.95`,
   excluding the new validation set by HDF5 to avoid leakage.
2. Fine-tune `runs/phase3_unet/phase3_v6_aux_only_w4/best_checkpoint.pt` for ~6 epochs,
   LR 5e-5, on the new training set with the same loss configuration that produced
   v6_aux_only (no radius head; the radius-head branch is on hold per
   `radius_head_post_mortem.md`).
3. Re-evaluate on `validation_stratified_mixed_geometry_v1` and on the original
   contained-only `validation_stratified_v1`. Promote the new model only if:
   - Mixed truncated recall increases by >= 0.10 absolute,
   - Mixed contained recall does not regress by more than 0.03 absolute,
   - Original contained validation AUROC does not regress by more than 0.01.

If those gates fail, fall back to a per-geometry calibration: keep v6_aux_only for
contained candidates and explicitly hand truncated candidates to matched_template,
which currently has equal or better Dice in that regime.

## What this does not solve

- The `visible_fraction_low` cell will remain SNR-limited. No retraining will recover
  recall above ~0.20 in that cell with current beam/noise; it has to be addressed by
  the downstream Bayesian template-fit verifier (Feeney 1012.1995 step 2).
- The radius-head branch is still parked. Mixed-geometry retraining must be run
  *without* the radius head so we can attribute any change to geometry alone.
- The decision to drop v5_consensus / boundary_v4 / hard_w15 from the operating
  table is unchanged; their relative ranking in the contained regime already deemed
  them inferior or redundant.
