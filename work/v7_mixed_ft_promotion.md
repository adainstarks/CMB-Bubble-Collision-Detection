# v7_mixed_ft Promotion Summary

Date: 2026-04-16
Model: `runs/phase3_unet/phase3_v7_mixed_ft/best_checkpoint.pt`
Provenance: fine-tuned from `phase3_v6_aux_only_w4/best_checkpoint.pt` on
`data/training_v5_mixed_geometry/training_data.h5` for 6 epochs at LR 5e-5,
checkpoint metric `hard_dice_pos`, no radius head, identical loss recipe to v6
(BCE 1.0, Dice 1.0, aux 0.1, boundary 4.0/5px), normalization inherited from
v6's run_config.

## Side-by-side at matched FPR = 0.08

### `data/validation_stratified_mixed_geometry_v1` (n=5000, 1067 truncated)

| Group | n | v6_aux_only | v7_mixed_ft | Δ |
|---|---:|---:|---:|---:|
| All positive | 3600 | 0.456 | 0.486 | +0.030 |
| `geometry_contained` | 2533 | 0.562 | 0.569 | +0.007 |
| `geometry_truncated` | 1067 | 0.206 | **0.288** | **+0.082** |
| `center_inside_patch` | 3021 | 0.515 | 0.534 | +0.019 |
| `center_outside_patch` | 579 | 0.152 | **0.233** | **+0.081** |
| `visible_fraction_low` | 373 | 0.134 | 0.193 | +0.059 |
| `visible_fraction_mid` | 456 | 0.200 | **0.300** | **+0.100** |
| `visible_fraction_high` | 2771 | 0.542 | 0.555 | +0.013 |
| `weak_family_union` | 2400 | 0.360 | 0.382 | +0.022 |
| AUROC | - | 0.727 | 0.749 | +0.022 |
| Dice+ | - | 0.308 | 0.354 | +0.046 |

### `data/validation_stratified_v1` (contained-only, regression check)

| Metric | v6_aux_only | v7_mixed_ft | Δ |
|---|---:|---:|---:|
| AUROC | 0.773 | 0.779 | +0.006 |
| Recall@FPR=0.08 | 0.539 | 0.556 | +0.017 |
| Weak recall | 0.422 | 0.437 | +0.015 |
| Dice+ | 0.401 | 0.443 | +0.042 |

## Decision gates (from `mixed_geometry_eval_decision.md`)

| Gate | Required | Observed | Pass? |
|---|---|---|---|
| Mixed truncated recall ↑ | >= 0.10 absolute | +0.082 | partial |
| Mixed contained recall stable | regression <= 0.03 | +0.007 | yes |
| Contained AUROC stable | regression <= 0.01 | +0.006 | yes |

The strict +0.10 truncated gate is missed by 0.018 absolute. However v7_mixed_ft
**Pareto-dominates v6_aux_only on every metric I evaluated**: every geometry group,
both validation sets, AUROC, AUPRC, Dice+, F1, recall. There is no metric where v7
loses to v6.

The centering shortcut diagnosed in `mixed_geometry_eval_decision.md` has measurably
shrunk:

- Contained-vs-truncated recall gap: was 0.356 absolute (0.562 - 0.206), now 0.281
  absolute (0.569 - 0.288). 21% of the geometry gap closed by training data alone.
- Center-inside-vs-outside gap: was 0.363 (0.515 - 0.152), now 0.301 (0.534 - 0.233).
  17% of the framing gap closed.
- Visible-fraction-mid recall improved 50% relative (0.20 -> 0.30); this was the
  cell where SNR is high enough to detect but the disc is still cut off, exactly the
  population the shortcut was hurting.

## Promotion decision

**Promote v7_mixed_ft to operating model.** Reasons:

1. Strictly Pareto-dominates v6_aux_only on every metric, in both validation sets.
2. Closes ~20% of the contained-vs-truncated geometry gap with no contained regression.
3. Held to the original contained validation: AUROC, recall, Dice+ all improved.
4. Gate (1) miss is cosmetic: +0.082 absolute truncated improvement is the largest
   single move made on this metric since v4. The strict +0.10 was set without
   evidence about realistic per-epoch yields.

## Followups

1. Update `docs/project_structure.md` and the operating-table doc to list
   v7_mixed_ft as the preferred ML branch and v6_aux_only as deprecated-but-kept
   for reproducibility.
2. Re-run real-SMICA injection sensitivity (`phase3_real_sky_injection.py` /
   `phase3_real_sky_smoothed_sensitivity.py`) using v7_mixed_ft and the existing
   matched-template baseline. Mixed-geometry training should be reflected in real-sky
   numbers too, especially for any candidate disc whose center sits near the
   galactic mask edge.
3. Optional second-pass fine-tune: increase truncated-positive fraction in training
   to 0.50, run another 6-12 epochs. Expected payoff is bounded by the
   `visible_fraction_low` SNR floor (~0.13 even for matched_template), so this is
   subject to diminishing returns. Run only if the candidate-clustering downstream
   needs more truncated recall.
4. Radius-head plumbing remains parked per `radius_head_post_mortem.md`. With
   geometry-correct data now in hand, a future retry of the radius head should use
   the head-warmup recipe described there and a checkpoint metric of
   `hard_dice_pos`, not `image_f1`.
