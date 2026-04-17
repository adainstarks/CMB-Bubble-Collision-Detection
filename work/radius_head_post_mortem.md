# Radius-Head Branch Post-Mortem (phase3_scale_radius_head_w02)

Status: failed as trained. Plumbing kept in code with default `--radius-head-weight 0.0`.
This note documents *why* it failed, so we do not waste another run on the same mistake.

## Setup recap

- Resumed from `phase3_v6_aux_only_w4/best_checkpoint.pt` via `--model-only-resume`.
- Added radius-bin cross-entropy head over theta_crit with bin edges `[5,10,15,20,25]` (4 bins).
- `--radius-head-weight 0.2`, `--aux-head-weight 0.1`, `--boundary-weight 4.0`, `--boundary-width-pixels 5`.
- Optimizer: cosine schedule, LR `5e-5`, weight decay `1e-4`, 8 epochs, batch 64, 2x3090.
- Loss scale at start: `bce ~1.41`, `dice ~0.77`, `radius ~1.63` (~log 4, i.e. near-uniform).
- Checkpoint-metric: `image_f1`.

## Training history (from history.json)

| ep | lr | tr_dice_loss | tr_rad | tr_rAcc | val_dice_loss | val_rad | val_rAcc | val_image_f1 | val_hard_dice_pos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.0e-05 | 0.766 | 1.630 | 0.314 | 0.735 | 1.856 | 0.324 | 0.637 | **0.016** |
| 2 | 4.8e-05 | 0.722 | 1.468 | 0.364 | 0.726 | 1.891 | 0.336 | 0.428 | 0.014 |
| 3 | 4.3e-05 | 0.715 | 1.400 | 0.386 | 0.719 | 1.778 | 0.344 | 0.200 | 0.009 |
| 4 | 3.5e-05 | 0.711 | 1.360 | 0.398 | 0.713 | 1.619 | 0.374 | 0.447 | 0.064 |
| 5 | 2.5e-05 | 0.709 | 1.329 | 0.416 | 0.718 | 1.524 | 0.396 | 0.459 | 0.089 |
| 6 | 1.5e-05 | 0.708 | 1.315 | 0.421 | 0.715 | 1.416 | 0.378 | 0.529 | 0.157 |
| 7 | 7.3e-06 | 0.707 | 1.282 | 0.423 | 0.717 | 1.445 | 0.376 | 0.488 | 0.124 |
| 8 | 1.9e-06 | 0.707 | 1.295 | 0.427 | 0.714 | 1.386 | 0.386 | 0.535 | 0.156 |

## Diagnosis

The smoking gun is epoch 1: resume point was `v6_aux_only`'s best (stratified-external `Dice+ = 0.401`, `recall = 0.539`, `AUROC = 0.773`), and after a single epoch of training with the radius head attached, `val_hard_dice_pos` collapsed from roughly `0.40` down to `0.016`. That is a 25x segmentation regression in one epoch.

Mechanism. Three compounding causes, not one:

1. Radius head is cold-started (random weights). `tr_rad_accuracy = 0.31` and `tr_radius_loss = 1.63` at the first epoch match a near-uniform softmax over 4 bins (`log(4) ~= 1.386`). The gradient from that head is essentially random noise propagating backward into the encoder.

2. `--radius-head-weight 0.2` is large for a cold-started head. Effective contribution `0.2 * 1.63 ~= 0.33` is roughly comparable to `dice_loss` itself. Combined with (1), that means nontrivial random gradient magnitude was flowing into the backbone from the very first step.

3. No head-warmup discipline. The run used `--model-only-resume` + uniform LR (`5e-5`) for the whole model, including the cold-started head. Standard practice when adding a new classification head to a pretrained backbone is either:
   - Freeze the backbone for 1-2 epochs while the head warm-starts, or
   - Use a head-specific LR 10-100x larger than the backbone LR, or
   - Initialize the head so its initial loss contribution is tiny.

None of that was done.

4. Checkpoint-metric choice masked the damage. `image_f1` peaked at `0.535` (epoch 8) while `FPR = 0.482` and `image_precision = 0.589`. The selected "best" model therefore had nearly 50% false-positive rate on validation and was still selected because the F1 metric was permissive of that tradeoff. `hard_dice_pos` would have been a much more honest checkpoint metric. The external stratified gate used the standard FPR-matched operating point and correctly reported the collapse (`Dice+ 0.006`, `recall 0.302`).

## What this rules out

- The radius-head collapse is *not* a sign that scale-awareness is intrinsically bad for this task.
- The radius-head collapse is *not* caused by the Feeney signal model or by training-data shortcuts.
- The radius-head collapse is a straightforward training-hygiene failure: cold head + nontrivial weight + no warmup + permissive checkpoint metric.

## What this rules in (only if scale-awareness is re-attempted later)

Gate this on the mixed-geometry evaluation first. Do not retry unless that evaluation shows a scale-specific failure cell that the data correction does not fix. If and only if that happens, the minimum viable retry is:

1. Freeze `encoder` and `decoder` parameters for 1-2 epochs, training only the new `radius_classifier` head plus the existing `aux_classifier` head.
2. Unfreeze everything with `--radius-head-weight 0.05` (one fifth of the failed weight).
3. Encoder/decoder LR `1e-5`, classifier-head LR `3e-4` (separate parameter group).
4. `--checkpoint-metric hard_dice_pos` (not `image_f1`).
5. Rerun the stratified external gate.

If that still regresses, the scale-awareness hypothesis is dead and the plumbing should be removed for clarity.

## Recommendation right now

Leave the plumbing in, default-disabled. Move on to the mixed-geometry evaluation. Do not retry the radius head until after the geometry-correct data has been evaluated and, if necessary, used for training.
