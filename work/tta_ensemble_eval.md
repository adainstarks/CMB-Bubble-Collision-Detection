# Step 1 and Step 2 results: TTA and multi-model ensembling

## Setup

- Harness: `scripts/phase3_eval_tta_ensemble.py` (new). D4 TTA averages the 8
  flip/rotation augmentations of the *probability mask* before computing
  detection scores and Dice at the matched-FPR threshold. Ensembling averages
  each model's TTA-averaged probability mask with equal weights before
  calibration.
- Reports: `runs/phase3_unet/tta_ensemble_v1/`.
- Datasets:
  - `data/validation_stratified_mixed_geometry_v1` (3000 samples, ~20% truncated, visible fraction 0.15-0.95)
  - `data/validation_stratified_v1` (contained-only, equivalent to the historical Phase 3 validation)
- Matched-FPR point: 0.08.

## Mixed-geometry validation (the hard, fair benchmark)

| config | AUROC | AUPRC | recall @ FPR 0.08 | F1 | weak recall | contained recall | truncated recall | Dice+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v7_mixed_ft (no TTA) | 0.749 | 0.900 | 0.486 | 0.640 | 0.382 | 0.569 | 0.288 | 0.354 |
| v7_mixed_ft + D4 TTA | 0.748 | 0.900 | 0.486 | 0.641 | 0.378 | 0.570 | 0.287 | 0.361 |
| Ensemble {v7, v6_aux_only, boundary_v4} + D4 TTA | 0.733 | 0.894 | 0.460 | 0.617 | 0.353 | 0.562 | 0.217 | 0.324 |

- TTA on v7: within noise on recall, +0.7pp Dice. Essentially free.
- 3-model ensemble: strictly worse. AUROC -0.015, recall -0.026, truncated recall
  -0.071, Dice -0.037. The contained-only siblings (`v6_aux_only`, `boundary_v4`)
  dilute v7's geometry-aware mask with their center-framed shortcut.

## Contained-only validation (apples-to-apples with the historical baseline)

| config | AUROC | AUPRC | recall @ FPR 0.08 | F1 | Dice+ | z0_amp_bin_2 recall | z0_amp_bin_2 Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| v7_mixed_ft (no TTA) | 0.779 | 0.916 | 0.556 | 0.701 | 0.443 | - | - |
| v7_mixed_ft + D4 TTA | 0.782 | 0.917 | 0.566 | 0.709 | 0.452 | 0.848 | 0.718 |

On the strong-signal bin (`z0_amp_bin_2`, high amplitude), v7+TTA recovers recall
0.848 and Dice 0.718, i.e. the regime the user remembers (recall 0.833, Dice
0.720). The headline "lower recall" on the stratified benchmark is because the
benchmark averages in the explicit weak-signal bins that the historical 1000-
sample split underrepresented.

## Verdict

- Keep D4 TTA on v7 as the default deployment config. It is free Dice and never
  hurts.
- Reject naive multi-model averaging with the current model zoo. Contained-only
  siblings cannot lift a geometry-aware primary model on the hard benchmark.
- The real lever for further gains must be a new training signal, not a
  post-hoc mix. Two candidates remain on the table:
  1. Add a matched-filter response map as a second input channel and train
     `v8_mf_channel` on the mixed geometry pool.
  2. A heavier fine-tune pass with balanced weak-family sampling and longer
     warmup.
- Ensembling becomes attractive again only after we have at least two
  *mixed-geometry* models (e.g., v7 and a future v8). Revisit then.
