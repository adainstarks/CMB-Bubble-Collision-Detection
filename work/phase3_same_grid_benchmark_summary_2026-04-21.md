# Phase 3 Same-Grid Benchmark Summary

Date: 2026-04-21

## Assumptions

- This summary refers to the merged same-grid artifact in
  `runs/phase3_unet/remediated_v1_same_grid_fullsky_manifest/`.
- The benchmark is a fixed-manifest screening comparison, not a masked-sky
  Bayesian evidence calculation.
- Thresholds are calibrated on the matched negative split at realized
  `FPR = 0.0498` for every method.

## Closure Status

- `scripts/phase3_classical_same_grid_status.py` now reports `status=complete`.
- The merged benchmark report is
  `runs/phase3_unet/remediated_v1_same_grid_fullsky_manifest/same_grid_fullsky_report.json`.

## Main Outcome

Mean same-grid recall over the stratified `(A, theta_crit)` manifest:

- `wiener_feeney_matched_filter`: `0.3841`
- `imagenet_b64_aux`: `0.3517`
- `random_b64_aux`: `0.2994`
- `smhw_screen`: `0.2734`

Interpretation:

- The true Wiener/Feeney matched filter is the strongest **mean** screener on
  this fixed same-grid benchmark.
- The ImageNet U-Net is still competitive and remains cell-dependent rather
  than uniformly inferior.

## ImageNet Versus True Wiener

Raw cell-wise comparison over `35` amplitude/radius cells:

- ImageNet better in `17` cells.
- Wiener better in `16` cells.
- Tied in `2` cells.

Non-overlapping exact 95% CI comparison:

- ImageNet better in `7` cells.
- Wiener better in `8` cells.
- `20` cells overlap within uncertainty.

Representative gain/loss cells for ImageNet vs Wiener:

- best gain: `A=5e-5`, `theta=25 deg`, `0.760 -> 0.970` (`+0.210`)
- worst loss: `A=5e-5`, `theta=5 deg`, `0.760 -> 0.425` (`-0.335`)

## Scientific Framing Consequence

The same-grid closure does **not** support "ML beats the matched filter" as the
headline claim.

It **does** support the narrower statement:

> The ML screeners remain competitive with full-sky Wiener/SMHW screening on a
> matched injected benchmark and provide localized wins in selected
> amplitude/radius cells, but the Wiener/Feeney matched filter remains the
> strongest average classical screener on this benchmark.

That conclusion should replace any manuscript language implying uniform ML
superiority over the true Wiener baseline.
