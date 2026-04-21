# Phase 3 Ablation Summary (2026-04-21)

Assumptions
-----------
- These are fixed-FPR sensitivity-grid screening comparisons against the
  current `remediated_v1` ImageNet baseline, not Bayesian evidence statements.
- Baseline method: `imagenet_b64_aux` from
  `runs/phase3_unet/remediated_v1_sensitivity_curve/sensitivity_report.json`.
- Comparison artifact:
  `runs/phase3_unet/remediated_v1_ablation_compare/sensitivity_method_compare.{json,md}`.

## Closed ablations

### 1. SNR-guided hard-example weighting

Run:
- `runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux_snr_gap_ft/`
- Sensitivity eval:
  `runs/phase3_unet/remediated_v1_snr_gap_sensitivity_eval/`

Result:
- Overall mean `P_det` delta vs baseline: `-0.00068`.
- Hard-cell mean delta (`A <= 5e-6`): `-0.00375`.
- Moderate/high-amplitude mean delta (`A >= 1e-5`): `+0.00163`.
- Large-radius mean delta (`theta >= 15 deg`): `-0.00208`.

Interpretation:
- The weighting scheme did not produce a robust recall gain.
- It helps a few moderate-amplitude cells, but the aggregate effect is slightly
  negative and the low-amplitude regime remains worse than baseline.

### 2. Focal loss (`gamma = 2`, `alpha` disabled)

Run:
- `runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux_focal_g2_lr1e4_ft/`
- Sensitivity eval:
  `runs/phase3_unet/remediated_v1_focal_g2_sensitivity_eval/`

Result:
- Overall mean `P_det` delta vs baseline: `+0.00071`.
- Hard-cell mean delta (`A <= 5e-6`): `-0.00225`.
- Moderate/high-amplitude mean delta (`A >= 1e-5`): `+0.00294`.
- Large-radius mean delta (`theta >= 15 deg`): `-0.00119`.

Interpretation:
- Focal loss is the better of the two completed ablations, but the effect is
  small.
- The gain is concentrated in moderate/high-amplitude cells; it does not solve
  the lowest-amplitude recall failure that motivated the ablation.

## Decision

- Keep focal loss as a plausible secondary branch worth citing as a mild
  improvement, not as a core scientific fix.
- Do not promote the current SNR-guided reweighting schedule into the baseline.
- The next high-value test remains the true Wiener extra-channel two-stream
  model, because the completed loss/sampling ablations did not materially
  change the hard-signal regime.
