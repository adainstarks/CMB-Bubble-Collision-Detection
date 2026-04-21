# Phase 3 True-Wiener Two-Stream Summary (2026-04-21)

Assumptions
-----------
- This is a fixed-FPR synthetic sensitivity comparison against the current
  `remediated_v1` ImageNet baseline, not a Bayesian evidence statement.
- The comparison uses the corrected full sensitivity Wiener cache with
  `33000 / 33000` finite rows in
  `runs/phase3_unet/remediated_v1_sensitivity_curve/sensitivity_data.h5`.
- The model branch is the remediated two-channel fine-tune
  `runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux_true_wiener_ft_v2/`
  with input channels `patches` and `features/wiener_feeney_response`.

## Guardrail

- The first two-stream sensitivity comparison was invalid because only the
  first `12000` sensitivity rows had a populated Wiener auxiliary channel.
- That failure mode is now blocked in code:
  `scripts/phase3_train_unet.py` raises if an auxiliary input channel contains
  non-finite values.

## Corrected result

- Sensitivity report:
  `runs/phase3_unet/remediated_v1_true_wiener_sensitivity_eval_v2/sensitivity_report.json`
- Comparison report:
  `runs/phase3_unet/remediated_v1_ablation_compare_with_true_wiener_v2/sensitivity_method_compare.md`

Corrected `true_wiener_ft` versus `imagenet_b64_aux`:

| subset | baseline mean | true-Wiener mean | mean delta |
|---|---:|---:|---:|
| overall | `0.34875` | `0.36839` | `+0.01964` |
| hard (`A <= 5e-6`) | `0.05967` | `0.05375` | `-0.00592` |
| moderate (`A >= 1e-5`) | `0.56556` | `0.60437` | `+0.03881` |
| large-radius (`theta >= 15 deg`) | `0.39792` | `0.42018` | `+0.02226` |

Cell accounting:

- Improved cells: `18`
- Worsened cells: `13`

Largest gains:

- `A = 5e-5`, `theta = 5 deg`: `+0.124`
- `A = 2e-5`, `theta = 25 deg`: `+0.098`
- `A = 2e-5`, `theta = 10 deg`: `+0.093`
- `A = 1e-5`, `theta = 20 deg`: `+0.091`
- `A = 1e-5`, `theta = 25 deg`: `+0.085`

Largest losses:

- `A = 5e-6`, `theta = 10 deg`: `-0.024`
- `A = 1e-6`, `theta = 5 deg`: `-0.021`
- `A = 2e-6`, `theta = 10 deg`: `-0.018`
- `A = 2e-6`, `theta = 15 deg`: `-0.010`
- `A = 5e-6`, `theta = 5 deg`: `-0.009`

## Interpretation

- The true-Wiener auxiliary channel is a real improvement, but not a universal
  one.
- It helps in the regime where the branch was meant to help most plausibly:
  moderate/high amplitude and larger radii.
- It does not solve the hardest low-amplitude recall problem, so it should not
  replace the baseline blindly.
- The next defensible use is either:
  1. keep it as a secondary branch in the paper,
  2. test calibrated score fusion against the baseline and Wiener classical
     screener, or
  3. audit deployment burden before promoting it operationally.
