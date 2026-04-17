# Batch 1: Real-SMICA Validation Gate, v7_mixed_ft vs v6_aux_only

## TL;DR

**v7 does not dominate v6 on real SMICA.** On synthetic CAMB backgrounds, the
earlier benchmark had v7 Pareto-dominating v6. On real SMICA, the story
reverses for contained/high-SNR geometry but confirms v7's lead on truncated
and edge-crossing geometry. The correct next move is a two-model portfolio, not
a single-model promotion.

## Setup

- Gate harness: `scripts/phase3_real_sky_v7_gate.py` (new).
- Injection dataset: 500 unmasked SMICA patches, 7 amplitudes × 5 θ × 4
  sign-quadrants = 17500 positives per geometry mode.
- Null source: `data/training_v4/smica_null_controls_all.h5` (5000 SMICA
  background patches, untouched across the gate).
- Thresholds calibrated to actual SMICA null FPR at 0.05, 0.08, 0.10.
- Two geometries evaluated:
  - Contained (default Feeney 2011-style).
  - Mixed (30% truncated, visible fraction 0.15–0.95).

## Global recall on real SMICA

### Contained geometry

| FPR | v7 recall | v6 recall | delta |
|---:|---:|---:|---:|
| 0.05 | 0.286 | 0.348 | **−0.062** |
| 0.08 | 0.357 | 0.372 | −0.014 |
| 0.10 | 0.386 | 0.389 | −0.003 |

### Mixed geometry (30% truncated)

| FPR | v7 recall | v6 recall | delta |
|---:|---:|---:|---:|
| 0.05 | 0.248 | 0.305 | **−0.057** |
| 0.08 | 0.328 | 0.331 | −0.003 |
| 0.10 | 0.355 | 0.347 | +0.008 |

## Geometry-conditional recall on mixed-geometry SMICA at FPR 0.08

| group | v7 recall | v6 recall | delta | n |
|---|---:|---:|---:|---:|
| all_positive | 0.328 | 0.331 | −0.003 | 17500 |
| geometry_contained | 0.360 | 0.380 | −0.020 | 12586 |
| geometry_truncated | 0.246 | **0.205** | **+0.041** | 4914 |
| center_inside_patch | 0.349 | 0.360 | −0.011 | 14875 |
| center_outside_patch | 0.207 | **0.163** | **+0.044** | 2625 |
| visible_fraction_low | 0.196 | **0.145** | **+0.051** | 1708 |
| visible_fraction_mid | 0.265 | 0.235 | +0.030 | 2100 |
| visible_fraction_high | 0.354 | 0.369 | −0.015 | 13692 |

## Interpretation

### v6 wins contained geometry at FPR ≤ 0.08.

Look at the contained mid-SNR regime, which is where most physically plausible
bubble signatures live:

| A | θ (deg) | v7 | v6 | v6 advantage |
|---:|---:|---:|---:|---:|
| 2e-5 | 15 | 0.274 | 0.400 | +46% relative |
| 2e-5 | 20 | 0.504 | 0.620 | +23% relative |
| 2e-5 | 25 | 0.658 | 0.780 | +19% relative |
| 5e-5 |  5 | 0.374 | 0.400 | +7% relative |

v7's mixed-geometry fine-tune trades away some contained-geometry sharpness on
real backgrounds. The synthetic benchmark did not expose this because CAMB
backgrounds are cleaner than SMICA foreground residuals.

### v7 wins truncated / edge-crossing geometry, as intended.

Truncated recall climbs from 0.205 (v6) to 0.246 (v7), a +20% relative gain on
4914 positives. The "center outside patch" subgroup, which v6 was never trained
to see, climbs from 0.163 to 0.207 (+27% relative). These are the geometries
v6_aux_only literally cannot solve; v7 is the only path here.

### v7 also wins at very high amplitude.

At A = 1e-4, v7 > v6 across all θ, both geometries. This is a minor point for
the paper (these signals are already well above any physical threshold) but
confirms v7's pure-template pattern matching is fine; the regression is
specifically on mid-SNR contained patterns where the real SMICA foreground
complexity interferes.

## Decision

Adopt a two-model portfolio for Phase 5 deployment:

1. **v6_aux_only** as the contained-geometry screener.
2. **v7_mixed_ft** as the truncated / edge-geometry screener.
3. Run both on every patch at Phase 5 tiling time. Route candidates by whether
   the bubble center is estimated to be inside the patch.
4. Combine via OR for initial candidate generation, then use geometry
   classification to assign the authoritative score per candidate.

This is not a v7 promotion and not a v7 retirement. It is the correct response
to a documented domain-specific tradeoff.

## Implications for Batch 2

Batch 2 (probability-mask smoothing, matched-filter rescoring, per-θ thresholds,
isotonic calibration) applies to **both** models. The ablation harness must run
the full matrix for v6_aux_only and v7_mixed_ft separately. Expected ceiling
targets:

- v6 contained recall at FPR 0.08 today: 0.372. Post-Batch-2 target: 0.45–0.55.
- v7 truncated recall at FPR 0.08 today: 0.246. Post-Batch-2 target: 0.33–0.40.

If Batch 2 closes the contained-geometry gap (i.e., v7+smoothing matches
v6_aux_only on contained), then v7 becomes the sole deployment model and the
portfolio collapses to single-model. If not, the portfolio stays.

## Artifacts

- `runs/phase3_unet/real_sky_v7_gate_v1/contained/v7_vs_v6_contained_report.{md,json}`
- `runs/phase3_unet/real_sky_v7_gate_v1/mixed/v7_vs_v6_mixed_report.{md,json}`
- `runs/phase3_unet/real_sky_v7_gate_v1/{contained,mixed}/smica_real_sky_injection_*.h5`
- `runs/phase3_unet/real_sky_v7_gate_v1/{contained,mixed}/score_cache/*.npz`
