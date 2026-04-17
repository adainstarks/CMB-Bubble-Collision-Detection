# Batch 4: Router Feature Expansion (Truth-Free Geometry Proxies)

## TL;DR

Adding 4 truth-free geometry proxies per model (8 new features total) to the
PR #8 learned-GBT router lifts recall **+0.043 on mixed geometry at
FPR 0.08** (0.365 → 0.408), clearing the pre-registered +0.01 ship gate
by more than 4x. The gain is robust across 4 null-split seeds (mean
+0.032, min +0.021, all pass gate) and is largest on the hardest
geometry subgroups: `geometry_truncated` +0.060, `center_outside_patch`
+0.058, `visible_fraction_low` +0.056.

Gate: **PASS**. Ship as Batch 4.

New features (4 per model, computed from `P = sigmoid(mask_logits)`):

- `mask_area_at_0.5`: fraction of pixels with `P >= 0.5`.
- `centroid_offset_px`: Euclidean distance from probability-weighted
  centroid to patch center.
- `compactness`: perimeter / sqrt(area) on the thresholded binary mask.
  Disc-like masks give ~3.5 (2√π), scattered/fragmented masks give
  10-40+.
- `edge_touching_fraction`: fraction of mask-at-0.5 pixels within ≤4 px
  of the patch boundary.

None of these use injection truth. All derive from the frozen U-Net
probability mask alone. They are legal to deploy in Phase 5 on
real-sky patches.

Harness: `scripts/phase3_postprocess_ablation.py` (extended to compute
and cache geometry features) + `scripts/phase3_geometry_router.py`
(new `--feature-set {scores_only, all}` flag).

Artifacts: `runs/phase3_unet/batch4_router_features_v1/`.

## Headline results

`learned_gbt` policy under the exact PR #8 honest-eval protocol
(cross-geometry positive training, 2500/2500 disjoint null split at
fixed seed 20260417, threshold calibrated on held-out null half):

| geometry | FPR | `v6_only` | `gbt_6` | `gbt_14` | Δ vs `gbt_6` | Δ vs `v6_only` |
|---|---:|---:|---:|---:|---:|---:|
| contained | 0.05 | 0.348 | 0.359 | **0.388** | **+0.029** | +0.040 |
| contained | 0.08 | 0.372 | 0.403 | **0.434** | **+0.031** | +0.063 |
| contained | 0.10 | 0.389 | 0.431 | **0.466** | **+0.035** | +0.077 |
| mixed | 0.05 | 0.305 | 0.322 | **0.352** | **+0.030** | +0.047 |
| mixed | 0.08 | 0.331 | 0.365 | **0.408** | **+0.043** | +0.077 |
| mixed | 0.10 | 0.347 | 0.392 | **0.438** | **+0.046** | +0.091 |

Every (geometry, FPR) cell improves. Deltas are consistent in sign and
magnitude, which is the fingerprint of a real feature gain rather than
a lucky threshold jitter.

## Per-geometry-group breakdown on mixed at FPR 0.08

| group | n | `gbt_6` | `gbt_14` | Δ |
|---|---:|---:|---:|---:|
| all_positive | 17500 | 0.365 | **0.408** | +0.043 |
| geometry_contained | 12586 | 0.409 | **0.445** | +0.036 |
| geometry_truncated | 4914 | 0.253 | **0.313** | **+0.060** |
| center_inside_patch | 14875 | 0.393 | **0.433** | +0.040 |
| center_outside_patch | 2625 | 0.207 | **0.265** | **+0.058** |
| visible_fraction_low | 1708 | 0.191 | **0.246** | **+0.056** |
| visible_fraction_mid | 2100 | 0.281 | **0.340** | +0.059 |
| visible_fraction_high | 13692 | 0.400 | **0.439** | +0.039 |

The geometry features do exactly what the hypothesis said they would.
The truncated, center-outside, and low-visible-fraction groups — where
the probability mask is most spatially non-trivial and where the 6
scalar scores lose the most information — pick up +5 to +6 points of
recall. That is physically sensible: for a partially-clipped bubble
disc, the mask shape (edge-touching, elongated centroid, non-disc-like
compactness) carries information that `max(mask)` and disc-kernel MF
do not.

## Seed sensitivity

Same data, same protocol, four independent null-split seeds. `mixed`
recall at FPR 0.08:

| seed | `gbt_6` | `gbt_14` | Δ |
|---:|---:|---:|---:|
| 20260417 (default) | 0.365 | 0.408 | +0.043 |
| 111 | 0.370 | 0.403 | +0.034 |
| 222 | 0.349 | 0.370 | +0.021 |
| 333 | 0.382 | 0.411 | +0.028 |

Mean Δ = **+0.032**, min **+0.021**, max **+0.043**, stdev 0.009.
Every seed clears the +0.010 gate. The worst seed (222) still gives
2.1x the required delta, so the result is not a lucky draw on one null
split.

## Feature importances (14-feature GBT)

Default seed, mixed geometry:

| feature | importance | |
|---|---:|---|
| v6_baseline | 0.327 | primary score (down from 0.437 in the 6-feature GBT — not because v6 matters less but because variance is now distributed across 14 features) |
| v7_mf_on_mask | 0.089 | disc-coherence signal on v7 mask |
| v6_smooth_multi | 0.074 | v6 scale consistency |
| v7_centroid_offset | 0.067 | **new geometry feature** |
| v6_centroid_offset | 0.066 | **new geometry feature** |
| v7_edge_touching | 0.061 | **new geometry feature** |
| v7_mask_area | 0.056 | **new geometry feature** |
| v7_smooth_multi | 0.051 | |
| v6_mf_on_mask | 0.049 | |
| v6_mask_area | 0.046 | **new geometry feature** |
| v7_baseline | 0.036 | |
| v6_compactness | 0.033 | **new geometry feature** |
| v7_compactness | 0.031 | **new geometry feature** |
| v6_edge_touching | 0.013 | **new geometry feature** |

The 8 new geometry features collectively account for **~37%** of the
GBT's used signal on mixed geometry (sum of the new rows above). The
four most informative single additions are the centroid offsets (both
models, ~6.6% each) and v7's edge-touching fraction (6.1%). That
pattern is consistent with "v7's fine-tuned spatial sensitivity picks
up which candidates are edge-crossing, and the centroid offset tells
the router whether the peak lives near the patch center or near the
boundary."

Contained-geometry GBT importances show a similar story: centroid
offsets are the top two new features, then mask_area and edge_touching.
Compactness is the weakest new feature on both geometries — useful but
below baseline scores.

## `learned_logistic` behaviour (for completeness)

| geometry | FPR | log_6 | log_14 | Δ |
|---|---:|---:|---:|---:|
| contained | 0.05 | 0.348 | 0.361 | +0.013 |
| contained | 0.08 | 0.374 | 0.384 | +0.010 |
| contained | 0.10 | 0.397 | 0.403 | +0.006 |
| mixed | 0.05 | 0.308 | 0.325 | +0.017 |
| mixed | 0.08 | 0.335 | 0.348 | +0.013 |
| mixed | 0.10 | 0.357 | 0.367 | +0.010 |

`learned_logistic` gets smaller gains (+0.013 on the gate cell vs
+0.043 for GBT). The optimal decision boundary on the 14 features
is non-linear, same as it was on 6 features in Batch 3. GBT remains
the right model class for this router.

## Why this works (mechanism, not just numbers)

The 6-feature GBT had only model-score information. It could combine
v6 and v7 baselines and their two post-processing transforms, but it
could not ask "where is the peak?" or "is the mask disc-shaped or
scattered?" The 4 new features provide exactly that complementary
information at zero additional inference cost.

Specifically:

1. `centroid_offset_px` separates central-disc positives from
   edge-peak positives. A centered disc has offset ≈ 0; a truncated
   disc with center outside the patch has offset of 80-120 px. The
   router uses this to reweight the v6 vs v7 scores per-candidate.
2. `edge_touching_fraction` is the learned-router's version of the
   `mf_on_mask - baseline` heuristic from Batch 2, but on the raw
   mask-at-0.5 footprint, not on a disc-kernel response. It tells
   the GBT explicitly "this mask hugs the patch boundary", letting it
   up-weight the v7 score specifically on truncated candidates.
3. `mask_area_at_0.5` gives an implicit scale signal. Weak-signal
   false positives often show a single high-prob pixel (tiny area);
   real truncated discs show an elongated patch against the boundary
   (larger, edge-biased area). The GBT uses mask_area jointly with
   baseline to separate those cases.
4. `compactness` is the weakest new feature but still useful. Very
   disc-like masks (low compactness, ~3.5) get a boost over scattered
   noise masks (high compactness, 20+).

None of these are correlated 1:1 with the truth labels in training.
They are frozen-mask geometry derived at inference time from the
probability image, computable on real-sky patches without any
knowledge of injection geometry. Legal for Phase 5 deployment.

## What this closes

- Section 25 "expand learned-router feature set" entry. Measured
  positive.
- `scripts/phase3_postprocess_ablation.py` transform caches are now
  the source of 14 per-patch features. PR #8's 6-feature variant
  remains available via `--feature-set scores_only` for the published
  comparator.

## What still remains

- **v8 retrain** with matched-filter response channel on mixed
  geometry. Still the only untouched training-signal lever. Expected
  +4-10pp truncated recall on top of the Batch 4 router gain. Smoke
  test mandatory before the 6-10 hour run.
- **Isotonic score calibration** on real-SMICA nulls. Threshold
  hygiene for paper candidate-volume statistics.
- **Matched-filter response channel as a geometry feature** (rather
  than an input channel). Could be evaluated in a Batch 5 router
  expansion if v8 retrain does not happen soon.

## Deployment advice (current, revised from Section 22)

- **Primary policy: `learned_gbt` with the 14-feature set.** Same
  200-tree, depth-3, LR 0.05 GBT architecture as PR #8. Beats the
  6-feature GBT by +0.029 to +0.046 recall at every (geometry, FPR)
  cell. Real-SMICA threshold depends on the null-split seed; the
  reproducible operating points are reported in the artifact JSON.
- **Single-model fallback: `v6_aux_only` @ 0.873 (FPR 0.08).**
  Unchanged from PR #8.
- **`v7_mixed_ft`**: unchanged — router input + Phase 5 truncated
  specialist, not a parallel screener.
- **`matched_template`**: unchanged.

## Artifacts

- `runs/phase3_unet/batch4_router_features_v1/batch3_router_report.json`
  — `scores_only` feature set (6 features, PR #8 baseline on these
  caches).
- `runs/phase3_unet/batch4_router_features_v1/batch3_router_report_all.json`
  — `all` feature set (14 features, shipped).
- `runs/phase3_unet/batch4_router_features_v1/batch3_router_report.md`,
  `batch3_router_report_all.md` — markdown summaries.
- `runs/phase3_unet/batch2_postprocess_ablation_v1/score_cache/*.npz`
  — regenerated caches now include the 4 geometry features per model.
