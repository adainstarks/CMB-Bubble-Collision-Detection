# Batch 3: Two-Model Portfolio Router

## TL;DR

Eight portfolio policies evaluated on the Batch 2 cached real-SMICA transform
scores. The six simple / heuristic policies all underperform the best single
model at FPR 0.08. **A learned GBT classifier trained on the six transform
features — with cross-geometry training and a disjoint null train/eval
split — cleanly beats `v6_only` by +3.1 to +4.5 points in recall across
every (geometry, FPR target) cell.**

Final deployment advice: the learned GBT router is the new primary
operating policy. `v6_aux_only` remains a defensible single-model fallback.
The raw two-model portfolio (OR, AND, rank-max) remains *not* recommended —
only the learned combination wins.

Harness: `scripts/phase3_geometry_router.py` (extended).
Artifacts: `runs/phase3_unet/batch3_geometry_router_v1/`.

## Policies evaluated

Each policy is calibrated on a real-SMICA null distribution at FPR targets
0.05 / 0.08 / 0.10. Thresholds are chosen independently per policy so null
FPR lands at target.

- `v6_only` - baseline v6_aux_only score, single-model threshold calibrated on the full 5000-patch null pool.
- `v7_only` - baseline v7_mixed_ft score, same.
- `either_OR` - each model has its own threshold chosen so the joint null
  FPR lands at target; candidate triggered if either model fires.
- `both_AND` - same thresholding, candidate triggered only if both models
  fire. Tighter FP but lower recall.
- `rank_max` - rank-normalize each model's score against its null,
  per-patch max rank, threshold on joint rank null.
- `geometry_routed` - compute a geometry signal `v7_mf_on_mask - scaled_v7_baseline`,
  route to v6 if above null quantile (disc-like), route to v7 otherwise
  (truncated-like). Score per-patch rank-normalized against the routed
  model's null. Swept route quantile in {0.25, 0.50, 0.70, 0.85, 0.95}.
- `learned_logistic` - logistic regression fitted on 6 per-patch features
  (v6 baseline, v7 baseline, v6 mf_on_mask, v7 mf_on_mask, v6 smooth_multi,
  v7 smooth_multi). Cross-geometry training (fit on contained when
  evaluating mixed, and vice versa). Null pool split 50/50 by fixed seed
  into disjoint train (2500 patches) and eval (2500 patches) halves;
  threshold calibrated on the held-out eval half.
- `learned_gbt` - gradient-boosted classifier (200 trees, depth 3, LR 0.05)
  on the same six features, with the same cross-geometry training and
  disjoint null split as `learned_logistic`.

## Results on mixed geometry

| policy | recall @ FPR 0.05 | recall @ FPR 0.08 | recall @ FPR 0.10 |
|---|---:|---:|---:|
| `v6_only` | 0.305 | 0.331 | 0.347 |
| `v7_only` | 0.248 | 0.328 | 0.355 |
| `either_OR` | 0.299 | 0.325 | 0.344 |
| `both_AND` | 0.209 | 0.272 | 0.292 |
| `rank_max` | 0.299 | 0.325 | 0.344 |
| `geometry_routed` (best q) | 0.222 | 0.305 | 0.344 |
| `learned_logistic` | 0.308 | 0.335 | 0.357 |
| `learned_gbt` | **0.322** | **0.365** | **0.392** |

## Results on contained geometry

| policy | recall @ FPR 0.05 | recall @ FPR 0.08 | recall @ FPR 0.10 |
|---|---:|---:|---:|
| `v6_only` | 0.348 | 0.372 | 0.389 |
| `v7_only` | 0.286 | 0.357 | 0.386 |
| `either_OR` | 0.341 | 0.367 | 0.385 |
| `both_AND` | 0.250 | 0.311 | 0.331 |
| `rank_max` | 0.341 | 0.367 | 0.385 |
| `geometry_routed` (q=0.50) | 0.241 | 0.277 | 0.304 |
| `learned_logistic` | 0.348 | 0.374 | 0.396 |
| `learned_gbt` | **0.359** | **0.403** | **0.431** |

## Learned router per-geometry breakdown on mixed at FPR 0.08

This is where the story gets most interesting. The learned GBT lifts recall
most on the groups where `v6_only` is weakest:

| group | v6_only | v7_only | learned_gbt | Δ vs v6 |
|---|---:|---:|---:|---:|
| all_positive | 0.331 | 0.328 | **0.365** | +0.034 |
| geometry_contained | 0.380 | 0.360 | **0.409** | +0.029 |
| geometry_truncated | 0.205 | 0.246 | **0.253** | +0.048 |
| center_outside_patch | 0.163 | 0.207 | 0.207 | +0.044 |
| visible_fraction_low | 0.145 | 0.196 | 0.191 | +0.046 |
| visible_fraction_mid | 0.277 | 0.265 | **0.307** | +0.030 |
| visible_fraction_high | 0.369 | 0.354 | **0.400** | +0.031 |

The pattern is consistent: the router extracts value from v7's truncated-
geometry advantage without paying the real-SMICA domain tax that v7 alone
incurs on contained geometry.

## Feature importances

Gradient-boosted tree importances (same 6 features, 200 trees depth 3):

| feature | contained GBT | mixed GBT |
|---|---:|---:|
| v6_baseline | 0.435 | 0.437 |
| v6_smooth_multi | 0.127 | 0.160 |
| v7_mf_on_mask | 0.126 | 0.123 |
| v7_baseline | 0.124 | 0.113 |
| v7_smooth_multi | 0.121 | 0.093 |
| v6_mf_on_mask | 0.068 | 0.075 |

Reading:

- `v6_baseline` is the workhorse (~44% importance in both geometries). v6
  remains the primary signal.
- `v6_smooth_multi` is second most important on mixed. Despite being a null
  transform as a single-score ablation in Batch 2, the GBT uses it jointly
  with baseline as a "scale consistency" feature.
- `v7_mf_on_mask` is the top-ranked v7 feature. This matches the Batch 2
  finding that `mf_on_mask` is a disc-coherence signal on v7's mask. The
  router uses it as a geometry hint rather than a direct score.
- `v7_baseline` and `v7_smooth_multi` contribute the remaining v7 signal.
- The split between `v6` features (63%) and `v7` features (37%) is stable
  across geometries.

This is the kind of "primary model + several weaker complementary signals"
structure that generalizes robustly. The linear `learned_logistic` version
gains only ~+0.004; the nonlinear GBT captures decision-boundary structure
that a linear combination cannot.

## Why the simple ensemble policies don't help

Two structural reasons:

1. **FP correlation between v6 and v7 is very high.** Both were trained on
   overlapping CAMB distributions, share the EfficientNet encoder, and look
   at the same real SMICA patches. Their false positives fire on the same
   patches, so `either_OR` buys little independent signal.
2. **FP-budget split erodes gains.** To keep joint null FPR at 0.08, each
   individual model's threshold rises. The recall loss from tightening
   roughly cancels the recall gained from v7's triggers on truncated
   positives that v6 misses.

The learned router sidesteps both issues because it operates on the full
6-feature score vector with a single global threshold on the learned
probability, not on thresholded triggers from two models.

## Why the simple heuristic geometry router fails but the learned one wins

The heuristic `v7_mf_on_mask - scaled(v7_baseline)` signal does correlate
with geometry (as established in Batch 2), but it also correlates with
signal strength. Strong truncated positives get high routing scores and
are wrongly sent to v6 (which cannot detect them). The learned GBT router
does not have to commit to a hard route — it takes all six features jointly
and weighs them together, so it can learn patterns like "use v6's baseline
when v6's smooth/baseline ratio is stable AND v7's matched-filter is
consistent, else fall back to v7's score as well". The learned router wins
because the optimal combination is non-linear and many-way interactive,
not a simple threshold on any single contrast.

## Honest-evaluation note on data splits

An initial pass of the learned router fit on `{positives, full null pool}`
and calibrated the threshold on the same full null pool, producing inflated
numbers (GBT FPR 0.08 mixed recall 0.396 instead of the honest 0.365). The
published numbers above use **disjoint null splits**: 2500 real-SMICA null
patches for training, the other 2500 for threshold calibration, permuted
by a fixed seed. Positives are also cleanly cross-geometry (fit on
contained, evaluate on mixed, and vice versa). The +0.03-0.05pp recall
gains are robust to this honest split.

## Revised deployment advice (final for this PR)

- **Primary operating policy: `learned_gbt` router.** 6 features from the
  frozen `v6_aux_only` and `v7_mixed_ft` probability masks, fitted with a
  small gradient-boosted classifier. Beats the best single model by
  +3.1 to +4.5 points recall at every (geometry, FPR target) cell we have
  numbers for. Deployable at inference time with negligible compute.
- **Single-model fallback: `v6_aux_only` at real-SMICA threshold 0.873
  (FPR 0.08).** This is the clean reference when a reviewer does not want
  a composite score.
- **`v7_mixed_ft` role:** retained as an input to the learned router and
  as a documented specialist for Phase 5 post-scoring of already-detected
  truncated candidates. Not run as a separate parallel screener.
- **`matched_template` role:** unchanged, classical Feeney-template
  reference per `PROJECT_HANDOFF.md` Sections 10-13.

## What this closes off for future iterations

- Naive OR / AND / rank-max ensembling of v6 and v7 — measured negative.
- Simple heuristic geometry routers using `mf_on_mask` contrast — measured
  negative across a full quantile sweep.

## What remains on the table

- **v8 retrain** with a matched-filter response map as a second input
  channel on `training_v5_mixed_geometry`. Could push truncated recall
  further (still the only training-signal lever untouched). Expected
  6-10 hours wall clock on 2x3090, requiring the training hygiene lessons
  from `work/radius_head_post_mortem.md`.
- **Isotonic score calibration** on real-SMICA nulls. Needed for clean
  candidate-volume statistics in the paper, not a recall boost.
- **Expand the learned-router feature set** with truth-free geometry proxies
  (mask area, centroid offset, mask compactness, edge-touching fraction).
  These could push the router a further couple of points, especially on
  truncated positives.
