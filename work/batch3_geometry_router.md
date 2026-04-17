# Batch 3: Two-Model Portfolio Router

## TL;DR

Six portfolio policies evaluated on the Batch 2 cached real-SMICA transform
scores. None beat the best single model at our deployment FPR target (0.08)
on either geometry. The "two-model portfolio" decision in PR #6 does not
survive the fair operating-point comparison this router enables.

**Revised deployment advice:** `v6_aux_only` alone is the best operating
policy at FPR 0.08 on both contained and mixed geometry. `v7_mixed_ft` is
retained as a documented specialist useful only at FPR >= 0.10 on truncated
positives. The two-model portfolio stays in the repo for Phase 5 post-scoring
but should not be claimed as a global deployment advance.

Harness: `scripts/phase3_geometry_router.py` (new).
Artifacts: `runs/phase3_unet/batch3_geometry_router_v1/`.

## Policies evaluated

Each policy is calibrated on the 5000-patch real-SMICA null distribution at
FPR targets 0.05 / 0.08 / 0.10. Thresholds are chosen independently per
policy so null FPR lands at target.

- `v6_only` - baseline v6_aux_only score, single-model threshold.
- `v7_only` - baseline v7_mixed_ft score, single-model threshold.
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

## Results on mixed geometry

| policy | recall @ FPR 0.05 | recall @ FPR 0.08 | recall @ FPR 0.10 |
|---|---:|---:|---:|
| `v6_only` | **0.305** | **0.331** | 0.347 |
| `v7_only` | 0.248 | 0.328 | **0.355** |
| `either_OR` | 0.299 | 0.325 | 0.344 |
| `both_AND` | 0.209 | 0.272 | 0.292 |
| `rank_max` | 0.299 | 0.325 | 0.344 |
| `geometry_routed` (q=0.50) | 0.202 | 0.237 | 0.275 |
| `geometry_routed` (q=0.95) | 0.222 | 0.305 | 0.344 |

## Results on contained geometry

| policy | recall @ FPR 0.05 | recall @ FPR 0.08 | recall @ FPR 0.10 |
|---|---:|---:|---:|
| `v6_only` | **0.348** | **0.372** | **0.389** |
| `v7_only` | 0.286 | 0.357 | 0.386 |
| `either_OR` | 0.341 | 0.367 | 0.385 |
| `both_AND` | 0.250 | 0.311 | 0.331 |
| `rank_max` | 0.341 | 0.367 | 0.385 |
| `geometry_routed` (q=0.50) | 0.241 | 0.277 | 0.304 |

Best single model wins every cell on contained. On mixed, v6_only wins at
FPR 0.05 and 0.08, v7_only wins at FPR 0.10 (by 0.008).

## Why the ensemble doesn't help

Two structural reasons:

1. **FP correlation between v6 and v7 is very high.** Both were trained on
   overlapping CAMB distributions, share the EfficientNet encoder, and look
   at the same real SMICA patches. Their false positives fire on the same
   patches, so `either_OR` buys little independent signal.
2. **FP-budget split erodes gains.** To keep joint null FPR at 0.08, each
   individual model's threshold rises. For v6, that moves from the
   single-model threshold (0.873) to a tighter value. The recall loss from
   this tightening roughly cancels the recall gained from adding v7's
   triggers on truncated positives that v6 misses.

## Why geometry routing failed

The routing signal `v7_mf_on_mask - scaled(v7_baseline)` did show the right
geometry correlation in Batch 2 (higher on contained positives than
truncated). But the router failed for two reasons:

1. The signal is dominated by the `mf_on_mask` magnitude, which correlates
   with signal strength as much as with geometry. Strong truncated positives
   get high routing scores and are sent to v6, which does not detect them.
   Weak contained positives get low routing scores and are sent to v7, which
   scores them lower than v6 would.
2. The optimal operating boundary for routing is not any simple null
   quantile. Sweeping quantile in {0.25, 0.50, 0.70, 0.85, 0.95} did not
   produce any setting that exceeded best single model at any FPR target.

A proper geometry router would need a learned classifier using more
features (mask compactness, centroid offset, edge fraction, matched-filter
per theta), and would need to be trained with explicit geometry labels, which
is a different project.

## Revised deployment advice

**Single-model operational path:** `v6_aux_only` at real-SMICA-calibrated
threshold 0.873 (FPR 0.08). This beats or ties every portfolio policy at
the deployment operating point.

**v7_mixed_ft role:** retained as a documented specialist for Phase 5
post-scoring of already-detected truncated candidates, where its per-group
advantage on truncated (0.246 vs 0.205) and center-outside-patch (0.207 vs
0.163) is real and measurable. Not the primary screener.

**matched_template role:** unchanged, remains the classical Feeney-template
reference and independent ranking score per `PROJECT_HANDOFF.md` Sections
10-13.

## What this closes off for future iterations

- Naive OR / AND / rank-max ensembling of v6 and v7. Measured negative here.
- Simple heuristic geometry routers using mf_on_mask contrast. Measured
  negative across a full quantile sweep.

## What remains on the table

- A learned geometry classifier trained on the mixed-geometry validation
  truth labels, used to gate which model scores each patch. More invasive
  but the only remaining pure-post-processing path that could improve over
  `v6_only`.
- A proper `v8` retrain with a matched-filter response map as a second
  input channel, on training_v5_mixed_geometry. This is the only training-
  signal lever that hasn't been touched. Expected 6-10 hours wall clock.
- Batch 2 small-sigma smoothing sweep (sigma in (0.5, 1, 2) pix):
  confirmed null. See `work/batch2_postprocess_ablation.md` update.
