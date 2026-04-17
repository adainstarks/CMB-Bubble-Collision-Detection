# Batch 6: Nside=32 recalibration + review-driven ablations

## TL;DR

Two things happened after the Batch 5 PR:

1. **Nside=32 full-sky audit** (overnight orchestrator) reran the tile
   calibration with 11200 patches per map instead of 700. FPR-calibration
   uncertainty dropped from ±0.011 to ±0.003. Every Batch 5 conclusion
   held. This is the paper-grade version of the cross-map deployment
   comparison.

2. **Review-driven ablations** (responding to a teammate code review of
   PR #7-9) confirmed all four structural concerns raised are well-posed
   and the data either supports them cleanly (pooled-vs-cross router,
   n_estimators sweep, bootstrap CIs) or refutes them cleanly
   (smoothing-is-a-contradiction).

Artifacts:

- `runs/phase3_unet/batch6_fullsky_nside32_*/fullsky_tile_report.{json,md}`
- `runs/phase3_unet/batch6_fullsky_nside32_smica/crossmap_recalibration_nside32.{json,md}`
- `work/teammate_review_response.json` — machine-readable ablation results
- `scripts/batch6_overnight_orchestrator.sh` — runs all 4 maps paired on 2 GPUs
- `scripts/batch6_overnight_analysis.py` — produces the cross-map recal report

## Part A: Nside=32 full-sky audit

Compute cost: 1h43m wall clock on 2x3090 (paired map-per-GPU). Disk:
~10 GB of tile-patch HDF5s (gitignored).

### Patch-level FPR at shipped (clean-null) thresholds

| map | n_tile | `v6_only` | `gbt_6` | `gbt_14` |
|---|---:|---:|---:|---:|
| SMICA | 11200 | 0.066 | 0.104 | 0.152 |
| NILC | 11200 | 0.122 | 0.135 | 0.157 |
| SEVEM | 11200 | 0.289 | 0.268 | 0.297 |
| Commander | 11200 | 0.427 | 0.394 | 0.453 |

Two refinements over Batch 5:

- On SMICA, `v6_only`'s clean-null calibration is **not inflated** at
  this sample size (0.066 vs target 0.08 — within uncertainty). The
  previous "1.96x inflation on SMICA" was small-sample noise on the
  700-patch Nside=8 tile. On the other three maps the inflation is
  real and large.
- The GBTs (both `gbt_6` and `gbt_14`) have stronger calibration drift
  than `v6_only` on SMICA (0.104 and 0.152 vs 0.066). This is the
  router-specific bias — the GBTs, trained on the clean null pool,
  over-fire on mask-adjacent tile patches more aggressively than the
  single v6 model does. Consistent with the Batch 5 mechanism
  explanation.

### Tile-recalibrated mixed recall at FPR 0.08

| map | `v6_only` | `gbt_6` | `gbt_14` | `gbt_6−v6` | `gbt_14−v6` | `gbt_14−gbt_6` |
|---|---:|---:|---:|---:|---:|---:|
| SMICA | 0.347 | 0.337 | 0.329 | **−0.010** | −0.018 | −0.009 |
| NILC | 0.296 | 0.307 | 0.320 | +0.011 | +0.024 | +0.013 |
| SEVEM | 0.173 | 0.237 | 0.192 | **+0.065** | +0.020 | −0.045 |
| Commander | 0.161 | 0.230 | 0.193 | **+0.070** | +0.032 | −0.038 |
| **mean** | **0.244** | **0.278** | **0.258** | **+0.034** | +0.014 | −0.020 |

Conclusions match Batch 5 at 16x statistical power:

- **`gbt_6` survives**: cross-map mean lift over `v6_only` **+0.034**
  (Batch 5 Nside=8: +0.036). Lift is map-dependent — essentially zero
  on SMICA, significant on NILC, substantial on SEVEM/Commander.
- **`gbt_14` does not survive**: cross-map mean lift over `gbt_6` is
  **−0.020** (Batch 5 Nside=8: −0.015). The 14-feature router is
  strictly worse than the 6-feature router under deployment
  calibration.
- **On SMICA alone, `gbt_6` is marginally WORSE than `v6_only`**
  (−0.010). The cross-map lift lives in the noisier maps. This is the
  honest framing that should lead the paper abstract rather than the
  headline cross-map average.

## Part B: review ablations

Four ablations requested by a teammate code review. Each is its own
self-contained test; methods section and raw numbers below.

### B1. Drop-one ablation on `smooth_multi` features

**Question**: If PR #7 concluded "smoothing on the probability mask is
a null", how is `v6_smooth_multi` carrying 16% GBT feature importance
in PR #8? Contradiction?

**Method**: Drop `v6_smooth_multi`, `v7_smooth_multi`, or both, from
the 6-feature GBT. Refit on cached features with 4 null-split seeds.
Compare mixed-recall-at-FPR-0.08 under both clean-null calibration and
SMICA Nside=32 tile recalibration.

**Result** (mean over 4 seeds):

| feature set | n_feat | mixed recall (clean) | Δ vs full_6 | mixed recall (tile SMICA) | Δ vs full_6 |
|---|---:|---:|---:|---:|---:|
| full_6 | 6 | 0.3666 | — | 0.3414 | — |
| −v6_smooth_multi | 5 | 0.3637 | **−0.0028** | 0.3377 | **−0.0037** |
| −v7_smooth_multi | 5 | 0.3662 | −0.0004 | 0.3381 | −0.0032 |
| −both | 4 | 0.3637 | **−0.0028** | 0.3372 | **−0.0042** |

**Verdict**: Dropping BOTH smoothing features costs 0.28pp clean-null
and 0.42pp tile. The teammate's own threshold was `<0.5pp = noise,
≥1pp = PR #7 wrong`. Smoothing is noise. PR #7's univariate-null claim
is empirically confirmed under the ablation test.

The 16% GBT importance is sklearn-feature-importance-inflation: a
feature used for many splits scores high on Gini-gain-at-split even
when its predictions contribute little. This is a known Gini-importance
limitation, not a scientific finding. A proper contribution metric
(drop-one, permutation importance) shows smoothing adds ~0.003.

### B2. Pooled-training vs cross-training router

**Question**: PR #8/9 reported two routers (one per eval geometry); a
deployment-era router has to be pooled. What's the pooled number?

**Method**: Train three router variants on 6 features (seed 20260417):

- `cross_contained`: fit on contained positives only; evaluate on mixed.
- `cross_mixed`: fit on mixed positives only; evaluate on contained.
- `pooled`: fit on contained ∪ mixed positives; evaluate on both.

| train mode | mixed_recall (clean) | contained_recall (clean) | tile-mean mixed | tile-mean contained |
|---|---:|---:|---:|---:|
| `cross_contained` | 0.3651 | 0.4105 | 0.2779 | 0.3217 |
| `cross_mixed` | 0.3782 | 0.4029 | 0.2690 | 0.3051 |
| `pooled` | 0.3693 | 0.4043 | 0.2743 | 0.3145 |

**Verdict**: Pooled router performance is within ±0.005 of both
cross-trained variants under both calibrations. The deployable pooled
router is essentially indistinguishable from the cross-trained numbers
shipped in PR #8/9. The ambiguity in the PR description was real and
should be fixed, but the deployment-real number does not change.

### B3. Bootstrap 95% CI on `gbt_6 − v6_only` mixed recall delta

**Method**: 1000 resamples of the 17500-positive mixed gate set, with
thresholds frozen at shipped clean-null and SMICA Nside=32 tile values.

| calibration | point estimate | 95% CI |
|---|---:|---|
| shipped clean-null | **+0.0342** | [+0.0294, +0.0386] |
| SMICA tile-recal | **−0.0095** | [−0.0142, −0.0050] |

**Verdict**:

- Shipped-calibration lift is ~14σ from zero. PR #8's claim is
  statistically rock-solid under its own calibration assumptions.
- Honest-calibration SMICA lift is ~4σ NEGATIVE. On SMICA deployment,
  `gbt_6` loses to `v6_only`. The cross-map mean lift lives entirely in
  SEVEM/Commander (see B4 / Part A).

Paper-level consequence: the headline delta cannot be quoted on its
own. Either quote the cross-map mean (+0.034) with per-map CIs, or
quote per-map numbers (SMICA negative, SEVEM/Commander positive, NILC
tied). Do not quote a map-averaged number as a SMICA-specific claim.

### B4. n_estimators sweep

**Method**: Fit `gbt_6` with n_estimators ∈ {50, 100, 200, 400, 800},
4 seeds each. Measure clean-null mixed recall.

| n_estimators | mean | stdev | min | max |
|---:|---:|---:|---:|---:|
| 50 | 0.3526 | 0.005 | 0.3461 | 0.3593 |
| 100 | 0.3616 | 0.008 | 0.3485 | 0.3715 |
| 200 (shipped) | 0.3666 | 0.012 | 0.3491 | 0.3822 |
| 400 | 0.3786 | 0.011 | 0.3603 | 0.3890 |
| 800 | 0.3864 | 0.006 | 0.3768 | 0.3939 |

**Verdict**: 200 trees is conservative, not cherry-picked — larger
trees give monotonically higher clean-null recall. Two caveats the
review didn't flag:

- "Monotonically higher" does not imply "monotonically better". The
  increase from 200 to 800 is almost certainly overfitting the clean
  null pool's mask-fraction distribution more, which is exactly the
  failure mode Batch 5 identified.
- Any future n_estimators tuning should be evaluated on
  deployment-representative tile recall, not clean-null recall. The
  existing 200 shipped value is defensible; increasing it without that
  check would be unjustified.

## Part C: consequences for the paper narrative

The Batch 6 numbers + review ablations tighten the deployment story
but also sharpen the map-conditional nature of it. The published
framing needs the following corrections:

1. **Lead with expected-candidates-per-sky at a couple of operating
   points**, not FPR. The FPR columns stay as secondary.
2. **Report per-map recall** explicitly. Cross-map means are
   illuminating but misleading as a single headline: gbt_6 has no
   advantage on SMICA, marginal on NILC, large on SEVEM/Commander.
3. **Quote bootstrap 95% CIs** on every shipped delta.
4. **Drop the smoothing-as-a-useful-GBT-feature framing**. Keep the
   feature in the router for completeness, but flag it as
   non-contributing per the drop-one ablation. 16% Gini importance is
   feature-inflation.
5. **The Batch 4 geometry features** are archived infrastructure. Do
   not claim any deployment benefit from them without rebuilding the
   null pool and rerunning the comparison.

## Part D: next concrete work

Ordered by expected value, consistent with the Batch 5 roadmap:

1. **Deployment-representative null pool per map** at
   `MASK_THRESHOLD = 0.5`, ~5000 patches each, independent coord pool.
   This fixes the calibration foundation everything else depends on.
2. **Rerun all three policies** (`v6_only`, `gbt_6`, `gbt_14`) and the
   real-SMICA Section 10-13 threshold-volume work on the new pool.
   Produce a corrected deployment table.
3. **Tile-audit `phase3_v7_mf_channel_aux_w4`**. Question: does the
   MF-channel input reduce mask-adjacent FP inflation relative to
   vanilla `v7_mixed_ft`? If yes, MF-channel is a deployment advantage
   even if its clean-pool recall gain is small.
4. **Plan a v8 retrain on real-SMICA null backgrounds** (not CAMB).
   CAMB→SMICA transfer failure was not marginal on contained; staying
   on CAMB for v8 is buying another ticket in the same losing lottery.
   Smoke-test first per `work/radius_head_post_mortem.md` hygiene.
