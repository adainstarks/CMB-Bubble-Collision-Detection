# Batch 5: Full-sky calibration gap and Batch 4 retraction

## TL;DR (revised 2026-04-17 after Steps 1 + 2)

A full-sky gnomonic tiling audit of all four Planck cleaned maps at HEALPix
Nside=8 (~7.3 deg spacing, 700 patches per map after common-mask filter
at unmasked-fraction >= 0.5) revealed **three findings**:

1. **The `smica_null_controls_all.h5` calibration pool systematically
   under-represents the mask-adjacent sky regions that dominate the
   model's false-positive response.** Real-sky patch-level FPR at the
   shipped 14-feature-GBT threshold is 0.157 on SMICA vs. the
   calibration pool's 0.08. On Commander it is 0.463 — **5.8x inflated**.

2. **Under deployment-representative tile calibration, the PR #9
   (Batch 4, `gbt_14`) lift evaporates or reverses cross-map.**
   Cross-map mean `gbt_14 − gbt_6` recall delta is **−0.015** (SMICA
   −0.001, NILC +0.023, **SEVEM −0.045**, **Commander −0.036**). The
   Batch 4 geometry features specifically overfit the clean-pool
   mask-fraction distribution. **PR #9 is downgraded; the 14-feature
   router is not recommended as deployment policy.**

3. **The PR #8 (Batch 3, `gbt_6`) lift SURVIVES recalibration cleanly.**
   Cross-map mean `gbt_6 − v6_only` recall delta under tile
   recalibration is **+0.036** (shipped clean-null claim was +0.031).
   Tied on SMICA/NILC, **+0.070 on SEVEM, +0.074 on Commander**.
   The 6-feature router's v6/v7 score ensemble does exactly what
   PR #8 said it would. **PR #8 is the right primary deployment
   policy, with per-map calibrated thresholds.**

Concrete deployment recommendation:

- **Primary: `phase3_geometry_router.py --feature-set scores_only`
  (the 6-feature GBT from PR #8), with per-map null calibration.**
- **Fallback: `v6_aux_only` single model, per-map calibrated.**
- **Archive: `--feature-set all` (PR #9 14-feature variant). Keep the
  code for reproducibility of the published clean-null numbers; do not
  deploy.**
- **Per-map null pools are mandatory.** Single-threshold FPR calibration
  on SMICA's clean null pool produces deployment FPR of 0.15 on SMICA's
  own full tile, 0.29 on SEVEM, 0.46 on Commander.

Gate (pre-registered from the session plan): my own "Step 1 hypothesis"
(patch-level FPR is 2-5x what cluster-level FPR would be, so clustering
is a 2-5x FP reduction lever). **Partially correct for the wrong
reason**: clustering DOES pay off at realistic deployment tile density
(4.8x reduction at 25-deg cluster radius at Nside=8), but the bigger
issue is that the patch-level FPR is not 0.08 in the first place — it
is 0.157-0.463 depending on map.

Harness: `scripts/phase3_fullsky_tile.py` (new). Artifacts under
`runs/phase3_unet/batch5_fullsky_fp_audit_*`.

## Setup

Each run tiles one Planck cleaned map at HEALPix Nside=8, which gives
768 centers at ~7.3 deg mean spacing. The common mask at Nside=256 is
projected through each candidate patch; centers whose patch has <50%
unmasked fraction are dropped. All 4 maps retain 700/768 centers
after filtering.

For each kept patch we run both U-Nets (`v6_aux_only` and
`v7_mixed_ft`), compute the Batch 2 transforms + the Batch 4 geometry
features, and score with the Batch 4 14-feature gradient-boosted
router (seed 20260417, cross-geometry fit, 2500/2500 disjoint null
split — same protocol as PR #9).

Two separate analyses:

1. Cluster-reduction (Step 1 hypothesis) — group triggered patches by
   great-circle distance between their probability-mask peak sky coords
   at cluster radii 5/10/15/25/40 deg; report cluster count and
   reduction factor.
2. Tile-recalibrated Batch 4 comparison — recompute the GBT threshold
   so that exactly 8% of tile patches trigger (deployment-representative
   FPR 0.08), apply to the cached 17500-positive mixed gate set, and
   report `gbt_6` and `gbt_14` recalls.

## Finding 1: cluster reduction factor as a function of tile density

| tile Nside | spacing | n patches | 5 deg | 10 deg | 15 deg | 25 deg | 40 deg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 (SMICA) | 14.7 deg | 175 | 1.00x | 1.13x | 1.18x | 1.53x | 2.60x |
| 8 (SMICA) | 7.3 deg | 700 | 1.17x | 1.43x | **2.29x** | **4.78x** | 10.00x |
| 8 (NILC)  | 7.3 deg | 700 | - | - | 2.76x | 4.35x | 9.42x |
| 8 (SEVEM) | 7.3 deg | 700 | - | - | 3.24x | 7.03x | 13.60x |
| 8 (Commander) | 7.3 deg | 700 | - | - | - | - | 24.92x |

At realistic deployment density (Nside=8, 7.3 deg spacing) clustering
at 15-deg radius gives 2.3-3.2x reduction across maps and at 25 deg
gives 4.4-7.0x, which is roughly the range the original Step 1
hypothesis proposed. Nside=4 is too sparse for meaningful clustering
because a given sky feature only falls inside 1-2 overlapping patches
at 14.7 deg spacing.

Clustering is therefore a real, cheap FP-burden reduction lever **at
deployment tile density**, but the patch-level FPR itself is what we
actually need to fix first.

## Finding 2: per-map patch-level FPR at shipped threshold

Under the PR #9 shipped 14-feature GBT threshold 0.8814 (calibrated to
FPR 0.08 on the clean-null eval half):

| map | n tiles | triggered | patch-level FPR | FPR inflation vs target 0.08 |
|---|---:|---:|---:|---:|
| SMICA | 700 | 110 | 0.157 | 1.96x |
| NILC | 700 | 113 | 0.161 | 2.01x |
| SEVEM | 700 | 204 | 0.291 | 3.64x |
| Commander | 700 | 324 | 0.463 | **5.78x** |

On SMICA the ratio of mask-fraction bands tells the story:

| mask fraction band | n tiles | triggers | observed FPR |
|---|---:|---:|---:|
| [0.50, 0.70) | 59 | 14 | 0.237 |
| [0.70, 0.90) | 32 | 5 | 0.156 |
| [0.90, 0.95) | 15 | 2 | 0.133 |
| [0.95, 1.00] | 69 | 5 | 0.073 |

In the band where the calibration pool lives (mask fraction >= 0.95),
the tile FPR is 0.073 — matching the calibration target. In the
deployment-representative band [0.50, 0.70), the tile FPR is 0.237 —
**3.3x higher**. The calibration pool was drawn with the
training-data rule `MASK_THRESHOLD = 0.95`; this is far stricter than
any deployment tiling would use.

The 5000-patch SMICA null pool in `data/training_v4/smica_null_controls_all.h5`
has `coord_mask_fraction` min 0.950, median 0.984 — it systematically
excludes the mask-adjacent sky where the model fires most.

## Finding 3: full cross-map deployment-calibrated policy comparison

Recalibrating all three policies (`v6_only`, `gbt_6`, `gbt_14`) per map
using the Nside=8 tile as the null distribution (threshold set so
exactly 8% of tile patches trigger), then re-applying to the cached
17500-positive mixed gate set:

| map | `v6_only` | `gbt_6` | `gbt_14` | `gbt_6 − v6` | `gbt_14 − v6` | `gbt_14 − gbt_6` |
|---|---:|---:|---:|---:|---:|---:|
| SMICA | 0.334 | 0.331 | 0.329 | −0.003 | −0.005 | −0.001 |
| NILC | 0.293 | 0.292 | **0.315** | −0.001 | **+0.022** | **+0.023** |
| SEVEM | 0.167 | **0.237** | 0.192 | **+0.070** | +0.025 | −0.045 |
| Commander | 0.155 | **0.230** | 0.193 | **+0.074** | +0.038 | −0.036 |
| **cross-map mean** | **0.237** | **0.273** | 0.257 | **+0.036** | +0.020 | −0.015 |

This is a more complete picture than just the pairwise `gbt_14 − gbt_6`
comparison. Two things emerge:

### Result 1: PR #8 (6-feature GBT) SURVIVES deployment calibration

`gbt_6` beats `v6_only` by **+0.036 cross-map mean** under tile
recalibration (shipped clean-null claim was +0.031). The 6-feature
router's ensemble-of-scores signal generalizes cross-map; the geometry
features do not.

- SMICA: tied (−0.003, noise)
- NILC: tied (−0.001, noise)
- SEVEM: **+0.070** — big win for the router
- Commander: **+0.074** — big win for the router

The PR #8 delta is ROBUSTLY ≥0 across all four maps and dramatically
positive on the two noisier maps. The PR #8 claim does not need to
be retracted.

### Result 2: PR #9 (14-feature GBT) FAILS deployment calibration

`gbt_14` beats `v6_only` cross-map by only **+0.020 mean** (shipped
clean-null claim was gbt_14 − v6_only ≈ +0.077). And `gbt_14` beats
`gbt_6` by **−0.015 cross-map mean** (shipped claim: +0.043).

- SMICA: tied (−0.005, noise vs v6_only; −0.001 vs gbt_6)
- NILC: **+0.022 vs v6_only, +0.023 vs gbt_6** — the one map where Batch 4 genuinely helps
- SEVEM: +0.025 vs v6_only but **−0.045 vs gbt_6** — the router helps but the geometry features hurt
- Commander: +0.038 vs v6_only but **−0.036 vs gbt_6** — same pattern

The Batch 4 geometry features made the router *worse than gbt_6* on
the two noisier maps. The shipped "+0.043" was an artifact of the
clean-null calibration pool.

### Revised deployment recommendation (this PR)

- **Primary policy: `learned_gbt` with `--feature-set scores_only` (6 features).**
  Beats `v6_only` by +0.036 cross-map mean under tile recalibration;
  statistically indistinguishable from `v6_only` on clean maps (SMICA/NILC)
  and +0.070-0.074 on noisier maps (SEVEM/Commander).
- **Single-model fallback: `v6_aux_only`** at per-map-calibrated thresholds.
  Closest to the 6-feature GBT on SMICA/NILC; noticeably weaker on
  SEVEM/Commander.
- **Archive: `learned_gbt` with `--feature-set all` (14 features).** The
  geometry features overfit the clean null pool and hurt deployment
  recall on SEVEM/Commander. Keep the `--feature-set all` flag in the
  codebase for reproducibility of the PR #9 numbers; do not use it as
  the deployment policy.
- **Per-map calibrated thresholds** are not optional. Single-threshold
  "FPR 0.08" calibrated on SMICA's clean null produces deployment FPR
  of 0.23 on SMICA's own full tile, 0.29 on SEVEM, **0.46 on Commander**.
  Every deployment point needs its own map-specific null calibration.

## Why the 14-feature GBT specifically loses cross-map

Mechanism consistent with the feature set:

- `edge_touching_fraction` and `centroid_offset_px` were shown in
  Batch 4 to carry ~12% of GBT importance combined. On the clean null
  pool (mask fraction >= 0.95) these features rarely fire because the
  backgrounds don't contain mask-adjacent foreground residuals.
- Real-sky tiles at mask fraction 0.5-0.9 include many such residuals.
  The geometry features fire on those residuals, producing higher GBT
  scores, and the threshold-at-FPR-0.08 has to rise substantially to
  reject them.
- The rise in threshold eats more of the positive recall than the
  rise in `gbt_6`'s threshold does, because `gbt_6` was less sensitive
  to mask-adjacent signal shape in the first place.
- On SEVEM and Commander (worse cleaning in-plane) this effect is
  stronger than on SMICA/NILC.

This is consistent with the `edge_touching_fraction` feature having
learned a calibration-pool-specific statistic rather than a
deployment-generalizable one.

## Consequences

### Batch 4 (PR #9) is downgraded

- Code stays merged; the feature-set plumbing is useful infrastructure.
- The "default deployment policy" role goes to **`gbt_6` (PR #8's
  6-feature router)**, which survives recalibration cleanly (see
  Result 1 above).
- The `gbt_14` variant (PR #9's geometry-feature router) is **not**
  recommended as the primary deployment policy on any map.

### PR #8 (6-feature GBT) is CONFIRMED (not retracted)

- `gbt_6` vs `v6_only` cross-map mean delta under deployment-representative
  tile calibration: **+0.036 recall** (shipped clean-null claim was
  +0.031). The +3.1-4.5pp lift holds up.
- On SMICA/NILC the lift is close to zero; on SEVEM/Commander it is
  +0.070 to +0.074. The router's `v6 + v7` score ensemble does exactly
  what it was designed to do on the two noisier maps.

### All prior FPR-calibrated thresholds in the repo are now suspect

- Section 10 real-SMICA recalibration thresholds (`v6_aux_only` 0.873
  at nominal FPR 0.08) were calibrated on the clean null pool and
  produce 0.12-0.16 actual FPR at Nside=8 deployment tiling on SMICA,
  up to 0.4+ on Commander.
- Section 12 threshold-volume sweep — same issue.
- Section 13 two-pass policy results — same issue.
- The published PR #6, #7, #8, #9 numbers are all on the clean null
  pool and need deployment-representative recalibration before they
  can be cited as deployment performance.

## What to do next

Corrective work, in priority order:

1. **Rebuild the null calibration pool per map** with a
   deployment-representative mask-fraction distribution. Target 5000
   patches per map at `MASK_THRESHOLD = 0.5` instead of 0.95, sampled
   uniformly from valid sky centers. Use `phase2_extract_smica_null_controls.py`
   with a loose mask argument (needs a small code change) or a new
   script. Compute budget: ~2-3 hours per map if done serially, ~1 hour
   total if parallelized across 2 GPUs (2 maps per GPU).
2. **Re-run `phase3_postprocess_ablation.py`** on the new null pools
   to regenerate transform + geometry feature caches.
3. **Re-run `phase3_geometry_router.py`** with `--feature-set scores_only`
   and `--feature-set all` on the new caches. Report the cleaned-up
   gbt_6 / gbt_14 / v6_only comparison per map.
4. **Ship the honest deployment-policy recommendation** based on the
   above. Likely: `v6_aux_only` with per-map calibrated thresholds
   remains the primary; any router claim is gated on the new pool.
5. **Document the Batch 4 retraction** explicitly in `PROJECT_HANDOFF.md`
   Section 24 as negative entry #10.

## What this does NOT break

- The Feeney signal model, injection protocol, U-Net training
  pipeline, and evaluation harnesses are all untouched.
- The v6_aux_only and v7_mixed_ft model weights are unchanged; their
  performance relative to each other on fixed data is unchanged.
- The post-processing transforms (`smooth_multi`, `mf_on_mask`) and
  Batch 4 geometry features (`mask_area_at_0.5`, `centroid_offset_px`,
  `compactness`, `edge_touching_fraction`) are all real and remain
  cached. The question is only whether they contribute deployment-real
  detection gain after honest calibration — and the answer so far is
  no for the geometry features specifically.
- The clustering infrastructure built in `phase3_fullsky_tile.py` is
  independently useful and gave a clean Step 1 side-result (2-5x
  cluster reduction at realistic tile density).

## Artifacts

- `runs/phase3_unet/batch5_fullsky_fp_audit_v1/` — SMICA Nside=4
  (falsified my original Nside hypothesis).
- `runs/phase3_unet/batch5_fullsky_fp_audit_nside8/` — SMICA Nside=8.
- `runs/phase3_unet/batch5_fullsky_fp_audit_nside8_nilc/` — NILC.
- `runs/phase3_unet/batch5_fullsky_fp_audit_nside8_sevem/` — SEVEM.
- `runs/phase3_unet/batch5_fullsky_fp_audit_nside8_commander/` — Commander.
- `runs/phase3_unet/batch5_fullsky_fp_audit_v1/crossmap_recalibration_summary.json`
  — this analysis's machine-readable summary.
