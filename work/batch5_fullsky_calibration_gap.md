# Batch 5: Full-sky calibration gap and Batch 4 retraction

## TL;DR (revised 2026-04-17 after Steps 1 + 2)

A full-sky gnomonic tiling audit of all four Planck cleaned maps at HEALPix
Nside=8 (~7.3 deg spacing, 700 patches per map after common-mask filter
at unmasked-fraction >= 0.5) revealed **two serious findings**:

1. **The `smica_null_controls_all.h5` calibration pool systematically
   under-represents the mask-adjacent sky regions that dominate the
   model's false-positive response.** Real-sky patch-level FPR at the
   shipped 14-feature-GBT threshold is 0.157 on SMICA vs. the
   calibration pool's 0.08. On Commander it is 0.463 — **5.8x inflated**.

2. **Under deployment-representative tile calibration, the Batch 4
   (PR #9) +0.043 gate-cell lift of `gbt_14` over `gbt_6` evaporates and
   in some cases reverses.** Cross-map tile-recalibrated mixed-recall
   delta (gbt_14 − gbt_6) is −0.001 on SMICA, +0.023 on NILC, **−0.045
   on SEVEM**, **−0.036 on Commander**. Mean across maps **−0.015**,
   stdev 0.027. The Batch 4 lift is indistinguishable from zero on
   average and negative on 2 of 4 maps.

These findings together mean:

- Batch 4's PR #9 numbers are correct *as evaluated on the clean null
  pool*. They are not correct as deployment expectations. The shipped
  claim "+0.043 recall at FPR 0.08" should be revised to "+0.043 on
  clean-null calibration, ~0.00 under deployment-representative
  calibration on SMICA, negative on SEVEM/Commander."
- The 14-feature GBT is **not a deployment improvement over the
  6-feature GBT** on any map except arguably NILC, and even there the
  gain is within one tile-recalibration-uncertainty of zero.
- The clean-null calibration bias also applies to the PR #8 numbers
  (6-feature GBT vs `v6_only`). That comparison also needs re-running
  on a deployment-representative pool before we can state its real-sky
  deployment recall.

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

## Finding 3: tile-recalibrated Batch 4 delta (the important one)

Recalibrating the `gbt_6` and `gbt_14` thresholds on the full-sky tile
to land at the same 0.08 tile FPR, then re-applying to the cached
17500-positive mixed gate set:

| map | fpr6@shp | fpr14@shp | thr6_tile | thr14_tile | rec6_tile | rec14_tile | delta (14-6) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMICA | 0.121 | 0.157 | 0.8907 | 0.9038 | 0.3306 | 0.3291 | **−0.001** |
| NILC | 0.147 | 0.161 | 0.9012 | 0.9085 | 0.2923 | 0.3150 | **+0.023** |
| SEVEM | 0.260 | 0.291 | 0.9893 | 0.9920 | 0.2369 | 0.1921 | **−0.045** |
| Commander | 0.393 | 0.463 | 0.9934 | 0.9915 | 0.2297 | 0.1934 | **−0.036** |

Cross-map summary of the tile-recalibrated `gbt_14 - gbt_6` delta:
mean **−0.015**, stdev **0.027**, min **−0.045**, max **+0.023**.

Contrast with the shipped clean-null delta: **+0.043** on the gate
cell, claimed in PR #9 Section 22 / Section 26.

**The Batch 4 lift does not survive deployment-representative
calibration on any map except weakly NILC.** On SEVEM and Commander
the 14-feature GBT is actively *worse* than the 6-feature variant.

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
- The "default deployment policy" role goes back to either `v6_aux_only`
  (per-map calibrated) or `gbt_6` (also per-map calibrated), pending
  deployment-representative recalibration.
- The `gbt_14` variant is **not** recommended as the primary
  deployment policy on any map.

### PR #8 (6-feature GBT) is unverified but not retracted

- `gbt_6` vs `v6_only` delta was +0.031 on the gate cell under clean-null
  calibration. Under tile recalibration on 700-patch SMICA, `gbt_6`
  mixed recall is 0.331; `v6_only` single-model tile-recalibrated recall
  has not yet been computed but will likely be similar or slightly
  lower. Full re-evaluation is Step 2b.

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
