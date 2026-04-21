# Phase 5 Half-Mission Sign-Flip Null

## Assumptions

- This module is a real-data calibration stage for screened candidates, not a
  training-time fix and not a replacement for synthetic null controls.
- The required inputs are Planck half-mission cleaned maps, a common analysis
  mask, the frozen candidate-scoring command, and a candidate table.
- The target statistic is a per-candidate screening score or calibrated
  candidate score. It is not a Bayesian evidence ratio.
- The method assumes the HM1/HM2 difference is a usable local noise and
  residual-systematics proxy after applying the same beam, mask, and patch
  projection used by the deployment scorer.

## Decision

This is implemented as `scripts/phase5_half_mission_signflip_null.py`, but it
remains a downstream Phase 5 calibration step. It should run after candidate
emission and score-policy selection are frozen, not during model training or
threshold search.

This is scientifically valuable because it preserves the observed CMB signal
while randomizing half-mission noise structure inside each candidate patch. It
does not solve the low-amplitude simulation-null recall problem, and it requires
HM1/HM2 data products that are intentionally outside the synthetic remediated
artifact graph.

## Estimator

For a candidate patch, define

```text
T_mean = 0.5 * (T_HM1 + T_HM2)
T_diff = 0.5 * (T_HM1 - T_HM2)
```

Draw sign fields `s_i` with values in `{-1, +1}` on valid pixels, optionally
block-correlated to the effective beam or patch-pixel scale. Construct null
realizations

```text
T_i = T_mean + s_i * T_diff
```

Run the frozen candidate scorer on `T_i` and compare the observed score
`S_obs = S(T_mean)` against the null score ensemble. A conservative one-sided
empirical p-value is

```text
p = (1 + count(S_i >= S_obs)) / (1 + N_null)
```

The output belongs in the candidate table as `hm_signflip_p_value`,
`hm_signflip_num_null`, and score quantiles. Multiple-candidate reporting should
use false-discovery-rate or family-wise correction, not single-candidate
threshold shopping.

## Interface

Preflight the frozen candidate/policy/model inputs before HM maps are
available:

```bash
python scripts/phase5_half_mission_signflip_null.py \
  --preflight-only \
  --candidate-jsonl runs/phase3_unet/remediated_v1_tile_constrained_candidates/cluster_representatives_15deg.jsonl
```

The current local preflight validates the `24` cluster representatives and
their `tile_constrained_rank1_2_of_3` policy slug, but reports `blocked`
because HM1/HM2 cleaned-map paths have not been supplied. This is expected until
the Planck half-mission component-separated products are staged locally.

Current full calibration CLI pattern:

```bash
python scripts/phase5_half_mission_signflip_null.py \
  --hm1-map data/HM1_cleaned_map.fits \
  --hm2-map data/HM2_cleaned_map.fits \
  --candidate-jsonl runs/phase3_unet/remediated_v1_tile_constrained_candidates/cluster_representatives_15deg.jsonl \
  --num-realizations 1024
```

The default policy JSON is the current tile-constrained policy-search report:

```text
runs/phase3_unet/remediated_v1_tile_constrained_policy_search/tile_constrained_policy_search.json
```

Older Policy-Pareto tile-audit candidate files are still supported when an
older policy JSON is passed explicitly.

Required inputs:

- Candidate table with sky coordinates, patch geometry, map name, score mode,
  and frozen scorer identifier.
- HM1 and HM2 cleaned maps for each map family being claimed.
- Analysis mask and projection settings matching deployment.
- Frozen scorer command for ML, classical, or score-ensemble candidates.

Outputs:

- Per-candidate observed score.
- Per-candidate null score distribution summary.
- Empirical p-value and multiple-testing-ready metadata.
- Reproducibility manifest containing map versions, mask version, scorer hash,
  RNG seed, and sign-flip policy.

## Implementation Notes

- Use `float64` for map differencing and projection, then convert only if the
  frozen scorer requires `float32` tensors.
- Reject patches with non-finite pixels, insufficient valid mask fraction, or
  inconsistent HM beam/projection metadata.
- Keep this module downstream of candidate emission. It should not tune model
  thresholds, architecture, or training data.
