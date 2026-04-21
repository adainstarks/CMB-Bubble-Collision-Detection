# Manuscript Framing: Candidate Screening, Not Cosmological Detection

## Assumptions

- The current repository output is a candidate-screening and follow-up pipeline.
- It does not yet compute the Feeney/OSS masked-sky Bayesian evidence ratio.
- Screening calibration, clustered candidate counts, template-fit seeds, and
  screening-derived upper-limit proxies are operational quantities, not direct
  cosmological posteriors.

## Allowed High-Level Claim

Use language at this level:

> We present a reproducible Planck-era candidate-screening pipeline for
> localized bubble-collision signatures, combining ML and classical filters to
> generate a manageable set of follow-up targets for downstream
> template-likelihood or Bayesian analysis.

This is the central framing rule for the paper, README, and talk slides.

## Claims That Are Safe

- The U-Net and classical baselines can be compared as **screeners** on matched
  synthetic sensitivity grids.
- The same-grid Wiener/SMHW benchmark is a **candidate-screening benchmark**,
  not a full optimal-likelihood ceiling.
- Real-map null calibration provides **candidate-ranking metadata** and
  candidate-volume accounting.
- The full-sky candidate list, clustering audit, and template-fit packet are
  **handoff products** for later inference.
- HM sign-flip and frequency-jackknife follow-up can be used to say which
  screened candidates remain viable follow-up targets.
- The upper-limit calculator yields a **screening-derived detectable-collision
  sensitivity proxy** under explicit assumptions.

## Claims That Are Not Safe

- Do not claim a bubble-collision detection.
- Do not claim a posterior on the expected number of collisions from this repo
  alone.
- Do not describe empirical null-survival p-values or BH q-values as global
  significance.
- Do not describe the screening-derived upper limit as a competitive Planck
  cosmological bound unless it is replaced by a proper masked-sky Bayesian
  evidence analysis.
- Do not describe greedy candidate clustering as a physical source association
  likelihood or an independent-trials correction.

## Required Wording For Key Artifacts

- Same-grid benchmark:
  "same-grid screening comparison against Wiener/SMHW on identical injected
  skies"
- Candidate calibration:
  "empirical null-survival ranking for frozen screened candidates"
- Template-fit handoff:
  "deterministic local Feeney-template seed for downstream likelihood or
  Bayesian follow-up"
- Projection audit:
  "geometry/systematics caution study for patch-vs-spherical follow-up"
- Upper limits:
  "screening-derived detectable-collision sensitivity proxy"

## Recommended Abstract / Results Framing

1. State the problem as candidate generation under Planck-era complexity.
2. State the method as ML plus classical screening with matched synthetic and
   real-null calibration.
3. State the output as sensitivity maps, candidate burden, frozen candidate
   representatives, and template-fit/Bayesian handoff artifacts.
4. State the non-claim explicitly: no cosmological detection claim is made
   without downstream Bayesian evidence.

Use results wording at this level:

- The same-grid Wiener/Feeney filter is the strongest average screener on the
  fixed synthetic benchmark; ML remains locally competitive rather than
  uniformly better.
- The corrected true-Wiener two-stream branch is promising on synthetic recall
  but is not promoted because deployment burden, candidate calibration, and
  frozen-candidate follow-up evidence are still missing.
- The frozen real-sky follow-up is harsh: `23 / 24` representatives remain
  stable under the frequency jackknife, but only `3 / 24` remain stable under
  both HM sign-flip and frequency follow-up, collapsing to `2 / 10` adaptive
  sky systems.
- The upper-limit number is a screening-derived sensitivity proxy, not a
  standalone cosmological bound.
- No standalone detection claim is being made.

## Paper-Facing Checklist

- The introduction cites Feeney/McEwen/OSS as the downstream inference target,
  not as something already achieved here.
- The methods section separates screening, calibration, clustering, and
  follow-up handoff.
- The results section reports recall, null burden, candidate burden, and
  same-grid screening comparisons separately, then gives the HM-plus-frequency
  survivor table as a follow-up triage result.
- Every table that contains p-values, q-values, or cluster counts states that
  these are screening/follow-up quantities.
- The conclusion says the next scientific step is projection-robust Bayesian or
  template-likelihood analysis of the surviving adaptive systems.
