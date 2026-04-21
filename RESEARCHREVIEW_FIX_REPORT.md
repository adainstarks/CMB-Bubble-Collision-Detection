# RESEARCHREVIEW Fix Report

Date: 2026-04-21
Reviewed file: `RESEARCHREVIEW`
Repo baseline inspected: `remediated_v1`

## Assumptions First

- This review treats the repository as a candidate-screening and follow-up
  pipeline, not as a standalone cosmological detection or model-selection
  result.
- Local evidence has priority for implementation status. Primary papers and
  official Planck data pages have priority for scientific validity. Search
  snippets, conference abstracts, and non-primary web pages are treated as
  leads, not as final evidence.
- All local Planck PR3 component-separated maps inspected in `data/` have
  `TUNIT1 = K_CMB`; patch tensors are CMB anisotropies in Kelvin.
- The current `remediated_v1` products use `Nside=256`, `256 x 256` gnomonic
  patches, `13 arcmin` pixels, `5 arcmin` beam handling, and the canonical
  `mask_fraction >= 0.9` science footprint.
- The current ML scores are screening scores. They are not Bayesian evidence,
  posterior probabilities for bubble collisions, or global detection
  significances.
- Report verdicts:
  - `valid`: claim is supported and actionable.
  - `partly valid`: scientific concern is real, but the review overstates or
    conflates details.
  - `invalid as stated`: the claim conflicts with local code or primary source
    evidence.
  - `needs verification`: plausible but not safe to use as a paper claim until
    the repo records a source-backed reproduction or search log.

## Executive Verdict

- The review is right that the patch-RMS noise-floor diagnostic cannot justify
  a paper-facing "physics limit" claim. The current report already warns that
  it is not an impossibility proof, but `PROJECT_HANDOFF.md` still phrases weak
  recall as a physical/noise-floor limitation. This must be softened and
  replaced with a matched-filter SNR and ROC comparison.
- The review is right that the current `Nbar_s^95 = 10.2674` result is not a
  competitive Feeney-style bound. Feeney et al. 2013 report fewer than `4.0`
  detectable bubble collisions at 95% confidence on WMAP 7-year data. The repo
  should frame its result as an efficiency-weighted candidate-screening
  post-processing number, not a headline cosmological upper limit.
- The review was right that the true Wiener/SMHW same-grid benchmark had to be
  closed before any ML-vs-classical claim. That benchmark is now complete and
  shows `wiener_feeney_matched_filter` as the strongest average screener on the
  fixed stratified manifest, with the ImageNet U-Net retaining localized wins
  rather than uniform superiority.
- The review is too strong on injection. Local code does not implement the
  plainly wrong `T_anisotropy *= (1 + f)` operation. It implements the exact
  Feeney 2011 modulation of full temperature:
  `patch + f*T_CMB + f*patch`. The `f*T_CMB` term is the desired first-order
  additive perturbation; `f*patch` is the second-order modulation term that
  McEwen/OSS discard for filter construction. This is a deliberate convention
  issue, not a confirmed physics bug. It must still be documented explicitly so
  the McEwen/OSS additive filter benchmark is transparently a first-order
  approximation to Feeney-style injected maps.
- The data-handling concerns are mostly useful: unit validation should become a
  first-class gate; PR4/NPIPE should not be assumed to have four official
  component-separated maps; constant `15 deg` clustering needs a scale-dependent
  sensitivity audit; large-radius gnomonic distortion needs a quantified
  projection systematic.
- The "scoop landscape is clear" claim is not safe as a blanket statement.
  Phys. Rev. D 108, 043525 / arXiv:2303.08869 is a CNN search for pairwise
  CMB hotspots from cosmological particle production. It is not a
  bubble-collision segmentation pipeline, but it is scoop-neighboring enough
  that the paper introduction must cite and distinguish it before claiming
  novelty.

## High-Severity Findings

### Implementation Update

- `runs/phase3_unet/remediated_v1_classical_same_grid_status/classical_same_grid_status.json`
  now reports `status = complete`.
- On the closed same-grid manifest, mean recall is `0.3841` for
  `wiener_feeney_matched_filter`, `0.3517` for `imagenet_b64_aux`, `0.2994`
  for `random_b64_aux`, and `0.2734` for `smhw_screen`.
- The SNR-guided hard-example ablation is complete and does not justify
  promotion into the baseline. Its mean sensitivity delta versus the ImageNet
  baseline is slightly negative.
- The focal-loss ablation (`gamma = 2`, no `alpha`) is complete and yields only
  a mild aggregate sensitivity gain.
- The corrected true-Wiener two-stream rerun is complete. It improves mean
  sensitivity overall (`0.34875 -> 0.36839`) and helps moderate/high-amplitude
  and large-radius cells, but it still worsens the hardest low-amplitude
  subset.
- The original catastrophic true-Wiener result was traced to a partially
  populated auxiliary-channel cache (`12000 / 33000` finite rows). The active
  dataset loader now raises on non-finite auxiliary channels so this benchmark
  failure mode cannot silently recur.
- The deterministic template-fit and Bayesian handoff now exist for the frozen
  `24` cluster representatives, with `17` candidates flagged for projection
  caution in downstream follow-up.
- Real-sky follow-up is now materially closed on that frozen set:
  - common-resolution frequency jackknife stable: `23 / 24`;
  - HM sign-flip stable at `p <= 0.05`: `3 / 24`;
  - combined HM-plus-frequency survivors in the merged handoff: `3 / 24`.
- A new adaptive follow-up table now groups the `24` representatives into
  `10` sky systems using `r_assoc = clamp(theta_crit_fit / 2, 5 deg, 15 deg)`;
  only `2 / 10` systems contain any real-sky survivor, and `8 / 10` are still
  geometry-limited until projection-robust re-extraction.
- The true-Wiener two-stream branch is now explicitly `not promotable` as the
  default deployment branch. It remains a secondary synthetic result until
  deployment burden, candidate calibration, and follow-up stability are closed.

### P0.1 Patch-RMS Noise Floor Is Not A Physics Proof

Verdict: `valid`, with one overstatement.

Local evidence:

- `runs/phase3_unet/remediated_v1_noise_floor/noise_floor_report.md:3` says
  the diagnostic is not Bayesian evidence or an information-theoretic
  impossibility proof.
- `scripts/phase3_noise_floor_analysis.py:8-11` records the same limitation in
  code.
- `README.md:115-118` is appropriately cautious.
- `PROJECT_HANDOFF.md` now separates the low-amplitude CMB-confusion regime from
  higher-SNR algorithmic target cells.

Source evidence:

- McEwen et al. 2012 derive spherical matched filters that maximize filtered
  SNR for a known template in isotropic Gaussian CMB background and show the
  matched filter is superior to needlets and the unfiltered field.
- This does not prove the review's exact SNR numbers in this repo. Those
  numbers must be reproduced under this repo's beam, `C_ell`, mask, radius
  grid, and noise conventions before they are cited.

Required fix:

- Replace "physical/noise-floor limitation" wording with "current
  patch-screener limitation; not yet benchmarked against the covariance-aware
  matched-filter limit."
- Add a reproducible matched-filter SNR report:
  - New script target: `scripts/phase3_matched_filter_snr_curve.py`.
  - Inputs: same CAMB TT spectrum, beam, `noise_sigma_uk_arcmin`, radius grid,
    sign quadrants, and amplitude grid used by `remediated_v1`.
  - Outputs: SNR vs amplitude/radius table, matched-filter ideal recall at
    fixed FPR, and comparison against current ML recall on the same cells.

Acceptance criteria:

- `PROJECT_HANDOFF.md` no longer states weak recall as a proven physics limit.
- An artifact under
  `runs/phase3_unet/remediated_v1_matched_filter_snr/` reports SNR curves and
  explicitly states the assumptions under which they are valid.
- Current README/handoff wording says the lowest amplitudes are low-SNR under
  the ideal diagnostic, while higher-SNR cells remain model/pipeline targets
  until the same-grid benchmark closes.

### P0.2 Upper-Limit Framing Is Under-Contextualized

Verdict: `valid`.

Local evidence:

- `runs/phase3_unet/remediated_v1_upper_limits/upper_limits.md:3` correctly
  says this is not Bayesian evidence.
- The current ImageNet value is `Nbar_s^95 = 10.2674` at
  `runs/phase3_unet/remediated_v1_upper_limits/upper_limits.md:17-20`.
- `README.md:147-152` describes an efficiency-weighted Poisson upper limit and
  not a universal `lambda/B` mapping.
- `PROJECT_HANDOFF.md:112-121` lists the values without enough benchmark
  context.

Source evidence:

- Feeney et al. 2013 state that WMAP 7-year data constrain the expected number
  of detectable bubble collisions to fewer than `4.0` at 95% confidence.

Required fix:

- Rename paper-facing language from "upper limit" to
  "candidate-screening detectable-collision sensitivity" unless it is combined
  with a Feeney/OSS-style Bayesian evidence calculation.
- Add a benchmark table comparing:
  - This repo's candidate-screening Poisson number.
  - Feeney et al. 2013 WMAP `Nbar_s < 4.0`.
  - OSS-style amplitude-space constraints, clearly marked as not the same
    statistical object.
- Do not lead with `lambda/B` unless the exposure factor is model-specific and
  documented.

Acceptance criteria:

- README and handoff explicitly say the current `Nbar_s` number is weaker than
  Feeney 2013 and is not a Planck-collaboration or Feeney-style bound.
- The upper-limit artifact records the literature benchmark and the prior
  assumptions next to the project result.

### P0.3 True Classical Same-Grid Benchmark

Verdict: `valid`, now closed.

Local evidence:

- `runs/phase3_unet/remediated_v1_classical_same_grid_status/classical_same_grid_status.md:3`
  now says status is `complete`.
- `runs/phase3_unet/remediated_v1_same_grid_fullsky_manifest/same_grid_fullsky_report.json`
  provides the fixed-manifest comparison table across ImageNet, random-init,
  Wiener/Feeney, and SMHW methods.
- `README.md` and `PROJECT_HANDOFF.md` now state the closed-benchmark outcome:
  Wiener is the strongest average classical screener, while ML remains locally
  competitive in selected cells.
- `scripts/phase3_classical_filters.py:146-173` has a real harmonic
  covariance-weighted filter implementation with explicit Gaussian-beam and
  HEALPix-pixel-window transfer handling, and it has now been evaluated on the
  same stratified same-grid manifest as the ML branch.

Required fix:

- Generate or retain full-sky injected positive and negative maps for the
  `remediated_v1` amplitude/radius/zcrit grid.
- Run `scripts/phase3_classical_filters.py` on those maps with the same beam,
  pixel-window policy, mask, `lmax`, and `C_ell/noise` model.
- Extract scores at the same tile centers or candidate coordinates used by ML.
- Calibrate thresholds on the same null split and report recall/FPR, cell
  stratification, bootstrap intervals, and multiple-testing correction.

Implementation outcome:

- `scripts/phase3_same_grid_fullsky_benchmark.py` now streams full-sky
  injected maps, writes patches projected from those same maps, scores
  `wiener_feeney_matched_filter` and `smhw_screen` as blind local maxima, and
  can optionally score ML checkpoints on the generated HDF5.
- `scripts/phase3_classical_filters.py` now exposes `--pixel-window-policy`,
  so remediated products generated with `synfast(pixwin=True)` can be filtered
  with the matching `B_l P_l` transfer function.
- The closed stratified manifest reports mean recall `0.3841` for Wiener,
  `0.3517` for ImageNet, `0.2994` for random-init, and `0.2734` for SMHW.
  ImageNet beats Wiener in `17 / 35` raw cells, loses in `16 / 35`, and ties
  in `2`; under non-overlapping exact 95% CIs, ImageNet is better in `7` cells
  and worse in `8`.

Acceptance criteria:

- `scripts/phase3_classical_same_grid_status.py` returns `complete`.
- The report includes ImageNet U-Net, random-init U-Net, circular-template
  screen, Wiener Feeney matched filter, and SMHW screen on the same rows and
  thresholds.
- No README/handoff sentence claims ML beats classical filters unless this
  artifact supports it. Current wording correctly limits the claim to
  complementary screening.

### P0.4 Injection Convention Needs Provenance, Not A Bug Rewrite

Verdict: `invalid as stated`, but still paper-blocking until documented.

Local evidence:

- `scripts/phase2_signal_model.py:161-182` injects
  `(1 + signal) * (T_CMB_K + patch) - T_CMB_K`.
- `scripts/phase2_observing_model.py:230-232` uses the same spherical injection.
- `scripts/phase2_physics_checks.py:85-113` explicitly tests the Feeney
  full-temperature convention and quantifies the additive cross term.
- Several downstream scripts previously reconstructed injected positives with
  `signal * (T_CMB_K + base)`, including
  `scripts/phase3_real_sky_injection.py:482-492` and
  `scripts/phase3_two_pass_policy.py:169-179`; this should remain centralized
  through the shared signal-model convention helper.
- A direct search found no pure `T_anisotropy *= (1 + f)` path. The code's
  algebra is the Feeney full-temperature modulation:
  `patch + signal*T_CMB_K + signal*patch`.

Source evidence:

- Feeney et al. 2011 PRD writes the temperature modulation as
  `(1 + f(n)) * (1 + delta(n)) - 1` and simulates
  `(1 + f(n)) * (T0 + delta T_syn(n)) - T0`.
- McEwen et al. 2012 states that bubble collisions have modulative and
  additive contributions, the modulative component is second order, and the
  optimal-filter search uses the additive contribution.

Scientific interpretation:

- The current implementation is consistent with Feeney 2011's physical
  reheating/modulation interpretation.
- The review is right that the true Wiener matched filter is an additive
  source-plus-noise detector. For small `|f|`, the Feeney multiplicative form is
  first-order equivalent to `patch + f*T_CMB`; the final `f*patch` term is the
  second-order modulation term and must be quantified and named.
- The paper should not imply that all primary sources use one identical
  convention.

Required fix:

- Add an explicit `injection_convention` metadata field everywhere generated
  data is written:
  - `feeney2011_full_temperature_modulation`: current convention.
  - `mcewen2012_first_order_additive`: `patch + f*T_CMB_K`, for filter-design
    and same-grid classical benchmark tests.
- Update docstrings from generic "temperature modulation" to specify whether
  the artifact is Feeney full-temperature modulation or McEwen first-order
  additive.
- Extend `scripts/phase2_physics_checks.py` to report the maximum and RMS
  fractional difference between exact Feeney modulation and first-order
  additive injection across the production amplitude/radius grid.
- For the same-grid classical benchmark, either:
  - score additive-injected full-sky maps, or
  - score Feeney-modulated maps and explicitly show that the first-order
    additive delta is negligible for all benchmark cells.

Acceptance criteria:

- Every newly generated HDF5 artifact has `summary/injection_convention`; legacy
  artifacts without the field emit an audit warning until regenerated. Artifacts
  with `provenance_schema_version = injection_convention_v1` or
  `created_utc >= 2026-04-20T00:00:00Z` fail closed if the metadata is missing.
- Generator CLIs expose `--injection-convention` so Feeney remediated products
  and McEwen-additive benchmark products can be generated without editing source
  constants.
- README and handoff explain why `remediated_v1` uses Feeney full-temperature
  modulation and why the classical benchmark's effective template is additive
  or equivalent to first order.
- No paper-facing text says the project uses a nonstandard injection unless the
  selected artifact actually does.

Benchmark-design caveat:

- If P0.3 generates McEwen-additive full-sky maps directly, there is no
  Feeney-vs-additive cross term. The `phase2_physics_checks.py` cross-term
  ceiling applies only to the alternate design where Feeney-modulated products
  are scored with an additive matched filter.

## Medium-Severity Findings

### P1.1 Unit Handling Needs A Hard Gate

Verdict: `valid`.

Local evidence:

- Astropy FITS-header inspection confirms all local PR3 files
  `COM_CMB_IQU-{commander,nilc,sevem,smica}_2048_R3.00_full.fits` have
  `TUNIT1 = K_CMB`.
- `scripts/phase3_classical_filters.py:45-57` rejects maps with max absolute
  anisotropy above `1 K`, which catches many microkelvin/Kelvin confusions but
  is not a full FITS unit audit.

Required fix:

- Add `scripts/phase0_planck_unit_audit.py`.
- It should read FITS headers, record `TUNIT*`, `BUNIT`, `NSIDE`, `ORDERING`,
  map min/max/RMS in K and uK, and fail if the units do not match the requested
  map family.
- It should write a JSON report and be called by `scripts/audit_remediated_flow.py`.

Acceptance criteria:

- Local PR3 maps pass and record `K_CMB`.
- Any PR4/frequency-map path must pass an explicit unit conversion policy
  before injection or filtering.

### P1.2 PR4/NPIPE Four-Map Consistency Claim Needs Guardrails

Verdict: `valid`.

Source evidence:

- The official Planck Legacy Archive wiki says PR3 provides CMB maps for
  Commander, NILC, SEVEM, and SMICA.
- The same Planck Legacy Archive wiki says the 2020 NPIPE CMB full-frequency
  and A/B maps were component separated using Commander and SEVEM.

Required fix:

- Keep the four-method cross-consistency result scoped to PR3 unless the repo
  has source-backed SMICA/NILC NPIPE products.
- Add a map-release field to candidate and null-control metadata:
  `PR3_R3.00`, `PR4_NPIPE_official`, or `PR4_NPIPE_community`.

Acceptance criteria:

- No PR4/NPIPE report assumes four official component-separated maps unless
  all four products are present and provenance-tagged.

### P1.3 Constant 15 Deg Clustering Is A Convention, Not A Final Science Choice

Verdict: `partly valid`.

Local evidence:

- `scripts/phase3_emit_tile_constrained_candidates.py:69` defaults to
  `--cluster-radius-deg 15.0`.
- `PROJECT_HANDOFF.md:286-288` already says the paper narrative still needs a
  final cluster-radius convention.
- The current clustering is candidate-volume accounting on overlapping tiles,
  not a source-detection matching radius. Calling it a direct violation of
  Planck catalog matching conventions is overstated.

Required fix:

- Add scale-dependent candidate clustering and sensitivity reports:
  - fixed radii: `5, 10, 15, 20 deg`.
  - scale-dependent radii: `theta_crit / 2`, `theta_crit`, and one beam-like
    lower bound for small objects.
- Report candidate burden and duplicate-merge rates under all conventions.

Acceptance criteria:

- The paper-facing candidate table states the clustering convention and includes
  a robustness appendix.

### P1.4 Gnomonic Large-Radius Systematics Need Quantification

Verdict: `valid`.

Local evidence:

- `scripts/phase2_physics_checks.py:116-138` already computes gnomonic geometry
  diagnostics.
- Current patches are `55.47 deg` wide, so large signals near patch edges can be
  distorted in Euclidean patch space.

Required fix:

- Add a projection-systematics report for radii `15, 20, 25 deg` and offsets
  from patch center.
- Compare gnomonic patch score shifts against native-sphere score shifts.
- Decide one of:
  - restrict large-radius candidate claims to a central safe zone;
  - switch large-radius templates to ZEA/equal-area patches;
  - run large-radius filters natively on the sphere.

Acceptance criteria:

- `runs/phase3_unet/remediated_v1_projection_systematics/` documents recall and
  score bias versus radius and offset.

### P1.5 Mask Threshold After Apodization Needs Documentation

Verdict: `partly valid`.

Local evidence:

- The active generator/audit contract uses `mask_threshold = 0.9`
  (`README.md:46-48`, `PROJECT_HANDOFF.md:72-73`).
- The review's claim that thresholding "defeats apodization" may be true for
  final spherical likelihoods, but current patch screening uses projected
  mask-fraction eligibility and does not yet implement a final likelihood.

Required fix:

- Record whether every mask is binary, apodized, thresholded, or a projected
  fraction.
- For final spherical filters and evidence calculations, use the unthresholded
  apodized mask as a weight unless a source-backed alternative is documented.

Acceptance criteria:

- Mask provenance appears in HDF5 metadata and classical-filter artifacts.
- Same-grid classical benchmark states whether it uses binary exclusion or
  weighted apodized masking.

### P1.6 Frequency Jackknife Is Worth Adding

Verdict: `valid`.

Reasoning:

- Four component-separated maps are not independent enough to rule out shared
  frequency-domain foreground residuals.
- Candidate-level frequency jackknifes are a scientifically cheap way to stress
  shared-foreground failure modes.

Required fix:

- For frozen candidates, rerun extraction and scoring after dropping each
  informative frequency channel, or after using source-backed foreground-reduced
  frequency products.
- Normalize the jackknife by the expected score response under the
  no-foreground-contamination hypothesis. Frequency channels have different
  beams and noise levels, so a raw score drop is not by itself evidence for a
  foreground residual.
- Implement this as either:
  - a common-resolution analysis, with all maps smoothed to the worst beam
    before scoring; or
  - a frequency-specific transfer-function analysis, with expected score shifts
    calibrated from injected sources and null skies using each channel's beam
    and noise model.
- Record normalized score stability, sign stability, and fitted-template
  amplitude stability.

Acceptance criteria:

- Every paper-facing candidate has cross-map, half-mission, and
  SNR-normalized frequency-jackknife rows.

### P1.7 Radius Prior Needs A Split Report

Verdict: `valid`.

Reasoning:

- The current grid is useful for ML sensitivity and small-radius screening.
- Theory-motivated bubble-size priors can favor larger angular scales, where
  gnomonic patching and CMB cosmic variance behave differently.

Required fix:

- Split all prior-weighted summaries into:
  - `small_radius_screening`: current `5-25 deg` operational grid.
  - `large_radius_theory_weighted`: source-backed prior extending toward
    half-sky scales, likely requiring native-sphere filters.

Acceptance criteria:

- Upper-limit and sensitivity summaries state which radius prior they use and
  never mix the two without a weight table.

### P1.8 Matched-Filter + ML Score Fusion Should Be Tested

Verdict: `valid`.

Reasoning:

- Closing the true classical benchmark should not only answer "ML versus
  matched filter." If the U-Net and Wiener-Feeney filter make partially
  independent errors, the scientifically optimal screen is a calibrated score
  ensemble.
- The current repo already found value in constrained composite policies, so a
  same-grid ML/classical fusion test is a natural extension rather than a new
  model family.

Required fix:

- After the same-grid Wiener/SMHW benchmark exists, fit a small pre-registered
  fusion model on the calibration split only:
  - inputs: U-Net score, circular-template score, Wiener-Feeney SNR, SMHW score,
    mask fraction, and radius/scale metadata if allowed by the benchmark design;
  - candidates: logistic regression, monotone gradient boosting, or a
    constrained `k-of-n` rule;
  - outputs: recall/FPR, precision, null burden, and per-cell gains against each
    standalone score.
- Report whether fusion improves recall at fixed FPR without increasing
  full-sky candidate burden beyond the deployment constraints.

Acceptance criteria:

- Fusion is evaluated on the same held-out rows, full-sky tile audit, and
  candidate-burden constraints as standalone ML and classical scores.
- Any paper claim says whether the promoted policy is ML-only, classical-only,
  or fused.

## Lower-Severity Or Roadmap Findings

### P2.1 GKF, SMHW Cold Spot, Minkowski, S2LET

Verdict: `partly valid`.

Action:

- Add GKF/peak-statistics and SMHW Cold Spot reproduction as validation
  benchmarks before submission.
- Treat Minkowski functionals and directional S2LET wavelets as optional unless
  the paper expands into broad non-Gaussianity testing or oriented features.

### P2.2 SBI And Posterior Handoff

Verdict: `valid roadmap`, not a blocker for the current screening paper if the
paper does not claim posterior inference.

Action:

- Keep empirical null-survival scores as screening p-values.
- Add SBI or template-likelihood posterior only after HM/frequency-vetted
  candidates are frozen.

### P2.3 Domain Adaptation And Self-Supervised Pretraining

Verdict: `valid model-development direction`.

Action:

- Prioritize real-null backgrounds plus injected signals before another
  CAMB-only branch.
- Add PySM3/ForSE-style foreground randomization and a DANN/Sinkhorn domain
  adaptation branch only after injection provenance and the same-grid classical
  benchmark are stabilized.

### P2.4 Scoop Landscape Claim

Verdict: `partly valid`, with a verified scoop-neighboring exception.

Evidence:

- CMBubbles is real as a UNSW/Indico conference project and appears classical,
  not a peer-reviewed ML segmentation pipeline.
- Phys. Rev. D 108, 043525 / arXiv:2303.08869 is a peer-reviewed CNN search for
  pairwise CMB hotspots from cosmological particle production. It is not a
  bubble-collision pipeline, but it is close enough that the introduction must
  cite it and explain the distinction: paired hotspot classification in
  idealized simulations versus bubble-collision candidate screening on Planck
  component-separated maps with real-null calibration.
- Other non-primary search results surfaced possible ML bubble-collision claims.
  They were not validated as arXiv/DOI-backed publications in this pass and
  should not be cited as science, but they are enough to make "scoop landscape
  is clear" unsafe without a formal search log.

Action:

- Add `docs/literature_watch.md` with query strings, dates, inclusion/exclusion
  criteria, and direct links.
- Monitor arXiv, ADS, INSPIRE, CMB-S4/SO proceedings, and GitHub for:
  `bubble collision CMB machine learning`, `DeepSphere bubble collision`,
  `CMBubbles Maheshwari Hamann`, and `remote dipole bubble collision`.
- Before drafting the introduction, write a comparison paragraph for
  arXiv:2303.08869 that states why this project is not claiming the same signal,
  dataset, null model, or inference target.

### P2.5 Polarization And Remote-Field Future Direction

Verdict: `valid discussion item`.

Evidence:

- Cai, Zhang, and Guan 2025/2026, arXiv:2510.12134, forecast bubble-collision
  constraints using CMB remote dipole and quadrupole fields. Their abstract
  reports that RQF reconstruction can improve constraining power by about an
  order of magnitude for CMB-S4-like and LSST-like experiments.

Action:

- Add a Discussion paragraph that temperature-only Planck screening is a
  reproducible current-data baseline, while next-decade sensitivity should use
  polarization-assisted and remote-field observables from SO/CMB-S4 plus LSST.
- Do not imply the current repo reaches the remote-field sensitivity floor.

## Claim-By-Claim Verdict Table

| Review claim | Verdict | Fix |
|---|---|---|
| Patch RMS is the wrong detectability metric for a localized template. | valid | Add matched-filter SNR/ROC artifact; soften "physics limit" language. |
| Exact SNR values in the review can be quoted now. | needs verification | Reproduce under repo beam, `C_ell`, noise, mask, and radius assumptions first. |
| Multiplicative injection is inconsistent with every primary source. | invalid as stated | Feeney 2011 uses full-temperature modulation; McEwen 2012/OSS use a first-order additive approximation for filters. Add explicit convention metadata and tests. |
| Filters expect an additive source-plus-noise template. | valid | Same-grid classical benchmark must state additive convention or prove the Feeney second-order cross term is negligible. |
| `Nbar_s^95 = 10.27` is weaker than Feeney 2013 WMAP `4.0`. | valid | Reframe as candidate-screening sensitivity, not competitive cosmological bound. |
| Planck did not publish a dedicated Feeney-style bubble-collision `Nbar_s` bound. | likely valid | Keep as source-backed literature statement after final bibliography check. |
| OSS exact likelihood is the optimal ceiling under masked/inhomogeneous noise. | valid direction | Use as the final classical/Bayesian benchmark target, not just McEwen isotropic filters. |
| Radius prior should split small-radius screening from theory-weighted large radii. | valid | Add split prior reports and avoid one pooled headline. |
| PR4/NPIPE four-map consistency is unavailable with official maps. | valid caution | Tag map release and method provenance; do not assume four official PR4 maps. |
| Constant `15 deg` clustering violates catalog conventions. | partly valid | It is a candidate-volume convention, not source matching; add scale-dependent sensitivity. |
| Gnomonic projection is a large-radius systematic. | valid | Quantify and either restrict, switch projection, or use native-sphere filtering. |
| Mask thresholding after apodization is nonstandard. | partly valid | Document mask provenance; use weighted apodized masks for final likelihood/filter stages. |
| GKF/SMHW/SBI/domain adaptation should be considered. | valid roadmap | Add after P0 benchmark closure; do not let these replace core physics fixes. |
| No competing ML bubble-collision pipeline exists. | partly valid | No direct peer-reviewed ML bubble-collision pipeline was verified, but arXiv:2303.08869 is a scoop-neighboring CNN pairwise-hotspot paper that must be cited and differentiated. |
| Frequency jackknife can use raw score drops. | invalid | Normalize frequency jackknife score changes against beam/noise-specific expected shifts or use common-resolution maps. |
| Matched filter and ML should only be compared sequentially. | partly valid | Close standalone same-grid benchmarks, then test calibrated ML/classical score fusion on the same null and burden constraints. |
| Polarization/remote-field extensions can be ignored. | invalid for discussion | Add a next-decade SO/CMB-S4/LSST paragraph, citing arXiv:2510.12134. |

## Stale Or Risky Local Files To Update

- `README.md`
  - `README.md:46`: keep the wording on Feeney full-temperature modulation and
    the McEwen/OSS first-order additive classical-filter approximation.
  - `README.md:97-119`: keep diagnostic and link to the matched-filter SNR
    report.
  - `README.md:147-152`: add Feeney 2013 benchmark context.
  - `README.md:448-506`: keep the now-closed same-grid benchmark framed as a
    screening comparison and do not promote it into a central physical-limit
    claim.
- `PROJECT_HANDOFF.md`
  - `PROJECT_HANDOFF.md:100-110`: keep softened noise-floor interpretation.
  - `PROJECT_HANDOFF.md:112-121`: add Feeney 2013 benchmark context.
  - `PROJECT_HANDOFF.md:281-285`: keep the split between low-amplitude
    CMB-confusion and higher-SNR algorithmic targets.
- `scripts/phase2_signal_model.py`
  - `scripts/phase2_signal_model.py:149-183`: keep the Feeney implementation,
    name the convention explicitly, and expose the first-order additive
    alternative for benchmark generation.
- `scripts/phase2_observing_model.py`
  - `scripts/phase2_observing_model.py:192-233`: keep convention metadata and
    additive alternative.
- `scripts/phase2_physics_checks.py`
  - `scripts/phase2_physics_checks.py:85-113`: test Feeney full-temperature
    modulation and McEwen first-order additive; quantify the cross term.
- `scripts/phase3_real_sky_injection.py`,
  `scripts/phase3_real_sky_smoothed_sensitivity.py`,
  `scripts/phase3_two_pass_policy.py`,
  `scripts/phase3_visualize_smoothed_examples.py`, and
  `scripts/phase3_nside512_probe.py`
  - These should reconstruct positives through the shared convention helper and
    write convention metadata.
- `docs/phase3_matched_fpr_sensitivity_decision.md:140` and
  `docs/project_structure.md:12`
  - Update multiplicative wording and stale comparator naming.

## Comprehensive Fix Queue

1. P0: Add injection-convention metadata and dual Feeney full-temperature
   modulation/McEwen first-order additive tests.
2. P0: Maintain the completed same-grid Wiener/SMHW benchmark as a regression
   artifact and rerun it when the screening stack changes materially.
3. P0: Implement matched-filter SNR curves under repo assumptions and publish
   SNR-vs-ML ROC tables. The initial artifact exists at
   `runs/phase3_unet/remediated_v1_matched_filter_snr/`; final claims should be
   updated against the now-closed same-grid benchmark rather than against
   pre-benchmark assumptions.
4. P0: Reframe upper-limit reports against Feeney 2013 and mark the current
   value as candidate-screening post-processing.
5. P0: Update handoff/README language using the completed same-grid benchmark
   and SNR report, not pre-benchmark guesswork.
6. P1: Add FITS unit/provenance audit and fail on unit ambiguity.
7. P1: Add map-release provenance and PR4/NPIPE guardrails.
8. P1: Add scale-dependent cluster-radius and candidate-burden sensitivity.
9. P1: Quantify gnomonic projection bias for large radii and off-center
   injections.
10. P1: Record mask provenance and use weighted apodized masks for final
    spherical likelihood/filter stages.
11. P1: Add SNR-normalized frequency jackknife for frozen candidates once
    frequency products are selected.
12. P1: Split small-radius operational and large-radius theory-weighted prior
    reports.
13. P1: Test matched-filter + ML score fusion after the standalone same-grid
    benchmark is closed.
14. P2: Add GKF peak-statistics and SMHW Cold Spot reproduction benchmarks.
15. P2: Build SBI/template-likelihood posterior handoff after candidates are
    HM/frequency-vetted.
16. P2: Add real-null/domain-adaptation model branch only after P0 benchmark
    closure.
17. P2: Maintain a formal `docs/literature_watch.md` novelty and scoop-risk log,
    including arXiv:2303.08869.
18. P2: Add polarization/remote-field future-work discussion, citing
    arXiv:2510.12134.

## Dependency-Ordered P0 Plan

Do not finalize README or paper-facing scientific wording before the benchmark
results exist. That benchmark now exists, so the remaining P0 dependency order
is:

1. Close P0.4 injection-convention metadata/tests so `remediated_v1` is explicitly
   Feeney full-temperature modulation and the first-order additive benchmark
   path is available.
2. Close P0.1 matched-filter SNR curves and compare them with the same-grid
   recall/FPR results.
3. Then perform P0.2 upper-limit reframing and P0.1 README/handoff wording fixes
   using the actual benchmark outcome: ML matches, ML lags, ML exceeds, or fused
   ML/classical policy wins.

## Source Links Used

- Feeney, Johnson, Mortlock, Peiris 2011 PRD:
  https://arxiv.org/abs/1012.3667
- McEwen, Feeney, Johnson, Peiris 2012:
  https://arxiv.org/abs/1202.2861
- Feeney et al. 2013 WMAP hierarchical constraints:
  https://arxiv.org/abs/1210.2725
- Osborne, Senatore, Smith 2013 search:
  https://arxiv.org/abs/1305.1964
- Osborne, Senatore, Smith 2013 methodology:
  https://arxiv.org/abs/1305.1970
- Planck PR3 CMB maps:
  https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/matrix_cmb.html
- Planck Legacy Archive CMB maps / NPIPE notes:
  https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_maps
- CMBubbles Indico entry:
  https://indico.global/event/12919/contributions/113513/
- UNSW CMBubbles profile:
  https://www.unsw.edu.au/science/our-schools/physics/about-us/our-people/research-students/jahanvi-maheshwari
- Adjacent CMB CNN localized-signal paper:
  https://arxiv.org/abs/2303.08869
- Remote dipole/quadrupole bubble-collision forecast:
  https://arxiv.org/abs/2510.12134
