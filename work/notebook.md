# Research notebook: CMB bubble-collision screener (continued, November 2026)

Rules: dated entries, appended in order, never silently rewritten (corrections are new entries that name what
they correct). Time stamps come from the clock at the moment of writing. Predictions are written before the
analysis they concern. Bugs found are logged, not quietly patched.

Layout on Hamilton (`/projects/$USER/cmb`, private):
- `CMB-Collision-Bubbles/` = the project tree, restored from the lab-box backup. Restored files are
  write-protected, so a rerun cannot silently overwrite an April result; new outputs get new names.
- `restore/RESTORE_MANIFEST.tsv` = size, CRC-32 and date of every restored file.
- New code goes into the existing `scripts/` folders as new files; this notebook is mirrored to
  `work/notebook.md`.

---

## (a) 2026-11-21 01:10 CST. Restore started

- Source: the lab-server backup zip (124,789,975,292 bytes), holding the lab-box tree as of 2026-04-23
  (2,592 project files, 132.9 GB).
- Upload to Hamilton in 30 chunks of 4 GiB, each compared by SHA-256 on arrival. Measured 56 to 63 MiB/s.
- Extraction reads the chunks directly and checks every file's CRC-32 (script tested locally on a synthetic
  split zip: identical output). The one `.env` file (old API keys) is not extracted.
- Environment: the project's own `environment.yml` versions (python 3.11, torch 2.5.1, healpy 1.19.0,
  camb 1.6.6, segmentation-models-pytorch 0.5.0, timm 1.0.26), built with venv + pip.
  First attempt (Slurm job 82902 on node c2) failed: c2 has no outbound internet. Rebuilt on the login node.

## (b) 2026-11-21 01:10 CST. The edge term z_crit: what the literature says

Read from the primary papers (texts downloaded during the literature review).

| Year | Source | Statement |
|---|---|---|
| 2009 | Chang, Kleban, Levi, arXiv:0810.5128 | Disc with temperature linear in cos(theta). No temperature jump predicted. |
| 2011 | Feeney et al., arXiv:1012.3667 | Introduces z_crit as a free parameter: "we allow bubble collisions to produce modulations with and without discontinuities"; argues ISW and Doppler terms could make the temperature discontinuous even if the potential is not. Their edge detector finds no circular discontinuities in WMAP 7-year data. |
| 2011 | Kleban, Levi, Sigurdson, arXiv:1109.3473 | Temperature "decreases linearly in cos theta, and then smoothly transitions to a constant at theta = theta_c (with no edge or discontinuity)"; the kink spreads at the sound speed and stays within the sound horizon. |
| 2012 | Gobbetti and Kleban, arXiv:1201.6380 | During inflation "an initial discontinuity becomes a kink"; the kink is sharp for thin walls and smoother for thick walls. |
| 2012 | McEwen et al., arXiv:1202.2861 | "we restrict our attention to z_crit ~ 0". |
| 2013 | Feeney et al., arXiv:1210.2725 | "remove the edge discontinuity parameter from our analysis ... there is no discontinuity in temperature at the causal boundary". |
| 2013 | Osborne, Senatore, Smith, arXiv:1305.1970 and 1305.1964 | Keeps two models at the level of the primordial curvature: "ramp" and "ramp+step"; "there is some disagreement in the literature as to which ... is better motivated". The CMB transfer function changes the ramp profile by about 10% and the step profile at order unity. For the step, "Planck can potentially improve the optimal WMAP constraints". |
| 2014 | Wainwright et al., arXiv:1312.1357, 1407.2950 | Full-GR simulations: "the collision boundary does not result in a discontinuous jump"; "there is no sharp edge to the collision boundary"; the kink "is smoothed by the subsequent evolution". |
| 2015 | Feeney, Elsner, Johnson, Peiris, arXiv:1506.01716 | Template = linear plus quadratic curvature profile through the full transfer functions. The linear term gives "a comparatively sharp feature" at the edge from the discontinuous first derivative, strongest in polarization. |
| 2015 | Zhang and Johnson, arXiv:1501.00511 | In the kSZ signal the slope discontinuity appears as a velocity discontinuity at the edge. |

Conclusion: a free temperature jump at the edge was dropped by the search team in 2012 and 2013 and is
contradicted by the 2014 simulations. It came back once as an agnostic alternative (the 2013 ramp+step
model, defined in curvature and smoothed by the transfer function), and an edge signature survives as a kink
that matters for polarization and kSZ. The project's generator injects a temperature jump z_crit directly
on the sky, which no post-2011 theory paper supports.

Planned fix (not yet done): build the template from the curvature profile through CAMB transfer functions
(ramp; ramp+step as a separate labelled hypothesis), and check it three ways: the Sachs-Wolfe limit, the
Osborne et al. profiles, and the 2015 forecast templates.

## (c) 2026-11-21 01:10 CST. Pre-registration: how much do the detectors lean on the edge?

Data: the existing sensitivity grid (`runs/phase3_unet/remediated_v1_sensitivity_curve/`), 35 amplitude x
radius cells, 200 positives per cell per edge ratio |z_crit| / |z0| in {0, 0.5, 1, 2}, 5,000 negatives,
per-row scores already saved for the ImageNet U-Net, the random-initialised U-Net and the circular template
screen. Thresholds: each method's own 5% false-positive threshold on the negatives, as in the original
report. No retraining, no new scoring.

Quantity: detection probability by edge ratio, pooled over the cells with z0 from 1e-5 to 5e-5 (where
detection is neither at chance nor saturated), with binomial 95% intervals.

Predictions, written before looking:
1. The U-Net's detection probability rises with the edge ratio. At ratio 0 it is at least 20% lower
   (relative) than the value pooled over all four ratios.
2. The circular template screen also rises with the ratio (it has a ring term).
3. The U-Net's lead over the circular screen is smaller at ratio 0 than pooled.
I am least sure of 3.

If 1 holds, the published-style numbers (0.35 mean recall and the comparison with the Wiener filter)
overstate sensitivity to the signal theory actually predicts, and every benchmark must be redone on the
corrected template before any claim.

## (d) 2026-11-21 01:10 CST. Verification list before anything new is built

Nothing from April counts as verified until it reproduces from the restored files.

1. Restore integrity: file count and CRC-32 of every file (from the extraction job).
2. The project's own quality gates and flow audit run clean in the new environment.
3. Re-evaluate the saved ImageNet U-Net checkpoint on the held-out test split: expect precision 0.888,
   recall 0.396, false-positive rate 0.050 at threshold 0.96.
4. Re-score the sensitivity grid with the saved checkpoint: expect the saved per-row scores to within
   numerical noise.
5. Code read against the papers: template formula, units (K against uK), beam and pixel window, coordinate
   conventions, injection, Wiener filter normalisation, split leakage, threshold calibration, mask handling.

## (e) 2026-11-21 01:19 CST. Three geometry bugs found by reading the code, then measured

(Stamp note: first typed as 01:26; the clock read 01:19 when this was written, so it was corrected on the spot.)

Method: `scripts/analysis/check_patch_geometry.py` (new) projects maps of the HEALPix pixel directions
through the same `hp.gnomview(reso=13 arcmin, xsize=256, return_projected_map=True)` call the project
uses, which gives the true sky direction of every patch pixel. Run on Hamilton with the project's own
`scripts/core` modules (on the login node, about 01:17; output `patch_geometry_check.json`).

**B1. The signal's angle grid is not the projection's.** `make_angular_distance_grid` (in
`phase_dataset_utils.py` and `phase2_signal_model.py`) places pixel n at angle n x 13 arcmin and then takes
the tangent. A gnomonic patch is linear in the tangent-plane coordinate, so pixel n is at arctan(n x 13
arcmin). Measured: code grid against truth differs by up to 2.2 deg for a centred signal and up to 3.2 deg
off-centre; a linear-plane grid matches truth to 0.06 to 0.16 deg. A centred disc labelled 25 deg is truly
23.6 deg on the axes and 24.2 deg on the diagonals (20 -> 19.3 / 19.6; 15 -> 14.7 / 14.8; 10 -> 10.0).
Off-centre it is worse: a disc labelled 5 deg centred at pixel (30, 220) is truly 4.4 deg.
Affects: every patch-injected data set (training, calibration, test, sensitivity grid): radius labels are
up to 6% (centred) or 13% (off-centre) too large and discs are slightly non-circular. Backgrounds are fine.

**B2. `offset_signal_vector` is mirrored left-right.** Asked for pixel x = 60, the signal lands at x = 196
(x = 200 -> 53; x = 210 -> 42); y is correct. In the array healpy returns, column index increases to the
west and row index increases to the north. Affects: anything injected on the sphere at an off-centre pixel
position (the same-grid benchmark), where the stored nominal position and any truth mask drawn from it are
on the wrong side. Image-level detection rates are not affected; position-based quantities are.

**B3. `peak_sky_coord` has both axes reversed.** `phase3_fullsky_tile.py` converts a candidate's peak
pixel to Galactic coordinates assuming row 0 is north and columns increase eastward. Both are the opposite
of the measured orientation, so every reported candidate position is the true position reflected through
its patch centre (moved by 9 to 53 deg). Reproducing the reported positions with the code's convention
gives 0.00 deg error for all 24, so the bug is modelled exactly.

Corrected positions of the 24 frozen candidates (col -> west, row -> north, true gnomonic inverse):

| location | corrected (l, b) | tiles (maps) | scatter between tiles | nearest published WMAP candidate |
|---|---|---|---|---|
| A | (184.3, -50.9), (165.2, -48.9), (184.2, -51.6) | 9628 (nilc, commander), 9629 (all four), 10138 (smica, sevem) | 9628 and 10138 agree to 0.7 deg | McEwen 10 = Feeney feature 1 (172.6, -46.6; radius 20 deg): 5.5 and 8.8 to 9.1 deg |
| B | (203.2, -57.1), (203.3, -56.9), (201.1, -56.0) | 10058 (nilc, commander, sevem), 10411 (sevem) | 1.6 deg | Cold Spot, McEwen 7 (207.1, -57.4; radius 5 deg): 2.1 to 3.5 deg |
| C | (150.4, -52.6), (148.3, -52.7), (148.3, -53.6) | 9847 (nilc, smica, commander), 10140 (sevem), 10398 (commander) | 1.5 deg | none within its own radius; 15 to 17 deg west of McEwen 10 |
| D | (205.2, -24.2), (204.8, -24.3), (212.4, -20.3) | 9293, 9289 (smica), 9034 (nilc) | 0.4 deg (smica pair) | none; this is the Orion region near the mask edge |
| E | (277.9, -39.3), (277.8, -38.3), (274.8, -38.6), (271.3, -35.2) | 8410, 10416, 9178, 9981 (smica only) | 1 to 7 deg | McEwen 13 (282.0, -46.6; radius 40 deg): 7.9 to 13.9 deg; the LMC is at (280.5, -32.9) |

Independent confirmation that the corrected orientation is right: under the reported coordinates,
candidates from different tiles were scattered by 15 to 50 deg; under the corrected ones, different tiles
and different component-separated maps converge on the same sky positions to within about 1.5 deg.

Consequences:
- The 24 candidates are about five sky locations, not ten "systems". The April 15 deg clustering, the
  scale-dependent grouping and the paper survivor table were built on reflected coordinates.
- The candidate-centred equal-area re-extraction was centred 9 deg (tile 9629) and 21 deg (tile 9628) away
  from the true peaks. The verdict that 9628 is "centre-unstable" is therefore unreliable and must be redone.
- Half-mission and frequency tests work on whole tiles and are not affected.
- Correction to what was said earlier in this review: the first cross-match (4.0 deg from McEwen 10; three
  candidates 4 to 6 deg from the Cold Spot) used the reported coordinates. The association of the top
  location with McEwen 10 survives (the tile centre itself is 1.6 deg from it) but the separations are 5.5
  and 9 deg, not 4.0. The three candidates first matched to the Cold Spot were chance matches of wrong
  coordinates; the real Cold Spot hits are tiles 10058 and 10411, at 2.1 to 3.5 deg.
- Lesson for the chance-coincidence estimate: wrong coordinates produced three "matches" within 6 deg, so
  the estimate must respect how concentrated the candidates are on the sky. To be pre-registered.

Not yet done: fix the three functions (as new, tested helpers, with the old ones kept for provenance),
re-run clustering and follow-up on corrected coordinates.

## (f) 2026-11-21 01:34 CST. Collision template through CAMB transfer functions: first results, one open discrepancy

New module `scripts/core/collision_template_transfer.py` and check script
`scripts/analysis/check_collision_template.py`. Model: Osborne, Senatore, Smith (arXiv:1305.1970) Eqs. 4-8,
curvature ramp a(z - r) for z >= r, optional step, harmonic coefficients
b_l = 2 * integral dk Delta_l(k) [i^l zeta~(k) + (-i)^l zeta~(-k)], which for the ramp is
-4a * integral dk Delta_l(k) cos(kr - l pi/2) / k^2 and for the step 4 a_s * integral dk Delta_l(k)
cos(kr - (l-1) pi/2) / k. Monopole and dipole dropped.

Checks run (Hamilton login node, output `template_check/`):
1. Integrator against the analytic answer, using a pure Sachs-Wolfe transfer -j_l(kD)/5: ramp reproduced to
   2e-3 (10 deg), 4e-5 (25 deg), 3e-5 (60 deg) of the central amplitude; step to 1.2e-2, 1.8e-3, 4e-4 of the
   step height when compared with exact Legendre coefficients. (A first comparison showed 4 to 8% for the
   step; that was the reference, a quadrature of a discontinuous function, not the integrator.)
2. CAMB normalisation: C_l rebuilt as 4 pi * integral dlnk P(k) Delta_l^2 equals CAMB's own unlensed TT C_l
   with ratio 1.0000 at l = 2, 10, 30, 100, 220, 500. CAMB gives 1,067 k values, median spacing 1.27e-4 /Mpc
   (3.6 per oscillation period 2 pi / D*), D* = 13,872.8 Mpc. Low-l transfer = -0.91 * j_l(kD)/5 at l = 10.
3. Profiles through the full transfer function, in units of the Sachs-Wolfe central amplitude:

| theta_c | model | centre | at theta_c | 2 deg outside | Sachs-Wolfe centre / edge |
|---|---|---:|---:|---:|---|
| 10 | ramp | 2.89 | 1.54 | 0.66 | 1.01 / 0.03 |
| 20 | ramp | 1.38 | 0.29 | -0.10 | 0.94 / -0.05 |
| 40 | ramp | 0.84 | -0.12 | -0.22 | 0.78 / -0.18 |
| 70 | ramp | 0.45 | -0.28 | -0.30 | 0.45 / -0.30 |

   (Values are not exactly 1 and 0 because the monopole and dipole are removed.)
   The ramp through the transfer function has no sharp kink. Its boundary sits about one sound horizon
   outside the causal edge (10 -> about 13 deg, 20 -> about 21.7 deg, as cos(theta) = cos(theta_c) - r_s/D
   predicts with r_s = 145 Mpc), and small discs are much brighter than the Sachs-Wolfe ramp. The step
   shows a peak of about 2.5 step heights just outside the edge.

Analytic cross-check of the size of the effect. In the long-wavelength limit the baryon-photon velocity is
v = -(tau/3) grad(Phi) = a tau*/5 along the collision axis, so the Doppler term adds a depth-equivalent of
tau* = 280 Mpc everywhere inside the disc, with the same sign as the Sachs-Wolfe term, falling to zero
across +- r_s around the wall. The monopole term with baryon loading R = 0.6 gives (1+3R) r_s / 2 = 203 Mpc
depth-equivalent at the wall. For a 10 deg disc (211 Mpc deep at the centre) this predicts about 2.3 units at
the centre and 1.6 at the edge; CAMB gives 2.89 and 1.54. For a 40 deg disc (3,246 Mpc deep) the Doppler
term is a 9% correction, matching the "10%" Osborne et al. quote for the ramp.

**Open discrepancy, not resolved.** Osborne et al. Fig. 3 (bubble of 11.39 deg) shows the same outward
shift of the boundary (their ramp reaches zero near 15 deg) but a much tamer shape: their ramp at the
nominal edge is about 0.17 of its central value, mine is 0.53; their step has no 2.5x peak. Their text
calls the Sachs-Wolfe ramp "excellent for most bubble sizes". Either their figure omits part of the
transfer function that mine includes, or my calculation has an error the three checks above do not catch
(candidates: interpolation of CAMB's coarse k sampling in a phase-sensitive integral; something specific to
small discs). Until this is settled, no data are generated from the new template.
Next check: decompose CAMB's temperature source into monopole, Doppler and ISW parts (`camb.symbolic`) and
see which part produces the edge feature; repeat with higher CAMB accuracy and a denser k grid.

What this already says about z_crit (entry (b)): the edge is neither a free sharp jump (the April generator)
nor absent (z_crit = 0). In the ramp model the temperature at the causal edge is tied to the central
amplitude, has the same sign, and falls to zero over the acoustic scale. For 5 to 25 deg discs that is a
large effect; for discs above about 40 deg the simple ramp is good to about 10%.

## (g) 2026-11-21 01:38 CST. Template: convergence and source-term checks; the discrepancy with Osborne et al. Fig. 3 stays open

Two further checks of the template in entry (f).

Convergence. CAMB AccuracyBoost 1 (1,067 k values) against 3 (3,232 k values, spacing 4.2e-5 /Mpc):
ramp at 10 deg centre 2.892 / 2.877, edge-to-centre ratio 0.53 / 0.53; at 11.39 deg centre 2.445 / 2.434,
ratio 0.48 / 0.47; at 20 deg 1.380 / 1.380, ratio 0.21 / 0.20. A 30 arcmin beam changes nothing.
The result is numerically converged.

Source terms (`scripts/analysis/check_template_source_terms.py`, `camb.symbolic`): the monopole, ISW,
Doppler and quadrupole parts add up to CAMB's total temperature transfer to 4e-15. Ramp at 11.39 deg, in
units of the Sachs-Wolfe central amplitude:

| part | centre | half radius | edge | edge + 2 deg | edge + 4 deg |
|---|---:|---:|---:|---:|---:|
| total | 2.445 | 2.188 | 1.168 | 0.387 | -0.048 |
| monopole | 1.077 | 0.817 | 0.524 | 0.186 | -0.011 |
| ISW | 0.095 | 0.119 | 0.087 | 0.013 | -0.004 |
| Doppler | 1.270 | 1.249 | 0.551 | 0.180 | -0.034 |
| quadrupole | 0.002 | 0.004 | 0.006 | 0.009 | 0.000 |

At 20 deg: total 1.380 / 0.288 (centre / edge), monopole 1.047 / 0.114, Doppler 0.389 / 0.141.
The Doppler part is a plateau across the disc (the fluid inside the collision region moves at one velocity)
that falls off across the wall. In depth-equivalent it is 347 Mpc at 11.39 deg and 325 Mpc at 20 deg, close
to the conformal time at recombination (280 Mpc) as the long-wavelength estimate predicts. Caveat: CAMB
splits the source in its own gauge, so only the total is physical; the split is a guide to the mechanism.

This is the same mechanism Feeney et al. (2011) gave for allowing an edge term ("This discontinuity arises
from the ISW and Doppler contributions"), and it is consistent with Kleban, Levi and Sigurdson (2011): a
Sachs-Wolfe term proportional to (cos theta - cos theta_c) plus a Doppler term proportional to cos theta is
still linear in cos theta inside the disc, non-zero at the causal edge, with a smooth transition outside.

Status of the discrepancy with Osborne et al. Fig. 3: unchanged. Their plotted ramp has an edge-to-centre
ratio near 0.17 at 11.39 deg; this calculation gives 0.48, supported by (1) the integrator test, (2) the
C_l normalisation, (3) numerical convergence, (4) the source terms adding up, (5) the analytic size of the
Doppler and baryon-loading terms. Their figure is drawn with "arbitrary normalization" and the cause of the
difference cannot be read from it. To settle: reproduce one of their published numbers that depends on the
profile shape, compute the same template with a second Boltzmann code (CLASS), or ask the authors. This is
a good question for a physicist co-author.

Working answer to "was z_crit dropped, and was it ever added back": the sharp free jump was dropped
(2012, 2013) and is unphysical. A non-zero temperature at the causal edge is physical for the ramp model,
has the sign of the central amplitude, a size fixed by the Doppler plateau (about 330 to 350 Mpc of
depth-equivalent against a central depth of D(1 - cos theta_c)), and a width set by the sound horizon. In
the old parameters that is z_crit / z0 of roughly 0.5 at 10 to 11 deg, 0.2 at 20 deg and below 0.1 above
40 deg, smoothed over about r_s / (D sin theta_c): 3.4 deg at 10 deg, 1.75 deg at 20 deg.

## (h) 2026-11-21 01:43 CST. Corrected geometry helpers tested; Doppler term confirmed in CAMB; pre-registration of the candidate re-analysis

Geometry. New `scripts/core/patch_geometry.py` (pixel_directions, pixel_to_lonlat, lonlat_to_pixel,
angular_distance_grid) with `scripts/analysis/check_patch_geometry_v2.py`. At five patch centres:
agreement with directions projected through `hp.gnomview` 0.044 to 0.059 deg (interpolation level);
agreement with healpy's own `GnomonicProj` object 1.2e-6 deg; round trip 1e-11 px or better; a blob placed
on the sphere is found at exactly the predicted pixel; angular-distance grid within 0.084 deg of truth.
So the orientation in entry (e) (columns increase west, rows increase north) is confirmed by healpy's own API.

Doppler term, template-independent. Fitting CAMB's low-k temperature transfer as
A j_l(kD) + B k j_l'(kD) gives A = -0.18 (l = 10; Sachs-Wolfe -0.20) and B = -90 to -99 Mpc for l = 10 to
60, that is 450 to 495 Mpc in depth-equivalent, with the sign that adds to the Sachs-Wolfe term. The
long-wavelength estimate is tau* = 280.5 Mpc; the two-term fit is crude (residual 15 to 20%) and absorbs
early ISW. Either way the plateau in the template of entry (g) is CAMB's standard physics.

Pre-registration: re-analysis of the frozen candidates on corrected coordinates (written before running).
- Input: the 102 frozen tile candidates and their 24 representatives
  (`runs/phase3_unet/remediated_v1_tile_constrained_candidates/`), unchanged scores and peaks.
- Step 1: recompute every position with `patch_geometry.pixel_to_lonlat`.
- Step 2: greedy clustering by score at 5 deg (primary), also 10 and 15 deg, pooling the four maps.
- Step 3: cross-match the cluster centres with McEwen et al. (2012) Table I. A match is a separation of at
  most max(5 deg, theta_crit / 2) of the published candidate. Only published candidates with theta_crit
  between 5 and 26 deg count (labels 7, 8, 9, 10, 11), because that is the range the screener was trained on.
- Step 4, chance probability: draw the same number of locations as primary clusters at random from the
  eligible tile centres (mask fraction >= 0.9, the footprint the candidates were emitted from), 100,000
  times; statistic = number of the five in-range published candidates matched by at least one location.
  Report P(statistic >= observed). A second version restricts the random draws to the southern Galactic
  hemisphere, where all 24 representatives lie, as the conservative case.
- Expectation from the hand calculation in entry (e): two matched (labels 7 and 10). I expect the chance
  probability to be a few per cent in the all-sky version and larger in the southern version.
- What would count against the result: no match after the official recomputation, or a chance
  probability above about 20% in the conservative version.

## (i) 2026-11-21 02:18 CST. Restore verified; edge-ratio split (pre-registered in (c)); the Wiener filter bank has no edge-free template; candidate re-analysis (pre-registered in (h))

Restore. Upload finished 01:40 (30 chunks, each SHA-256 verified; whole-file SHA-256
3d49964d6cfa9a77fb1c81790c5aba685f375c3344653994d841d22db55d4d69). Extraction job 82903 (node c9, 29 min 34 s):
2,591 files, 132,860,464,597 bytes, every CRC-32 checked by the extractor, the one `.env` skipped. This is
the zip listing minus that file. Restored files are now write-protected (job 82904).

Edge-ratio split, sensitivity grid (33,000 rows, patch-injected; scores from April, thresholds at 5% false
positives). Detection probability, amplitudes 1e-5 to 5e-5, with 95% intervals:

| method | ratio 0 | 0.5 | 1 | 2 | pooled |
|---|---|---|---|---|---|
| U-Net, ImageNet | 0.299 [0.283, 0.316] | 0.330 | 0.429 | 0.631 [0.614, 0.648] | 0.422 |
| U-Net, random init | 0.234 | 0.292 | 0.357 | 0.530 | 0.353 |
| circular template screen | 0.237 [0.222, 0.252] | 0.256 | 0.308 | 0.442 | 0.311 |

Predictions from (c): 1. U-Net at ratio 0 at least 20% below pooled: **held** (0.299 against 0.422, 29% lower).
2. Circular screen rises with the ratio: **held**. 3. U-Net lead over the circular screen smaller at ratio 0:
**held** (0.062 against 0.111). Over all amplitudes the U-Net is 0.293 at ratio 0 against 0.349 pooled. Below
5e-6 everything is at chance at every ratio; at 1e-4 the U-Net is at 0.99 or above at every ratio.

Edge-ratio split, same-grid benchmark (12,000 rows, sphere-injected, includes the Wiener filter):

| method | band | ratio 0 | 0.5 | 1 | 2 | pooled |
|---|---|---|---|---|---|---|
| Wiener matched filter | 1e-5 to 5e-5 | 0.058 [0.043, 0.077] | 0.430 | 0.686 | 0.949 | 0.529 |
| U-Net, ImageNet | 1e-5 to 5e-5 | 0.300 [0.268, 0.334] | 0.350 | 0.454 | 0.616 | 0.429 |
| Wiener matched filter | all | 0.135 | 0.351 | 0.456 | 0.604 | 0.384 |
| U-Net, ImageNet | all | 0.289 | 0.315 | 0.365 | 0.442 | 0.352 |
| Wiener matched filter | 1e-4 | 0.620 | 1.000 | 1.000 | 1.000 | 0.905 |

**B4. The Wiener filter bank contains no edge-free template.** `phase3_classical_filters.py` builds its bank
from `feeney_template_l0(theta, z0 = +-1, zcrit = +-1)` over the four sign quadrants only, so every
template has |z_crit| = |z0|. With inverse-CMB weighting such a template is dominated by its sharp edge.
On signals with no edge the filter is at chance in the mid band (0.058) while the U-Net detects 0.300.
Consequences:
- The April headline (Wiener 0.384, U-Net 0.352, "the filter is the strongest average screener") is an
  average over four edge ratios with equal weight. The filter's lead comes from ratios 0.5 and above; at
  ratio 0 the U-Net is ahead by a factor of two. Neither statement is a test of machine learning against
  the optimal filter, because for three of the four ratios the filter is not matched to the signal.
- Where the filter is matched (ratio 1), the U-Net reaches about two thirds of its detection rate in the
  mid band (0.454 against 0.686). That is the relevant measure of how far the network is from optimal on a
  sharp-edged signal, and it is a much larger gap than the pooled numbers suggested.
- Correction to something said earlier in this review: "the U-Net is 8% below the matched filter, where
  theory says it should be" compared pooled numbers and does not stand.
- The April "ideal matched-filter recall 0.83 at 1e-5, 15 deg against U-Net 0.106" used the median over
  profile variants; most of that apparent headroom is the edge.

Two verification steps in job 82904 did not count and are being rerun as job 82905: the test-split
evaluation failed because `phase3_evaluate_run.py` takes the data path from `run_config.json`, which holds
the lab-box path; and `phase3_score_models_on_sensitivity.py` reuses saved scores unless `--overwrite` is
given, so its "identical to April" result was not a recomputation.

Candidate re-analysis (pre-registered in (h); `scripts/followup/recompute_candidate_positions.py`).
All 102 tile candidates recomputed; the April positions are reproduced by the reflected convention to
0.0000 deg. Clusters at 5 deg: 10 (at 10 deg: 6; at 15 deg: 4).

| cluster (l, b) | score | members | maps | scatter | nearest McEwen 2012 candidate |
|---|---:|---:|---|---:|---|
| (184.3, -50.9) | 1.599 | 8 | all four | 0.7 | 10 (radius 20), 8.8 deg: match |
| (165.2, -48.9) | 1.594 | 7 | all four | 0.1 | 10, 5.5 deg: match |
| (150.9, -51.5) | 1.559 | 41 | all four | 3.3 | 10, 15.0 deg |
| (163.4, -56.2) | 1.460 | 9 | all four | 4.3 | 10, 11.1 deg |
| (205.2, -24.2) | 1.459 | 8 | smica | 2.3 | none (Orion region) |
| (212.4, -20.3) | 1.405 | 3 | commander, nilc | 3.5 | none |
| (150.5, -57.9) | 1.322 | 2 | commander, smica | 0.0 | none |
| (277.9, -39.3) | 1.300 | 11 | smica | 3.1 | 13 (radius 40, outside the tested range), 7.9 deg |
| (201.5, -57.2) | 1.239 | 8 | all four | 1.6 | 7 = Cold Spot (radius 5), 3.0 deg: match |
| (273.1, -35.0) | 1.095 | 5 | smica | 2.0 | 13, 13.4 deg |

Two thirds of the 102 tile candidates lie in one region, l = 150 to 185, b = -49 to -58, with McEwen
candidate 10 (Feeney feature 1) in the middle of it.
In-range published candidates matched: 7 and 10, as expected in (h). Chance probability of matching two or
more with 10 random locations: 0.0236 over all 5,298 eligible tiles; 0.0695 over the 2,623 southern ones.
Both are on the side I expected (a few per cent, larger in the south) and below the 20% line I set as
counting against the result. Reading: the positional agreement with the two best-known WMAP features is
unlikely to be chance at the few per cent level; it is evidence that the screener responds to real sky
structure, not evidence for a collision.

## (j) 2026-11-21 02:50 CST. Checkpoints reproduce April's scores; how detectable each template is

Verification, second pass (job 82905, node c3, RTX 5000 Ada).
- Sensitivity grid really re-scored on the GPU from the saved checkpoints (`--overwrite`), 33,000 rows each:
  ImageNet U-Net: largest score difference 1.7e-3, median 5.4e-5, correlation 0.9999997, 2 decisions change
  at the 5% threshold. Random-initialised U-Net: 5.0e-4, 1.5e-5, 0.9999999, 3 decisions.
  The April per-row scores are reproduced on different hardware and a rebuilt environment.
- Geometry helper tests pass inside the restored tree at five patch centres.
- Test-split evaluation failed a second time (my mirrored run folder lacked `split_indices.npz`); third
  attempt is job 82907.

Ideal matched-filter signal-to-noise by template (`scripts/analysis/template_snr_comparison.py`, job 82906).
One collision at a known position on the full sky, noise = the CMB itself (CAMB TT) plus 30 uK-arcmin,
5 arcmin beam, lmax 767, monopole and dipole excluded. Central amplitude z0 = 1e-5 throughout.

| theta_c | April, z_crit = 0 | April, z_crit = z0 | transfer-function ramp, same curvature slope | transfer-function ramp, same observed peak | peak / z0 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.77 | 2.62 | 7.27 | 0.91 | 8.00 |
| 10 | 0.87 | 3.71 | 2.94 | 1.02 | 2.89 |
| 15 | 0.89 | 4.54 | 1.95 | 1.09 | 1.79 |
| 20 | 0.89 | 5.21 | 1.52 | 1.10 | 1.38 |
| 25 | 0.88 | 5.79 | 1.29 | 1.10 | 1.17 |
| 40 | 0.84 | 7.13 | 0.96 | 1.14 | 0.84 |
| 70 | 0.66 | 8.61 | 0.68 | 1.49 | 0.45 |

Central amplitude needed for a signal-to-noise of 5.2 (April's blind-search threshold): edge-free April
template 5.8e-5 to 6.8e-5 for 5 to 25 deg; sharp-edged disc 0.9e-5 to 2.0e-5; transfer-function ramp
4.7e-5 to 5.7e-5 in observed peak amplitude.

Reading:
- A sharp edge is worth a factor of 3 to 10 in signal-to-noise. April's detections between 1e-5 and 5e-5
  were detections of the edge.
- Per unit of observed peak amplitude, the transfer-function ramp is as hard to detect as the edge-free
  ramp: signal-to-noise about 1.0 per 1e-5 for every size from 5 to 40 deg. Even a perfectly matched filter
  at a known position needs an observed central Delta T / T of about 5e-5 (about 135 uK). This is set by
  the CMB's own fluctuations and no instrument or algorithm changes it in temperature.
- For a fixed underlying curvature slope, small discs are easier than the Sachs-Wolfe picture suggests,
  because the acoustic and Doppler terms brighten them (8 times at 5 deg, 2.9 times at 10 deg).
- Consistency with a published number: McEwen et al. (2012) candidate 10 has z0 = 0.11 mK = 4.0e-5,
  theta_crit = 20 deg, filter significance 3.94. The table predicts 0.89 x 4.04 = 3.6 (edge-free) to
  1.10 x 4.04 = 4.4 (transfer-function ramp) for that amplitude and size. The WMAP candidates sit exactly at
  this limit.

Where this leaves the comparison of methods. Nothing measured so far says how a network compares with the
right matched filter on the right signal: April's filter had no edge-free template (B4), April's signals
had unphysical sharp edges of random sign, and the signal geometry was slightly off (B1). The honest
target for the next benchmark: transfer-function templates injected on the sphere, a filter bank built
from the same templates, the existing and new networks scored on the same skies, amplitudes quoted as
observed peak Delta T / T, with the ideal numbers above as the ceiling.

## (k) 2026-11-21 03:11 CST. April test-split results reproduced exactly

Job 82907 (node c3). `phase3_evaluate_run.py` on the held-out test split (2,000 samples), thresholds fixed
at April's calibration values, run from a mirrored run folder whose `run_config.json` has the data path
rewritten (the original run folders are untouched):

| model | threshold | precision | recall | false-positive rate | TP / FP / FN / TN | April |
|---|---:|---:|---:|---:|---|---|
| U-Net, ImageNet EfficientNet-B0 | 0.96 | 0.887892 | 0.396 | 0.050 | 396 / 50 / 604 / 950 | identical counts |
| U-Net, random initialisation | 0.99 | 0.819005 | 0.362 | 0.080 | 362 / 80 / 638 / 920 | identical counts |

The only difference is in a mask-overlap score at the fifth decimal (3e-5 and 1e-5).

State of the verification list in (d): 1 restore integrity: done. 3 test-split evaluation: reproduced.
4 sensitivity-grid scores: reproduced to numerical noise. 5 code read: four defects found so far (B1 to B4).
2 the project's own quality gates: `phase2_physics_checks.py` runs; the full gate script has not been run
(it rewrites report files that are now write-protected; to be run with outputs redirected).

So the April numbers are real outputs of the saved models on the saved data. What they mean is another
matter: entries (e), (i) and (j) show the signals had a wrong geometry and an unphysical edge, and the
classical comparator lacked the edge-free template.

## (l) 2026-11-21 03:22 CST. Pre-registration: benchmark on transfer-function templates with a matched filter bank

Committed and pushed at 03:17: branch `geometry-template-fixes`, four commits (the April 21-23 working tree;
geometry; template; verification and this notebook through entry (k)).

Purpose. Nothing measured so far compares a network with the right matched filter on the right signal
(entries (i), (j)). `scripts/analysis/benchmark_transfer_template.py` does that as a first, limited test.

Design.
- Backgrounds: 450 Gaussian CMB skies (CAMB lensed TT, `synfast` with 5 arcmin beam and pixel window,
  Nside 256, 30 uK-arcmin white noise), eight patch centres per sky at least 50 deg apart: 3,600 patches,
  alternating signal-free and signal.
- Signals: the ramp template through CAMB transfer functions, sizes 5, 10, 15, 20, 25 deg, amplitudes 1e-5,
  2e-5, 3e-5, 5e-5, 7e-5, 1e-4 in observed peak Delta T / T, random sign, centre within 5 deg of the patch
  centre, 60 patches per (amplitude, size) cell.
- Matched filter bank: the same five templates; patch statistic = largest |SNR| over the bank and over all
  sky pixels within 27 deg of the patch centre.
- Networks: the two April U-Nets (ImageNet and random initialisation), scored as in April (largest mask
  probability over the patch). They were trained on sharp-disc templates with the geometry defect B1.
- Every method thresholded at 5% false positives on the 1,800 signal-free patches.
- Built-in checks: null SNR maps must have unit standard deviation; the SNR at an injected signal's centre
  must average A * sqrt(norm); the fast projector must return exactly what `hp.gnomview` returns.

Limits, stated beforehand: Gaussian skies only (no real-sky residuals); one template family (ramp, no step);
signal centres near the patch centre; the networks were not trained on these templates, so this measures
how the April networks transfer, not what a network trained on the right signal can do.

Predictions.
1. The filter bank's 50% detection amplitude lies between 3e-5 and 6e-5 for every size.
2. The filter bank detects at least as much as either U-Net in at least 80% of the cells where either
   method is between 10% and 90%.
3. Between 3e-5 and 5e-5 the ImageNet U-Net's detection rate is below 0.7 of the filter bank's.
I am unsure of 3: the networks did see edge-free discs in training and may transfer better than that.
If 2 fails, the first suspect is my filter (search region or normalisation), and the self-checks decide.

## (m) 2026-11-21 03:35 CST. Benchmark on transfer-function templates: first result; two of three predictions failed

Job 82909 (node c3, 10 min). Run at 1,500 skies instead of the 450 in (l): the test run (job 82908) showed
0.2 s per sky, and the size was raised before any full result existed. 12,000 patches: 6,000 signal-free,
6,000 with a signal (200 per cell).

Self-checks: projector output identical to `hp.gnomview`; standard deviation of the null SNR maps 0.998,
0.993, 0.989, 0.979, 0.970 for the 5 to 25 deg templates; SNR at the injected centre against A * sqrt(norm):
worst of 30 cells 2.15 standard errors. Ideal SNR per 1e-5 of peak: 0.908, 1.015, 1.088, 1.102, 1.104.

Detection probability at 5% false positives, all sizes pooled (1,000 signals per amplitude), 95% intervals:

| observed peak | matched filter bank, 27 deg search | U-Net, ImageNet (April) | U-Net, random init (April) |
|---:|---|---|---|
| 1e-5 | 0.061 [0.048, 0.078] | 0.075 [0.060, 0.093] | 0.067 [0.053, 0.084] |
| 2e-5 | 0.147 [0.126, 0.170] | 0.241 [0.216, 0.268] | 0.157 [0.136, 0.181] |
| 3e-5 | 0.321 [0.293, 0.351] | 0.447 [0.416, 0.478] | 0.295 [0.268, 0.324] |
| 5e-5 | 0.905 [0.885, 0.922] | 0.898 [0.878, 0.915] | 0.741 [0.713, 0.767] |
| 7e-5 | 0.996 | 0.990 | 0.959 |
| 1e-4 | 1.000 | 1.000 | 0.999 |

Filter threshold 4.36 sigma. By size at 3e-5: filter 0.17, 0.20, 0.38, 0.40, 0.46; ImageNet U-Net 0.20, 0.35,
0.56, 0.56, 0.56 (5 to 25 deg).

Predictions from (l):
1. Filter 50% detection amplitude between 3e-5 and 6e-5 for every size: **held** (between 3e-5 and 5e-5).
2. Filter at least as good as either U-Net in 80% of the informative cells: **failed**. The ImageNet U-Net
   is ahead at 2e-5 and 3e-5 with separated intervals and level from 5e-5 up.
3. ImageNet U-Net below 0.7 of the filter between 3e-5 and 5e-5: **failed**; it is at 1.4 and 1.0 of it.

What this does and does not show. The April ImageNet network transfers to physically motivated templates
it never saw, and detects them at least as well as this filter bank. It does not show a network beating
the optimal test: as written in (l), the first suspect is the filter's search region. The filter takes its
maximum over every pixel within 27 deg and pays a look-elsewhere price for that (threshold 4.36 sigma),
while every signal here lies within 5 deg of the patch centre. A network trained on discs that had to fit
in the patch may have learned to expect large discs near the centre, which matches this benchmark's layout.
Follow-up, not pre-registered and labelled as such: job 82910 repeats the run with the filter's search
radius at 27, 12 and 6 deg on identical data.

## (n) 2026-11-21 03:47 CST. Follow-up to (m), not pre-registered: the filter's search radius explains the result

Job 82910, identical data (the 27 deg numbers reproduce (m) exactly). The filter statistic is now also taken
over sky pixels within 12 and 6 deg of the patch centre. Signals lie within 5 deg of the centre.
Detection probability at 5% false positives, all sizes pooled, 1,000 signals per amplitude:

| observed peak | filter, 27 deg | filter, 12 deg | filter, 6 deg | U-Net, ImageNet (April) | U-Net, random init (April) |
|---:|---:|---:|---:|---:|---:|
| 1e-5 | 0.061 | 0.076 | 0.087 | 0.075 | 0.067 |
| 2e-5 | 0.147 | 0.245 | 0.317 [0.289, 0.346] | 0.241 [0.216, 0.268] | 0.157 |
| 3e-5 | 0.321 | 0.499 | 0.594 [0.563, 0.624] | 0.447 [0.416, 0.478] | 0.295 |
| 5e-5 | 0.905 | 0.959 | 0.976 [0.965, 0.984] | 0.898 [0.878, 0.915] | 0.741 |
| 7e-5 | 0.996 | 0.999 | 1.000 | 0.990 | 0.959 |

Thresholds: 4.36, 3.93 and 3.59 sigma for 27, 12 and 6 deg.

Reading.
- With the search region matched to where signals can be, the filter bank is ahead of the April ImageNet
  U-Net at every informative amplitude: by a factor of 1.3 at 2e-5 and 3e-5 and by 1.09 at 5e-5.
  Prediction 2 of (l) holds for the 6 deg filter and fails for the 27 deg one; the comparison depends on the
  positional prior, which (l) did not specify. That was a flaw in the pre-registration.
- The April network performs like a filter with a search radius of about 12 deg. It carries a positional
  prior from its training geometry, where large discs had to lie near the patch centre.
- The ImageNet U-Net reaches about 75% of the 6 deg filter's detection rate at 2e-5 and 3e-5 and 92% at
  5e-5, on templates it was never trained on. The random-initialised U-Net is clearly behind
  (0.157, 0.295, 0.741), so pretraining matters.
- This sets a measurable target: a network trained on these templates with the correct geometry and the
  same positional prior should approach 0.32, 0.59 and 0.98 at 2e-5, 3e-5 and 5e-5, and by detection
  theory should not exceed them on Gaussian skies.

Note on entry (f). I wrote there that no data would be generated from the new template until the difference
from Osborne et al. Fig. 3 was settled. Entries (g) and (h) then showed the template is CAMB's standard
physics (source terms add up; the Doppler term is visible directly in the transfer function), and on that
basis this benchmark was run. The comparison with the published figure is still open, and the exact edge
shape would change these numbers somewhat; the ordering of the methods is less likely to depend on it.

## (o) 2026-11-21 03:50 CST. Pre-registration: fine-tune the ImageNet U-Net on the transfer-function templates

Committed and pushed at 03:48: the benchmark script, its report and entries (l) to (n) (commit ab1bf2e).

Question. Entry (n) left a gap: the April ImageNet U-Net detects 0.24, 0.45, 0.90 at 2e-5, 3e-5, 5e-5 where the
6 deg matched filter detects 0.32, 0.59, 0.98. Does training on the right signal close it?

Design (`scripts/ml/finetune_on_transfer_template.py`).
- Start from the April ImageNet U-Net checkpoint; fine-tune all weights for 12 epochs, AdamW, learning rate
  1e-4 with cosine decay, batch 64, loss = cross-entropy + Dice on the mask + 0.2 x cross-entropy on the
  image-level head, dihedral augmentation.
- Training data made in memory with the benchmark's recipe and a different seed (20261122): 3,000 skies,
  24,000 patches, half with a signal; 21 template sizes from 5 to 25 deg; amplitudes log-uniform from 1e-5 to
  1.2e-4 in observed peak; signal centre within 5 deg of the patch centre; target mask = where the template
  exceeds 10% of its peak. Validation: 300 more skies.
- Test: the fixed benchmark of entry (m) (seed 20261121), untouched by training. The last epoch is used; no
  selection on the test set. Score = largest mask probability, as before; the image-level head is reported
  as a second score.
- Check built in: the starting model scored through this code path must reproduce the benchmark's saved
  scores.

Predictions.
1. The fine-tuned network beats the April ImageNet U-Net at 2e-5 and at 3e-5 by more than the interval
   half-width (about 0.03).
2. It does not exceed the 6 deg filter by more than 0.03 at any amplitude.
3. At 3e-5 it reaches at least 0.50 (85% of the filter's 0.594).
If 2 fails, the filter is not the optimum for this prior (a maximum over a bank is not the Bayes test), and
the right reference becomes the marginal likelihood ratio.

## (p) 2026-11-21 04:16 CST. Fine-tuning on the transfer-function templates closes most of the gap to the matched filter

Job 82911 (node c3, 24 min in total; 12 epochs at 84 s each on one RTX 5000 Ada).
Check: the starting model scored through the fine-tuning code path reproduces the benchmark's saved scores
(largest difference 8e-4). Validation detection between 2e-5 and 5e-5 rose from 0.62 after one epoch to 0.69
after two and stayed there. The last epoch is used.

Test on the fixed benchmark of (m), 12,000 patches, 5% false positives, all sizes pooled:

| method | 1e-5 | 2e-5 | 3e-5 | 5e-5 | 7e-5 |
|---|---:|---:|---:|---:|---:|
| matched filter bank, 6 deg search | 0.087 | 0.317 | 0.594 | 0.976 | 1.000 |
| matched filter bank, 12 deg search | 0.076 | 0.245 | 0.499 | 0.959 | 0.999 |
| U-Net, ImageNet, April | 0.075 | 0.241 | 0.447 | 0.898 | 0.990 |
| U-Net, ImageNet, fine-tuned (mask maximum) | 0.083 | 0.283 | 0.568 | 0.970 | 1.000 |
| U-Net, ImageNet, fine-tuned (image head) | 0.093 | 0.289 | 0.571 | 0.960 | 0.998 |

Intervals are about +-0.03 at 2e-5 and 3e-5 and +-0.012 at 5e-5 (1,000 signals per amplitude).

Predictions from (o): 1. beats the April network at 2e-5 and 3e-5 by more than 0.03: **held** (+0.042 and
+0.121). 2. does not exceed the 6 deg filter by more than 0.03 anywhere: **held** (it is at or below the
filter at every amplitude with the mask score; the image head is 0.006 above it at 1e-5, within noise).
3. at least 0.50 at 3e-5: **held** (0.568).

Reading.
- Seventeen minutes of fine-tuning on the right signal takes the network from 75% to 96% of the matched
  filter's detection rate at 3e-5, from 76% to 89% at 2e-5, and from 92% to 99% at 5e-5.
- It does not pass the filter, as detection theory says it should not on Gaussian skies with a known
  template family. This is the first measurement in the project of a network against the right filter on
  the right signal, and the answer is "about equal".
- The image-level head, which was the weaker score in April, now matches the mask score.
- What a network can add beyond the filter is therefore not sensitivity on Gaussian skies. The open
  questions are real-sky backgrounds, where the filter's Gaussian assumption fails and April saw the
  network's false-positive rate rise from 0.05 to 0.185, and discs wider than the patch.

Limits: Gaussian skies; ramp template only; signal centres within 5 deg of the patch centre in both training
and test; one training run, one seed.

## (q) 2026-11-21 05:17 CST. Pre-registration: the real-sky test

Merged at 04:55: `geometry-template-fixes` into `main` (merge commit dd6365d). The public repository's main
branch now holds everything through entry (p).

State when this was written: the script below exists and passes a syntax check; it has not been run. The
connection to the cluster dropped at about 04:59 and nothing can run until it is restored.

Questions. Entries (m) to (p) used Gaussian skies only. On the Planck maps:
1. Do thresholds set on simulated skies give the same false-positive rate on the real sky?
2. Do the detectors keep their sensitivity when the background is the real sky?
3. How unusual are the strongest responses of the corrected filter bank on the real sky?

Design (`scripts/analysis/benchmark_real_sky.py`).
- Maps: PR3 SMICA (primary), NILC, SEVEM, Commander, temperature, `ud_grade` to Nside 256. PR3 common
  intensity mask, kept where the degraded value is at least 0.9.
- Preparation, the same for real and simulated skies: monopole and dipole fitted on unmasked pixels and
  removed; masked pixels filled with a smooth local mean (10 deg Gaussian) and relaxed by 200 sweeps of
  neighbour averaging so the fill meets the sky without a step.
- Tiles: Nside 16 centres whose 256 x 256 patch is at least 90% unmasked and whose 6 deg search disc is at
  least half unmasked.
- Methods: matched filter bank with 6 deg search (statistic = largest |SNR| over the bank and the unmasked
  pixels of the disc); April ImageNet U-Net; fine-tuned U-Net of entry (p). Networks scored by the largest
  mask probability.
- Simulations: 300 Gaussian skies (the benchmark's recipe) with the same mask, preparation and tiles. The
  filter is evaluated on all tiles, the networks on a fixed random 300 tiles.
- Thresholds: (a) 5% of simulated tiles; (b) 5% of the real map's own tiles.
- Question 1 is answered per sky: the fraction of a sky's tiles above threshold (a), real against the 300
  simulated skies on the same tiles (rank probability, resolution 1/301).
- Question 2: ramp templates, 5 to 25 deg, six amplitudes, centre within 5 deg of the tile centre, one
  position per (tile, size) shared by all amplitudes. Added analytically (fast path); the exact path
  (template added to the raw map, then the full preparation and filtering) on 24 tiles measures the fast
  path's error. Reference: the same injections on 8 simulated skies at threshold (a).
- Question 3: the largest |SNR| over the searched region, per bank size and over the bank, ranked among the
  300 simulated skies; the eight strongest peaks at least 10 deg apart, with the nearest McEwen 2012 entry
  and the distance to the Cold Spot.
- Added after writing the first draft of this entry and before any run: an "unfilled" variant for the
  networks, where masked pixels keep the map's own values. April's tiles were made that way, and April
  measured the ImageNet network's false-positive rate rising from 0.05 on simulations to 0.185 on real-sky
  patches. Simulated counterpart: the plain Gaussian sky. Trigger fractions are also split into tiles at
  least 97% unmasked and the rest.

Limits, stated beforehand.
- One sky. Tiles overlap heavily (centres 3.7 deg apart, patches 55 deg wide), so tile counts are not
  independent trials; the only valid yardstick for a trigger fraction is the spread across simulated skies.
- The simulations are Gaussian with white noise. They have no foreground residuals and no real noise
  pattern. Planck's own FFP10 simulations would be the proper null and are not on the cluster.
- The fill is my choice; simulated skies get the same one, which removes it from the comparison but not
  from the sensitivity.
- Threshold (b) is set on the same tiles it is used on.
- Ramp template only; signals near the tile centre; no step, no discs wider than the patch.

Predictions.
1. Filter, SMICA: the trigger fraction at threshold (a) lies inside the central 95% of the simulated skies.
   I expect it at or below their median, because the real sky has less power at low multipoles than the model.
2. Networks on filled patches, SMICA: both inside the central 95% of their simulated skies. I am less sure
   for the fine-tuned network, which has only ever seen Gaussian skies.
3. April ImageNet U-Net on unfilled patches, SMICA: above the 97.5th percentile of its simulated skies (the
   April rise reproduces), with the excess concentrated in tiles that are less than 97% unmasked. If 2 and 3
   both hold, the April false-positive problem came from what lies under the mask and filling removes it.
4. Detection on the real sky at threshold (b): filter >= fine-tuned > April ImageNet at 2e-5 and 3e-5
   (differences under 0.03 count as equal), and each method's pooled detection at 3e-5 is within 0.10 of its
   value on the simulated skies.
5. Strongest filter response on SMICA: not unusual, P(simulated >= real) above 0.05 for the bank.
6. Fast-path injection: mean difference from the exact path below 5% of the filter statistic. Above 10%, the
   detection tables are approximate and are labelled so.
7. The four maps agree: trigger fractions within 0.02 of SMICA's for the filter, and the strongest peak at
   the same place (within 3 deg) in at least three maps.

If 1 fails on the high side: suspects are foreground residuals near the mask edge and the fill; the split by
unmasked fraction and the agreement between maps decide. If a network fails 2 while the filter passes 1, the
network sees a difference between simulated and real skies that the filter does not, and thresholds from
simulations cannot be used for it.

Addition at 05:46, before any run. Job 82913 (submitted before the connection dropped) shows what the April
real-sky training patches held under the mask: exact zeros (`null_controls_smica_mask090.h5`, about 1,900 zero
pixels per patch on average; `project_patch` refuses a map with unseen pixels, so the masked array was filled
with 0). The April full-sky tiles, made earlier, hold the map's own values there (job 82912). The networks are
therefore scored on three versions of each tile: filled (new), unfilled (April tile scan) and zeroed (April
training). Prediction for the zeroed version, April ImageNet U-Net, SMICA: inside the central 95% of its
simulated skies, since this is the form it was trained on. No prediction for the fine-tuned network on it.

## (r) 2026-11-21 06:24 CST. Real-sky test: the filter behaves as on simulated skies; the April false-positive rise comes from masked pixels, not from the sky

Job 82914 (node c3, 33 min; a small run first, then the full run of (q)). 1,287 of 3,072 tile centres are
eligible, 770 of them at least 97% unmasked; the searched region is 47.6% of the sky; the mask keeps 77.2%.
300 simulated skies. The connection came back at 05:44; the job was submitted at 05:46.

Checks. Patch indices reproduce `hp.gnomview`. Standard deviation of the simulated SNR maps over the searched
region: 0.997, 0.998, 0.996, 0.991, 0.981 (5 to 25 deg). Exact injection against the fast path, 24 tiles at
5e-5: filter statistic 5.42 against 5.44 on average, mean difference 0.021 (0.4%), largest 0.46; patches differ
by 1.9 uK rms; network scores differ by 0.001 on average.

**1. False-positive rate.** Thresholds at 5% of simulated tiles. "Same tiles" is the fraction on the tiles the
simulations used (all 1,287 for the filter, a fixed 300 for the networks); the rank is the share of the 300
simulated skies with a fraction at least as high.

| method, input | threshold | SMICA, all tiles | SMICA, same tiles | rank P | central 95% of simulated skies | tiles >=97% unmasked | other tiles |
|---|---:|---:|---:|---:|---|---:|---:|
| matched filter bank | 3.58 sigma | 0.039 | 0.039 | 0.605 | 0.007 to 0.117 | 0.026 | 0.058 |
| April U-Net, filled | 0.9958 | 0.056 | 0.053 | 0.419 | 0.030 to 0.082 | 0.001 | 0.137 |
| April U-Net, zeroed | 0.9953 | 0.057 | 0.060 | 0.216 | 0.033 to 0.075 | 0.001 | 0.141 |
| April U-Net, unfilled | 0.9475 | 0.042 | 0.040 | 0.605 | 0.003 to 0.130 | 0.014 | 0.083 |
| fine-tuned, filled | 0.99996 | 0.071 | 0.057 | 0.206 | 0.040 to 0.063 | 0.000 | 0.176 |
| fine-tuned, zeroed | 1.0000 | 0.067 | 0.060 | 0.056 | 0.040 to 0.060 | 0.000 | 0.166 |
| fine-tuned, unfilled | 0.8981 | 0.052 | 0.060 | 0.349 | 0.010 to 0.113 | 0.019 | 0.101 |

The other three maps: filter 0.035, 0.040, 0.040 (NILC, SEVEM, Commander); networks on filled and zeroed
input within 0.003 of SMICA; on unfilled input lower (April 0.023, 0.024, 0.036; fine-tuned 0.026, 0.026,
0.037), so what the maps hold under the mask differs between them. The fine-tuned network on zeroed input sits
at 0.063 on the same tiles for those three maps, rank P 0.010: above all but 1% of simulated skies, by a
small amount.

**2. Sensitivity.** Detection probability of injected ramp templates, all sizes pooled.

| method | where, threshold | 1e-5 | 2e-5 | 3e-5 | 5e-5 | 7e-5 | 1e-4 |
|---|---|---:|---:|---:|---:|---:|---:|
| matched filter bank | SMICA, its own 5% (3.43 sigma) | 0.087 | 0.297 | 0.664 | 0.987 | 1.000 | 1.000 |
| | SMICA, simulation threshold (3.58) | 0.063 | 0.230 | 0.586 | 0.982 | 1.000 | 1.000 |
| | simulated skies, same mask | 0.105 | 0.286 | 0.610 | 0.976 | 1.000 | 1.000 |
| | Gaussian benchmark, no mask (entry (n)) | 0.087 | 0.317 | 0.594 | 0.976 | 1.000 | 1.000 |
| April U-Net, filled input | SMICA, its own 5% (0.9969) | 0.053 | 0.066 | 0.140 | 0.572 | 0.792 | 0.839 |
| | simulated skies, same mask | 0.065 | 0.102 | 0.216 | 0.604 | 0.805 | 0.856 |
| | Gaussian benchmark, no mask (entry (n)) | 0.075 | 0.241 | 0.447 | 0.898 | 0.990 | 1.000 |
| fine-tuned, filled input | SMICA, its own 5% (0.999999) | 0.051 | 0.051 | 0.053 | 0.055 | 0.072 | 0.090 |
| | SMICA, simulation threshold (0.99996) | 0.072 | 0.072 | 0.079 | 0.215 | 0.607 | 0.822 |
| | simulated skies, same mask | 0.051 | 0.051 | 0.063 | 0.229 | 0.607 | 0.821 |
| | Gaussian benchmark, no mask (entry (p)) | 0.083 | 0.283 | 0.568 | 0.970 | 1.000 | 1.000 |

Filter by template size at 3e-5, SMICA, own threshold: 0.47, 0.63, 0.73, 0.73, 0.77 (5 to 25 deg); simulated
skies 0.42, 0.58, 0.68, 0.69, 0.68. NILC, SEVEM and Commander give the same table within 0.02.

**3. Strongest filter responses, SMICA.** Standard deviation of the real SNR maps over the searched region:
0.956, 0.937, 0.911, 0.908, 0.866 (5 to 25 deg), against 0.98 to 1.00 on simulated skies. Largest |SNR| over
the bank: 4.68. 32 of 300 simulated skies have a larger one (P = 0.11; median simulated maximum 4.17). SEVEM
4.73 (22 of 300), Commander 4.70 (23), NILC 4.68 (32). Per bank size, P = 0.027, 0.26, 0.43, 0.13, 0.69.

| peak (lon, lat) | SNR | bank size | P for that tile | nearest McEwen 2012 entry | separation | their z0 |
|---|---:|---:|---:|---|---:|---:|
| (213.7, -59.1) | -4.68 | 5 deg | 0.003 | no. 7, 5 deg (the Cold Spot is 3.3 deg away) | 3.9 deg | -0.16 mK |
| (159.0, -58.7) | +4.26 | 20 deg | 0.013 | no. 10, 20 deg | 14.6 deg | +0.11 mK |
| (176.1, -49.3) | +3.90 | 15 deg | 0.020 | no. 10, 20 deg | 3.6 deg | +0.11 mK |
| (215.0, -41.0) | -3.59 | 25 deg | 0.050 | no. 11, 26 deg | 4.1 deg | -0.09 mK |
| (325.9, -34.0) | +3.54 | 5 deg | 0.050 | no. 8, 5 deg | 3.9 deg | +0.15 mK |
| (136.1, 35.1) | -3.51 | 15 deg | 0.066 | no. 5, 3 deg | 21.9 deg | |
| (229.8, 47.9) | -3.43 | 20 deg | 0.073 | no. 9, 7 deg | 21.9 deg | |
| (188.8, 25.1) | -3.39 | 5 deg | 0.086 | no. 1, 2 deg | 17.8 deg | |

Four of the five strongest peaks lie within 4.1 deg of a published WMAP 7-year candidate (McEwen et al. 2012,
Table I, numbers 7, 10, 11, 8), with the published size in the same class and the same sign each time. The
four maps give the same list; in Commander the eighth peak is (217.3, 26.4), 2.9 deg from number 9. These are
the same features of the sky seen by a second satellite through a different pipeline, which is what a correct
pipeline should return. None is significant after the search over the region is counted (P = 0.08 to 0.11
for the largest), which agrees with the published conclusion of no detection. The matching was not
pre-registered as a test; the script only reports the nearest entry.

Predictions of (q).
1. Filter trigger fraction inside the central 95%, at or below the median: **held** (0.039; 60% of simulated
   skies are higher).
2. Both networks on filled patches inside the central 95%: **held** (rank P 0.42 and 0.21).
3. April U-Net on unfilled patches above the 97.5th percentile, excess in the less clean tiles: **failed**.
   0.040, rank P 0.61: no excess at all. I had the mechanism backwards; see below.
4. Detection at own thresholds: order filter >= fine-tuned > April, each within 0.10 of the simulated skies
   at 3e-5: the second part **held** (0.664 against 0.610; 0.140 against 0.216; 0.053 against 0.063); the
   order **failed**: on masked skies the fine-tuned network is the worst of the three.
5. Largest filter response not unusual (P above 0.05): **held** (0.11; 0.08 on SEVEM and Commander).
6. Fast-path error below 5%: **held** (0.4%).
7. Four maps agree: **held** (filter fractions 0.035 to 0.040; the strongest peak is the same pixel in all
   four).
Addition of 05:46 (April U-Net on zeroed patches inside the central 95%): **held** (rank P 0.22).

Not pre-registered, computed from the saved scores after seeing the table above.
- The April situation rebuilt: a threshold taken from simulated patches without holes (0.9475 for the April
  U-Net) and applied to real patches whose masked pixels are zero, as in the April training file. Fraction
  of real tiles above it: 0.146 on SMICA (0.145 to 0.147 on the others), against the nominal 0.05. The same
  threshold on simulated skies with the same holes gives 0.161. So the rise appears on simulated skies too and
  is the holes: 0.017 of real tiles that are at least 97% unmasked trigger, 0.338 of the rest (simulated: 0.054
  and 0.348). April measured 0.185 on its own tiles (Nside 32 centres, a different mix). For the fine-tuned
  network the same calculation gives 0.216 real, 0.220 simulated.
- With holes, the networks' triggers follow the mask and not the sky. Over 300 different simulated skies the
  5% most-triggering tiles carry 72% (April U-Net) and 88% (fine-tuned) of all triggers on filled input, 89%
  of tiles never trigger for the fine-tuned network, and the per-tile trigger rate correlates with the masked
  fraction at +0.5 to +0.6. On unfilled input and for the filter the same 5% of tiles carry 8%.
- Saturation: on filled or zeroed input the fine-tuned network gives exactly 1.0 on 2.7% of simulated null
  tiles and 3.9% of SMICA tiles (0.999 or more on 8% and 11%), so its 5% threshold is 0.99996 to 0.999999
  and an injected signal has nowhere to go. On unfilled input nothing saturates.

Reading.
- The matched filter needs no correction on the real sky. Its false-positive rate at simulation thresholds is
  3.5 to 4.0% for a nominal 5%, inside the spread of simulated skies, and its sensitivity is what the
  simulations give (0.66 at 3e-5, 0.99 at 5e-5). The real sky gives a slightly quieter filter output than the
  model (standard deviation 0.87 to 0.96), most for the largest templates, which is the known shortage of
  large-scale power; that is why the trigger fraction is below 5% and the real sky's own threshold is lower.
- The April false-positive problem was a difference in preparation between real and simulated patches, not
  something in the sky. The networks treat masked pixels, zeroed or smoothly filled, as signal. Real patches
  had them, simulated patches did not, and a threshold set on the simulated ones then lets three times too
  many real tiles through. When simulated skies get the same mask the rate transfers (5.3 to 6.0%).
- The cost is sensitivity. On tiles with masked pixels neither network can be trusted, and the fine-tuned
  network, which never saw a hole in training, is the worst: 0.06 at 3e-5 on masked simulated skies against
  0.57 without a mask. On the real sky as prepared here the order is filter 0.66, April U-Net 0.14, fine-tuned
  0.05. The "about equal to the filter" of entry (p) holds for skies without a mask only.
- Two ways out, both open: give the networks the values the map itself holds under the mask (the unfilled
  input, where nothing saturates and false positives are 0.04 to 0.05), or train with holes in the patches.
- Limits as in (q). The April-situation numbers use the tiles of this test, not those of April.

Follow-up, not pre-registered, written before it runs: `benchmark_real_sky.py --inject-inputs
filled,unfilled,zeroed` repeats the injection test for the networks on all three inputs (same design
otherwise; job file `realsky_v2.sbatch`). Expectation: on the unfilled input both networks get back, on
simulated skies, roughly what they reach without a mask (about 0.45 and 0.57 at 3e-5), the real sky is within
0.10 of the simulated skies, and the order filter >= fine-tuned > April returns. On the zeroed input I expect
the same picture as on the filled one.

## (s) 2026-11-21 07:21 CST. Follow-up to (r), not pre-registered: with the map's own values under the mask the networks recover, and the fine-tuned network is level with the filter on the real sky

Job 82915 (node c3, 51 min): the run of (r) repeated with `--inject-inputs filled,unfilled,zeroed`, so the
networks also get injections on patches whose masked pixels keep the map's own values (unfilled) or are set
to zero (zeroed). Same seed. Every number the two runs share is identical (largest difference 0), so the
tables of (r) stand as written and this entry only adds rows.

Detection probability, all template sizes pooled. Real maps: each method at the 5% threshold of that map's
own signal-free tiles. Simulated skies: 8 skies, same mask and tiles, simulation thresholds.

| where | method, input | 1e-5 | 2e-5 | 3e-5 | 5e-5 | 7e-5 | 1e-4 |
|---|---|---:|---:|---:|---:|---:|---:|
| simulated skies | matched filter bank | 0.105 | 0.286 | 0.610 | 0.976 | 1.000 | 1.000 |
| | fine-tuned, unfilled | 0.100 | 0.281 | 0.576 | 0.964 | 1.000 | 1.000 |
| | April U-Net, unfilled | 0.085 | 0.218 | 0.457 | 0.886 | 0.994 | 1.000 |
| | April U-Net, zeroed | 0.063 | 0.090 | 0.180 | 0.535 | 0.757 | 0.837 |
| | fine-tuned, zeroed | 0.048 | 0.049 | 0.050 | 0.100 | 0.286 | 0.546 |
| SMICA | matched filter bank | 0.087 | 0.297 | 0.664 | 0.987 | 1.000 | 1.000 |
| | fine-tuned, unfilled | 0.066 | 0.203 | 0.497 | 0.940 | 0.983 | 0.995 |
| | April U-Net, unfilled | 0.061 | 0.164 | 0.386 | 0.870 | 0.965 | 0.987 |
| NILC | matched filter bank | 0.090 | 0.292 | 0.659 | 0.989 | 1.000 | 1.000 |
| | fine-tuned, unfilled | 0.127 | 0.371 | 0.703 | 0.988 | 1.000 | 1.000 |
| | April U-Net, unfilled | 0.073 | 0.227 | 0.504 | 0.927 | 0.997 | 1.000 |
| SEVEM | matched filter bank | 0.096 | 0.316 | 0.684 | 0.989 | 1.000 | 1.000 |
| | fine-tuned, unfilled | 0.115 | 0.366 | 0.701 | 0.987 | 1.000 | 1.000 |
| | April U-Net, unfilled | 0.074 | 0.234 | 0.501 | 0.932 | 0.998 | 1.000 |
| Commander | matched filter bank | 0.095 | 0.305 | 0.671 | 0.989 | 1.000 | 1.000 |
| | fine-tuned, unfilled | 0.086 | 0.282 | 0.614 | 0.979 | 1.000 | 1.000 |
| | April U-Net, unfilled | 0.063 | 0.186 | 0.438 | 0.911 | 0.996 | 1.000 |

At the simulation thresholds instead, on the four real maps at 3e-5: filter 0.59 to 0.60, fine-tuned unfilled
0.51 to 0.53, April unfilled 0.35 to 0.36. The real maps' own thresholds are lower than the simulation ones for
every method (filter 3.41 to 3.44 against 3.58 sigma; fine-tuned 0.78 to 0.90 against 0.90; April 0.88 to 0.93
against 0.95), which is the quieter real sky of entry (r). On zeroed input the real maps give the same picture
as on filled input (April 0.12, fine-tuned 0.05 at 3e-5).

By template size at 3e-5 on simulated skies (5, 10, 15, 20, 25 deg): filter 0.42, 0.58, 0.68, 0.69, 0.68;
fine-tuned unfilled 0.44, 0.61, 0.67, 0.64, 0.52; April unfilled 0.21, 0.42, 0.58, 0.59, 0.48. The fine-tuned
network matches the filter up to 15 deg and falls behind at 25 deg, where the disc with its offset no longer
fits inside the 55 deg patch.

How sure. Tiles overlap, so tile counts overstate the evidence. Block bootstrap: the sky cut into 30 x 30 deg
cells (48 hold tiles), cells resampled 1,000 times, every threshold recomputed in each resample
(`scripts/analysis/real_sky_post_hoc.py`). Difference to the filter at 3e-5, with 95% intervals:

| map | fine-tuned, unfilled | April U-Net, unfilled |
|---|---|---|
| SMICA | -0.17 [-0.39, +0.03] | -0.28 [-0.42, -0.07] |
| NILC | +0.04 [-0.06, +0.12] | -0.16 [-0.25, +0.01] |
| SEVEM | +0.02 [-0.06, +0.11] | -0.18 [-0.26, -0.03] |
| Commander | -0.06 [-0.28, +0.07] | -0.23 [-0.31, -0.07] |

At 5e-5 the fine-tuned network is within 0.01 of the filter on NILC, SEVEM and Commander and 0.05 below on
SMICA (interval -0.14 to -0.003); the April network is 0.06 to 0.12 below on all four.

Expectations written in (r) before the run.
- Unfilled input on simulated skies gives back what the networks reach without a mask (0.45 and 0.57 at
  3e-5): **held** (0.457 and 0.576).
- The real sky within 0.10 of the simulated skies: **held** for the April network on all four maps and for
  the fine-tuned network on SMICA and Commander; **not held** for the fine-tuned network on NILC and SEVEM,
  where it is 0.13 higher than on simulated skies.
- Order filter >= fine-tuned > April: **held** on simulated skies, SMICA and Commander. On NILC and SEVEM
  the fine-tuned network is nominally above the filter by 0.02 to 0.04, inside the interval.
- Zeroed input like filled input: **held**.

Reading.
- The masked pixels were the whole problem. Given the map's own values there, both networks behave on the
  real sky as they do on simulated skies without a mask, false positives included (2.3 to 5.2% at simulation
  thresholds, entry (r)).
- On the real sky the fine-tuned network and the matched filter cannot be told apart at 2e-5 and 3e-5 on any
  of the four maps, and at 5e-5 on three of them. The April network is behind the filter by 0.16 to 0.28 at
  3e-5. This is the same ordering as on Gaussian skies in (p), now on Planck data.
- Which map matters for the networks and not for the filter. The filter never sees the masked pixels. The
  networks do: on SMICA and Commander the partly masked tiles trigger more often without any signal (0.083
  and 0.070 for the April network against 0.039 and 0.041 on NILC and SEVEM), which raises their thresholds
  and costs sensitivity. What each map holds under the mask is that map's processing, not sky.
- For use on real data: do not zero or fill masked pixels in the network input; calibrate on simulated skies
  only if they go through the same preparation; prefer NILC or SEVEM for the networks, or train with masked
  pixels in the patches. April's full-sky tile scan already used the unfilled input. The 0.185 came from the
  null-control patches, whose masked pixels were zero.
- The 25 deg result is a patch-size limit, not a network limit, and is the first measured reason to widen
  the footprint.

Limits. One sky, 1,287 overlapping tiles, intervals from 48 sky cells. Thresholds at a map's own 5% are set
on the tiles they are applied to. Simulated skies are Gaussian with white noise. Ramp template only, signal
centres within 5 deg of the tile centre, one fine-tuning run. The unfilled input is only as good as what a
map holds under the mask. Nothing here is a search result: the injected signals are synthetic, and the
strongest real responses are those of entry (r).

Two corrections to (s), 07:32, found while checking it before publication.
- The sentence on thresholds is wrong for one case. The real maps' own thresholds are lower than the
  simulation ones for the filter and the April network on all four maps, and for the fine-tuned network on
  three. On SMICA the fine-tuned network's own threshold is 0.903 against 0.898 from the simulations.
- "The 0.185 came from the null-control patches" names the wrong file. The April figure (recall 0.372,
  false-positive rate 0.185 for the ImageNet branch) is from the real-SMICA injection diagnostic of 19 April,
  `runs/phase3_unet/remediated_v1_real_sky_injection_smica_mask090`: 200 real backgrounds with 90.3 to 98.8%
  of the patch unmasked. Job 82916 opened its patch file: every sampled patch holds exact zeros, and their
  number equals the number of masked pixels (1343 against 1343, 2592 against 2604, 2706 against 2718). So the
  April measurement was a threshold from hole-free simulated patches applied to real patches with zeroed
  masked pixels, which is the situation rebuilt in (r): 0.146 here on a different tile mix, 0.185 there on 200
  backgrounds. The null-control training file was made the same way.

## (t) 2026-11-21 16:09 CST. Where the open notes stand

Entries are not rewritten, so several notes above still read "not yet done" although a later entry did the
work. This entry lists them once, then what is open now. No new measurement.

Closed by a later entry.
- (b) "Planned fix (not yet done)", the template from the curvature profile through CAMB: built in (f),
  shown converged and split into source terms in (g), Doppler term confirmed in (h). Of the three checks
  planned there, the Sachs-Wolfe limit is done, the comparison with Osborne et al. is done and disagrees
  (item 1 below), and the comparison with the 2015 templates was never made (item 1 below).
- (e) "Not yet done: fix the three functions ... re-run clustering and follow-up": helpers written and
  tested in (h); positions, clustering, cross-match and chance probability redone in (i). The
  candidate-centred re-extraction is not redone (item 2 below).
- (f) "Next check: decompose CAMB's temperature source ... denser k grid": done in (g).
- (f) "no data are generated from the new template" until the Osborne question is settled: lifted in (n),
  with the reason given there.
- (i) two verification steps "being rerun": done in (j) and (k), both reproduce April.
- (p) "open questions are real-sky backgrounds ... and discs wider than the patch": the real sky is (q) to
  (s); wider discs are item 6 below.
- (r) "Two ways out, both open": the unfilled input is (s); training with masked pixels is item 5 below.

Open now: loose ends in what exists.
1. The template against Osborne et al. Fig. 3, entries (f) and (g): edge-to-centre ratio 0.48 here against
   about 0.17 in their figure for an 11.39 deg disc. Every number from (l) on uses this template. Ways to
   settle it named in (g): a second Boltzmann code, a published number that depends on the profile shape,
   or the authors. The comparison with the templates of Feeney et al. 2015 planned in (b) is also missing.
2. The candidate-centred re-extraction at the corrected peaks, entry (e): April's "centre-unstable" verdict
   for tile 9628 was measured 21 deg from the true peak. The half-mission and frequency tests work on whole
   tiles and stand.
3. Item 2 of the verification list in (d): the project's full quality-gate script has not been run in the
   new environment, only `phase2_physics_checks.py`, entry (k).
4. The April code and data are unchanged. B1 to B4 are still in the April scripts, which is why April's
   numbers reproduce; the April data sets were not regenerated; the README below its status note is the
   April text. New work uses `scripts/core/patch_geometry.py` and the filter bank built inside the
   benchmark scripts.

Open now: not started.
5. A network trained with masked pixels in its patches, entry (r), so that it no longer depends on what a
   map holds under the mask.
6. Discs of 25 deg and wider, where the patch is the limit, entry (s).
7. The ramp+step template is built but not benchmarked; the quadratic term of the 2015 template is not
   built. The fine-tuned network is one run with one seed. Simulated skies are Gaussian with white noise.

None of 1 to 7 is running or scheduled.

## (u) 2026-11-22 00:35 CST. Osborne et al. Fig. 3 read from its vector file; pre-registration of a depth fit

Work resumes on the open list of entry (t), item 1 first: the template against Fig. 3 of Osborne, Senatore
and Smith (arXiv:1305.1970).

The figure as numbers. The arXiv source of the paper holds the figure as a vector file
(`bubble_profile.pdf`). Its drawing commands give every plotted curve as coordinates, and the tick marks
drawn in the same file calibrate both axes (0, 5, 10, 15 deg and 0.0 to 1.0, all major ticks reproduced to
0.001). The comparison no longer rests on reading the figure by eye. The cosine profile is 1 at the centre;
the other normalisations are arbitrary (their caption):

| theta (deg) | cosine | ramp | step |
|---:|---:|---:|---:|
| 0 | 1.000 | 0.892 | 0.311 |
| 4 | 0.876 | 0.802 | 0.316 |
| 5.695 | 0.749 | 0.710 | 0.319 |
| 8 | 0.506 | 0.526 | 0.351 |
| 9 | 0.375 | 0.422 | 0.356 |
| 10 | 0.229 | 0.308 | 0.353 |
| 11 | 0.067 | 0.186 | 0.332 |
| 11.39 | 0.000 | 0.141 | 0.271 |
| 12 | | 0.098 | 0.132 |
| 13.39 | | 0.052 | 0.056 |
| 15 | | 0.024 | 0.025 |
| 16 | | 0.014 | 0.014 |

Ramp edge-to-centre 0.141 / 0.892 = 0.158 (entry (f) read "about 0.17" by eye). The step curve is flat near
0.31, rises to 0.356 at 9 deg and drops between 11 and 12 deg.

What the paper states. Caption: "bubbles at comoving distance r = 13886.6 Mpc (corresponding to angular size
theta_bubble = 11.39 deg) and arbitrary normalization". With cos(theta) = r / D_dc that fixes D_dc = 14,165.5
Mpc and a depth D_dc - r = 278.9 Mpc at the disc centre (r / D_dc = 249/254, a grid value). Their printed
b_l formulas for the ramp and the step, even and odd l, are the ones in `collision_template_transfer.py`,
checked term by term, and they also use CAMB. Their text: the Sachs-Wolfe approximation for the ramp is
"excellent for most bubble sizes" and breaks down only for theta below about 1 deg and near 90 deg.

How the two curves differ. Inside the disc both are linear in cos(theta). Their ramp follows
(cos theta - 0.97592) / 0.02408 to within 0.01 from 0 to 8 deg, so its straight part runs to zero at
12.6 deg. In depth-equivalent that is the Sachs-Wolfe ramp of a 278.9 Mpc deep disc plus a constant of about
60 Mpc. The calculation of entries (f), (g) gives about 400 Mpc for the same disc (centre 2.445 against 1 for
the Sachs-Wolfe ramp), so its straight part runs to zero near 17.7 deg and is cut off at the wall.

Why about 60 Mpc cannot be right for a 279 Mpc deep disc. For a uniform gradient of the potential the
standard long-wavelength result (the cancellation of the dipole of a superhorizon mode: Grishchuk and
Zel'dovich 1978; Erickcek, Carroll and Kamionkowski 2008, both quoted from memory, to be re-read before they
are cited) is
    dT/T = (1/3) n.grad(Psi) r_dec  -  n.v_dec  +  n.v_obs,   v = -(tau/3) grad(Psi) in matter domination,
and the three terms cancel only because r_dec + tau_dec = tau_0. The middle term is the Doppler term of the
emitting fluid: it has the sign of the Sachs-Wolfe term and the size tau_dec / r_dec of it. Inside a
collision region the observer term is absent (the observer sits outside), so the temperature is
proportional to (r_dec + tau_dec) cos(theta) - r: the Sachs-Wolfe ramp plus a plateau of tau_dec, about 280
Mpc before radiation corrections. This does not depend on any Boltzmann code. Entry (g) measured 325 to 347
Mpc for the Doppler part in CAMB.

Hypothesis H. The published curves were computed with a transfer function whose last-scattering distance D*
is not the D_dc = 14,165.5 Mpc used for the label, so the wall in the figure sits at a depth d = D* - r
different from 278.9 Mpc. Small discs are sensitive to this: 100 Mpc is 0.7% of D but a third of the depth.

Pre-registered test (script `scripts/analysis/compare_osborne_fig3.py`, written before any fit was run):
- Model: ramp and step profiles from `collision_template_transfer.py`, CAMB with every multipole to
  l = 2000, no beam, for depths d = D* - r from -60 to 320 Mpc in steps of 5 and for d = 278.9. Two
  cosmologies: Planck 2018 (project default) and WMAP 7-year. The monopole and dipole, which the template
  drops, are restored by subtracting a constant and a cos(theta) term fitted to the profile at theta >= 30
  deg, where the true signal is zero.
- Fit: each published curve against the model with one free amplitude, theta from 0 to 16 deg. Statistic:
  RMS residual divided by the maximum of the published curve. A second fit with a free constant is reported
  as a robustness check only.
- Outcome A, "the figure's label and its transfer function are inconsistent": one depth other than 278.9
  fits both curves with RMS at or below 3% each, the best depths of the two curves agree within 20 Mpc, and
  d = 278.9 is worse by a factor of 3 or more on both.
- Outcome B, "this template disagrees with the published one": no depth reaches 5% on the ramp. Item 1 then
  stays open and the next step is a second Boltzmann code and the authors.
- Anything else is reported as inconclusive.
- Also reported, with no criterion: D* and tau_0 for Planck 2018, Planck 2013, WMAP 7-year, WMAP 9-year and
  the CAMB default parameters of 2013, to see which distance equals 14,165.5 Mpc.

## (v) 2026-11-22 00:54 CST. Result of the depth fit; Feeney et al. 2015 Fig. 6 reproduced

Pre-registered test of entry (u), job 82917 (`runs/template_check/osborne_fig3/`). RMS residual divided by the
maximum of the published curve, one free amplitude:

| cosmology | best depth, ramp | best depth, step | at the label depth 278.9 Mpc |
|---|---|---|---|
| Planck 2018 | 140 Mpc, 3.06% | 130 Mpc, 16.4% | ramp 13.9%, step 46.0% |
| WMAP 7-year | 140 Mpc, 3.06% | 130 Mpc, 16.5% | ramp 13.7%, step 45.7% |

With a free constant: ramp 130 Mpc at 2.5%, step 125 Mpc at 11.5%. The vectorised integrator agrees with
`ct.b_ell` to 2e-15; the profiles are zero beyond 30 deg to 3e-3 (ramp) and 9e-3 (step) of their maximum.

Outcome by the criteria fixed in (u): inconclusive. A is not met (the ramp reaches 3.06%, not 3%, and the
step is far from 3%). B is not met either (the ramp does get below 5%).

What the run shows all the same:
- The published curves are not reproduced at the labelled depth. The ramp residual there is 4.5 times the
  best one and the step residual 2.8 times.
- The published ramp is matched best by a wall 130 to 140 Mpc inside the last-scattering surface, in both
  cosmologies. Model against published at 130 Mpc (Planck 2018): 0.877 / 0.892 at the centre, 0.801 / 0.802
  at 4 deg, 0.547 / 0.526 at 8 deg, 0.300 / 0.308 at 10 deg, then the model falls faster: 0.129 / 0.186 at
  11 deg, 0.059 / 0.141 at 11.39 deg, 0.020 / 0.052 at 13.39 deg.
- The published step is smoother than the model at every depth: the model has a peak near 11 deg (0.49
  against 0.26 at its centre) and drops to 0.02 by 12 deg; the published curve rises from 0.31 to 0.36 and
  falls over 3 deg.

Distances (CAMB background, Mpc), against the figure's D = 14,165.5 and r = 13,886.6:

| parameters | D* | tau_0 | D* - r | distance to z* without radiation |
|---|---:|---:|---:|---:|
| Planck 2018 | 13,872.8 | 14,153.3 | -13.8 | 13,937.2 |
| Planck 2013 | 13,883.3 | 14,163.7 | -3.3 | 13,947.7 |
| WMAP 7-year | 14,049.3 | 14,333.8 | +162.7 | 14,118.7 |
| WMAP 9-year | 14,002.4 | 14,286.1 | +115.8 | 14,070.8 |
| CAMB default 2013 | 14,077.6 | 14,362.2 | +191.0 | 14,147.2 |

No parameter set has D* = 14,165.5. Two things come close: the conformal age tau_0 of the Planck 2013
parameters (1.9 Mpc away, and then r is 3.3 Mpc from that D*), and, more loosely, the WMAP sets, for which
the stated r puts the wall 116 to 163 Mpc deep, which brackets the fitted 130 to 140 Mpc. I cannot tell from
the paper which, if either, happened.

Not pre-registered, job 82919 (`runs/template_check/osborne_fig3_with_smoothing/`): the same fit with a
smoothing of the model. Gaussian smoothing from 0 to 4 deg FWHM lowers the residuals steadily (ramp 3.2% to
0.9%, step 16.4% to 6.5%, both cosmologies) and the best value is the last one tried, so this does not
determine a smoothing scale; a sharp cut in l does not help. The fitted depth stays at 120 to 140 Mpc for
every variant, and the label depth never does better than 12.8% (ramp) and 35% (step).

Second published curve, also not pre-registered (`scripts/analysis/compare_feeney2015_kernel.py`,
`runs/template_check/feeney2015_fig6/`). Fig. 6 of Feeney, Elsner, Johnson and Peiris (arXiv:1506.01716) shows
r^2 alpha_l(r) for l = 20, alpha_l(r) = (2/pi) integral dk k^2 Delta_l(k) j_l(kr), the real-space form of the
transfer function. A pure Sachs-Wolfe transfer function would give a spike at D*; a Doppler term moves
weight beyond D* and leaves a negative lobe inside. The figure is a vector file in the arXiv source and was
read the same way (139 and 138 points). Computed from this project's transfer functions with their
parameters (Planck 2013 + WP + highL + BAO, D* = 13,907.8 Mpc):

| | published | this work |
|---|---:|---:|
| temperature, peak position | 14,058 Mpc | 14,057 Mpc (D* + 149) |
| temperature, negative / positive area | 0.67 | 0.70 |
| temperature, first moment about D* / integral | +496 Mpc | +494 Mpc |
| temperature, correlation of the two curves | | 0.963 |
| polarization, peak position | 13,783 Mpc | 13,785 Mpc |
| polarization, correlation; RMS / maximum | | 0.996; 2.9% |

The temperature residual (8.5% of the maximum) is ringing in my curve next to the spike: CAMB gives
Delta_20(k) only up to k = 0.132 /Mpc. The overall sign is opposite, as it should be (they use
dT/T = +R/5). So an independent implementation shows the same displaced spike and negative lobe, with the
same first moment of about +495 Mpc at l = 20. That is the feature that produces the plateau.

Where item 1 of the open list stands.
- For the template: five internal checks (entries (f) to (h)), the long-wavelength argument of entry (u),
  and now the reproduction of a published transfer-function curve from a different group.
- Against it: nothing measured. Fig. 3 of Osborne et al. is not reproduced as labelled, but it is
  reproduced in its main part (the ramp to 3%, or to 1% with smoothing) by the same template with the wall
  130 to 140 Mpc deep instead of 279.
- Not explained: why their figure corresponds to that depth, and why their curves are smoother. Only the
  authors can answer that. Not done: a second Boltzmann code (all three groups and this project use CAMB).
- The template is unchanged, and so is every number from (l) on.
- The full-sky search that comes next will be run with the edge-free template of McEwen et al. as well, so
  its results do not rest on this point.

## (w) 2026-11-22 01:10 CST. Pre-registration: full-sky filter search, WMAP 7-year W band first, then Planck

Purpose. The real-sky test of entries (q) to (s) searched 48% of the sky for five sizes between 5 and 25 deg,
because it was built around the network's patches. The filter itself runs on the whole sphere. This entry
fixes a search done the way McEwen, Feeney, Johnson and Peiris (arXiv:1202.2861) did theirs: every unmasked
position, their 33 sizes from 1 to 90 deg, significance per position from simulated skies with the same
mask. It is run first on their own data, as a check of this implementation against their published table,
and then on the four Planck PR3 maps.

Script: `scripts/analysis/fullsky_filter_search.py`; the procedure and every choice of mine are in its
header. Written from the text of their Secs. III C and IV. State when this entry was written: the code had
run on simulated skies only (test jobs 82920, 82922: template coefficients against quadrature 4e-5; region
finder exact on three test caps, one over a pole and one over the longitude seam; injected amplitude
recovered to 0.2% on an unmasked sky; the per-pixel mean of the filtered simulations consistent with zero).
No real map had been searched with it.

One thing the test changed. I first set the threshold of a size as the lowest value above which a simulated
sky has on average at most two connected regions. On the small test that gave 3.6 to 4.9, far above the
significances in their table (3.03 to 4.67), and about 40 candidates per simulated sky. Their text counts
detections after the step across sizes ("detections on one scale can be eliminated by stronger detections
made on adjacent scales"), so their thresholds are lower than a count before that step allows. The same
group printed its thresholds in Feeney et al. (arXiv:1210.2725, Table 1). Those are used here.

Fixed before the runs.
- Sizes: 1, 1.5, 2, 3 ... 10, 12 ... 30, 35 ... 90 deg (their grid). Adjacent sizes: within 5 deg (theirs).
- Two templates, each with its own filters, thresholds and candidate list: "cosine", their template with
  z_crit = 0; "transfer", the ramp through the transfer functions (entries (f), (v)).
- Preparation of every sky: monopole and dipole fitted on the unmasked pixels and removed, masked pixels set
  to zero. The search runs over all unmasked pixels.
- Simulated skies: sigma maps from 3,000 (their number); the distribution under no signal from 1,000 more.
  Seed 20261122. Gaussian, no foreground residuals.
- Thresholds: Feeney et al. (2013) Table 1, for both templates and every data set: 4.25 at 1 to 2 deg, 4.00
  at 3 to 8, 3.75 at 9 to 16, 3.50 at 18 to 26, 3.25 at 28 to 75, 3.00 at 80 to 90 deg. They were set on
  one end-to-end simulation of WMAP with the KQ85 mask, "at most three false detections on each scale",
  which gave 12 false detections on that simulation. McEwen et al. (2012) do not print theirs; they are
  described as more conservative, with 13 false detections on the same kind of simulation.
- A potential detection is one connected region above the threshold. It is dropped if a region of an
  adjacent size overlaps it, has the same sign and a larger amplitude estimate inside it (their rule, as I
  read it). What is left is a candidate.
- Match to a published candidate, and between maps: same sign, size within 5 deg, centres within the larger
  of 3 deg and a quarter of the size.
- Masks at Nside 256, pixel kept where the degraded mask is at least 0.9: WMAP KQ75y7 (69.7% of the sky;
  70.6% at full resolution, the number in their paper), Planck common mask (77.2%).

Part 1, the check on WMAP. Data: the foreground-reduced W-band map of the 7-year release
(`wmap_band_forered_iqumap_r9_7yr_W_v4.fits`), noise sigma_0 / sqrt(N_obs) with sigma_0 = 6.549 mK, 13.2
arcmin beam, l up to 256 as in their paper, WMAP 7-year parameters. Their table lists 16 candidates with S
from 3.03 to 4.67. Differences I cannot remove: Nside 256 instead of 512; the noise enters my filter;
thresholds from Gaussian skies instead of one end-to-end simulation; the mask threshold; my reading of
"localized peaks".
- Criterion (a), filters and significance maps: at their 16 positions and sizes, the largest S within the
  larger of 1.5 deg and a quarter of the size is within 0.5 of the published S for at least 12 of the 16.
- Criterion (b), the whole procedure: 11 of the 16 have a published S at or above the threshold of their
  size in the table used here (nos. 0, 4 to 10, 12, 14, 15; three of them, nos. 0, 5 and 14, by 0.03 or
  less). At least 7 of these 11 come out as candidates by the match rule.
- If (a) fails, the implementation is not validated and no Planck number from this script is used until
  the cause is found. If (a) holds and (b) fails, the filters agree and the difference lies in the
  thresholds or in the step across sizes; it is reported as that.
- Reported without a criterion: the number of candidates next to the 1,000 simulated skies (they found 16,
  and 13 on their end-to-end simulation), and the same search on the W-band map without foreground
  reduction.

Part 2, Planck: SMICA, NILC, SEVEM, Commander, 5 arcmin beam, the simulated skies of entry (q), l up to 256
as in their search (data set "planck"). The same with l up to 767, the limit of the Nside 256 maps, is run
as a second data set ("planck767") and reported separately; the like-for-like result is the first.
Reported per map and template: the candidates; their number and its rank among the simulated skies; the
largest S over all sizes and its rank; which of the 16 published candidates are recovered; which candidates
are found in all four maps.
- What would count as an excess: the number of candidates, or the largest S, with p below 0.01 in all four
  maps for the same template. An excess would not be a detection. It would go to the frequency and
  half-mission checks first.
- Expectation, written before the run: no excess. The Cold Spot (their no. 7) is the strongest feature at
  the small sizes. Published candidates that are masked here or have no counterpart are listed as such.

Not part of this stage: injection tests over the full searched area (the sensitivity as a function of size
and amplitude), the Planck end-to-end simulations, any Bayesian step.

## (x) 2026-11-22 02:00 CST. Full-sky filter search: the WMAP check misses its criteria; Planck shows no excess

Runs of entry (w): jobs 82923, 82926, 82927 (WMAP, three runs), 82924 (Planck, l up to 256), 82925 (Planck,
l up to 767). Outputs `runs/fullsky_search/<data set>_v<n>/`, tables from
`scripts/analysis/fullsky_search_summary.py`, short versions in `work/results_2026-11-22/`.

Part 1, the check on WMAP. Both criteria fail. Three runs, cosine template, foreground-reduced W band:

| run | what changed | rms of S over the unmasked sky at 1, 4, 8 deg | (a) within 0.5, of 16 | (b) recovered, of 11 | candidates | largest S (p) |
|---|---|---|---:|---:|---:|---|
| 82923 | as pre-registered | 0.86, 0.92, 0.89 | 4 | 4 | 10 | 4.50 (0.99) |
| 82926 | power amplitude at the right scale | 0.91, 0.98, 0.94 | 10 | 5 | 18 | 4.78 (0.73) |
| 82927 | published beam instead of a Gaussian | 1.00, 0.99, 0.95 | 10 | 6 | 27 | 4.88 (0.56) |

Needed: 12 for (a), 7 for (b). Simulated skies give 51 candidates on average (5% to 95%: 20 to 107).

The first two runs had two errors of mine in the WMAP simulations, not in the search itself.
- WMAP quotes the primordial amplitude 2.441e-9 at k = 0.002 /Mpc; I passed it to CAMB, whose reference
  scale is 0.05 /Mpc. That is 12.6% too much power, so every S was 6% low.
- I used a 13.2 arcmin Gaussian beam. The published W-band beam transfer functions (mean of W1 to W4) are
  0.929, 0.798, 0.727 at l = 100, 200, 256, where that Gaussian has 0.987, 0.948, 0.916.
Both were found from a number that does not use the published table: the rms of S over the unmasked pixels
of the real map, which is 1 if the simulated skies have the power of the map. It is now part of the output.
For Planck it was 1.00 from the start.

What agrees after the third run. The amplitude estimates at their positions: +0.228 here against +0.24 mK
published (no. 0), -0.195 / -0.19 (no. 3), -0.172 / -0.17 (no. 4), -0.160 / -0.16 (no. 6), +0.143 / +0.15
(no. 8), +0.091 / +0.11 (no. 10). The strongest features: no. 6, S 4.88 here against 4.56; no. 4, 4.14 / 4.26;
no. 0, 4.03 / 4.25; no. 10, 3.99 / 3.94.

What does not.
- For sizes up to 40 deg my S is lower than the published one by 0.27 on average, with a scatter of 0.33.
  Four are 0.6 to 0.7 low: no. 2 (3.48 / 4.08), no. 5 (3.41 / 4.02), no. 7, the Cold Spot at 5 deg
  (3.96 / 4.67; here it peaks at 6 and 10 deg with S 4.43 and 4.68), no. 9 (3.77 / 4.39).
- The two largest, nos. 14 and 15 at 85 and 90 deg, are not there: S 1.8 and 2.0 against 3.03 and 3.11,
  amplitudes near zero against +0.15 and -0.16 mK. Their paper notes that this pair looks "somewhat like
  a dipole contribution". This pipeline removes a monopole and dipole fitted outside the mask and drops
  both from the template. I have not tested which choice of theirs produces the pair.
- Not established: why the rest is lower. Candidates are the differences listed in (w) (Nside, the noise
  in my filter, the mask threshold) and the fact that their S values are the peaks of their own maps.
By the rule of (w) the implementation is therefore not validated against their table. The Planck numbers
below are what this pipeline gives; they are not a validated repeat of their search.

A second difference from their papers, in the counting. With the published thresholds a Gaussian sky gives
2.1 regions per size before the step across sizes and 51 candidates after it. They report at most three
per size and 12 or 13 candidates on their end-to-end simulation. So their step across sizes removes far
more than mine, which lets one structure count several times here (the feature near (160 to 167, -49 to
-60) four times, at 18 to 26 deg). The comparison with the simulated skies is not affected, since they are
counted the same way.

Part 2, Planck, l up to 256, the same thresholds. Per map, SMICA / NILC / SEVEM / Commander:

| template | candidates | simulated skies: mean, 5% to 95% | p | largest S | p |
|---|---|---|---|---|---|
| cosine | 33 / 34 / 32 / 31 | 52.6, 23 to 106 | 0.77 / 0.76 / 0.79 / 0.82 | 4.80 / 4.78 / 4.79 / 4.78 at 4 deg | 0.73 to 0.76 |
| transfer | 40 / 39 / 34 / 38 | 99.0, 19 to 269 | 0.78 / 0.79 / 0.84 / 0.80 | 4.56 / 4.56 / 4.61 / 4.59 at 5 deg | 0.38 to 0.44 |

- No excess by the criterion of (w) (p below 0.01 in all four maps): every p is between 0.38 and 0.84.
- The four maps agree: 32 of the 33 SMICA candidates (cosine) and 36 of the 40 (transfer) are in all four,
  with S equal to 0.05.
- By size: no candidate at 45 deg or more, where a simulated sky has 1.5 to 3.2 regions per size; at 50 to
  70 and 80 to 90 deg the largest S is lower than in 94 to 100% of the simulated skies (at 75 deg, in 54%).
  The rms of S falls from 1.00
  at 1 to 4 deg to 0.90 at 30 deg and 0.61 at 90 deg. That is the low large-angle power of the real sky.
  The smallest p of a single size is 0.07 (10 deg) and 0.08 (4 deg).
- Strongest candidates (cosine, in all four maps): 4 deg at (l, b) = (269.4, -77.0), z0 -0.18 mK, S 4.80
  (their no. 6); 2 deg at (22.5, -24.8), -0.22 mK, 4.63 (not in their table); 1.5 deg at (245.0, +25.4),
  +0.26 mK, 4.58 (not in their table); 10 deg at (213.5, -59.5) and 6 deg at (207.2, -57.2), -0.14 and
  -0.16 mK, 4.56 and 4.55 (the Cold Spot, their no. 7); 6 deg at (264.6, -19.5), +0.16 mK, 4.56.
- Of their 16: nos. 0, 4, 5, 6, 7, 8, 10, 11 have a counterpart (and no. 2 by a 6 deg feature 2.8 deg
  away). Not found: nos. 1 and 3 (S 3.98 and 3.89 against a threshold of 4.25), no. 9 (3.80 against 4.00),
  nos. 12 to 15 (2.94, 2.47, 1.87, 1.68). S is within 0.5 of the published value for 10 of 16.
- Several of the large candidates lie next to the Galactic mask with half of their disc masked (30 deg at
  (228.7, +6.6), 64% unmasked; 35 deg at (166.5, +6.0), 55%).

Planck, l up to 767 (second data set): 94 / 104 / 96 / 95 cosine candidates against 152 on average (59 to
345), p 0.66 to 0.73; largest S 5.23 at 1.5 deg, (245.2, +25.3), +0.27 mK, p 0.23 to 0.26. Transfer: 72 / 69 /
70 / 71 against 265 (41 to 781), p 0.86 to 0.88. 93 of 94 SMICA candidates are in all four maps. At this
resolution S is within 0.5 of the published value for 12 of 16, and no. 9 is recovered as well.

Summary. On every Planck map, with either template and either band limit, the number of candidates and the
largest significance are ordinary among Gaussian skies with the same mask. Eight of the 16 WMAP candidates
of McEwen et al. are found again at l up to 256. The check against their table on their own data is
partial: amplitudes and the strongest features agree, the significances are lower by about 0.3, and their
two largest candidates do not appear.

Open after this entry: why my significances are lower than the published ones; which treatment of the
lowest multipoles produces their nos. 14 and 15; the count after the step across sizes. Not done, as fixed
in (w): injection tests over the searched area, the Planck end-to-end simulations, any Bayesian step.

## (y) 2026-11-22 02:38 CST. The Osborne figure again, with this project's template as the suspect; pre-registration

Entry (v) left the comparison with Fig. 3 of Osborne, Senatore and Smith at "their figure is matched at a
different depth". That reads the disagreement in this project's favour. This entry turns it round: assume
their figure is right as labelled and look for the error here.

What separates the two. For a wall 279 Mpc inside the last-scattering surface the template gives a centre
2.4 times the Sachs-Wolfe value and an edge-to-centre ratio of 0.48. Their figure has 0.16, and their text
calls the Sachs-Wolfe ramp "excellent for most bubble sizes". The whole difference is a plateau of about
400 Mpc depth-equivalent that the template assigns mostly to the motion of the plasma inside the collision
region (the Doppler term). If that plateau is not real, the template is wrong for every disc below about 30
deg and so is every number from entry (l) on.

Where an error here could sit, and what has been checked so far:
1. In the use of CAMB's transfer function in the harmonic formula (phases, the sum over multipoles, the k
   range CAMB provides per multipole). Checked only against CAMB-based results: entries (f) to (h), and Feeney
   et al. Fig. 6 in (v), which is also CAMB.
2. In CAMB itself. Not checked; all three published groups and this project use it.
3. In the physical argument of entry (u) for the Doppler term. It is an argument, not a calculation.

Test, pre-registered (`scripts/analysis/realspace_ramp_check.py`). The profile for depths of 279 and 135 Mpc
is rebuilt in real space by two routes that use neither CAMB nor a transfer function Delta_l(k) nor a sum
over multipoles:
- "toy": a fluid model written from the textbook equations (CDM, tightly coupled photons and baryons, a
  neutrino fluid, instantaneous recombination, two Gaussian damping factors). No Boltzmann code at all.
- "class": the Boltzmann code CLASS in Newtonian gauge: photon density, baryon velocity divergence and
  potentials on a grid of times, CLASS's visibility function, the integrated Sachs-Wolfe term.
In both the velocity of the plasma comes from its divergence, that is from the continuity equation, so no
sign convention of a code enters, and the code checks that equation numerically for CLASS.

Outcomes fixed in advance, for the depth of 279 Mpc:
- "The template is wrong": both routes give a centre within 20% of the Sachs-Wolfe value (at most 1.2) and
  an edge-to-centre ratio at or below 0.25.
- "The template is confirmed": CLASS agrees with the CAMB template to 5% of the central value at every
  angle from 0 to 16 deg, the toy model to 20%, and both have a centre above 2 and an edge-to-centre ratio
  between 0.4 and 0.6.
- Anything else is reported as mixed, with the numbers.

Also started, with no criterion: reading the published profiles of Kleban, Levi and Sigurdson
(arXiv:1109.3473, Fig. 2) from their figure, which is a rendered image and not vector data; and trying to
reproduce the Osborne et al. curves by procedures they could have used.

## (z) 2026-11-22 03:00 CST. Verdict on the Osborne figure: the template stands; the figure's label does not match its curve

Result of the test pre-registered in (y), job 82929 (`runs/template_check/realspace_ramp/`). Profile of a wall
279 Mpc inside the last-scattering surface, in units of the Sachs-Wolfe central value:

| route | centre | half radius | edge | edge + 2 deg | edge + 4 deg | edge / centre | largest difference from CAMB, of the centre |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAMB, harmonic template (entries (f), (g)) | 2.509 | 2.252 | 1.236 | 0.461 | 0.043 | 0.493 | |
| fluid model written here, no Boltzmann code | 2.577 | 2.286 | 1.269 | 0.489 | 0.000 | 0.492 | 2.7% |
| CLASS, real space | 2.501 | 2.240 | 1.206 | 0.413 | 0.000 | 0.482 | 2.0% |

At 135 Mpc: centre 4.041 / 4.052 / 4.017, edge-to-centre 0.633 / 0.651 / 0.622, differences 2.9% and 2.7%.

Outcome by the criteria of (y): the template is confirmed. CLASS agrees to 2.0% (5% allowed), the fluid
model to 2.7% (20% allowed), both have a centre above 2 and an edge-to-centre ratio between 0.4 and 0.6.

What the three routes share and do not share. CAMB route: transfer function Delta_l(k), the harmonic formula
of Osborne et al., a sum over multipoles. CLASS route: a different Boltzmann code, fields in real space, the
plasma velocity taken from its divergence (CLASS satisfies the continuity equation to 0.6% at last
scattering), a line-of-sight sum over 286 emission times. Fluid route: seven ordinary differential equations
per wavenumber from Ma and Bertschinger, instantaneous recombination. Parts in the CLASS route, at the
centre: density 1.09, Doppler 1.31, integrated Sachs-Wolfe 0.10. The Doppler part is the plateau: 1.31 at the
centre, 1.29 at half radius, 0.58 at the edge, as entry (g) found from CAMB (1.27, 1.25, 0.55). Its
depth-equivalent at long wavelengths is 317 Mpc in the fluid model and 311 Mpc in CLASS.

Two errors of mine on the way, both in the new fluid model, neither in the template. The first run (82928)
started the fluid model from the leading radiation-era forms; those break the density constraint at order
a / a_eq and the equations then hold the potential constant through equality (density constraint off by 19%).
Starting on the exact superhorizon solution fixes it (constraint to 0.1%, potential 0.62403 against the
expected 0.62412). CLASS needed the setting its error message named.

The three tests passed on from outside, on what the published figure is:
1. Does the Feeney et al. curve carry a depth? No. Their Fig. 6 is the transfer function against radius and
   holds no collision. But their Fig. 2 maps the distance of the wall to the angular radius, and that curve
   reaches zero at 13.913 Gpc and follows acos(x_c / 13.914 Gpc) to 0.06 deg; CAMB puts last scattering at
   13,908 Mpc for their parameters. So their size convention is this project's. And the depth handling here
   is now tested by a route in which the wall is placed geometrically (the CLASS route).
2. Was the 3% match of (v) on shape or on height? On shape: the fit had a free amplitude. With both curves
   scaled to a peak of 1 and nothing fitted, the ramp still picks 135 Mpc (RMS 3.4%; 110 to 160 Mpc within
   1.5 times of that), and 279 Mpc gives 17.9%. Half maximum of the published ramp at 8.78 deg; of the
   template at 9.05 deg for 135 Mpc and 11.45 deg for 279 Mpc. A geometry or definition problem, not a
   normalisation.
3. Any other number in their text for the same collision? The paper states r = 13,886.6 Mpc, the label
   11.39 deg, and that it uses "the WMAP7+BAO+H0 cosmological parameters throughout". For those parameters
   CAMB gives a last-scattering distance of 14,049 Mpc. With it, r is 163 Mpc inside the surface and
   theta = sqrt(2 d / R) = 8.7 deg. The label needs 14,165.5 Mpc, which is 116 Mpc more.

Verdict.
- On this project: the template is what the ramp model gives. Two Boltzmann codes and a model with no
  Boltzmann code agree to 3%, the published transfer function of Feeney et al. is reproduced (entry (v)),
  and the plateau has a plain cause: the plasma on the two sides of the wall falls towards it.
- On the figure: the curve is consistent with this calculation, the label is not. As labelled (11.39 deg,
  279 Mpc deep) the figure is not reproduced by any of the four calculations. Its shape is that of a wall
  110 to 160 Mpc deep, a disc of 7 to 9 deg, and the stated r with the stated cosmology gives 163 Mpc.
- Not explained, and only the authors can say: where the distance 14,165.5 Mpc of the label comes from; why
  their curves are smoother than these (the step fits at 16% at best); and their sentence that the
  Sachs-Wolfe ramp is "excellent for most bubble sizes", which this calculation contradicts below about
  30 deg. A last-scattering distance too large by a hundred Mpc would make the Sachs-Wolfe ramp of the
  labelled size look like the true profile of the smaller disc; that is my guess at how the sentence and
  the figure hang together, not something shown.

A second published figure, read with less certainty. Kleban, Levi and Sigurdson (arXiv:1109.3473) Fig. 2 is
a rendered image of the profiles for all disc sizes, not vector data. The outline of its smallest disc can
be measured from pixels (tip of the surface at the peak, zero level from the edge of the flat part, 9.75
pixels per degree from the axis ticks): half maximum at 5.6 deg, 2% of the peak at 9.6 deg. From the axis
ticks that disc is labelled 8.0 deg, or 9.7 deg if the axis has the same 2% padding as the other one. As
read, the outline is close to a Sachs-Wolfe ramp of 8.0 deg (RMS 7%) and far from this template at the
labelled sizes (30 to 37%); this template gives such a compact spot only for a wall that just touches the
surface (best depth -15 Mpc, RMS 6%). In this calculation a wall touching the surface makes a spot of 8.3
deg, the sound horizon seen side-on, and no ramp collision makes a smaller one, so their axis starting at
8 deg may be exactly that. I cannot decide this from an image.

Consequences here. The template, the benchmark and the search are unchanged. Two things follow from the
plateau that the earlier entries did not spell out: a ramp collision cannot look smaller than about 8 deg,
and a disc of nominal radius below about 30 deg looks like a Sachs-Wolfe disc of a larger radius. Published
candidates of 1.5 to 7 deg found with the edge-free template are therefore not ramp collisions in this
calculation, whatever else they are.

Not done: reproducing the Osborne et al. curves by a procedure they could have used (their smoothing is the
part still missing). A question for the authors is the direct way.
