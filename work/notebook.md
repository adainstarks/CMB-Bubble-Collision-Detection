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
