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

## (aa) 2026-11-22 03:05 CST. Addition to (z): numerical shortcuts do not give the smoother published curves

Entry (z) left one thing untried: whether a shortcut in the numerics turns this calculation into the
smoother curves of Osborne et al. Fig. 3. Tried, not pre-registered, with the WMAP 7-year parameters their
paper states (job 82931, `compare_osborne_fig3.py --emulate`):

| procedure | ramp: best depth, RMS | ramp at r as stated (163 Mpc) | ramp at the label (279 Mpc) | step: best depth, RMS |
|---|---|---:|---:|---|
| every multipole, refined k (the template) | 140 Mpc, 3.1% | 4.0% | 13.7% | 130 Mpc, 16.5% |
| every multipole, CAMB's own k grid | 145 Mpc, 3.6% | 4.1% | 13.5% | 130 Mpc, 16.8% |
| CAMB's default 77 multipoles, b_l interpolated | 140 Mpc, 3.2% | 4.2% | 13.7% | 120 Mpc, 22.6% |
| both shortcuts | 145 Mpc, 3.7% | 4.3% | 13.5% | 125 Mpc, 22.7% |

None of them brings the step curve closer, and none moves the depth: the ramp keeps picking 140 to 145 Mpc
and stays at 13.5% or worse at the labelled depth. So the extra smoothness of their curves is not one of
these shortcuts, and the verdict of (z) does not depend on how the integrals are done. What produced their
smoothing stays a question for the authors.

## (ab) 2026-11-22 03:16 CST. Correction to (z) and (aa): the stated cosmology has massless neutrinos

In (z) I wrote that the distance r = 13,886.6 Mpc of the Osborne et al. caption, with the WMAP7+BAO+H0
parameters their paper states, puts the wall 163 Mpc inside the last-scattering surface (a disc of 8.7 deg)
and that the label needs a distance 116 Mpc larger. Those numbers came from CAMB runs that carried one
0.06 eV neutrino, the default of this project's Planck set-up. The WMAP 7-year parameters assume massless
neutrinos. With that:

| | last-scattering distance | wall depth for the stated r | disc | label distance minus this |
|---|---:|---:|---:|---:|
| one 0.06 eV neutrino (used in (v), (z), (aa)) | 14,049.3 Mpc | 162.7 Mpc | 8.73 deg | 116.3 Mpc |
| massless neutrinos (the stated cosmology) | 14,071.9 Mpc | 185.3 Mpc | 9.31 deg | 93.7 Mpc |

14,072 Mpc is also the value in the WMAP 7-year tables for that parameter set (14,073, quoted from memory).

What changes. In (z) the stated r landed at the edge of the range the shape of their ramp prefers (110 to
160 Mpc). At 185 Mpc it is outside it: the template fits their ramp with an RMS of about 6% there (5.5% at
180 Mpc, 7.3% at 200 Mpc in the table of job 82917), against 3.1% at the best depth of 140 Mpc and 13.7% at
the labelled 279 Mpc. So the literal reading of the caption is twice as far from the curve as the best
depth, and still more than twice as close as the label.

What does not change. The template and its three-way confirmation; that the figure as labelled is not
reproduced; that the shape of the published ramp picks 110 to 160 Mpc. What I can no longer say is that the
caption's r and the stated cosmology account for the curve. The curve sits about 45 Mpc shallower than even
that reading, and its smoothness is unexplained as before. The transfer functions of the WMAP runs in (v),
(z) and (aa) carried the same neutrino; the fitted depth is defined from each run's own last-scattering
distance, so it is not shifted by this.

## (ac) 2026-11-22 03:21 CST. Addition to (ab): the label's distance with and without radiation

The table of (v) had a column "distance to z* without radiation". It was computed with the 0.06 eV neutrino
counted as matter, the same slip as in (ab). For the cosmology the paper states (WMAP7+BAO+H0, massless
neutrinos, z* = 1089.0), distance to last scattering in Mpc, against the 14,165.6 the label needs:

| radiation in the distance integral | mean parameters | maximum-likelihood parameters | label minus this |
|---|---:|---:|---:|
| photons and three neutrinos (CAMB) | 14,071.9 | 14,069.6 | +94 to +96 |
| photons only | 14,100.7 | 14,098.3 | +65 to +67 |
| none | 14,145.7 | 14,143.2 | +20 to +22 |

Leaving radiation out of the integral closes 74 of the 94 Mpc. The remaining 20 Mpc would need the integral
to run to z = 1185 instead of 1089 (14,148 at z = 1100, 14,169 at z = 1200). The conformal age is 14,356
Mpc, far off. So a distance computed without radiation is the closest simple origin of the label that I can
find, and it is not a match. It stays a guess; the authors would know.

Nothing here touches this project's own runs: the Planck set-up carries photons, neutrinos and one 0.06 eV
neutrino, which is the Planck convention.

## (ad) 2026-11-22 03:39 CST. Where the label's distance comes from: the WMAP 5-year maximum-likelihood parameters

Entries (z) to (ac) left the distance of the Osborne et al. label, 14,165.5 Mpc, without an origin. Asked what
the WMAP 5-year distance to last scattering was, I computed it with CAMB (massless neutrinos) for the
parameter sets of every WMAP release:

| parameter set (H0, 100 Omega_b h^2, Omega_c h^2) | distance to last scattering | label minus this |
|---|---:|---:|
| WMAP 1-year (71.0, 2.24, 0.1126) | 14,044.0 Mpc | +121.6 |
| WMAP 3-year, WMAP alone (73.2, 2.229, 0.1054) | 14,267.0 | -101.5 |
| WMAP 5-year, WMAP alone, mean (71.9, 2.273, 0.1099) | 14,106.7 | +58.9 |
| WMAP 5-year, WMAP alone, maximum likelihood (72.4, 2.27, 0.108) | 14,168.5 | -2.9 |
| WMAP 5-year + BAO + SN (70.5, 2.267, 0.1131) | 14,032.2 | +133.4 |
| WMAP 7-year, WMAP alone (71.0, 2.258, 0.1109) | 14,106.9 | +58.7 |
| WMAP 7-year + BAO + H0 (70.4, 2.260, 0.1123) | 14,071.9 | +93.7 |
| WMAP 9-year, WMAP alone (70.0, 2.264, 0.1138) | 14,024.7 | +140.9 |
| WMAP 9-year + eCMB + BAO + H0 (69.32, 2.223, 0.1153) | 14,007.7 | +157.9 |

The parameter values are from memory except the 5-year maximum-likelihood row, which I read from Table 2 of
Dunkley et al. (arXiv:0803.0586): H0 72.4, 100 Omega_b h^2 2.27, Omega_c h^2 0.108. The published 5-year
distances are 14,115 (+188, -191) Mpc for WMAP alone and 14,034 Mpc with BAO and supernovae (table of Hinshaw
et al., arXiv:0803.0732, as returned by a web search).

The 5-year maximum-likelihood set gives 14,168.5 Mpc, 3 Mpc from the label. The rounding of the three
printed numbers allows 14,143 to 14,194 Mpc. With four-digit values that I remembered (2.268 and 0.1081) but
did not find printed, CAMB gives 14,165.1 Mpc. No other set is within 58 Mpc. So the label's distance is,
to within what the published table can show, the distance to last scattering of the WMAP 5-year
maximum-likelihood parameters. The paper's text says it uses the WMAP7+BAO+H0 parameters throughout
(14,071.9 Mpc).

What this changes in (z) and (ab).
- "Where the distance of the label comes from" is no longer open. The guess of (ac), a distance without
  radiation, is dropped.
- The label is not arbitrary: for the 5-year maximum-likelihood cosmology a wall at r = 13,886.6 Mpc is
  279 Mpc deep and the disc is 11.39 deg. The summary line of (z), "the curve is consistent with this
  calculation, the label is not", was too neat. Two readings remain, and I cannot tell them apart:
  1. The transfer functions of the figure were also computed with the 5-year maximum-likelihood
     parameters. Then label and curve belong together, the wall is 279 Mpc deep, and the published ramp
     (edge-to-centre 0.16) is not what CAMB, CLASS or the fluid model give there (0.48 to 0.49).
  2. The transfer functions were computed with the stated WMAP7+BAO+H0 parameters and only the conversion
     to an angle used the older distance. Then the wall is 185 Mpc deep, where the template fits the
     published ramp to about 6%.
  In neither reading is the published curve reproduced as well as at the best depth of 140 Mpc (3.1%),
  and its smoothness is unexplained in both.
- Unchanged: the template and its confirmation by CLASS and the fluid model; that the figure as labelled
  is not reproduced.

So the disagreement with the paper is now this, stated without a theory of how it arose: for a wall 279
Mpc inside the last-scattering surface three independent calculations give a profile with a centre 2.5
times the Sachs-Wolfe value and an edge at 0.49 of the centre, and the paper's figure for a bubble labelled
that way shows a ramp close to the Sachs-Wolfe one. Which parameters went into the figure is a question for
its authors.

## (ae) 2026-11-22 03:49 CST. Back to the search: pre-registration of its sensitivity (injection test)

The question about the Osborne et al. figure is parked: a one-page note for a physicist and a draft email to
the authors are written, not sent. Work returns to the list. Entry (x) left the full-sky search without a
statement of what it could have seen. This entry fixes how that is measured. Simulated skies only.

Set-up: the Planck like-for-like search of (x) (data set "planck", l up to 256, the published thresholds,
sigma maps of job 82924). A ramp template of known size and amplitude is added to a signal-free simulated
sky at a random unmasked position, and the significance maps are evaluated around it. Filtering and
preparation are linear, so the response to a unit-amplitude template is computed once per (template, size,
position) and scaled.

Fixed before the run (`fullsky_filter_search.py --stage inject`):
- Sizes: 2, 3, 5, 7, 10, 14, 20, 30, 40, 50, 60, 75, 90 deg. Amplitudes (observed peak dT/T): 0, 1, 1.5, 2,
  3, 4, 5, 6, 8, 10, 15, 20, 30 times 1e-5. Positive sign only (the pipeline is symmetric in sign).
- Positions: 150 pixels drawn uniformly from the unmasked pixels, the same for every size. A disc may be
  partly masked; that loss is part of the answer.
- Backgrounds: 100 new simulated skies, and separately the four real maps.
- Injected and searched: the transfer template searched with the transfer filters and with the cosine
  filters; the cosine template searched with the cosine filters.
- Where a match is looked for: filter sizes within 5 deg of the true size, pixels within the larger of
  3 deg and a quarter of the size of the true centre (every pixel up to 6,000, a regular subsample
  beyond), positive filtered value.
- Criterion (a), "would be a candidate": S at or above the published threshold of that filter size. This is
  before the step across sizes, which can only hand the detection to an overlapping neighbour.
- Criterion (b), "would stand out": S at or above the value that the largest S of a whole signal-free sky,
  over all 33 sizes of that template, exceeds in 5% of the 1,000 skies of (x).
- Reported: the fraction of (position, sky) pairs meeting each criterion per size and amplitude, the
  amplitudes at which that fraction reaches 50% and 95%, and the fraction at zero amplitude.

No pass or fail here; it is a measurement. Expectation written beforehand: the 50% amplitude of (a) is near
threshold divided by the ideal signal-to-noise per unit amplitude (about 4e-5 at 10 deg); (b) needs roughly
a third more; the cosine filters lose against the transfer filters on transfer signals below about 30 deg;
discs of 60 deg and more reach 95% late or not at all because the mask cuts them.

What it is for: criterion (b) at 95% is the amplitude above which a ramp collision in the unmasked 77% of
the sky would have been the strongest feature of the map. The real maps have no such feature (largest S
4.80, p about 0.75), so that curve is the statement the search supports.

## (af) 2026-11-22 04:05 CST. Pre-registration: three checks of the sensitivity measurement

Written while job 82933 (the measurement of (ae): 100 skies, 150 positions) is running and before any of
the numbers below exist. None of the published figures named here has been opened yet; only their file
names in the arXiv source of McEwen et al. (2012) and the text of their Sec. IV B have been read.

1. The shortcut, end to end (`fullsky_inject_check.py`). The measurement adds a scaled filtered response
   to a filtered signal-free sky and looks at S near the true position, before the step across sizes. The
   check does a sample the long way: template added to the simulated map, then preparation, all 66
   filters, significance, regions and the step across sizes. Cases: sizes 3, 10, 30, 60, 90 deg; transfer
   template with transfer filters and cosine template with cosine filters; two amplitudes per size, the
   measured 50% amplitude of criterion (a) and the 95% amplitude of criterion (b); 8 random (position,
   sky) pairs each, 160 cases, new skies.
   - The shortcut is confirmed if the largest difference in S between the two ways is below 0.02 and
     criterion (a) comes out the same both ways in every case.
   - Whether (a) means "the search returns a matching candidate" is measured, not assumed: of the cases
     where (a) holds, the fraction where the full algorithm returns a candidate that the match rule of (w)
     accepts. If that fraction is below 90%, the (a) curves are described as "reaches the threshold" and
     not as "becomes a candidate".

2. The signal-to-noise curve of the filters against McEwen et al. Their source has a figure of the filter
   signal-to-noise for a collision of 100 uK as a function of size. My number for the same thing: the
   cosine template on the `wmap7w` data set (published W beam, noise from N_obs, l from 2 to 256),
   z0 * peak_value * sqrt(norm) with z0 = 100 uK / 2.725 K. This has no random part.
   - Within 5% at the sizes they plot from 5 deg up: the filters are normalised the same way, and the
     lower S of (x) (0.27 on average) comes from somewhere else.
   - Mine lower by 5 to 10% across sizes: that accounts for the lower S of (x).
   - Anything else: reported as it is. If their caption shows that the curve is for a different set-up
     (another multipole range or noise level), the comparison is repeated for that set-up and both are
     given.

3. The sensitivity regions against McEwen et al. Their Sec. IV B places collisions at two positions,
   (l, b) = (99.2, +32.3) and (193.0, +33.4) deg, on three simulated WMAP W skies each, and calls a
   (size, amplitude) pair "exclusion" if all six are detected, "sensitivity" if some are, "unprobed" if
   none. Mine: the `wmap7w` data set with the sigma maps and thresholds of run v3, cosine template and
   filters, the same two positions, 200 new simulated skies, the amplitude grid and the sizes of (ae);
   detection = criterion (a).
   - Correspondence fixed beforehand: all six detected has probability one half when the detection
     probability is 0.89; none of six has probability one half at 0.11. So their exclusion boundary is
     compared with my amplitude at a detection fraction of 0.89 (the worse of the two positions), their
     lower boundary with my amplitude at 0.11 (the better of the two positions), both as z0 in uK.
   - Same sensitivity: both boundaries within a factor 1.3 at every size they tested between 5 and 60 deg
     (their amplitude grid is coarse; the factor is replaced by one grid step if the grid can be read).
     Mine higher by more than that: my search is less sensitive than the published one by that factor,
     and the Planck curve of (ae) carries the same doubt. Mine lower: reported, not claimed as a gain.

What these do not test: foreground residuals, anything about the real sky, and collisions centred inside
the mask.

## (ag) 2026-11-22 04:34 CST. Sensitivity of the full-sky search: the measurement of (ae) and the three checks of (af)

Jobs: 82933 (Planck, the measurement), 82934 (WMAP at the two published positions, 200 skies), 82935 (end
to end, 160 cases). Scripts: `fullsky_filter_search.py --stage inject`, `fullsky_inject_check.py`,
`fullsky_inject_summary.py`, `compare_mcewen_sensitivity.py`, `read_published_figure_curves.py`. Files in
`work/results_2026-11-22/` (`fullsky_inject_*`, `compare_mcewen_sensitivity_report.json`,
`mcewen2012_fig*`); figure `plots/10_fullsky_search_sensitivity.png`.

The measurement. Planck set-up of (x): 77% of the sky, l up to 256; 100 simulated skies x 150 positions per
size. Amplitude = the peak of the disc as it sits in the map, dT/T in units of 1e-5, mean and dipole
removed. "Threshold" is criterion (a); "stands out" is criterion (b), S above 5.13 for the transfer filters
and 5.46 for the cosine filters. The percentages are the share of (position, sky) pairs.

| size (deg) | transfer: threshold 50% | threshold 95% | stands out 95% | cosine: threshold 50% | threshold 95% | stands out 95% | the last as z0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4.28 | 7.09 | 8.01 | 6.64 | 9.88 | 14.24 | 15.1 |
| 3 | 4.16 | 6.88 | 7.95 | 5.51 | 8.74 | 13.10 | 13.2 |
| 5 | 3.84 | 6.09 | 7.84 | 4.49 | 7.52 | 9.72 | 9.8 |
| 7 | 3.59 | 5.82 | 7.71 | 4.12 | 7.02 | 9.28 | 9.3 |
| 10 | 3.25 | 5.33 | 7.46 | 3.68 | 5.95 | 8.63 | 8.8 |
| 14 | 2.98 | 4.92 | 7.16 | 3.50 | 5.78 | 8.06 | 8.3 |
| 20 | 2.60 | 4.63 | 6.98 | 3.11 | 5.23 | 7.89 | 8.4 |
| 30 | 2.19 | 3.99 | 6.75 | 2.61 | 4.70 | 7.75 | 8.9 |
| 40 | 2.01 | 3.85 | 5.98 | 2.37 | 4.21 | 7.43 | 9.5 |
| 50 | 1.88 | 3.69 | 5.72 | 2.13 | 3.87 | 6.43 | 9.5 |
| 60 | 1.78 | 3.48 | 5.24 | 1.89 | 3.59 | 5.71 | 10.1 |
| 75 | 1.38 | 2.80 | 4.34 | 1.44 | 2.85 | 4.77 | 12.0 |
| 90 | 1.08 | 2.24 | 3.77 | 1.09 | 2.22 | 3.86 | 15.5 |

- In words, for this project's template: a collision whose peak in the map is above about 8e-5 (220 uK)
  below 30 deg, falling to 4e-5 at 90 deg, would have been the strongest feature of the whole map in 95% of
  cases. The largest S of the real maps is 4.80 (cosine filters) and 4.56 to 4.61 (transfer filters), under
  both bars. Half of the collisions reach the candidate threshold at 3e-5 to 4e-5 (80 to 120 uK) up to
  14 deg and at 1e-5 to 2e-5 from 50 deg up.
- The amplitude grid is coarse where the 95% levels fall (6, 8, 10 times 1e-5); the levels are interpolated
  and good to about 0.3e-5.
- On the four real maps the same injections give "stands out" 95% within 0.25e-5 of the simulated skies
  below 60 deg, and 0.3e-5 to 0.4e-5 lower from 60 to 90 deg.
- Chance: with no template added, criterion (a) is met in under 1% of the trials up to 14 deg, in 1 to 3.5%
  from 20 to 75 deg and in 4.4% at 90 deg; criterion (b) in none.
- The transfer template searched with the cosine filters (the published filters on this project's signal)
  needs 3 to 24% more for "stands out" 95% (most at 2, 3 and 40 deg) and -4 to +16% for the threshold 50%.
- In z0, the usual parameter (the central value of the cosine ramp before the mean and dipole are removed),
  "stands out" 95% is 8e-5 to 10e-5 from 5 to 60 deg and 15e-5 at 2 deg and at 90 deg. For the transfer
  template the same primordial gradient gives a brighter disc below 30 deg (2.9 times at 10 deg), so
  expressed as the z0 of the cosine ramp with that gradient its "stands out" 95% is 2.6e-5 at 10 deg,
  5.1e-5 at 20 deg and 6.6e-5 at 30 deg; from 50 deg up the two templates agree to 16%. This last
  translation is derived, not measured (`fullsky_inject_summary.py`).

The expectations written in (ae), against this:
- "50% of (a) near threshold / ideal signal-to-noise, about 4e-5 at 10 deg": 3.25e-5 and 3.68e-5. Roughly.
- "(b) needs roughly a third more": a third more at 2 to 3 deg, 50% more at 10 deg, twice as much from
  30 deg up (50% levels). Wrong above a few degrees.
- "cosine filters lose on transfer signals below about 30 deg": they lose a little at every size. Wrong
  about where.
- "discs of 60 deg and more reach 95% late or not at all": they reach it at the lowest peak amplitudes
  (and the highest z0, because most of a large disc goes into the mean and dipole). Wrong.

Check 1, the shortcut end to end (160 cases). The largest difference in S between the shortcut and the long
way is 2e-6, and criterion (a) comes out the same in 160 of 160. Confirmed. Of the 118 cases where (a)
holds, the full algorithm returns a candidate that the match rule accepts in 105 (89.0%), a candidate at the
right place whose size is more than 5 deg off in 10, and none within the match radius in 3. That is under
the 90% fixed beforehand, so criterion (a) is called "reaches the threshold" everywhere, not "becomes a
candidate". At the "stands out" 95% amplitude: 80 cases, all reach the threshold, 78 leave a candidate at
the right place, 70 of them within 5 deg of the true size. The misses are large discs that the step across
sizes handed to another size, usually a smaller one (90 to 80 deg, 60 to 50, 30 or 2 deg, 30 to 7 or 4 deg).

Looking at why, I found a deviation of my step 4 from the published one. Their text compares "the maximum
absolute value of the filtered field", and their filtered field estimates z0. My code compared the
amplitude in peak-in-the-map units. The two are the same to 3% up to 14 deg and differ by 1.15 at 30 deg,
1.8 at 60 deg and 4 at 90 deg, so my step 4 favoured the smaller of two neighbouring sizes more than
theirs does. Nothing in the sensitivity numbers depends on step 4. The candidate lists and counts of (x)
do; see (ah).

Check 2, the filter signal-to-noise against McEwen et al. Fig. 4a. Their curve is for a CMB-only sky without
beam (caption and Sec. III C), so mine was computed for that: exact template coefficients, WMAP7+BAO+H0
spectrum, l from 2 to 256. Mine / published: 0.904 at 2 deg, 0.919 at 3, 0.930 at 5, 0.936 at 6, 0.952 at
8, 0.964 at 10, 0.965 to 0.968 from 14 to 55, 0.973 at 60, 0.987 at 75, 0.996 at 90. With the filters of
the `wmap7w` data set (beam, noise, pixel window) my curve is the same to 0.3% from 3 deg up.
- By the readings fixed beforehand this is the third case, "anything else": inside 5% from 8 deg up, 5 to 7%
  low at 5 to 7 deg. There is no normalisation error of a size that would matter (tens of percent).
- The offset is not explained. Tried and ruled out: a higher band limit (2% at 3 deg and 1% at 5 deg, of
  the 8% and 7% missing), a beam in the background spectrum only (0.5%), the template sampled on a HEALPix
  grid of Nside 128, 256 or 512 instead of exact coefficients (0.5% or less). Their second curve (the
  unfiltered field) is not reproduced by the centre value of my template over the CMB rms either: theirs is
  4 to 12% lower up to 30 deg, 23% lower at 60 deg and 45% lower at 90 deg. So their numerical template
  differs from the analytic one in a way I cannot read from the text.
- I had written that an offset of 5 to 10% "accounts for the lower S of (x)". For the ten published
  candidates of 7 deg and smaller the offset is 5.5 to 10% and their S in my run is lower by 8% on average,
  so the numbers fit. I now think that sentence was too quick and do not take it as the explanation: a
  different template normalisation changes the signal-to-noise of an injected template, not the S of a
  given patch of sky; and S that is lower at positions picked for being high in someone else's analysis is
  what any imperfectly correlated repeat gives.

Check 3, the sensitivity regions against McEwen et al. Fig. 6 (their grid: sizes 2, 5, 10, 25, 50, 90 deg;
z0 = 25, 50, 75, 100, 150, 200, 400 uK). Mine: `wmap7w`, run v3, their two positions, 200 skies. z0 in uK:

| size (deg) | their exclusion boundary lies in | mine at 0.89, worse position | their lower boundary lies in | mine at 0.11, better position |
|---:|---:|---:|---:|---:|
| 2 | 200 to 400 | 270 | 100 to 150 | 114 |
| 5 | 150 to 200 | 183 | 50 to 75 | 59 |
| 10 | 100 to 150 | 147 | 25 to 50 | 51 |
| 25 | 100 to 150 | 139 (mean of 20 and 30 deg) | 25 to 50 | 37 |
| 50 | 75 to 100 | 154 | 25 to 50 | 31 |
| 90 | 150 to 200 | 215 | 75 to 100 | 19 |

- The rule covered 5 to 60 deg. Seven of the 8 comparisons are within one step of their grid (6 inside
  their bracket, 1 less than 1 uK outside it). The eighth, the exclusion boundary at 50 deg, is outside by
  2% (154 against an allowed 150). By the rule as written the check is not passed at 50 deg.
- What that one is: they detected all six of their 100 uK collisions at 50 deg; my detection probability
  there is 0.57 and 0.52, which gives six out of six in 2.7% of cases. Either their search is more sensitive
  than mine at 50 deg, by a factor between 1.5 and 2, or their six skies were lucky. Six skies cannot tell.
  Added afterwards, not pre-registered: of the 42 cells of their grid, the class they published has a
  probability of 13% or more under my detection probabilities in 41; this cell is the other one.
- At 90 deg my lower boundary (19 uK) is far below theirs. Criterion (a) at 90 deg is met by chance in 4.5%
  of trials and its match region is wide, so my 0.11 level there is mostly chance. Their three "unprobed"
  cells at 90 deg have probabilities 0.45, 0.25 and 0.13 under mine.

Also looked at, not pre-registered: their two largest candidates (nos. 14 and 15, 85 and 90 deg), which my
search does not see.
- The W-band maps over the unmasked sky: monopole +0.2 uK and dipole 6.7 uK (foreground-reduced), +15.5 uK
  and 4.7 uK (plain band map). Removing them or not changes S at the four largest published candidates by
  0.05 or less on the foreground-reduced map and by up to 0.2 on the plain map. That is not it.
- Filtering without the mask lets the Galactic plane dominate (S up to 17) and does not give their nos. 14
  and 15 either (wrong sign at no. 15).
- At the nearest maxima my amplitudes for nos. 12 and 13 are theirs (0.087 and 0.084 against 0.09 and 0.10
  mK; S 2.9 and 2.7 against 3.3 and 3.1). For nos. 14 and 15 mine have their sign and half their amplitude
  (+0.07 and -0.08 against +0.15 and -0.16 mK; S 1.8 and 2.0 against 3.0 and 3.1). Still not explained.
- Their source names the displayed map `wmap_band_imap_r9_7yr_W_v4_maskKQ75`, the plain band map's file
  name, while the text says foreground-cleaned. On the plain map run v3 finds 8 of the 11 eligible
  candidates as candidates (criterion: 7) and 9 of 16 within 0.5 in S (criterion: 12). Recorded only: the
  map fixed in (w) was the foreground-reduced one, and its verdict stands.

Where this leaves the search: its sensitivity is measured, the shortcut behind the measurement is exact, and
on WMAP it agrees with the published sensitivity within their grid except for one cell at 50 deg. The
repeat of their candidate list is still not validated.

## (ah) 2026-11-22 04:34 CST. Correction to step 4 of the search, and what is rerun (written before the reruns)

The deviation found in (ag): in step 4 (a potential detection is dropped when an overlapping region on a
neighbouring size has the same sign and a larger amplitude) my code compared amplitudes in
peak-in-the-map units. McEwen et al. compare the filtered field, which in their normalisation estimates z0.
For the cosine template the code now compares z0 (amplitude / peak_value). The factor between the two is
1.00 to 1.03 up to 14 deg, 1.15 at 30 deg, 1.8 at 60 deg, 4.0 at 90 deg. The transfer template has no
published convention; it stays in peak units, unchanged. `--step4-amplitude peak` reproduces the earlier
runs.

What can change: which of two overlapping detections on neighbouring sizes survives, so the candidate lists
above about 20 deg, the number of candidates, and that number's rank among simulated skies. What cannot:
the filtered maps, S, the largest S and its p-value, the thresholds, the sensitivity numbers of (ag), and
everything for the transfer template.

Reruns (stages null and data; the sigma maps are reused): `planck` (run v2), `planck767` (v2), `wmap7w`
(v4, the settings of v3); and the end-to-end check of (ag) on the same 160 cases.

Fixed beforehand:
- The WMAP check keeps the criteria of (w). Criterion (a) (S within 0.5 for 12 of 16) cannot change.
  Criterion (b) (7 of the 11 eligible found as candidates) can. Whatever it gives is reported; the verdict
  of (x) on runs v1 to v3 stands as written.
- Planck: reported are the candidate counts and their p among 1,000 simulated skies, old next to new, and
  whether any candidate of 45 deg or more appears. "No excess" in (x) rested on the count and on the
  largest S; if the new count has p below 0.05 on any map, that sentence is withdrawn until understood.
- End-to-end check: the share of cases with (a) that end as a matching candidate, old 89.0%. The wording
  "reaches the threshold" stays whatever the new share is.

## (ai) 2026-11-22 04:48 CST. Step 4 corrected: nothing moves on Planck, one candidate less on WMAP

Jobs 82936 (planck v2), 82937 (planck767 v2), 82938 (wmap7w v4), 82939 (end to end). Report:
`work/results_2026-11-22/fullsky_step4_correction_report.json` (`fullsky_compare_runs.py`).

- Planck, l up to 256: the candidate lists of all four maps are the same as before, entry for entry, for
  both templates (cosine 33, 34, 32, 31; transfer 40, 39, 34, 38). Simulated skies: 52.59 -> 52.53 cosine
  candidates per sky, 5% to 95% still 23 to 106. p of the counts 0.76 to 0.81. No cosine candidate of 45 deg
  or more.
- Planck, l up to 767: the same lists as before (cosine 94, 104, 96, 95); simulated skies 152.4 -> 152.1.
- WMAP W band (v4): one candidate less on each map, an 8 deg one at (l, b) = (209.5, -59.3) with S 4.0 that
  now loses to a neighbouring size. The check of (w): criterion (a) 10 of 16 as before; criterion (b) 6 of
  11 as before (it needs 7). On the plain band map 9 and 8, as before.
- End to end, the same 160 cases: of the 118 that reach the threshold, 110 now end as a matching candidate
  (93.2%, before 89.0%), 6 at another size, 2 with none in the match radius. At the "stands out" 95%
  amplitude: 74 of 80 match, 79 of 80 leave a candidate at the right place. The shortcut is unchanged
  (2e-6, 160 of 160).

So the correction matters for strong large discs, which is where it was found, and for nothing in (x): on
the real maps overlapping detections on neighbouring large sizes are rare. The numbers of (x) stand. The
wording "reaches the threshold" stays, as fixed in (ah). `fullsky_filter_search.py` now compares z0 by
default.

## (aj) 2026-11-22 16:15 CST. z_crit behind the scenes: what McEwen et al.'s own figure shows, and four tests (written before running them)

William's question (22 Nov): does z_crit affect the sensitivity, the signal-to-noise curve (check 2 of (ag)),
or the agreement with published searches; "you dont know if a parameter could exist behind the scenes on
accident".

Known before any test:

1. This project's code. `cosine_coefficients` in `fullsky_filter_search.py` integrates
   (mu - mu_c) / (1 - mu_c) over the disc: zero at the edge by construction, hard cut. There is no z_crit
   parameter anywhere in the full-sky search. The "transfer" template has no free edge either; its temperature
   at the causal edge is what the transfer functions give (entry (g)).
2. Their text (arXiv:1202.2861, TeX source). Profile [c0 + c1 cos(theta)] s(theta; theta_crit) with
   z0 = c0 + c1 and z_crit = c0 + c1 cos(theta_crit). "We replace the discontinuous Heaviside step function
   of the bubble collision profile with a 'Schwartz' step function s(theta; theta_crit) that is infinitely
   differentiable but nevertheless exhibits a smooth but rapid transition to zero about theta_crit"; "we
   restrict our attention to z_crit ~ 0 uK". Neither the width of the step nor the number behind "~ 0" is
   given. Also: "on larger scales notice that the matched filter contains only the hot and cold rings that
   enhance the collision edge".
3. Their Fig. 1a, the radial profile for z0 = 100 uK, theta_crit = 10 deg (file
   `bubble_profile_thetac010_A100uK_psfrag.pdf` in the arXiv source; a vector curve of 177 points, read with
   the tick marks as calibration; the PDF writer rounded it to steps of 0.10 deg and 0.34 uK). This was looked
   at first and is what started the entry:
   - inside the disc the drawn curve is a straight line in cos(theta) to 0.3 uK rms, from 100 uK at the
     centre to 19.0 +- 0.2 uK at 10 deg (fits up to 6, 7, 8, 8.5, 9, 9.5, 9.9 deg give 18.4 to 19.1 uK);
   - it follows that line to 10.0 to 10.1 deg and then falls smoothly to zero at 10.9 to 11.1 deg; the line
     itself would reach zero at 11.1 deg, which is 10 deg / 0.9;
   - the plain z_crit = 0 cosine cut at 10 deg misses the drawn curve by 9.6 uK rms and 19.8 uK at most
     (5 uK at 5 deg, 12 uK at 8 deg).
   So the template drawn under "z_crit ~ 0" has about 19% of z0 at the causal edge, followed by a smooth
   fall about 1 deg wide. Two readings cannot be told apart at one size: (A) z_crit = 0.19 z0 with a smooth
   step just outside theta_crit; (B) the z_crit = 0 cosine of a disc of radius theta_crit / 0.9, cut
   smoothly between theta_crit and theta_crit / 0.9.

Not known: whether their filters and published numbers were made with the template as drawn or as
described. A figure can be an illustration made with other settings. The tests below ask the published
numbers. The template variants are fixed from Fig. 1a alone:

- `plain`: this project's cosine template (z_crit = 0, hard cut at theta_crit).
- `edge_rel`: straight line in cos(theta) from z0 to 0.19 z0 at theta_crit, smooth step from 1.005 to
  1.110 theta_crit.
- `edge_abs`: the same line, step from theta_crit + 0.05 deg to theta_crit + 1.10 deg.
- `scaled`: reading (B), step from 1.005 theta_crit to theta_crit / 0.9.
- `edge_sharp`, for reference only: the line cut at theta_crit, a plain jump of 0.19 z0.

Step shape: the integral of the bump exp(-1 / (1 - t^2)). At the drawing's resolution other smooth shapes
fit as well (0.55 uK rms each).

Test 1, their Fig. 1c (power spectrum of the 10 deg template, drawn next to the CMB spectrum). Model:
|a_l0|^2 / (2l + 1) for z0 = 100 uK. Expected if the drawn spectrum belongs to the drawn profile: the edge
variants follow it from l = 2 to 256, and `plain` falls below it at high l (a kink gives l^-5, a jump l^-3),
by more than a factor of 3 above l = 100. The red curve of the same figure is compared with the CAMB
spectrum used here, as a check of the background of check 2.

Test 2, their Fig. 4a, unfiltered curve (dashed red; their formula is the template value at the centre over
the rms of the background). At 1 to 4 deg it reads 0.96 to 0.97, which is 100 uK / 103 uK, while the CMB rms
for l up to 256 is 94.3 uK. So the band limit of that figure is in question as well; 256 and 512 are both
computed. Not a blind test: (ag) already has the plain template 23% too high at 60 deg and 45% at 90 deg,
and rough hand estimates made before writing this gave about the right level at 30 to 90 deg for an edge
of 0.2 z0. Counted as reproduced if a variant matches the curve to 3% rms over 2 to 90 deg.

Test 3, their Fig. 4a, matched-filter curve (solid blue). This is the blind one. Now: plain / published =
0.90 at 2 deg, 0.93 at 5 deg, 0.965 at 10 to 55 deg, 1.00 at 90 deg (check 2 of (ag), cause not found).
Rule fixed here: "explained" if one variant at one of the two band limits reproduces the published curve
with an rms deviation below 1.5% over 2 to 90 deg and no size off by more than 3%; "partly" if the rms
deviation is at most half of the plain template's but the bar is missed; otherwise "not explained". Six
combinations are tried (three variants, two band limits); the bar is meant to be too strict to pass by
trying six.

Test 4, only if test 3 gives "explained" or "partly": the WMAP W-band check of (w) rerun with the variant
that did best in place of the cosine template (filters, sigma maps, thresholds, 1,000 simulated skies), with
the criteria of (w) unchanged: (a) S within 0.5 of the published value for at least 12 of the 16 published
candidates; (b) at least 7 of the 11 eligible ones found as candidates. Also reported: S and z0 at their
nos. 14 and 15 (85 and 90 deg), and the detection fraction at 50 deg and 100 uK at their two positions (the
cell that failed check 3 of (ag)). Expectation: if their filters carry the edge, (a) and (b) are met and
nos. 14 and 15 appear. If they are not met, the drawn template is not what their search used, or something
else differs as well.

Cannot change, whatever comes out: the Planck results of (x) and the sensitivity numbers of (ag). They are
statements about this project's two templates and stay as measured. Can change: the sentence that the
cosine template is "their template", the reading of checks 2 and 3 of (ag), and the verdict of (x) on the
WMAP repeat. No change is made to the Planck search on the strength of these tests without William's word.

Also computed, as a description without a pass mark: the response of the plain filters to a signal with an
edge z_crit = r z0, relative to r = 0, for the Planck set-up (what an edge in a real signal would do to the
sensitivity numbers of (ag)).

## (ak) 2026-11-22 16:55 CST. z_crit behind the scenes: the published filter search runs on a template with 0.2 z0 at the edge

Answer to William's question of (aj), in one paragraph. In this project's code there is no hidden z_crit: the
cosine template is zero at the edge, checked from the formula and from its own coefficients. In the
published optimal-filter search of McEwen et al. (arXiv:1202.2861) there is one: the text says
"z_crit ~ 0 uK", the authors' public code sets z_crit = 0.2 z0 with a smooth cut-off out to 1.1 times the
size, and their figures and their candidate table follow the code. With that template every difference
between this project's numbers and theirs that was listed as unexplained in (x), (ag) and (ai) goes away or
shrinks to a few tenths in S. Nothing in the Planck results changes.

Figure: `plots/11_mcewen_template_edge.png`. Scripts: `scripts/analysis/check_mcewen_template.py` (tests 1 to
3), `fullsky_filter_search.py --cosine-edge` (test 4), `mcewen_template_diagnostics.py`,
`plot_mcewen_template.py`; the two figures are read by `read_published_figure_curves.py mcewen-profile` and
`mcewen-spectra`. Reports in `work/results_2026-11-22/`: `check_mcewen_template_report.json`,
`fullsky_search_summary_wmap7w_v4.json` (z_crit = 0), `_v5.json` (the fit of (aj)), `_v6.json` (their code),
`fullsky_inject_report_wmap7w_v5_mcewen.json`, `_v6_mcewen.json`,
`compare_mcewen_sensitivity_report_wmap7w_v5.json`, `_v6.json`, `mcewen_template_masked_positions.json`,
`mcewen_template_edge_response_planck.json`. Jobs 82940 to 82943.

### Their code (found after (aj) was written and after tests 1 to 3 had been run with the fitted template)

Public repositories of the first author's group: `github.com/astro-informatics/comb` (commit 3bd9996 of
2015-07-29) and `github.com/astro-informatics/s2fil` (commit aa545b9 of 2015-07-30).

- `comb/src/mod/comb_tmplmap_mod.f90`, function `comb_tmplmap_bubble` ("Written by Jason McEwen", November
  2011). Defaults: `z0 = 1e0`, `zc = 0.2e0 * z0`, `theta_c = 20e0 / 180e0 * PI`, `theta_0 = 1.1e0 * theta_c`.
  Value: `c1 = (z0 - zc) / (1 - cos(theta_c))`, `c0 = z0 - c1`, `val = c0 + c1 * cos(theta)`, times k with
  k = 1 inside theta_c, 0 outside theta_0 and `exp(-1/(1 - ts**2)) * exp(1)` in between,
  `ts = (theta - theta_c) / (theta_0 - theta_c)`. This is the "Schwartz step" of the paper.
- `comb/src/prog/comb_objgen.f90`, which draws bubbles of a given size: `bubble_params(1) = 1.0`,
  `bubble_params(2) = 0.2`, `bubble_params(3) = ` size, `bubble_params(4) = bubble_params(3) * 1.1`.
- The filters are built by `s2fil_filter_construct -tmpl bubble -tmpl_param FILE`. The only parameter file in
  the repository (`tmplparam01.txt`: 1.0, -1.0, 10.0, 10.0, a template from +1 to -1 with no cut-off) is not
  the paper's setting, so the files of the published run are not public. What that run used is read from its
  output in tests 3 and 4.
- Also in the repository: their background spectrum, `data_in/wmap_lcdm_pl_model_wmap7baoh0_CAMB.dat`. It
  agrees with the CAMB spectrum used here to 1.3% (ratio 1.011 at l = 2, 0.995 at 100, 1.004 at 256), so the
  background of check 2 of (ag) was right.

Three templates from here on: "text" (z_crit = 0, cut at the edge; this project's cosine template), "fit"
(0.19 z0 at the edge, cut-off from 1.005 to 1.110 of the size; the variant `edge_rel` fixed in (aj)), "code"
(the defaults above).

### This project's side

`cosine_coefficients` is zero at the edge by construction. Checked numerically: its Legendre series (to l = 4000) gives
0.0394 at 0.98 of the size for 10 deg (the exact form: 0.0395), 0.0009 at the edge and 0.0001 or less
outside, and a fit of its coefficients as a * top-hat + b * edge-term gives a = 1.000000, b = -1.000000 at
every size, which is z_crit = 0. The transfer template has no free edge; its edge value is what the transfer
functions give (0.5 of the centre at 10 deg, 0.2 at 20 deg, under 0.1 above 40 deg, entry (g)).

### Test 1: the spectrum they drew for the 10 deg template (Fig. 1c)

The model stated in (aj), |a_l0|^2 / (2l + 1), was wrong. The drawn quantity is |a_l0|^2: the ratio of the
drawn curve to |a_l0|^2 / (2l + 1) of the code template is 5.1, 7.1, 9.1, 11.8, 14.1 at l = 2 to 6 and 41.7
at l = 20, which is 2l + 1 (5, 7, 9, 11, 13, 41); for the text template it is 7.7 to 28.9 and follows no
such law. With |a_l0|^2:

| template | log10(drawn / model), l 2 to 20 | l 21 to 60 | largest value in band, drawn / model: l 61-100, 101-160, 161-256 |
|---|---|---|---|
| text | +0.090 (16 to 84%: -0.03 to +0.19) | -0.20 (-0.98 to +0.71) | 1.40, 2.26, 1.53 |
| fit | +0.013 (+0.008 to +0.028) | +0.003 (-0.12 to +0.07) | 1.02, 0.97, 0.66 |
| code | +0.010 (+0.006 to +0.024) | +0.019 (-0.001 to +0.031) | 0.97, 0.92, 0.63 |

The code template follows the drawn spectrum to 2 to 5% in power up to l = 60, zeros included. The text
template has its zeros at other multipoles (at l = 25, 28, 40 the drawn curve reads 0.065, 0.19, 0.096 uK^2;
text gives 0.83, 0.035, 0.33; code 0.063, 0.19, 0.096). The expectation of (aj) that the text template would
fall short by more than a factor of 3 above l = 100 was wrong: the smooth cut-off removes most of the l^-3
tail of the edge, the shortfall is 1.5 to 2.3. Above l = 160 every template is off by a third or more. Their CMB curve
in the same figure over C_l: median 1.04, range 0.99 to 1.09 (a dashed line of 82 points).

### Test 2: unfiltered curve of Fig. 4a

Their curve at 1 to 4 deg reads 0.963 = 100 uK / 103.8 uK. The rms of the CMB is 94.3 uK for l up to 256
(100 / sigma = 1.060), 101.8 uK up to 384 (0.982), 104.5 uK up to 512 (0.957). With l up to 512 the fit
template reproduces the curve to 0.8% rms over 2 to 90 deg (mean 0.999), the code template to 1.5% (mean
0.989; the last point, 90 deg, is 6% low where the curve itself is at 0.15 and is read in steps of 0.01).
The text template is 24% off (8% at 45 deg, 17% at 60 deg, 63% at 90 deg). With l up to 256 everything is
10% too high. Reproduced by the rule of (aj) (3%). So Fig. 4a was made with a band limit near 512, not the
256 of their search.

### Test 3: matched-filter curve of Fig. 4a (blind for the fit template)

Computed / published, CAMB spectrum unless said otherwise:

| template, band limit | 2 | 3 | 5 | 7 | 10 | 14 | 20 | 30 | 45 | 60 | 75 | 90 deg | rms dev | largest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text, 256 (check 2 of (ag)) | 0.904 | 0.919 | 0.930 | 0.946 | 0.964 | 0.967 | 0.967 | 0.965 | 0.966 | 0.973 | 0.987 | 0.996 | 3.4% | 9.6% |
| text, 512 | 0.935 | 0.933 | 0.936 | 0.950 | 0.967 | 0.969 | 0.968 | 0.966 | 0.966 | 0.974 | 0.988 | 0.997 | 3.2% | 6.8% |
| fit, 256 | 0.954 | 0.960 | 0.965 | 0.979 | 0.995 | 0.997 | 0.998 | 0.999 | 1.001 | 1.000 | 0.996 | 0.989 | 1.1% | 4.6% |
| fit, 512 | 0.991 | 0.984 | 0.981 | 0.990 | 1.000 | 0.999 | 0.998 | 0.999 | 1.001 | 1.000 | 0.996 | 0.989 | 0.6% | 2.0% |
| code, 512 | 0.995 | 0.988 | 0.984 | 0.992 | 1.002 | 1.001 | 1.001 | 1.002 | 1.005 | 1.005 | 1.004 | 1.004 | 0.5% | 1.7% |
| code, 512, their spectrum file | 0.996 | 0.988 | 0.983 | 0.991 | 1.000 | 0.998 | 0.997 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 0.4% | 1.7% |

By the rule of (aj) (rms below 1.5%, no size off by more than 3%): "explained" by the fit template at a
band limit of 512. The other two variants of (aj) fail (`edge_abs` 43 to 48% rms, `scaled` 3.3% with 0.914
at 90 deg): the edge is a fixed fraction of z0 and the cut-off grows with the size, which is what the code
does. The last row is not blind (made after reading the code) and has no free number in it: their template,
their spectrum file. It is the published curve to the accuracy with which the figure can be read from 10 deg
up. The 3 to 10% of check 2 of (ag) is the difference between z_crit = 0.2 z0 and z_crit = 0.

Fig. 1a against the code template: 0.71 uK rms, 2.5 uK at most (fit 0.40 and 1.70; text 9.60 and 19.84).
The value at the edge read from the drawing, 18.4 to 19.1 uK, against the code's 20: the drawing is rounded
to 0.10 deg and the slope there is 20 uK per degree, so 1 uK is the rounding. The "+- 0.2" of (aj) was the
scatter between fit ranges only and too small.

### Test 4: the WMAP W-band check of (w) with their template

Same data, mask, band limit, thresholds, 3,000 + 1,000 simulated skies as run v4 of (ai); only the cosine
template differs. v5 is the fit template as fixed in (aj), v6 the code template (added after the code was
found). Foreground-reduced map:

| | v4, text | v5, fit | v6, code | needed |
|---|---:|---:|---:|---:|
| (a) S within 0.5 of the published S, of 16 | 10 | 14 | 14 | 12 |
| (b) eligible candidates found, of 11 | 6 | 9 | 9 | 7 |
| largest difference in z0 over the 16, mK | 0.16 | 0.02 | 0.02 | |
| S within 0.5 at the published position itself, masked pixels included, of 16 | 5 | 13 | 13 | |

Both criteria of (w) are met with their template; on the plain band map the same (14 and 9). Run v6
against their table, S at the position (published) and z0 in mK (published):

| no. | size | S | z0 | | no. | size | S | z0 |
|---:|---:|---|---|---|---:|---:|---|---|
| 0 | 1.5 | 4.35 (4.25) | +0.235 (+0.24) | | 8 | 5 | 4.32 (4.43) | +0.153 (+0.15) |
| 1 | 2 | 3.68 (4.15) | +0.181 (+0.20) | | 9 | 7 | 4.36 (4.39) | +0.137 (+0.14) |
| 2 | 2 | 3.50 (4.08) | +0.195 (+0.20) | | 10 | 20 | 4.09 (3.94) | +0.106 (+0.11) |
| 3 | 2 | 3.97 (4.05) | -0.190 (-0.19) | | 11 | 26 | 3.30 (3.07) | -0.088 (-0.09) |
| 4 | 3 | 4.21 (4.26) | -0.166 (-0.17) | | 12 | 35 | 3.65 (3.33) | +0.089 (+0.09) |
| 5 | 3 | 3.47 (4.02) | +0.158 (+0.17) | | 13 | 40 | 2.93 (3.07) | +0.085 (+0.10) |
| 6 | 4 | 4.75 (4.56) | -0.154 (-0.16) | | 14 | 85 | 3.62 (3.03) | +0.153 (+0.15) |
| 7 | 5 | 4.47 (4.67) | -0.157 (-0.16) | | 15 | 90 | 3.24 (3.11) | -0.146 (-0.16) |

(Nos. 11 and 15 are centred on masked pixels; their S here is taken with masked pixels included.)

- Their two largest candidates, which the text template does not see (S 0.3 and 0.6 at the positions, z0
  -0.01 and -0.02 mK), are there: no. 14 with S 3.6 and z0 +0.153 mK (published 3.03, +0.15), a candidate
  of this search too; no. 15 with S 3.2 and z0 -0.146 mK (published 3.11, -0.16). No. 15 sits at
  b = +1.5 deg, inside the mask, and so does no. 11. Their search therefore did not leave masked pixels
  out; this one does (a stated choice of (w)), which is why no. 15 cannot be a candidate here. The z0
  values of nos. 10 to 13, 17 to 58% low with the text template, agree to 0.015 mK.
- What is left: at 2 to 3 deg three candidates (nos. 1, 2, 5) come out 0.5 to 0.6 lower in S at the
  published position (0.2 to 0.3 lower at the strongest pixel nearby), and nos. 12 and 14 come out 0.3 and
  0.6 higher. Not followed up. Unknown on their side: the
  parameter files, the pixel resolution (their scripts use Nside 128), the noise model in the filters.
- With their template the foreground-reduced map gives 40 candidates where simulated skies give 82 on
  average (p = 0.88), largest S 4.84 at 9 deg (p = 0.62). Ordinary, as with the text template.
- Check 3 of (ag), the published exclusion and sensitivity regions: with their template all 8 boundary
  comparisons from 5 to 60 deg are inside one grid step (before 7 of 8). The 50 deg exclusion boundary
  moves from 154 to 145 uK (allowed: 150). Their six detections out of six at 50 deg and 100 uK still have
  a probability of only 5% under the detection rate measured here (0.62 and 0.60 at the two positions;
  before 0.57 and 0.52, 2.7%). That cell stays the least likely one.

### What an edge does to the sensitivity numbers of (ag)

The numbers of (ag) are for the two templates of this project and stand. For a collision that did carry an
edge, at the same z0, on the Planck set-up (`mcewen_template_diagnostics.py edge-response`):

| size, deg | 2 | 5 | 10 | 20 | 30 | 50 | 60 | 75 | 90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| edge of their code: S in the z_crit = 0 filters / S of the no-edge signal | 1.03 | 0.98 | 0.93 | 0.88 | 0.84 | 0.77 | 0.73 | 0.63 | 0.48 |
| the same in the transfer filters | 1.24 | 1.26 | 1.31 | 1.34 | 1.13 | 0.89 | 0.81 | 0.69 | 0.51 |
| share of the best possible S reached by the z_crit = 0 filters | 0.97 | 0.94 | 0.90 | 0.85 | 0.81 | 0.75 | 0.71 | 0.63 | 0.48 |
| plain jump of 0.5 z0: share of the best possible S | 0.90 | 0.80 | 0.68 | 0.53 | 0.43 | 0.31 | 0.26 | 0.18 | 0.10 |
| plain jump of 0.5 z0: its own filters / the no-edge signal in its filters | 1.24 | 1.34 | 1.52 | 1.88 | 2.24 | 2.93 | 3.32 | 4.04 | 5.07 |

So z0 limits from the z_crit = 0 search do not carry over to signals with an edge: for the 20% smoothed
edge they would be up to a third weaker below 60 deg and a factor of two at 90 deg, and a sharp edge is a
different search altogether (its own filters gain a factor 1.2 to 5). The smoothed 20% edge itself adds only
3 to 6% to the best possible S at a given z0 up to 60 deg (nothing at 90 deg), which is the size of the
offset in Fig. 4a.

### Corrections to earlier entries

- (ag), check 2 ("anything else", cause not found): the cause is the template. Their curve is for
  z_crit = 0.2 z0 and a band limit near 512.
- (ag), check 3 (not passed at 50 deg): passes with their template; the 50 deg cell is still unlikely (5%).
- (x), (ag), (ai): "the WMAP check misses both criteria, not a validated repeat". True for the text
  template. With the template of their code the check meets both criteria. The runs with z_crit = 0 remain
  what they are: the search as their text describes it.
- (ag), diagnostics on nos. 14 and 15 ("still unexplained"): explained by the template; no. 15 also needs
  masked pixels in the search.
- README, the docstring of `fullsky_filter_search.py` and a panel title of
  `plots/10_fullsky_search_sensitivity.png` called the z_crit = 0 cosine "their template" or "the template
  and filters of the published searches". Changed to "as the published papers describe it"; the figure is
  redrawn from the same numbers.
- (aj): the "+- 0.2 uK" and the spectrum model, both above.

### What is and is not claimed

Claimed: the paper's text gives z_crit ~ 0; the authors' public code has 0.2 z0 as the default and in the
object generator; Figs. 1a, 1c and 4a and the candidate table are reproduced with 0.2 z0 and not with 0.
Not claimed: that this is a mistake. "~ 0" may have been meant loosely, or a default may have stayed in;
only the authors can say, and the parameter files of the published run are not public. Not checked: whether
later work that started from these candidates (Feeney et al. 2013, which removes the edge parameter from its
model) used the same filters.

For this project: no result changes. The cosine template of the Planck search is the z_crit = 0 one, the
transfer template is the physical one, and neither has a free edge. A repeat of their search on Planck with
their actual template would be one more run (`--cosine-edge 0.2 --cosine-step 1.0,1.1,schwartz`); it has
not been run and waits for William's word, like the other open steps.

## (al) 2026-11-22 17:26 CST. The Planck search with the template of their code: what is run and how it is read (written before the run)

William, 22 Nov: "run the planck search with their real template".

Run: `fullsky_filter_search.py --dataset planck --cosine-edge 0.2 --cosine-step 1.0,1.1,schwartz`, all
stages, run `planck_v3`; the same on `planck767` (l up to 767), run `planck767_v3`; then `--stage inject` on
`planck_v3`. Everything else as in the runs v2 of (ai): PR3 SMICA, NILC, SEVEM and Commander at Nside 256,
common mask (77% of the sky), 5 arcmin beam, thresholds of Feeney et al. (2013), step 4 in z0, 3,000
simulated skies for the sigma maps and 1,000 for the null statistics. The seeds are the same, so the
simulated skies are the same realisations as before and only the cosine filters differ.

Why: a like-for-like repeat of the published search on Planck. Not because the template is believed. A
temperature of 0.2 z0 at the edge is not supported by theory after 2011 (entry (b)).

Built-in check: the transfer template is not touched, so its half of the run must reproduce `planck_v2`
exactly: 40, 39, 34, 38 candidates (SMICA, NILC, SEVEM, Commander) and the same lists. If it does not, the
run is not used until that is understood.

Read-out, fixed here. For each of the four maps, cosine template:

1. the number of candidates and its p among the 1,000 simulated skies (share of simulated skies with at
   least as many);
2. the largest S over all sizes and its p (against the largest S of each simulated sky);
3. the number of candidates of 45 deg or more and its p (counted the same way in the simulated skies, from
   their stored candidate lists), and the largest S among those sizes with its p. With z_crit = 0 no map
   had a candidate of 45 deg or more;
4. the 16 WMAP candidates of McEwen et al. on Planck: S and z0 at their positions and sizes, masked pixels
   included as in (ak), and how many are candidates of this run (with z_crit = 0: 8 of 16);
5. the candidates common to the four maps, and the candidates gained and lost against `planck_v2`.

Verdict rule: "no excess" if on every map the p of (1), (2) and both parts of (3) are 0.05 or more. A p
below 0.05 on any of them is reported as "not ordinary, to be understood", not as a detection. These are
sixteen looks at four maps of the same sky; they are far from independent, and for independent looks one
below 0.05 would be expected by chance about half the time.

Expectations, before the run:

- The edge puts weight on the rim of the disc, so more regions pass the thresholds everywhere. On WMAP the
  simulated skies went from 51 to 82 candidates per sky with this template. Expected here: simulated skies
  from 52.5 to about 80 to 90 per sky, the real maps from 31 to 34 to about 40 to 60, p of the count
  between 0.5 and 0.95.
- Largest S near 4.8 to 5.0, most likely the 9 to 10 deg feature at (l, b) about (213, -59) that led the
  WMAP run with this template (S 4.84) and is the 10 deg candidate of the z_crit = 0 Planck run. p between
  0.3 and 0.8.
- Candidates of 45 deg or more now appear on the real maps: their no. 14 (85 deg) has S 3.6 on WMAP with
  this template and the threshold there is 3.0. Expected: one to four per map, and the same order in the
  simulated skies. If a small p turns up anywhere it is most likely here, because the thresholds are
  lowest for the largest sizes (3.0 to 3.25); it is judged against the simulated skies only.
- Of their 16 WMAP candidates about 10 or 11 are candidates here (WMAP run v6: 10 on the
  foreground-reduced map), with S within about 0.3 of the WMAP values of (ak): the same sky with less noise
  and a smaller mask.
- Sensitivity (the cosine / cosine pair of the inject stage now means their template in their filters): in
  z0 the 95% "stands out" level stays within 10% of the z_crit = 0 numbers of (ag) at every size, because
  the smoothed edge adds only 6% to the best possible S at 2 deg and nothing at 90 deg ((ak)).
- With l up to 767 the picture is the same as with 256, with more candidates on small sizes, as for
  z_crit = 0.

Cannot change: the transfer-template results, and the z_crit = 0 results of (x), which stay the search as
their text describes it. Whatever comes out, the README sentence "has not been run with the template of
their code" is replaced by the result.

## (am) 2026-11-22 18:14 CST. The Planck search with the template of their code: no excess, and their candidates are on Planck too

The runs of (al): jobs 82944 (`planck_v3`, l up to 256), 82945 (`planck767_v3`), 82946 (inject on
`planck_v3`). Reports in `work/results_2026-11-22/`: `fullsky_search_summary_planck_v3.json`,
`fullsky_search_summary_planck767_v3.json`, `fullsky_planck_code_template_comparison.json`
(`fullsky_compare_runs.py`, v2 against v3), `fullsky_large_sizes_report.json`
(`mcewen_template_diagnostics.py large-sizes`), `mcewen_template_masked_positions_planck_*.json`,
`fullsky_inject_report_planck_v3.json`, `fullsky_inject_summary_planck_v3.md` and `.json`.

Built-in check, passed: the transfer half of both runs is the same as in the v2 runs, entry for entry (40,
39, 34, 38 candidates at l up to 256 and 72, 69, 70, 71 at l up to 767; p and largest S to every digit;
simulated skies 98.975 and 265.036 per sky in both). The transfer / transfer levels of the inject stage are
identical as well.

### Read-out of (al), cosine template = the template of their code (0.2 z0 at the edge)

l up to 256, with the z_crit = 0 run (v2) below it:

| | SMICA | NILC | SEVEM | Commander | 1,000 simulated skies |
|---|---:|---:|---:|---:|---|
| 1. candidates (p) | 54 (0.750) | 52 (0.773) | 50 (0.798) | 52 (0.773) | mean 84.7, median 73, 5 to 95%: 33 to 171 |
| 2. largest S (p) | 4.72 (0.853) | 4.70 (0.871) | 4.74 (0.823) | 4.72 (0.849) | |
| 3. candidates of 45 deg or more (p) | 5 (0.931) | 5 (0.931) | 4 (0.943) | 4 (0.943) | mean 34.0, median 22, 5 to 95%: 3 to 102, none in 0.5% |
| 3. largest S at 45 deg or more (p) | 3.61 (0.810) | 3.62 (0.793) | 3.59 (0.821) | 3.60 (0.815) | 5, 50, 95%: 3.38, 3.94, 4.66 |
| z_crit = 0: candidates (p) | 33 (0.769) | 34 (0.755) | 32 (0.794) | 31 (0.814) | mean 52.5, 5 to 95%: 23 to 106 |
| z_crit = 0: largest S (p) | 4.80 (0.73) | 4.78 (0.76) | 4.79 (0.75) | 4.78 (0.76) | |
| z_crit = 0: 45 deg or more | 0 | 0 | 0 | 0 | mean 16.4, median 8, none in 12% |

l up to 767: 100, 94, 96, 94 candidates (p 0.810, 0.853, 0.836, 0.853; simulated skies mean 144.0, median
133, 5 to 95%: 76 to 240); largest S 5.11, 5.16, 5.12, 5.12 at 1.5 deg (p 0.467, 0.398, 0.451, 0.463); 5, 5,
4, 4 candidates of 45 deg or more (p 0.933, 0.933, 0.946, 0.946), largest S there 3.60 to 3.63 (p 0.79 to
0.82). For z_crit = 0 these were 94, 104, 96, 95 candidates (p 0.65 to 0.73) and largest S 5.21 to 5.23
(p 0.23 to 0.26).

Verdict by the rule of (al): no excess. The smallest of the thirty-two p values is 0.398; at l up to 256
they lie between 0.75 and 0.94.

4. Their 16 WMAP candidates on Planck. Candidates of this run: 13 of 16 at l up to 256 (not nos. 1, 3, 13),
14 of 16 at l up to 767 (not 3, 13), all 11 eligible ones, on each of the four maps; with z_crit = 0 it was
8 of 16. S at their positions within 0.5 of the published WMAP value for 13, 14, 13, 13 of 16 (masked
pixels included; with z_crit = 0: 6). SMICA, S at the position (published) and z0 in mK (published):

| no. | size | S | z0 | | no. | size | S | z0 |
|---:|---:|---|---|---|---:|---:|---|---|
| 0 | 1.5 | 4.38 (4.25) | +0.230 (+0.24) | | 8 | 5 | 4.36 (4.43) | +0.155 (+0.15) |
| 1 | 2 | 3.62 (4.15) | +0.175 (+0.20) | | 9 | 7 | 4.27 (4.39) | +0.137 (+0.14) |
| 2 | 2 | 2.60 (4.08) | +0.122 (+0.20) | | 10 | 20 | 4.07 (3.94) | +0.110 (+0.11) |
| 3 | 2 | 3.98 (4.05) | -0.184 (-0.19) | | 11 | 26 | 3.05 (3.07) | -0.085 (-0.09) |
| 4 | 3 | 4.34 (4.26) | -0.172 (-0.17) | | 12 | 35 | 3.51 (3.33) | +0.089 (+0.09) |
| 5 | 3 | 4.06 (4.02) | +0.153 (+0.17) | | 13 | 40 | 2.69 (3.07) | +0.081 (+0.10) |
| 6 | 4 | 4.72 (4.56) | -0.170 (-0.16) | | 14 | 85 | 3.57 (3.03) | +0.156 (+0.15) |
| 7 | 5 | 4.55 (4.67) | -0.158 (-0.16) | | 15 | 90 | 3.25 (3.11) | -0.147 (-0.16) |

Against the WMAP run v6 of (ak), same template: S within 0.25 for 14 of 16. The two exceptions are small
discs: no. 2 (2 deg) has S 2.60 and z0 +0.122 mK on Planck against 3.50 and +0.195 on WMAP (published 4.08,
+0.20), so a good part of it was noise of the W-band map; no. 5 (3 deg) has 4.06 on Planck against 3.47 on
WMAP (published 4.02).

5. Common to the four maps: 48 of the 54 SMICA candidates (l up to 256), 94 of 100 (l up to 767). The
strongest: 4 deg at (l, b) = (268.1, -77.2), z0 -0.170 mK, S 4.69 to 4.72; 9 deg at (213.7, -59.1), z0
-0.144 mK, S 4.70 to 4.74; 5 deg at (207.0, -57.6) and at (264.6, -19.5), S 4.6. Gained against z_crit = 0
on SMICA (33 to 54): 9 below 18 deg, 7 from 18 to 44 deg, and the 5 of 45 deg or more (z_crit = 0 had none
above 40 deg).

The candidates of 45 deg or more, the same on the four maps unless noted:

- 85 deg at (61.7, +20.4), S 3.59 to 3.62, z0 +0.157 mK. Their no. 14 (62.6, +20.4; +0.15 mK).
- 90 deg at (279.5, +4.6), S 3.35 to 3.38, z0 -0.152 mK. Their no. 15 (277.7, +1.5; -0.16 mK), 3.6 deg
  away at the nearest unmasked maximum; the Planck mask leaves pixels free there that the WMAP mask covers.
- 85 deg at (349.5, +81.4), S 3.45 to 3.54, z0 +0.064 mK.
- 65 deg at (178.6, -72.8), S 3.25 to 3.28, z0 -0.098 mK (not on Commander).
- 85 deg at (183.8, -79.0), S 3.0, z0 -0.094 mK (SMICA and NILC); 80 deg at (133.4, -79.9), S 3.0, z0
  +0.049 mK (Commander).

### What the read-out says

- The like-for-like repeat of the published search on Planck gives nothing beyond what Gaussian skies give.
  Their candidates are there, both large ones included, with the same amplitudes; they are features of the
  sky and not of the WMAP map (except part of no. 2). They are not unusual: a simulated sky run through the
  same search has 85 candidates on average where the real maps have 50 to 54, and 22 (median) of 45 deg
  or more where the real maps have 4 or 5.
- The real maps are on the low side at the largest sizes, with either template: 93 to 95% of simulated
  skies have at least as many candidates of 45 deg or more as the real maps (their template); with
  z_crit = 0 the real maps have none where 88% of simulated skies have at least one, and their largest S at
  those sizes is below that of 96% of simulated skies. In direction this is the known low power of the real
  sky at the largest angles; the simulated skies are drawn from the best-fit spectrum. The rule of (al) was
  one-sided, so this is recorded and not tested.
- Candidate counts are only comparable with simulated skies that went through the same search. McEwen et
  al. list 16 candidates on WMAP; this search, with their template and the threshold table of Feeney et al.
  (2013), gives 40 on WMAP and 50 to 54 on Planck. Their thresholds of 2012 are not in the paper as a table.

### Sensitivity with their template (inject stage, 100 skies x 150 positions; their template in their filters)

The level a signal-free sky exceeds in 5% of cases is S = 5.453 (z_crit = 0: 5.457); the real maps' largest
S is 4.70 to 4.74. Amplitude at which 95% of collisions would be the strongest feature of a whole sky, as
z0 in units of 1e-5:

| size, deg | 2 | 3 | 5 | 7 | 10 | 14 | 20 | 30 | 40 | 50 | 60 | 75 | 90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| their template | 14.4 | 11.3 | 9.5 | 8.9 | 8.2 | 8.2 | 8.4 | 8.9 | 8.9 | 9.3 | 10.1 | 12.9 | 16.3 |
| z_crit = 0, (ag) | 15.1 | 13.2 | 9.8 | 9.3 | 8.8 | 8.3 | 8.4 | 8.9 | 9.5 | 9.5 | 10.1 | 12.0 | 15.5 |
| half reach the threshold, their template | 6.6 | 5.3 | 4.3 | 3.9 | 3.6 | 3.4 | 3.1 | 2.8 | 2.8 | 2.9 | 3.1 | 3.2 | 3.7 |

In z0 the search with their template sees what the z_crit = 0 search sees: 8e-5 to 10e-5 (220 to 280 uK)
from 5 to 60 deg. Chance rate of reaching the threshold with no signal: under 1% up to 20 deg, 2 to 4% from
30 to 60 deg, 8% at 75 deg and 12.5% at 90 deg (z_crit = 0: 4.4% at 90 deg), so at the largest sizes
"reaches the threshold" means little with this template and only "stands out" (chance rate 0) is used.

### Expectations of (al) against the outcome

- Simulated skies 80 to 90 per sky, real maps 40 to 60, p 0.5 to 0.95: yes (84.7; 50 to 54; 0.75 to 0.80).
- Largest S 4.8 to 5.0 with p 0.3 to 0.8, at the 9 to 10 deg feature: partly. It is 4.70 to 4.74 with p
  0.82 to 0.87; the 9 deg feature leads on three maps, the 4 deg one at (268.1, -77.2) by 0.02 on SMICA.
- One to four candidates of 45 deg or more per map, "the same order in the simulated skies": four or five
  per map, and the second half was wrong. Simulated skies have a median of 22. I had not looked at that
  number before writing (al); the WMAP run v6 already has it (2 on the real map, median 23 simulated).
- Ten or eleven of their 16 as candidates, S within 0.3 of WMAP: 13, and within 0.25 for 14 of 16.
- Sensitivity in z0 within 10% of (ag) at every size: at 12 of 13 sizes (0.94 to 1.07); at 3 deg it is 15%
  lower, between two steps of the amplitude grid.
- l up to 767 the same picture: yes.

### Limits

As for (x): Gaussian simulated skies without foreground residuals, masked pixels left out of the search,
the Planck end-to-end simulations not used. The template of this run is the one of their code, which no
theory paper after 2011 supports; the run is a repeat of the published search, not a search this project
argues for. The z_crit = 0 and transfer results of (x) and (ag) stand unchanged.

## (an) 2026-11-23 14:37 CST. Correction to (al) and (am): theory does not rule out 0.2 at the edge, it rules out one fixed value at every size

Pointed out by William, 23 Nov. Two sentences of mine are wrong as written:

- (al): "A temperature of 0.2 z0 at the edge is not supported by theory after 2011 (entry (b))."
- (am), Limits: "The template of this run is the one of their code, which no theory paper after 2011 supports".

The same thought was in the README ("No theory paper after 2011 supports a temperature jump at the edge, so
this project keeps z_crit = 0 ...") and in the docstring of `fullsky_filter_search.py` ("not a physical
model: no theory paper after 2011 supports a temperature jump at the edge").

What this notebook itself measured, entries (f) and (g): the curvature ramp through the transfer functions
has a temperature at the causal edge that is not zero and has the sign of the centre. Edge over centre is
0.53 at 10 deg (1.54 / 2.89), 0.48 at 11.39 deg, 0.21 at 20 deg (0.288 / 1.380), and under 0.1 above 40 deg.
It falls smoothly to zero about one sound horizon outside the causal edge: the boundary is near 13 deg for a
10 deg disc and near 21.7 deg for a 20 deg disc. The theory papers after 2011 say the same in words: no
discontinuity, a smooth transition (Kleban, Levi and Sigurdson 2011; Gobbetti and Kleban 2012; Wainwright
et al. 2014; entry (b)).

So the correct statement has two parts. Theory after 2011 does not support (1) a sharp jump in temperature
at the edge, which is what the April generator injected and what (b) is about, and (2) one fixed edge value
at every size. It does support an edge value of about 0.2 of the centre for a disc of about 20 deg, with a
smooth fall just outside. The template of the published code is not a sharp jump either: it has 0.2 z0 at
the edge and a smooth cut-off out to 1.1 times the size, for every size. For a disc near 20 deg that is
close to what the ramp model gives ((f): boundary 21.7 deg, their cut-off 22 deg). At 10 deg the ramp model
has 0.5, at 40 deg and above under 0.1. How close the two templates are at each size is measured next
((ao), written before the measurement).

What changes: the wording only. (al) and (am) stand with this correction; no number in them depends on the
sentence. The README and the docstring are reworded today. The reason this project's cosine template keeps
z_crit = 0 is that it is the template as the published papers describe it; the size-dependent physical
edge is carried by the transfer template. That reason was right and stays.

## (ao) 2026-11-23 14:37 CST. Pre-registration: are the defaults of their code the transfer-function ramp of a 20 deg disc?

William's hypothesis, 23 Nov. The defaults of `comb_tmplmap_bubble` are zc = 0.2 z0, cut-off out to
1.1 theta_c, theta_c = 20 deg. Entries (f) and (g) give, for the transfer-function ramp of a 20 deg disc,
edge over centre 0.288 / 1.380 = 0.21 and a boundary near 21.7 deg; their cut-off ends at 22 deg. Either
the defaults were tuned to a 20 deg disc or round numbers happen to agree. A measurement only: nothing in
the Planck search changes. Nothing below has been computed when this is written.

### Looked up before writing this (reading, no computation)

History of the authors' public repositories (`gh api`):

- `comb_tmplmap_bubble` first appears in COMB commit 01a2d4f of 2011-11-10 ("Added functionality to
  simulate bubble templates"), with today's defaults already in it (zc = 0.2 z0, theta_c = 20 deg,
  theta_0 = 1.1 theta_c). The function has not changed since.
- The same commit adds five parameter files (z0, zc, theta_c, theta_0): (1, -1, 10, 10), (1, -1, 10, 12),
  (1, 0.2, 10, 12), (1, 0.2, 20, 22), (1, 0.2, 15, 17). So on the first day 0.2 was already used at 10, 15
  and 20 deg, each with a cut-off 2 deg wide; the fourth file is the function's default. The first two are
  the example of Feeney et al. (2011): z0 = 5e-5, z_crit = -5e-5, 10 deg.
- Commit 84d9cbf of 2011-12-06 ("Updated bubble input parameters") sets all five to zc = 0.2 and
  theta_0 = 1.1 theta_c: (10, 11) three times, (20, 22), (15, 16.5); 029fd48 of 2011-12-12 moves the fifth
  to (13, 14.3). So the cut-off went from 2 deg wide to 10% of the size within a month.
- Commit 9a31757 of 2011-12-12 puts 1.0, 0.2, size, 1.1 * size into the object generator; its message
  calls these "(restricted) bubble templates".
- S2FIL's `tmplparam01.txt` (1, -1, 10, 10) was added on 2011-11-10 as an "example template parameter file"
  and never changed. It is the first-day example, not a setting of the published run.
- No comment and no commit message gives a reason for 0.2 or for 1.1.

Papers (TeX sources from arXiv):

- Feeney et al. 2011 (arXiv:1012.3667 and 1012.1995): z0 and z_crit are independent, each with a uniform
  prior from -1e-4 to 1e-4, "Lacking a detailed theoretical prediction for the amplitude parameters". The
  discontinuity "arises from the ISW and Doppler contributions"; neglected effects "could smear out the
  causal boundary enclosing the collision on sub-degree scales". No 0.2 and no 1.1 anywhere.
- McEwen et al. 2012 (arXiv:1202.2861): the strings 0.2, 1.1, 20% and 10% do not occur. "we restrict our
  attention to z_crit ~ 0 uK"; the step is "a smooth but rapid transition to zero about theta_crit".

So no source states a reason. The history says 0.2 was treated as one number for every size from the first
day, which is against "tuned to 20 deg" as an intention, and says nothing about where the number came from.
The measurement can show whether the defaults coincide with the physics at 20 deg and only there. It cannot
show intent.

### Definitions

Set-up: the Planck data set of `fullsky_filter_search.py` (Planck 2018 parameters, lensed TT, 5 arcmin beam
times the Nside 256 pixel window, 30 uK-arcmin white noise, l = 2 to 256). For two templates with Legendre
coefficients a_l, b_l (window included),

    <a, b> = sum over l = 2..256 of (2l+1)/(4 pi) a_l b_l / P_l,     O(a, b) = <a, b> / sqrt(<a, a> <b, b>),

P_l the total power. O is the share of the best possible S that a filter matched to a reaches on a signal
of shape b at the same centre, the quantity of the edge table of (ak). Amplitudes drop out.

Templates, all at the same nominal size theta_c:
- T: the transfer-function ramp with the wall at D* cos(theta_c) (`collision_template_transfer`).
- C(zc, f): their family, [1 - (1 - zc)(1 - cos theta) / (1 - cos theta_c)] times the cut-off of their code
  from theta_c to f theta_c. Their defaults: C(0.2, 1.1).
- Z: the z_crit = 0 cosine, cut at theta_c.

Sizes: 5, 7, 10, 12, 14, 16, 18, 20, 22, 25, 30, 40, 50, 60 deg.

1. O(C(0.2, 1.1), T) and, as the baseline, O(Z, T), at each size.
2. At each size the zc and f that maximise O(C(zc, f), T): f from 1.00 (a sharp cut) to 2.00 in steps of
   0.005, zc any real number (the best zc for a given f has a closed form because C is linear in zc).
   Reported: zc*, f*, the overlap reached, what the defaults lose against it, and the range of f whose best
   overlap is within 0.002 of the maximum (how well the fit is located).
3. Picture: profiles at 10, 20 and 40 deg of T, C(0.2, 1.1) and Z, all three with l = 2 to 256 only (no
   monopole, no dipole) and the same beam and pixel window, centre scaled to 1.

Also reported, not part of the verdict:
- 1b. C(0.2, 1.1) at 20 deg against transfer templates of 10 to 30 deg in steps of 0.5 deg: the transfer
  size it matches best. If the defaults describe a 20 deg disc, that size is between 19 and 21 deg.
- 1c. Item 1 with the WMAP W-band set-up of the search (WMAP 7-year parameters, W beam and noise, l up to
  256), the data they had in front of them.
- 2b. The same two parameters fitted in real space: least squares between the profiles of item 3, uniform
  weight in angle from 0 to 1.5 theta_c. This is closer to a fit by eye than the filter weighting is.

### Verdict, fixed now

"Peak" is the size, among the fourteen, with the largest O(C(0.2, 1.1), T).

- Inconclusive, flat: over the sizes 10 to 40 deg the largest and the smallest O(C(0.2, 1.1), T) differ by
  less than 0.02.
- Support for "tuned to 20 deg": not flat; the peak is at 18, 20 or 22 deg; and at 20 deg the best fit of
  item 2 has zc* between 0.15 and 0.25 and f* between 1.05 and 1.15.
- Against: not flat and the peak is at 14 deg or below or at 30 deg or above; or the best fit at 20 deg has
  zc* outside 0.10 to 0.30 or f* outside 1.03 to 1.20.
- Anything between (peak at 16 or 25 deg, or a best fit between the two boxes): reported as partial,
  with the numbers.
- If the best fit at 20 deg is outside the inner box while the defaults lose less than 0.002 in overlap
  against it, the fit is too flat to locate; that half then counts as undetermined, not as against.

"Support" would mean the defaults coincide with the transfer ramp at about 20 deg and at no other size. It
would not show that they were chosen for that reason; the history above stands next to it.

### Known beforehand and expected

Known: the numbers of (f) and (g) quoted above, which are William's two coincidences. From the edge table
of (ak): at the same z0 the transfer filters respond 1.24 to 1.36 times more to the code template than to
z_crit = 0 up to 20 deg, 1.13 times at 30 deg, and less than 1 from 40 deg up. That is not the normalised
overlap, but it says the code template is nearer to T than Z is below 30 deg and not above 40 deg.

Expected:
- O(Z, T) between 0.85 and 0.98, lowest somewhere from 10 to 30 deg.
- O(C(0.2, 1.1), T) above O(Z, T) from 5 to 30 deg and below it from 40 deg up, with its peak between 14
  and 22 deg. My guess for the peak is 18 or 20 deg.
- Best fit: zc* near 0.5 at 10 deg, near 0.2 at 20 deg, between -0.1 and 0.1 from 40 deg up; f* near 1.3
  at 10 deg, near 1.1 at 20 deg, under 1.05 from 40 deg up. This is what (f) and (g) give in real space;
  the filter weighting favours the edge region and may move it.
- Verdict expected: support, with moderate confidence. The risk I see is the flat case: at l up to 256 a
  2 deg feature at the edge of a 20 deg disc is three resolution elements, and the overlap may not care
  much about it.

## (ap) 2026-11-23 14:49 CST. The defaults of their code are the physical template of a 20 to 24 deg disc and of no other size

Result of (ao). Script `scripts/analysis/mcewen_template_vs_transfer.py` (written after (ao) was pushed as
commit d5bcf83), report `work/results_2026-11-23/mcewen_template_vs_transfer.json`, picture
`plots/19_code_template_vs_transfer.png`. Run on the Hamilton login node, a few seconds.

Bug in the first run, logged: the raw transfer coefficients have the sign of the curvature ramp, which is a
cold disc for a positive ramp; the search flips them in `finish_template`, my new script did not. Every
overlap came out negative and the fit of item 2 was meaningless. Fixed by giving the transfer template a
positive centre as the search does; the numbers below are from the rerun. I saw the item 1 values with the
wrong sign before the rerun; nothing was chosen from them.

### Items 1 and 2, Planck set-up (l 2 to 256)

O is the overlap with the transfer template T at the same nominal size. "Default" is the template of
their code, C(0.2, 1.1); Z is z_crit = 0. zc*, f* are the edge value and cut-off factor of their family
that fit T best, O* the overlap they reach.

| size, deg | O(default, T) | O(Z, T) | zc* | f* | O* | default loses | real-space fit zc, f |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.362 | 0.300 | 0.877 | 2.000 (end of the grid) | 0.965 | 0.603 | 0.825, 2.000 |
| 7 | 0.520 | 0.422 | 0.699 | 1.705 | 0.996 | 0.476 | 0.690, 1.785 |
| 10 | 0.704 | 0.557 | 0.528 | 1.385 | 0.997 | 0.294 | 0.565, 1.375 |
| 12 | 0.791 | 0.612 | 0.446 | 1.275 | 0.996 | 0.205 | 0.500, 1.255 |
| 14 | 0.860 | 0.656 | 0.382 | 1.205 | 0.995 | 0.136 | 0.440, 1.185 |
| 16 | 0.916 | 0.695 | 0.332 | 1.155 | 0.994 | 0.078 | 0.390, 1.140 |
| 18 | 0.959 | 0.728 | 0.289 | 1.125 | 0.994 | 0.035 | 0.345, 1.115 |
| 20 | 0.983 | 0.758 | 0.254 | 1.100 | 0.994 | 0.011 | 0.305, 1.090 |
| 22 | 0.989 | 0.784 | 0.224 | 1.085 | 0.994 | 0.005 | 0.270, 1.070 |
| 25 | 0.976 | 0.817 | 0.188 | 1.065 | 0.995 | 0.019 | 0.225, 1.055 |
| 30 | 0.938 | 0.861 | 0.143 | 1.045 | 0.996 | 0.058 | 0.170, 1.040 |
| 40 | 0.869 | 0.918 | 0.090 | 1.025 | 0.999 | 0.130 | 0.100, 1.025 |
| 50 | 0.814 | 0.949 | 0.059 | 1.015 | 0.999 | 0.186 | 0.060, 1.030 |
| 60 | 0.762 | 0.968 | 0.041 | 1.015 | 0.999 | 0.237 | 0.030, 1.035 |

At 20 deg the fit is well located: f between 1.095 and 1.110 and zc between 0.250 and 0.255 stay within
0.002 of the best overlap. A sharp cut at the edge (f = 1) reaches only 0.853 there, so the smooth cut-off
carries most of the agreement.

### Verdict by the rule of (ao): partial, one number short of support

- Not flat: over 10 to 40 deg the overlap of the defaults runs from 0.704 to 0.989, a range of 0.285.
- Peak at 22 deg (0.989), with 0.983 at 20 and 0.976 at 25 deg: inside 18 to 22 as required for support.
- Best fit at 20 deg: f* = 1.100, inside 1.05 to 1.15. zc* = 0.254, which is 0.004 outside my inner box
  of 0.15 to 0.25 and inside the outer one. The defaults lose 0.011 against the best fit, so the fit is
  not too flat to locate.
- By the rule as written that is "partial". I do not move the box.

In words: the defaults coincide with the transfer-function ramp for discs of about 20 to 24 deg and for no
other size. The cut-off factor 1.1 is the best-fit value at exactly 20 deg; the edge value 0.2 is the
best-fit value at about 24 deg (0.224 at 22, 0.188 at 25); in overlap the pair is best at 22 deg. At 10 deg
the defaults reach 0.70 and at 40 deg 0.87.

### The other items

- 1b. The 20 deg default against transfer templates of other sizes: best at 20.0 deg (0.983), with 0.980
  at 19.5, 0.945 at 19, 0.947 at 20.5, 0.891 at 21, 0.857 at 18 and 0.774 at 22 deg. Inside 19 to 21 as
  stated in (ao), and sharply so.
- 1c. WMAP W-band set-up: the same. Peak at 22 deg (0.988), 0.982 at 20 deg, z_crit = 0 at 0.749; best
  fit at 20 deg zc* = 0.255, f* = 1.100.
- 2b. Real-space fit, last column of the table: 0.305 and 1.090 at 20 deg. The rms distance of the
  defaults from the transfer profile is smallest at 25 deg (0.015 of the centre; 0.045 at 20, 0.029 at 22,
  0.021 at 30 deg), against 0.123 for z_crit = 0 at 20 deg.
- William's two numbers, recomputed in this set-up (l 2 to 256, window applied): the transfer profile at
  the causal edge is 0.207 of its centre at 20 deg ((g): 0.21), and it crosses zero at 1.066 times the
  size, 21.3 deg ((f): about 21.7 deg from a finer profile). Their defaults: 0.2 and 22 deg.

Added afterwards, not in (ao):

- The width of the best-fit cut-off, (f* - 1) times the size, is 3.85, 2.87, 2.00, 1.35, 1.0 deg at 10,
  14, 20, 30, 40 deg. That follows the projected sound horizon, about 0.68 deg / sin(size) (3.9, 2.8,
  2.0, 1.4, 1.1 deg), 1.13 times the r_s / (D sin theta_c) of (g). The 2 deg wide cut-off of their
  first-day parameter files (10 to 12, 15 to 17, 20 to 22 deg) is the physical width at 20 deg and at no
  other size.
- With zc and f allowed to depend on the size, their two-parameter form reproduces the transfer template
  to better than 0.99 from 7 deg up. The columns zc* and f* are therefore the "physical z_crit" William
  asked for: 0.53 at 10 deg, 0.25 at 20, 0.14 at 30, 0.09 at 40, 0.04 at 60 deg.
- The 2011 theory paper they cite for a negligible discontinuity (Kleban, Levi and Sigurdson) shows the
  temperature profile for all sizes from 5 to 90 deg in one surface plot. It has no single example at
  20 deg and no 0.2.

### Tuned or coincidence

What is measured: both defaults sit where the physical template is for a disc of 20 to 24 deg, and the
20 deg default matches a 20.0 deg transfer disc more closely than any other size within half a degree.
Two round numbers landing there together is a narrow target: over the sizes of the table the best-fit edge
value runs from 0.04 to 0.88 and the best-fit cut-off factor from 1.015 to 2.0.

What the history says: the function's default size is 20 deg, the cut-off was 2 deg wide on the first day,
and the same 0.2 and the same 2 deg were given to 10 and 15 deg discs that day, then turned into "0.2 and
10% of the size" for every size within a month.

My reading, which the measurement cannot prove: the numbers look taken from a 20 deg example and then used
as one shape for every size. That would explain both the match at 20 deg and the fixed fractions. It is an
inference; neither paper states a reason, and only the authors can say.

### Expectations of (ao)

- O(Z, T) between 0.85 and 0.98, lowest from 10 to 30 deg: wrong. It rises steadily with size, from 0.30
  at 5 deg through 0.56 at 10 and 0.76 at 20 to 0.97 at 60 deg. At the same nominal size the physical
  template is much wider than the z_crit = 0 disc when the disc is small.
- Defaults above z_crit = 0 from 5 to 30 deg and below it from 40 deg: right. Peak between 14 and 22
  deg: right, at the upper end; my guess of 18 or 20 was one step low.
- Best fit near 0.5 and 1.3 at 10 deg (0.53, 1.39), near 0.2 and 1.1 at 20 deg (0.25, 1.10), small edge
  and a factor under 1.05 from 40 deg (0.09, 1.025): right, except that 0.25 is not 0.2.
- Verdict "support": by the letter it is partial. The flat case I feared did not occur.

### What this changes

Nothing in the Planck search or its numbers. Three statements are sharper now:
- (an) said their template is close to the physical one "near 20 deg at most". Measured: 0.98 to 0.99 at 20
  to 22 deg, 0.94 at 30, 0.87 at 40, 0.70 at 10 deg.
- The z_crit = 0 template of the papers' text is the poorer stand-in for the physical template below about
  35 deg at the same nominal size (0.76 at 20 deg), and the better one above (0.92 at 40, 0.97 at 60 deg).
  A size read from a z_crit = 0 filter is not the causal size of the physical disc for small discs.
- The published search (their code's template) was therefore closer to the physical template than the
  search their text describes for discs up to 30 deg (both are poor below about 12 deg), and further from
  it at 40 deg and above, the range of their two largest candidates (85 and 90 deg, beyond this table).

## (aq) 2026-11-23 15:11 CST. Pre-registration: overlaps with the size left free, and one definition of how small a ramp collision can look

An outside review that William passed on (23 Nov) makes four points about what I told him from (ap) and from
the earlier WMAP runs. Two are settled from files already in the repository and are recorded with the
result. Two need a computation; this entry is written before it.

Point: the overlaps of (ap) compare templates at the same nominal size, while a search scans sizes, so
"their filters reach 0.87 at 40 deg and 0.76 at 60 deg on a physical collision" overstates the loss.
Correct. Item 1b of (ao) did the scan only for their 20 deg default.

Point: (z) says a ramp collision "cannot look smaller than about 8 deg" (8.3 deg, a wall touching the
surface, the sound horizon seen side-on); I told William "roughly 12 to 13 deg". Two numbers for one idea.
My 12 to 13 deg was the first zero of the 5 to 10 deg profiles in the table of (ap) (13.9 deg for 7 deg,
13.3 deg for 10 deg), which is neither the definition of (z) nor a smallest possible value.

### Definitions

Set-up and overlap O as in (ao) (Planck, l 2 to 256).

A. Fitting factor. For the transfer template T of causal size s, at the fourteen sizes of (ao):
FF(s) = the largest O(filter of size t, T(s)) over t from 1 to 90 deg in steps of 0.5 deg, for two filter
families: their code template C(0.2, 1.1) and the z_crit = 0 cosine Z. Reported with the size t that gives
it, and the same restricted to the 33 sizes of the search bank.

B. Apparent size of the transfer template, from its profile as the search sees it (l 2 to 256, window,
centre scaled to 1), for causal sizes 0 (a wall touching the surface), 0.5, 1, 2, 3, 5, 7, 10, 14, 20 deg:
- R50: the angle where the profile has fallen to half its central value;
- R10 and R02: where it has fallen to 10% and to 2%;
- the first zero.
The same four numbers for their code template and for Z at the sizes of their small candidates (1.5, 2, 3,
4, 5, 7 deg). The check is repeated at l up to 767 to see that the band limit does not set the numbers.
The number to use from now on is R50, the half-maximum radius, with R02 as "where it fades out": both exist
for every template and do not depend on a zero crossing, which the removal of monopole and dipole moves.

### Known beforehand and expected

Known: the same-size overlaps of (ap); the first zeros quoted above; item 1b (their 20 deg default is best
on a 20.0 deg transfer disc); Gus's fitting factors for a Sachs-Wolfe bank on the transfer ramp at l up to
767 (0.99, 0.98, 0.96, 0.96, 0.93, 0.97 at 5, 10, 15, 20, 30, 60 deg, with the best filter 2 to 5 deg
larger than the causal size up to 30 deg); and from (z) the outline of the smallest disc in Kleban et al.'s
figure (half maximum at 5.6 deg, 2% at 9.6 deg), which this template matched for a wall touching the surface.

Expected:
- A, z_crit = 0 filters: FF between 0.92 and 0.99 from 5 to 60 deg, best filter 2 to 6 deg larger than
  the causal size up to 30 deg, the same size from 50 deg up.
- A, their code template: FF at least 0.93 from 7 to 30 deg, 0.90 to 0.95 at 40 deg, 0.85 to 0.92 at 60
  deg; best filter larger than the causal size below 20 deg, equal at 20 to 22 deg, smaller above 30 deg.
  So the loss of their search on physical collisions would be 5 to 15% at 40 to 60 deg, not 13 to 24%.
- B: for a wall touching the surface R50 near 5.6 deg and R02 near 9.6 deg; R50 about 7 deg for a causal
  size of 5 deg and about 10.3 deg for 10 deg; the numbers at l up to 767 within 0.3 deg of those at 256.
  Their template has R50 = 0.79 times its size, so their candidates of 1.5 to 5 deg have R50 of 1.2 to 4
  deg, below the smallest R50 of a ramp collision; their 7 deg candidate (5.5 deg) is at that limit.
