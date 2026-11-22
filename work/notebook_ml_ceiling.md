# Notebook: how far can a network go? (Gus's side, started 22 November 2026)

Same rules as `work/notebook.md`: dated entries, appended in order, never silently rewritten; predictions
are written before the analysis they concern; bugs are logged. One change answers the review of 22 November:
every prediction entry is committed and pushed before the job it concerns is submitted, so the git history
carries the time stamp.

Layout on Babbage: `/projects/gmarcum/cmb/` with `repo/` (this branch), `tests/`, `runs/`, `logs/`,
`cache/` (pretrained weights); environment `/projects/gmarcum/envs/cmb`.

---

## (A) 2026-11-22 08:36 CST. Why this study, what the yardsticks are

Starting point. November's entries (l) to (s) show a network that reaches the matched filter on Gaussian
skies, pooled over sizes, from one training run, against a filter whose search radius was chosen after the
first result. Those limits are listed in `work/review_2026-11-22.md` (local review, not on this branch).
The project's question is how far machine learning can go on this problem, so the comparison has to be one
that can carry an answer.

Two facts fix what "far" can mean.

1. For a known shape on Gaussian skies no detector beats the likelihood ratio. For a prior over size,
   position, sign and strength the best possible test is the marginal likelihood ratio (the Bayes test).
   The maximum over a filter bank is close to it and not identical. Both are computed here exactly, from
   full-sky filter outputs, and the Bayes test is the ceiling.
2. The shape is not known. New on 22 November (`scripts/analysis/template_fitting_factors.py`,
   `work/results_2026-11-22/template_fitting_factors.json`): the fraction of the ideal signal-to-noise that
   a bank recovers when the true shape differs from the bank's.
   - Ramp through the full transfer function, searched with a Sachs-Wolfe ramp bank (what the published
     and the Sydney searches use): 0.93 to 0.99 for 5 to 60 degrees. For detection the two ramps are nearly
     interchangeable; the full-transfer template matters for what an amplitude means, not for finding it.
   - Ramp plus a step of the same sign (step height = `mix` times the ramp's Sachs-Wolfe central depth),
     searched with a ramp-only bank: mix 0.25 gives 0.99, 0.95, 0.83, 0.61, 0.37 at 5, 10, 15, 20, 30
     degrees; mix 1 gives 0.96, 0.73, 0.52, 0.37, 0.20. A two-family bank (ramp and step) recovers 0.87 to
     1.00 everywhere.
   So a ramp-only search loses most of its sensitivity above about 15 degrees if the collision leaves even
   a modest step. Which of ramp and ramp+step is right is an open point in the literature (Osborne et al.
   2013; notebook entry (b)). This is the place where a detector that does not need the shape spelled out
   can be worth something, and it is measurable.

Caveat on the step shapes, measured today: CAMB's default k range (k < 0.17 per Mpc at lmax 767) holds a
step's multipoles only up to about 2370 sin(theta_c): 206 at 5 degrees, 411 at 10, 613 at 15, complete from
19 degrees. Extending the range does not help, because CAMB samples k too coarsely there for this integral
(a run with `max_eta_k` 14000 changed the 20 degree step by 39%, which is aliasing, not physics). The step
shapes below 19 degrees are therefore smoother at the edge than a full calculation would give. Signals,
filters and networks use the same templates, so the comparison between methods is not affected. The shapes
are a family for testing shape freedom, not a prediction.

## (B) 2026-11-22 08:36 CST. Pre-registration: ceiling study on Gaussian skies

Code: `scripts/ml/ml_ceiling_study.py` (local checks done before this entry: templates reproduce the
0.93 / 1.10 / 1.11 ideal SNR per 1e-5 at 5 / 15 / 25 degrees; a filter's response to its own template
equals sqrt(norm); the GPU signal injection agrees with `patch_geometry` to 8e-5 of the peak and gives
identical masks; threshold, table and log-mean-exp functions checked against known values. Nothing that
needs healpy has run yet.)

Prior, fixed here and used by training data, test sets and references alike: sizes 5 to 25 degrees in
1 degree steps; centre uniform within 5 degrees of the patch centre (covers the 3.7 degree spacing of the
Nside 16 tile scan); sign +-1; strength log-uniform in ideal matched-filter SNR from 1 to 12 (for the ramp
an observed peak of about 1e-5 to 1.2e-4). Skies as in November: Gaussian, 5 arcmin beam, pixel window,
30 uK-arcmin, Nside 256, 256 x 256 patches at 13 arcmin, no mask.

Test sets (made once, before any training; seeds 20261201 to 20261203; never used for training or
selection):
- T0: ramp, amplitudes 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, sizes 5, 10, 15, 20, 25; 1,500 skies, 12,000
  patches, 200 signals per cell.
- T35: ramp+step with mix 0.35 (a mix no network is trained on); ideal SNR 1, 2, 3, 5, 7, 10; 1,000 skies.
- T100: mix 1.0; same SNR grid; 1,000 skies.

References, computed with each test set: `glrt` (largest |SNR| over the bank and over sky pixels within the
5 degree prior disc) and `bayes_prior` (marginal likelihood ratio over bank, pixels, sign and the strength
prior), each for three banks: `ramp` (21 ramp templates, what the field uses), `family` (84 templates: mixes
0, 0.25, 0.5, 1), `matched` (21 templates of the test set's own shape; an oracle that knows the shape).
`bayes_known` is the same with the test level's strength known.

Networks: U-Net with EfficientNet-B0 encoder (segmentation-models-pytorch), one input channel, mask output
plus an image-level head; 32,000 signal-free training patches from 4,000 skies, signals re-drawn on the GPU
every batch; 30 epochs, AdamW, learning rate 3e-4 with one warm-up epoch and cosine decay, batch 64,
dihedral augmentation, loss = cross-entropy + Dice on the mask + cross-entropy on the image head. Last
epoch used. Training skies differ per seed.
- R: trained on the ramp only, ImageNet initialisation, seeds 1 to 5.
- N: as R with random initialisation, seeds 1 to 3.
- F: trained on the family (mixes 0, 0.25, 0.5, 1), ImageNet initialisation, seeds 1 to 5.

Scores: the image-head logit is primary; the largest mask logit is reported beside it. Thresholds: each
score's own 5% (and 1%) quantile on the test set's signal-free patches. All numbers below are detection
probabilities at 5%, pooled over sizes unless a size is named, means over seeds.

Predictions.
1. Ceiling reached. T0, arm R at 3e-5: between 0.93 and 1.00 of the ramp-bank `glrt`, and not above the
   ramp-bank `bayes_prior` by more than 0.02. Same at 2e-5 and 5e-5.
2. Seeds agree. Standard deviation over the five R seeds at 3e-5 is at most 0.02.
3. Every size. T0 at 3e-5, arm R is at least 0.90 of `glrt` at each of 5, 10, 15, 20, 25 degrees. I am
   least sure of 25 degrees, where November's network was 0.125 behind and the disc with its offset leaves
   the patch.
4. Bayes above the bank. `bayes_prior` is at or above `glrt` at every level of T0, by at most 0.04.
5. Pretraining. Arm N is below arm R by at least 0.03 at 3e-5 on T0.
6. Shape costs the ramp bank. T100 at ideal SNR 5: the ramp-bank `glrt` detects less than half of what the
   matched-bank `glrt` detects.
7. A network trained on the family keeps it. T100 at ideal SNR 5: arm F is at least 0.85 of the
   matched-bank `glrt` and at least 0.25 above the ramp-bank `glrt`.
8. Between the trained shapes. T35 at ideal SNR 3 and 5: arm F is at least 0.90 of the matched-bank `glrt`.
9. Price of generality. T0 at 3e-5: arm F is no more than 0.04 below arm R.
10. No prediction, reported as found: arm R on T35 and T100 (a ramp-trained network meeting a step).

What would count against "a network reaches the ceiling": prediction 1 failing low with flat validation
curves (longer training would then not be the answer), or prediction 3 failing at sizes below 25 degrees.
What would count against "a network handles shape freedom": prediction 7 or 8 failing while the `family`
bank's `glrt` stays near the matched bank.

Limits, stated beforehand: Gaussian skies with white noise; no mask (the masked-sky and real-map tests
come after this one and are not part of this entry); signal centres within 5 degrees; one architecture;
the band-limited step of entry (A); thresholds set on the test set's own signal-free patches.

## (C) 2026-11-22 08:52 CST. Environment on Babbage; first smoke job found a bug

Environment `/projects/gmarcum/envs/cmb`: python 3.11.18, torch 2.6.0+cu124, healpy 1.20.1, camb 2.0.5,
segmentation-models-pytorch 0.5.0, timm 1.0.31, scipy 1.17.1 (newer than `environment.yml`: healpy 1.19.0,
camb 1.6.6, torch 2.5.1). ImageNet weights for the encoder cached under `/projects/gmarcum/cmb/cache`.

Smoke job 93901 (node c11, RTX 2080 Ti): python starts on the compute node, CUDA is available, the
templates give 0.932 and 1.111 ideal SNR per 1e-5 at 5 and 25 degrees (as on the PC). Then it stopped:
`AttributeError: module 'healpy' has no attribute 'projector'`.

**Bug C1.** `ml_ceiling_study.project` used `hp.projector.GnomonicProj`. In healpy 1.20.1 the submodule is
not an attribute of the package until a plotting function has been imported; November's scripts call
`hp.gnomview` first, mine did not. Fixed by importing `GnomonicProj` from `healpy.projector` directly. No
result existed yet, so nothing is affected. The built-in guard (projected patch must equal `hp.gnomview`
output) stays and will run in the next smoke job.
