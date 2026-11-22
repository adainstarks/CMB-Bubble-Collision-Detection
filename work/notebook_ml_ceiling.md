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

Correction to the diagnosis of C1, 08:58 CST. Smoke job 93903 passed the projection and then stopped at the
guard itself: `module 'healpy' has no attribute 'gnomview'`. The cause of both errors is one thing: the
environment had no matplotlib, and healpy then loads without its plotting functions (and without the
`hp.projector` attribute path). The healpy version is not the cause. Fix: matplotlib added to the
environment, `MPLBACKEND=Agg` set in `cluster/babbage/env.sh`. The direct import of `GnomonicProj` stays;
it is correct either way.

## (D) 2026-11-22 09:03 CST. Smoke job 93904: test-set path passes its guards; training needs more than 11 GB

Job 93904 (node c11, RTX 2080 Ti, 1 min 26 s), six skies, three bank sizes:
- projector output identical to `hp.gnomview`: passed (both test sets);
- standard deviation of signal-free SNR maps: 1.002, 1.012, 1.034 (5, 15, 25 degrees; six skies);
- SNR at the signal centre against A * sqrt(norm): worst deviation 1.47 standard errors;
- reference tables are written for the ramp set and for a mix 1.0 set on the SNR grid (too few rows to read
  anything from: 12 signals per level).
- Training stopped at the first step: CUDA out of memory, 10.3 GiB in use on an 11 GiB card at batch 64.

Not a code defect. c11 is the only node with 11 GiB cards (2080 Ti); the others have 24 GiB (A5000) or
32 GiB (RTX 5000 Ada). Job files now exclude c11 as well as c14, and the training array runs at most four
jobs at a time so that it does not take every free GPU of the partition.

## (E) 2026-11-22 09:10 CST. Jobs submitted; one more job-file bug

Submitted at about 09:05 CST (15:05 UTC on the cluster clock), all from commit a2eec7b, after entry (B)
was pushed (commit 25a6edc, 08:40 CST):
- 93906_[0-2]: the three test sets T0, T35, T100 on CPU nodes c1 to c3. Running.
- 93905: smoke job on a 24 or 32 GiB GPU. Waiting; the scheduler's estimate is 23:44 CST tonight.
- 93907_[0-12%4]: the thirteen training runs, held until 93905 and all of 93906 end successfully.

**Bug C2 (job file only).** To record the code version I added `git log` to the job files (commit
37865b7) and ran an extra smoke job on the idle 2080 Ti node with batch 16 (93910). It failed in ten
seconds: compute nodes have no `git`. The job files now read the commit hash from `.git/refs` instead.
Jobs 93905, 93906 and 93907 were submitted before that edit and Slurm keeps the script as submitted, so
they are not affected.

## (F) 2026-11-22 09:14 CST. Smoke job 93912: training and scoring run; bug C3 in the report writer

Job 93912 (c11, batch 16): one epoch on 96 backgrounds in 7 s, both test sets scored, tables printed with
network and reference rows side by side. Then `TypeError: Object of type function is not JSON
serializable` while writing `report.json`.

**Bug C3.** The run's argument record included argparse's sub-command function. Fixed: that entry is
dropped before saving. Every real training run would have failed at this point, after training, so the
scores would have been lost; none had started. The pending jobs (93905 smoke, 93907 training) read the
script from the repository when they start and will run the fixed version.

## (G) 2026-11-22 09:20 CST. Whole path verified; state of the queue

Smoke job 93913 (c11, batch 16, commit 6c3a399): COMPLETED in 1 min 12 s, ends with `SMOKE_OK`. Test sets
made with all guards passing, one epoch trained, both test sets scored, `report.json` written, `summarize`
prints its tables. The only thing not yet exercised on the cluster is batch 64 on a 24 or 32 GiB card.

Queue at 09:20 CST:
- 93906_0 (T0): running, 3.3 s per sky, about 83 minutes for 1,500 skies. 93906_1 and _2 (T35, T100) running.
- 93905 (smoke on a large GPU): waiting for a free card, estimated 23:44 CST.
- 93907_[0-12%4] (thirteen training runs): held until 93905 and 93906 succeed.

Nothing in entries (C) to (G) changed the design or the predictions of entry (B). No test-set score of any
network has been looked at: none exists.

Next, when the runs are in: `python scripts/ml/ml_ceiling_study.py summarize --runs-dir
/projects/gmarcum/cmb/runs/ceiling_v1 --out-json ...`, then score the ten predictions of (B) one by one.

## (H) 2026-11-22 16:40 CST. The large cards are booked for days; the runs move to c11 with mixed precision

**State at 15:30 CST.** Test sets 93906_[0-2]: COMPLETED (T0 1 h 32 min, T35 1 h 06 min, T100 1 h 05 min).
Smoke job 93905 and the training array 93907: still pending. Every RTX 5000 Ada and A5000 node is occupied
by other users' jobs with one to three day limits, with more of the same queued behind them; c13 has been
drained since 11-18; our priority value is 1. Only c11 (two RTX 2080 Ti, 11 GiB each) is idle.

**Correction to (E) and (G).** The estimate "23:44 CST tonight" was my conversion error. `squeue` on the
login node prints local time (CST); only the compute nodes print UTC. The estimate was, and is, 05:44 CST
on 11-23, on c17, one card.

**Decision, made before any network has been trained or scored.** The thirteen runs of (B) go to c11, two
at a time (`cluster/babbage/train_c11.sbatch`). Batch 64 in float32 needs 10.3 GiB there and failed (job
93904), so training uses mixed precision: the forward pass runs under float16 autocast with a gradient
scaler, the loss is computed in float32. Everything else in the recipe of (B) is unchanged: batch 64,
AdamW at 3e-4, one warm-up epoch and a cosine schedule, 30 epochs, the same data and model seeds. All
scoring (validation and the three test sets) stays float32; the scoring batch drops from 256 to 128,
which cannot change a score (evaluation mode, no batch statistics).

This is a change to the training recipe after the pre-registration. It is logged here before the first
run starts, it applies to all thirteen runs so the three arms stay comparable, and the ten predictions of
(B) stand as written.

**Control.** One float32 run of arm R, seed 1, stays queued for a large card (`train.sbatch`, output in
`runs/ceiling_v1_fp32/R_seed1`). Stated now: its detection probability on T0 at 3e-5 (image head, 5%
false positives) should be within 0.04 of the mixed-precision R seed 1, that is twice the seed scatter
predicted in (B) item 2. It is reported whichever way it falls.

**Code.** `ml_ceiling_study.py train` gains `--amp` and `--eval-batch`, prints the time of the first 50
steps, and records the largest GPU memory in use. `smoke.sbatch` passes `SMOKE_TRAIN_FLAGS` through.
`train.sbatch` takes `RUNS_SUBDIR` and no longer excludes c14 (both of its cards have run other users'
jobs since 11-19).

Next: smoke job on c11 at batch 64 with `--amp`. Only if it passes: submit the c11 array, cancel 93905 and
93907, submit the float32 control.
