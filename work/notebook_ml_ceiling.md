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

## (I) 2026-11-22 16:30 CST. c11 check passes; the thirteen runs are started

**Test sets (jobs 93906_[0-2], read from their logs).** All three written. Standard deviation of the
signal-free filter maps per bank size, which should be 1: T0 1.003 to 1.035 (rising with size), T35 0.993
to 1.006, T100 1.009 to 1.018.

**Smoke job 93949** (c11, commit 88a3110, batch 64, `--amp --eval-batch 128`): COMPLETED in 1 min 17 s,
`SMOKE_OK`. Largest GPU memory in use during training 5.7 GiB, against 10.3 GiB and an out-of-memory
failure in float32 (job 93904).

**Submitted 16:27 CST.**
- 93951_[0-12%2]: the thirteen runs of (B) on c11, two at a time, mixed precision as described in (H).
  93951_0 and 93951_1 (arm R, seeds 1 and 2) started at once.
- 93952_[0]: the float32 control of (H), arm R seed 1, waiting for a large card (estimate 05:44 CST 11-23).
- Cancelled: 93905 (smoke on a large card, superseded by 93949) and 93907 (the float32 array).

Small thing seen in passing: c11's clock reads about 1 min 40 s ahead of the scheduler's (job start
16:27:35 CST by Slurm, 22:29:14 UTC in the job's own log). It affects nothing here.

No network score has been looked at. Next: when 93951 is done, `summarize` on `runs/ceiling_v1`, then the
ten predictions of (B), one by one, then the control.

## (J) 2026-11-22 18:10 CST. Hamilton is free: the float32 runs of (B) go there and are the primary set

**Why now.** Hamilton OnDemand is reachable again (a stale sign-in cookie in the browser, nothing to do with
this study). At 17:50 CST Hamilton had five idle nodes with one RTX 5000 Ada (32 GiB) each, and the A100
idle. The float32 recipe of (B) fits there, so the reason for the change of recipe in (H) does not apply.

**Set-up on Hamilton.** Same paths as on Babbage (`/projects/gmarcum/cmb`, environment
`/projects/gmarcum/envs/cmb`), a standalone Python 3.11.18, and the same pinned versions: torch 2.6.0+cu124,
torchvision 0.21.0, numpy 2.4.6, scipy 1.17.1, astropy 8.0.3, healpy 1.20.1, camb 2.0.5,
segmentation-models-pytorch 0.5.0, timm 1.0.31. Branch at f2cbd64. The job files under `cluster/babbage/`
are used as they are, with `--exclude=` and the partition given on the command line.
Smoke job 82949: `SMOKE_OK`, float32, batch 64, 10.7 GiB in use on an RTX 5000 Ada.

**A trap on Hamilton.** File locks on `/projects` hang there. A `uv python install` and a Hugging Face
download both sat in uninterruptible sleep until killed; the same commands were fine on Babbage. Both now
write to the login node's `/tmp` and are copied over. Jobs run with `HF_HUB_OFFLINE=1`, which reads the
cache without taking a lock.

**Test sets.** The two clusters have separate storage and no key between them, so T0, T35 and T100 are made
again on Hamilton with the same code, seeds and package versions: job 82950_[0-2]. When they are done their
`meta.npz` and `patches.npy` are compared with Babbage's (md5 of `meta.npz` there: T0 34a64af4..., T35
dfc051bd..., T100 c8bfdf64...). Whether the two copies are bit-identical is recorded either way. If they
are not, each set of runs is scored against the yardsticks computed on its own copy.

**Training.** Job 82951_[0-12%4]: `train.sbatch` unchanged (float32), output in `runs/ceiling_v1_fp32`,
held until 82950 ends successfully. Four at a time, which leaves two nodes and the A100 free.

**Which set counts, stated before any float32 run exists.** The float32 runs on Hamilton are the primary
set, because they are the recipe of (B) as written; the ten predictions are scored on them. The
mixed-precision runs on c11 (job 93951) are reported next to them as a replicate. The control of (H)
becomes a comparison of thirteen pairs instead of one, with the same criterion (0.04 on T0 at 3e-5).
I had seen the first two mixed-precision runs when writing this; their numbers follow so that the order of
events is on record.

**Early look, not a scored result** (arm R, seeds 1 and 2, mixed precision, jobs 93951_0 and _1; 1 h 06 min
and 5.8 GiB each; detection probability at 5% false positives):

| T0 (ramp) | 1e-5 | 2e-5 | 3e-5 | 5e-5 |
|---|---:|---:|---:|---:|
| network, image head | 0.103 | 0.318 | 0.603 | 0.974 |
| network, mask maximum | 0.100 | 0.303 | 0.595 | 0.974 |
| ramp filter bank, maximum | 0.093 | 0.295 | 0.613 | 0.983 |
| Bayes, with the prior | 0.108 | 0.334 | 0.623 | 0.979 |

| T100 (step-rich), ideal SNR | 3 | 5 | 7 | 10 |
|---|---:|---:|---:|---:|
| network, image head | 0.180 | 0.410 | 0.589 | 0.716 |
| ramp filter bank, maximum | 0.177 | 0.450 | 0.671 | 0.780 |
| matched filter bank, maximum | 0.149 | 0.671 | 0.980 | 1.000 |

The two seeds agree to 0.002 to 0.006. Nothing is concluded from two runs of one arm.

## (K) 2026-11-22 19:50 CST. The Hamilton test sets are the Babbage ones to rounding; float32 training under way

**Test sets made again on Hamilton** (job 82950: T0 39 min, T35 31 min, T100 32 min, at 1.5 s per sky
against 3.3 s on Babbage's CPU nodes).

The files are not bit-identical: every md5 differs (`patches.npy` T0 071a2291... on Babbage, a822b2c3... on
Hamilton, and likewise for T35 and T100; the `.npz` files would differ anyway because the archive stores
times). The contents agree to rounding. For T0, read with the same one-line script on both clusters:

| | Babbage | Hamilton |
|---|---|---|
| shape, type | (12000, 256, 256) float32 | the same |
| patch 0, row 100, pixels 100 to 103 | 1.9696623e-04, 1.9696623e-04, 9.4829600e-05, 4.7090085e-05 | the same digits |
| patch 5000, row 50, pixels 50 to 52 | -9.2066810e-05, -9.7516226e-05, -5.1702977e-05 | the same digits |
| positives | 6000 | 6000 |
| mean of the ramp-bank GLRT score | 4.209520419113489 | 4.209520419113488 |
| first four GLRT scores | 3.134655, 3.19520944, 2.82121637, 2.73012691 | the same digits |

So both clusters hold the same skies, the same injected signals and the same yardsticks, different only in
the last bits (different CPUs). The float32 runs and the mixed-precision runs are scored on what is in
effect one test set, and the thirteen pairs of (J) are a fair comparison. T35 and T100 were not printed
value by value; they come from the same code path with their own seeds.

**Float32 training on Hamilton** (job 82951_[0-12%4]) started at 18:47 CST, as soon as the test sets were
done: arm R seeds 1 to 4 on c3, c5, c6, c7, about 110 s per epoch. First three epochs of seed 1, loss and
validation detection (image head): 1.7346 / 0.408, 1.4129 / 0.678, 1.3699 / 0.543. The mixed-precision run
of the same seed on c11 gave 1.7355 / 0.421, 1.4128 / 0.676, 1.3703 / 0.552: the same training stream, as
intended.

**Mixed-precision set on c11** (job 93951) at 19:45 CST: arm R seeds 1 to 5 and arm N seed 1 finished;
N seeds 2 and 3 running; arm F waiting.

No score beyond the early look of (J) has been read.

## (L) 2026-11-23 00:30 CST. All 26 runs are in: the ramp ceiling is reached; the family-trained network does not use the sharp edge

**Runs.** Float32 on Hamilton (job 82951, the primary set of (J)): 13 of 13 completed, 1 h 02 to 1 h 06
each, 10.8 GiB of GPU memory in training. Mixed precision on Babbage c11 (job 93951): 13 of 13 completed,
1 h 05 each. The float32 control on Babbage (93952) was cancelled at 22:25 CST before it started; the
Hamilton set is that control thirteen times over.

**An earlier look, for the record.** At 22:22 CST I read the summary of 12 of the 13 float32 runs (arms R
and N complete, F four of five) for a status report. No setting, script or criterion was changed after it.
Every number below is from the complete sets.

**How read.** On each cluster: `ml_ceiling_study.py summarize --runs-dir <runs> --out-json <file>`. Files:
Hamilton `/projects/gmarcum/cmb/logs/summary_fp32.{txt,json}` and `history_fp32_compact.json`; Babbage
`/projects/gmarcum/cmb/logs/summary_amp.{txt,json}`. They are not in the repository yet (copying them off
the clusters is a download that has not been asked for). Each run folder also holds its per-patch scores
(`scores__T0.npz` and so on).

All numbers are detection probabilities at 5% false positives, pooled over sizes, image head, mean over
seeds, with the standard deviation over seeds after +-. Float32 set unless the mixed-precision one is named.

### T0, the ramp (what arm R is trained on)

| | 1e-5 | 2e-5 | 3e-5 | 5e-5 | 7e-5 | 1e-4 |
|---|---|---|---|---|---|---|
| arm R (5 seeds) | 0.109 +- 0.005 | 0.314 +- 0.005 | 0.599 +- 0.005 | 0.973 +- 0.002 | 1.000 | 1.000 |
| arm R, largest mask logit | 0.105 | 0.309 | 0.599 | 0.975 | 1.000 | 1.000 |
| arm N (3 seeds) | 0.110 +- 0.005 | 0.314 +- 0.001 | 0.593 +- 0.002 | 0.971 +- 0.001 | 1.000 | 1.000 |
| arm F (5 seeds) | 0.109 +- 0.005 | 0.303 +- 0.002 | 0.587 +- 0.008 | 0.968 +- 0.002 | 0.999 | 1.000 |
| `glrt`, ramp bank | 0.093 | 0.295 | 0.613 | 0.983 | 1.000 | 1.000 |
| `bayes_prior`, ramp bank | 0.108 | 0.334 | 0.623 | 0.979 | 1.000 | 1.000 |
| `bayes_known`, ramp bank | 0.111 | 0.327 | 0.623 | 0.983 | 1.000 | 1.000 |
| `glrt`, family bank | 0.075 | 0.178 | 0.440 | 0.907 | 0.997 | 1.000 |
| `bayes_prior`, family bank | 0.100 | 0.320 | 0.602 | 0.974 | 1.000 | 1.000 |

By size at 3e-5:

| | 5 | 10 | 15 | 20 | 25 degrees |
|---|---|---|---|---|---|
| arm R | 0.429 | 0.620 | 0.620 | 0.702 | 0.625 |
| arm R, largest mask logit | 0.512 | 0.670 | 0.608 | 0.646 | 0.558 |
| arm N | 0.430 | 0.613 | 0.627 | 0.687 | 0.610 |
| arm F | 0.414 | 0.613 | 0.609 | 0.683 | 0.616 |
| `glrt`, ramp bank | 0.450 | 0.635 | 0.620 | 0.710 | 0.650 |
| `bayes_prior`, ramp bank | 0.435 | 0.665 | 0.640 | 0.735 | 0.640 |
| arm R over `glrt` | 0.953 | 0.976 | 1.000 | 0.989 | 0.962 |

### T35, ramp+step with mix 0.35 (no network trained on this mix)

| ideal SNR | 1 | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|
| arm R | 0.064 | 0.140 | 0.289 | 0.645 | 0.800 | 0.919 |
| arm N | 0.065 | 0.134 | 0.279 | 0.628 | 0.786 | 0.909 |
| arm F | 0.069 +- 0.004 | 0.141 +- 0.007 | 0.282 +- 0.011 | 0.667 +- 0.006 | 0.837 +- 0.005 | 0.965 +- 0.006 |
| `glrt`, ramp bank | 0.066 | 0.143 | 0.280 | 0.669 | 0.839 | 0.952 |
| `glrt`, family bank | 0.058 | 0.100 | 0.185 | 0.741 | 0.988 | 1.000 |
| `glrt`, matched bank | 0.072 | 0.118 | 0.192 | 0.767 | 0.992 | 1.000 |
| `bayes_prior`, ramp bank | 0.064 | 0.139 | 0.302 | 0.662 | 0.821 | 0.940 |
| `bayes_prior`, family bank | 0.066 | 0.137 | 0.289 | 0.755 | 0.985 | 1.000 |
| `bayes_prior`, matched bank | 0.060 | 0.133 | 0.287 | 0.770 | 0.986 | 1.000 |

### T100, the pure step (arm F is trained on it, arm R is not)

| ideal SNR | 1 | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|
| arm R | 0.071 +- 0.008 | 0.115 +- 0.006 | 0.187 +- 0.011 | 0.417 +- 0.007 | 0.595 +- 0.004 | 0.717 +- 0.003 |
| arm N | 0.061 | 0.113 | 0.179 | 0.409 | 0.589 | 0.706 |
| arm F | 0.076 +- 0.005 | 0.104 +- 0.008 | 0.173 +- 0.007 | 0.447 +- 0.007 | 0.673 +- 0.007 | 0.791 +- 0.003 |
| arm F, largest mask logit | 0.079 | 0.109 | 0.193 | 0.438 | 0.655 | 0.773 |
| `glrt`, ramp bank | 0.076 | 0.107 | 0.177 | 0.450 | 0.671 | 0.780 |
| `glrt`, family bank | 0.051 | 0.078 | 0.152 | 0.651 | 0.976 | 1.000 |
| `glrt`, matched bank | 0.051 | 0.057 | 0.149 | 0.671 | 0.980 | 1.000 |
| `bayes_prior`, ramp bank | 0.067 | 0.116 | 0.188 | 0.432 | 0.633 | 0.749 |
| `bayes_prior`, family bank | 0.072 | 0.106 | 0.198 | 0.626 | 0.962 | 1.000 |
| `bayes_prior`, matched bank | 0.069 | 0.091 | 0.189 | 0.677 | 0.970 | 1.000 |

By size at ideal SNR 3 (133 signals per cell, so each number is good to about 0.03 to 0.04):

| | 5 | 10 | 15 | 20 | 25 degrees |
|---|---|---|---|---|---|
| T100, arm F | 0.421 | 0.229 | 0.113 | 0.057 | 0.044 |
| T100, arm R | 0.478 | 0.232 | 0.123 | 0.056 | 0.047 |
| T100, `glrt` matched bank | 0.226 | 0.211 | 0.098 | 0.128 | 0.083 |
| T35, arm F | 0.465 | 0.424 | 0.244 | 0.179 | 0.099 |
| T35, `glrt` matched bank | 0.308 | 0.241 | 0.150 | 0.158 | 0.105 |

### The ten predictions of (B), on the float32 set

1. **Ceiling reached: holds at 3e-5 and 5e-5, misses its upper bound at 2e-5.** Arm R over the ramp-bank
   `glrt`: 0.977 at 3e-5 and 0.990 at 5e-5, both inside 0.93 to 1.00. At 2e-5 it is 1.064 (0.314 against
   0.295): the network is above the filter bank, which the band did not allow for. The second clause holds
   at all three levels: arm R is never above `bayes_prior` (0.334, 0.623, 0.979). The prediction did not
   fail low anywhere.
2. **Seeds agree: holds.** Standard deviation over the five R seeds at 3e-5 is 0.005.
3. **Every size: holds.** Arm R over `glrt` at 3e-5 is 0.953, 0.976, 1.000, 0.989, 0.962 at 5 to 25
   degrees. The 25 degree case I was least sure of is at 0.962.
4. **Bayes above the bank: holds at five of six levels.** `bayes_prior` minus `glrt` on T0: +0.015, +0.039,
   +0.010, -0.004, 0.000, 0.000. At 5e-5 it is 0.004 below, four signals in 1,000. As written the
   prediction fails there.
5. **Pretraining: fails.** Arm N is 0.006 below arm R at 3e-5 (0.593 against 0.599), not 0.03 or more.
   ImageNet weights make no difference that this test can see, at any level or on any test set.
6. **Shape costs the ramp bank: fails.** T100 at ideal SNR 5: ramp bank 0.450, matched bank 0.671, a ratio
   of 0.67 where I predicted under 0.5.
7. **A network trained on the family keeps it: fails, both clauses.** T100 at ideal SNR 5: arm F 0.447,
   which is 0.67 of the matched bank (0.85 predicted) and 0.003 below the ramp bank (0.25 above predicted).
8. **Between the trained shapes: holds at SNR 3, fails at SNR 5.** T35: arm F over the matched-bank `glrt`
   is 1.47 at SNR 3 (0.282 against 0.192, where the bank is weak) and 0.870 at SNR 5 (0.667 against 0.767;
   0.90 predicted).
9. **Price of generality: holds.** T0 at 3e-5: arm F is 0.012 below arm R.
10. **No prediction, as found.** Arm R on a step behaves like the ramp bank and a little below it: on T100
    0.417, 0.595, 0.717 at SNR 5, 7, 10 against 0.450, 0.671, 0.780 for the ramp bank; on T35 0.645, 0.800,
    0.919 against 0.669, 0.839, 0.952.

### Mixed precision against float32 (the thirteen pairs of (J))

T0 at 3e-5, same seed, float32 / mixed precision:

- R: 0.601 / 0.599, 0.607 / 0.606, 0.600 / 0.605, 0.593 / 0.596, 0.595 / 0.598
- N: 0.591 / 0.591, 0.594 / 0.597, 0.595 / 0.603
- F: 0.594 / 0.596, 0.592 / 0.600, 0.592 / 0.588, 0.582 / 0.582, 0.575 / 0.575

The largest difference is 0.008, against the criterion of 0.04. Arm means of the mixed-precision set: R
0.107, 0.316, 0.601, 0.975 on T0 (1e-5 to 5e-5); N 0.107, 0.315, 0.597, 0.971; F 0.107, 0.302, 0.588,
0.967; on T100 at SNR 5, 7, 10, R 0.415, 0.594, 0.717 and F 0.443, 0.676, 0.792; on T35, F 0.665, 0.838,
0.966. Every prediction scores the same way on that set. Mixed precision on the small card is a fair
stand-in for float32 here.

### Training curves (float32)

Validation detection at the middle level (image head, each arm on its own training distribution) is flat
from epoch 5 to 10 onward in every run. Arm F seed 1 at epochs 1, 5, 10, 15, 20, 25, 28, 29, 30: 0.191,
0.381, 0.386, 0.381, 0.397, 0.397, 0.395, 0.388, 0.388; training loss 1.893, 1.557, 1.541, 1.526, 1.514,
1.512, 1.509, 1.502, 1.497. The other four F seeds end between 0.381 and 0.395. Arm R ends between 0.619
and 0.697 with loss 1.287 to 1.297; arm N between 0.647 and 0.658 with loss 1.303 to 1.311. The loss is
still creeping down at epoch 30 while validation is not moving.

### What this says

- **On the ramp the network is at the ceiling.** It is within 0.014 of the filter bank and 0.024 of the
  marginal likelihood ratio at 3e-5, above the filter bank and at the likelihood ratio at 1e-5 and 2e-5,
  the same at every size, with 0.005 between seeds. By the rule stated in (B), nothing here counts
  against "a network reaches the ceiling".
- **ImageNet initialisation buys nothing** on these skies (0.006), so the random-initialised network is as
  good a starting point for the masked-sky work.
- **Training on the family costs 0.012 on the ramp and gains almost nothing on the step.** On T100 arm F
  equals the ramp bank (0.447, 0.673, 0.791 against 0.450, 0.671, 0.780) while the family bank, which uses
  the same four mixes the network was trained on, is at 0.651, 0.976, 1.000 and sits next to the matched
  bank. That is the case (B) named beforehand as counting against "a network handles shape freedom":
  predictions 7 and 8 fail while the family bank stays near the matched bank. With flat validation curves,
  longer training of this recipe is not the likely remedy.
- **So the measured picture is two-sided.** For the smooth template that theory supports after 2011, a
  network matches the best available test and nothing is left to gain. For a sharp edge, a plain filter
  bank over the family finds what this network does not.

Not established, and not to be read into the above: why arm F misses the edge. Three candidates, none
tested: the information of the edge sits at the smallest scales and the encoder halves the resolution in
its first layer; the mask target is the disc and not its rim; at a fixed ideal SNR a step is faint
everywhere but its edge, so the network may be learning the part it can see. Each needs its own run.

Small things seen and not explained: at ideal SNR 3 the networks find far more 5 degree steps than the
matched bank (0.42 to 0.48 against 0.23) and fewer at 20 degrees (0.06 against 0.13), with the pooled
numbers equal within noise; and the network is above the filter bank at 1e-5 and 2e-5 on T0 by 0.016 and
0.019. The saved per-patch scores allow a paired test of both.

### Limits

As stated in (B): Gaussian skies, white noise, no mask, centres within 5 degrees, one architecture, one
training recipe, the band-limited step, thresholds from the signal-free patches of each test set. Levels
have 1,000 signals on T0 and 667 on T35 and T100, so an unpaired difference of 0.02 between a network and
a reference is within sampling error; they are scored on the same signals, and the paired error has not
been computed. Predictions 1 and 4 were written as bands on point estimates without that error in mind,
which is why two of them miss by amounts smaller than it.

### Next, none of it started

1. The paired test from the saved scores (network against filter bank at 1e-5 and 2e-5; the size pattern
   on the step).
2. Why arm F misses the edge: the three candidates above, one run each.
3. A fourth test set with the template of the published code (edge at 0.2 of the centre, main notebook
   entries (aj), (ak)), which sits between T0 and T35.
4. Masks and real maps, from random initialisation.

## (M) 2026-11-23 15:20 CST. Pre-registration: the networks on the real sky, with the Planck mask and the published WMAP candidates

Written before any training or scan run of `scripts/ml/ml_real_sky.py` (its self-test and smoke checks
excepted). Job files: `cluster/hamilton/realsky_{smoke,train,scan}.sbatch`.

### Why

The ceiling study, entries (A) to (L), is on unmasked Gaussian skies. The Planck maps add a mask (77% of
the sky kept) and named positions to test: the WMAP candidates of Feeney et al. (2011) and McEwen et al.
(2012). Gus asked on 23 November whether the search catches their top candidates. The full-sky filter
search on main does (main notebook entries (x), (am)). This entry asks the same of a network.

The four features Feeney et al. (2011, Table VIII) single out (2, 3, 7 and 10; 2 is the Cold Spot) are
McEwen et al. (2012) nos. 7, 8, 4 and 6: the same positions within a degree, the same sign and amplitude.
What the filter search of main gives there on Planck SMICA (largest S within the match radius, from
`work/results_2026-11-22/fullsky_search_summary_planck_v1.json` on main; the Feeney et al. 2013 threshold is
4.00 from 3 to 8 degrees):

| spot | size, deg | cosine template, z_crit = 0 | transfer template |
|---|---:|---:|---:|
| M4 = F7 | 3 | 4.23 | 2.08 |
| M6 = F10 | 4 | 4.80 | 1.93 |
| M7 = F2, Cold Spot | 5 | 4.11 | 3.26 |
| M8 = F3 | 5 | 4.05 | 2.82 |

With the physical template the classical filter does not pick out three of the four, so a network trained
on that template is not expected to either. Hence two networks, one per template.

### Design (the script header lists every assumption)

- Two networks, random initialisation (as planned in (L)), seed 1, otherwise the recipe of arm N: T is
  trained on the transfer-function ramp, C on the cosine cap with z_crit = 0 (the template the text of
  McEwen et al. describes).
- Sizes 2 to 25 degrees in steps of 1. The ceiling study used 5 to 25; M4 and M6 are 3 and 4 degrees.
- Mask: PR3 common intensity mask at Nside 256, a pixel kept where the value is at least 0.9. Every sky
  and map is prepared the same way: monopole and dipole fitted on the kept pixels and removed, masked
  pixels set to zero. The mask is the second input channel.
- Training patches: centres on kept pixels, 8 per sky, at least 45 degrees apart; 4,000 training and 300
  validation skies; signals drawn as in the ceiling study and hidden where the patch is masked.
- Scan: tiles at the Nside 16 pixel centres that fall on kept pixels; the four PR3 maps (SMICA, NILC,
  SEVEM, Commander) and 400 Gaussian skies with the same mask and preparation.
- Tile scores: image-head logit (primary) and largest mask logit (secondary), each standardised per tile
  by its mean and standard deviation over the 400 simulated skies.
- Spot score at a position: the largest standardised tile score within 5 degrees. Its p value is its rank
  among the 400 simulated skies at the same position. A position is caught when p < 0.01. Positions:
  McEwen et al.'s 16, Feeney et al.'s 11, and 300 random kept positions.
- Whole-sky statistics per map: the largest standardised tile score, and the number of tiles whose own p
  is below 0.01 (for a simulated sky, against the other 399). Their p values come from the simulated skies.
- Real-sky check by injection: sizes 3, 5, 10 and 20 degrees at ideal SNR 4, 6 and 8, at 100 random kept
  positions with random sign, each network with its own template, added on the sphere and prepared like the
  map, on each Planck map and on 20 of the simulated skies; detected when the spot p is below 0.01.
- Unmasked check: network T on the ceiling study's T0 test set, with the mask channel all kept.

Data: the four PR3 maps and the common mask were downloaded from IRSA onto Hamilton
(`/projects/gmarcum/cmb/data`, 7.45 GB in all), because the project's copy sits in a private folder. Gus
approved the download on 23 November.

### Predictions

1. Network C catches M6 (= F10) on all four maps, and at least two of the four top spots (M4, M6, M7, M8)
   on every map.
2. Network T catches none of M4, M6 and M8 on any map (p above 0.05 each). No prediction for the Cold Spot.
3. No excess: for neither network and neither whole-sky statistic is p below 0.01 on all four maps (the
   rule of main entry (al)).
4. The networks work on the real sky as on simulated skies: pooled over the four maps, the detected
   fraction of injections is within 0.10 of the simulated-sky fraction in at least 10 of the 12 (size, SNR)
   cells, for each network.
5. The four maps agree: for each network the set of caught published positions is the same on all four
   maps, up to one position.
6. Unmasked check: network T detects 0.56 to 0.62 of T0 at 3e-5 (5% false positives, image head); arm N
   gave 0.593.

### What would change the reading

If prediction 4 fails for a network, its results on the Planck maps are reported as affected by the real
sky (foreground residuals, noise, mask edges) and are not set against the filter search.

### Limits fixed now

One seed per network. The simulated skies are Gaussian, without foreground residuals or anisotropic noise,
as in the filter search of main. Sizes above 25 degrees are not searched (M11 to M15 are 26 to 90 degrees).
Positions with no tile centre on a kept pixel within 5 degrees are not scored. In training a signal does
not change the fitted monopole and dipole; in the scan it does.
