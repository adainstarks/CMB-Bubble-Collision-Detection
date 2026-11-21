# Pure ML Detector V1

## Assumptions

- Bubble-collision screening is treated as a fixed-false-positive-rate detection
  problem, not as Bayesian evidence or a cosmological posterior.
- The deployed score in this experiment is neural-only. Wiener, SMHW, and
  circular-template scores are not input channels and are not fused into the
  decision statistic.
- Classical methods may still be used as external comparators after scoring.
- Training and validation use Kelvin-scale CMB anisotropy patches. The trainer
  rejects non-finite patches and patches outside a physically plausible Kelvin
  scale.
- Float32 tensors are used for network training; score aggregation and
  threshold selection use float64 where practical.

## Goal

This folder isolates the attempt to make a pure learned detector beat the
current non-Bayesian mathematical screeners. If the track fails, this directory
and its output directory can be removed without touching the main remediated
pipeline.

The first experiment is intentionally detection-first rather than
segmentation-first:

- train on raw CMB map patches only;
- include optional real-map null controls from SMICA/NILC/SEVEM/Commander;
- optimize a neural image-level score directly;
- select checkpoints by recall at fixed FPR;
- report AUROC, AUPRC, and recall at FPR `0.05` and `0.08`.

Implemented architecture choices:

- `small_cnn`: custom residual CNN baseline.
- `resnet18`, `resnet34`: residual detector baselines.
- `resnet50`, `wide_resnet50_2`: the first serious residual backbones for
  this track. They are stronger, but not automatically better; they can overfit
  or learn nuisance structure if the data contract is weak.
- `resunet`: localization-aware residual U-Net-style detector that still emits
  one scalar screening score. This is the first serious next architecture after
  the plain ResNet50 result, because it keeps high-resolution evidence available
  to the score head.
- `convnextv2_small`, `convnextv2_base`: stronger ConvNeXt V2 candidates via
  `timm`. These are serious model swaps, not the main smoke path. `base` is the
  preferred run if memory permits.
- `efficientnet_v2_s`: efficient strong-image-backbone control via
  torchvision. It is useful for diversity, but not assumed to be better matched
  to bubble localization than `resunet`.
- `convnext_tiny`, `efficientnet_b0`: code-path and ablation controls only.
  They are not the main attempt to beat Wiener.

Pure-ML multi-model ensembles are allowed in this folder if every component
checkpoint uses raw patch pixels only. The ensemble scorer supports `mean_logit`,
`mean_prob`, and `max_prob`; none of these may include a classical score.

## Why This Is Different From The Current U-Net Baseline

The scalar detector path trains a neural likelihood-ratio-style detector
directly. It is useful because it tests whether a pure image classifier can beat
the classical screeners without any segmentation/component post-processing.

The `imagenet_b64_aux` result showed that the strongest pure-ML baseline in the
repo is not a plain image classifier: it is an EfficientNet-B0 ImageNet-backed
U-Net trained with dense mask supervision plus an auxiliary image-presence head.
That means the next serious pure-ML attempt should keep the neural-only
contract, but improve the segmentation+aux architecture and training discipline
instead of only trying larger classifier backbones.

## Experiment Ladder

1. `smoke`: proves the isolated trainer works and produces a calibrated
   fixed-FPR report. This is not a science result.
2. `synthetic_detector`: pure ML classifier on remediated synthetic
   train/validation splits.
3. `realnull_detector`: same model, but negatives include real-map null controls
   across all four cleaned maps.
4. `ranked_detector`: add pairwise ranking pressure so positives outrank nulls
   near the FPR operating point.
5. `hard_null_detector`: upweight false positives from the current ML and Wiener
   audits, while still using no classical score as an input.
6. `multiscale_detector`: add a larger-context raw-map patch or surrounding
   annulus as a second learned image input.
7. `sphere_native_detector`: replace flat-patch CNN with a HEALPix/spherical
   neighborhood model if flat projections remain the limiting factor.

Only steps 1-4 are implemented in the first trainer. Steps 5-7 are intentionally
kept here as pure-ML follow-up designs rather than mixed into the production
scripts.

## Smoke Test

```bash
python experiments/pure_ml_detector_v1/train_pure_ml_detector.py \
  --experiment-name smoke \
  --epochs 1 \
  --max-train-positives 64 \
  --max-val-positives 32 \
  --synthetic-negatives-per-positive 1 \
  --real-null-negatives-per-map 16 \
  --batch-size 16 \
  --num-workers 0 \
  --device cuda \
  --cache-patches
```

Outputs are written under:

```text
experiments/pure_ml_detector_v1/runs/<experiment-name>/
```

## Same-Grid Scoring

Score a trained pure detector on the fixed same-grid benchmark:

```bash
python experiments/pure_ml_detector_v1/score_pure_ml_detector.py \
  --checkpoint experiments/pure_ml_detector_v1/runs/<experiment-name>/best_checkpoint.pt \
  --batch-size 256 \
  --num-workers 0 \
  --device cuda \
  --cache-patches
```

The scorer writes `score_report.json`, `scores.npz`, and
`cell_recall_rows.csv` under the experiment run directory.

## Training Monitor

Training writes both completed-epoch metrics and live step progress:

- `live_metrics.jsonl`: one row per completed epoch with validation AUROC,
  AUPRC, recall, precision, threshold, confusion counts, train loss, validation
  loss, focal/BCE loss, and ranking loss.
- `progress.json`: latest train/validation/scoring step progress for terminal
  monitoring.
- `progress.jsonl`: append-only step progress history.

Monitor a run from another terminal:

```bash
python experiments/pure_ml_detector_v1/monitor_training.py \
  experiments/pure_ml_detector_v1/runs/<experiment-name> \
  --watch 10 \
  --lines 12
```

The trainer defaults to `--progress-interval 25`. Use a smaller interval for
short smoke tests and a larger interval if terminal logging becomes noisy.

Pure-neural ensemble scoring:

```bash
python experiments/pure_ml_detector_v1/score_pure_ml_ensemble.py \
  --checkpoint experiments/pure_ml_detector_v1/runs/<run-a>/best_checkpoint.pt \
  --checkpoint experiments/pure_ml_detector_v1/runs/<run-b>/best_checkpoint.pt \
  --aggregation mean_logit \
  --batch-size 256 \
  --num-workers 0 \
  --device cuda \
  --cache-patches
```

## Preflight Audit

Run the audit before treating any result as meaningful:

```bash
python experiments/pure_ml_detector_v1/preflight_pure_ml.py
```

Current preflight facts:

- Blocking row-split overlap in `training_data.h5`: none.
- `validation_idx` and `calibration_idx` overlap by `2000` rows by existing
  dataset contract; do not report validation and calibration as independent.
- `features/wiener_feeney_response` exists in the synthetic HDF5, but this track
  explicitly forbids using it as a deployed input.
- Real null-control files for SMICA/NILC/SEVEM/Commander contain zero positive
  labels.

## First Pilot Result

The first non-claim pilot was:

```bash
python experiments/pure_ml_detector_v1/train_pure_ml_detector.py \
  --experiment-name pilot_realnull_ranked \
  --epochs 3 \
  --max-train-positives 512 \
  --max-val-positives 128 \
  --synthetic-negatives-per-positive 1 \
  --real-null-negatives-per-map 128 \
  --batch-size 32 \
  --num-workers 0 \
  --device cuda \
  --base-channels 24 \
  --ranking-loss-weight 0.2 \
  --focal-gamma 1.0 \
  --augment-dihedral \
  --cache-patches \
  --normalization-samples 1024 \
  --save-scores
```

Internal tiny validation:

- AUROC: `0.7763`
- recall at FPR `0.05`: `0.3672`

Same-grid scoring of that pilot:

- mean amplitude-radius cell recall: `0.2717`
- AUROC: `0.6493`
- recall at FPR `0.05`: `0.2717`
- recall at FPR `0.08`: `0.3141`

This pilot does **not** beat the current true Wiener/Feeney mean same-grid
recall (`0.3684` in the current same-grid artifact). It is a fast feasibility check showing that a detection-first
pure-CNN track is executable and produces non-degenerate scores. It is not a
finished pure-ML attempt.

## Current ResNet50 Result

The dual-GPU ResNet50 resume run plateaued internally and was stopped early:

- best internal validation recall at FPR `0.05`: `0.6040`
- same-grid mean amplitude-radius recall at FPR `0.05`: `0.3231`
- same-grid AUROC: `0.6679`

This improves over the small pilot but does **not** beat the true
Wiener/Feeney same-grid mean recall (`0.3684` in the current same-grid artifact). The result is evidence against
spending major GPU time on plain image-classifier variants as the primary path.
The next serious pure-ML attempt should use `resunet` and then ConvNeXt V2.

The ResUNet scalar-detector run was worse than ResNet50 on the same grid
(`0.2853` mean recall), but the complementarity audit found real positives that
ResNet50 missed. The raw ResNet50+ResUNet union reached `0.3548` recall at an
unacceptable `0.079` FPR, so ResUNet is not a naive winning ensemble member.

ResNet50+`imagenet_b64_aux` had more useful complementarity: the raw union
reached `0.3883` recall, above the current Wiener same-grid mean, but with
`0.0882` FPR. Naive mean-logit/mean-prob ensembles did not beat ImageNet alone.
That points to a learned pure-neural gating/segmentation route, not a blind
average of classifier scores.

## ImageNet Baseline Autopsy

Local checkpoint metadata for
`runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux/best_checkpoint.pt`:

- architecture: U-Net with `efficientnet-b0` encoder and ImageNet weights;
- supervision: dense bubble mask + auxiliary image-presence head;
- loss: BCE + Dice, with auxiliary BCE weight `0.2`;
- optimizer setup: AdamW, learning rate `3e-4`, weight decay `1e-4`, gradient
  clipping `1.0`;
- batch size: `64` over GPUs `0,1`;
- split: predefined remediated-v1, `16000` train and `2000` calibration samples;
- best checkpoint: epoch `11`, selected by positive-sample hard Dice.

The key lesson is that ImageNet’s advantage probably came from representation
reuse plus dense localization supervision, not from being a larger generic
classifier. ConvNeXtV2 should therefore be tested as a stronger neural
segmentation+aux model before spending more time on scalar-only classifier
variants.

## ConvNeXtV2 Segmentation+Aux Track

`scripts/ml/phase3_train_unet.py` now supports `--model-architecture timm_unet`,
which builds a U-Net decoder on a `timm` feature encoder. This keeps the deployed
score pure ML: raw CMB patch pixels in, neural mask/aux scores out. No Wiener,
SMHW, or circular-template response is used as an input channel.

Validated code paths:

- `convnextv2_small` and `convnextv2_base` forward passes at `256 x 256`;
- dual-GPU `convnextv2_base` training smoke with AMP;
- gradient accumulation via `--grad-accum-steps`;
- Phase 3 evaluation loading and threshold sweeps;
- same-grid sensitivity scorer supports `--gpu-ids 0,1`;
- shared monitor supports both scalar detector runs and Phase 3 segmentation
  `history.json`.

First serious run:

```bash
python scripts/ml/phase3_train_unet.py \
  --run-name remediated_v1_timm_unet_convnextv2_base_aux \
  --model-architecture timm_unet \
  --encoder-name convnextv2_base \
  --encoder-weights imagenet \
  --epochs 36 \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --num-workers 8 \
  --device cuda \
  --gpu-ids 0,1 \
  --bce-weight 1.0 \
  --dice-weight 1.0 \
  --aux-head-weight 0.2 \
  --aux-head-dropout 0.2 \
  --checkpoint-metric hard_dice_pos \
  --grad-clip 1.0 \
  --cache-data
```

If pretrained `convnextv2_base` weights are unavailable from `timm` on this
machine, rerun the same command with `--encoder-weights none` and treat it as a
from-scratch control, not as the final attempt.

Monitor from another terminal:

```bash
python experiments/pure_ml_detector_v1/monitor_training.py \
  runs/phase3_unet/remediated_v1_timm_unet_convnextv2_base_aux \
  --watch 10 \
  --lines 12
```

Evaluate the calibration split with the same score families used by the
ImageNet run:

```bash
python scripts/ml/phase3_evaluate_run.py \
  --run-dir runs/phase3_unet/remediated_v1_timm_unet_convnextv2_base_aux \
  --checkpoint best \
  --split calibration \
  --device cuda \
  --gpu-ids 0,1 \
  --batch-size 8 \
  --num-workers 4 \
  --threshold-count 60 \
  --target-fpr 0.05 \
  --score-mode component_score
```

Same-grid scoring after training:

```bash
mkdir -p runs/phase3_unet/remediated_v1_timm_unet_convnextv2_base_aux/sensitivity_curve
ln -sf ../../remediated_v1_sensitivity_curve/sensitivity_data.h5 \
  runs/phase3_unet/remediated_v1_timm_unet_convnextv2_base_aux/sensitivity_curve/sensitivity_data.h5

python scripts/analysis/phase3_sensitivity_curve.py \
  --output-dir runs/phase3_unet/remediated_v1_timm_unet_convnextv2_base_aux/sensitivity_curve \
  --skip-existing-data \
  --model convnextv2_base_aux:runs/phase3_unet/remediated_v1_timm_unet_convnextv2_base_aux:best \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda \
  --gpu-ids 0,1 \
  --cache-ml-data
```

## Full Test To Run Next

The first serious runs should scale to the full remediated split and train a
strong architecture family. Tiny/B0 models are smoke tests only. The target is
not to make one convenient model look good; it is to find whether any honest
pure-neural score family can beat Wiener on the same-grid benchmark.

Run 1: ResNet50 residual detector.

```bash
python experiments/pure_ml_detector_v1/train_pure_ml_detector.py \
  --experiment-name resnet50_realnull_full_seed42 \
  --architecture resnet50 \
  --epochs 20 \
  --synthetic-negatives-per-positive 1 \
  --real-null-negatives-per-map 4000 \
  --batch-size 48 \
  --num-workers 0 \
  --device cuda \
  --ranking-loss-weight 0.2 \
  --focal-gamma 1.0 \
  --augment-dihedral \
  --cache-patches \
  --amp \
  --save-scores
```

Run 1b: wider ResNet50 stress test if memory allows.

```bash
python experiments/pure_ml_detector_v1/train_pure_ml_detector.py \
  --experiment-name wide_resnet50_2_realnull_full_seed45 \
  --architecture wide_resnet50_2 \
  --epochs 20 \
  --synthetic-negatives-per-positive 1 \
  --real-null-negatives-per-map 4000 \
  --batch-size 32 \
  --num-workers 0 \
  --device cuda \
  --ranking-loss-weight 0.2 \
  --focal-gamma 1.0 \
  --augment-dihedral \
  --cache-patches \
  --amp \
  --seed 45 \
  --save-scores
```

Run 2: ConvNeXt detector.

```bash
python experiments/pure_ml_detector_v1/train_pure_ml_detector.py \
  --experiment-name convnextv2_base_realnull_full_seed43 \
  --architecture convnextv2_base \
  --epochs 20 \
  --synthetic-negatives-per-positive 1 \
  --real-null-negatives-per-map 4000 \
  --batch-size 24 \
  --num-workers 0 \
  --device cuda \
  --gpu-ids 0,1 \
  --ranking-loss-weight 0.2 \
  --focal-gamma 1.0 \
  --augment-dihedral \
  --cache-patches \
  --amp \
  --seed 43 \
  --save-scores
```

Run 3: ResUNet localization-aware detector. This is the preferred next run
after the ResNet50 miss because it changes the inductive bias without changing
the benchmark contract.

```bash
python experiments/pure_ml_detector_v1/train_pure_ml_detector.py \
  --experiment-name resunet_b48_realnull_full_seed46 \
  --architecture resunet \
  --base-channels 48 \
  --epochs 24 \
  --synthetic-negatives-per-positive 1 \
  --real-null-negatives-per-map 4000 \
  --batch-size 64 \
  --num-workers 0 \
  --device cuda \
  --gpu-ids 0,1 \
  --ranking-loss-weight 0.2 \
  --focal-gamma 1.0 \
  --augment-dihedral \
  --cache-patches \
  --amp \
  --seed 46 \
  --save-scores
```

Run 4: EfficientNetV2 control.

```bash
python experiments/pure_ml_detector_v1/train_pure_ml_detector.py \
  --experiment-name efficientnet_v2_s_realnull_full_seed44 \
  --architecture efficientnet_v2_s \
  --epochs 20 \
  --synthetic-negatives-per-positive 1 \
  --real-null-negatives-per-map 4000 \
  --batch-size 48 \
  --num-workers 0 \
  --device cuda \
  --gpu-ids 0,1 \
  --ranking-loss-weight 0.2 \
  --focal-gamma 1.0 \
  --augment-dihedral \
  --cache-patches \
  --amp \
  --seed 44 \
  --save-scores
```

After each run, score it on same-grid. After at least two credible checkpoints,
score the pure-neural ensemble. If this family still cannot approach Wiener, the
next pure-ML move is not another short hyperparameter sweep; it is a larger-
context or sphere-native detector.

## Promotion Gate

A real pure-ML candidate must beat the true Wiener/Feeney matched-filter
screening benchmark under matched conditions:

- same-grid mean recall above the current true Wiener/Feeney artifact
  (`0.3684`) at realized FPR about `0.05`;
- no unacceptable held-out real-null FPR inflation;
- full-sky tile burden at or below the current deployment rule;
- no use of Wiener/SMHW/circular-template scores as deployed inputs.

If it only beats the circular-template screen, it is useful but not the goal.
