# Historical Nside=512 Probe Decision

This note is retained for provenance only. It predates the remediated-v1
transfer-function, observing-model, normalization, and calibration cleanup.
Do not treat it as a final statement that `Nside=512` is physically
uninformative.

The focused Nside=512 probe does not justify a full 512 retrain.

## Setup

- Training data: `data/training_v4_nside512_probe/training_data.h5`
- Training samples: 1000 total, 500 positive / 500 negative
- Geometry: `nside=512`, `512x512`, `6.5` arcmin/pixel
- CAMB realizations: 64
- `lmax`: 1536
- Resume checkpoint: `runs/phase3_unet/phase3_v6_aux_only_w4/best_checkpoint.pt`
- Architecture/loss: v6 auxiliary-head setup, boundary weight 4.0, auxiliary weight 0.1
- Focused eval cell: real-SMICA backgrounds, `A=2e-5`, `theta_crit=5 deg`, 200 positives
- Null sanity check: real-SMICA backgrounds, 500 negatives

## Result

Positive-only recall was misleading. The high-recall 512 checkpoints saturated on real-SMICA nulls.

| checkpoint | positives at threshold 0.80 | nulls at threshold 0.80 |
|---|---:|---:|
| image-F1 best | 200 / 200 | 500 / 500 |
| image-F1 last | 42 / 200 | 98 / 500 |
| Dice best | 200 / 200 | 500 / 500 |

Matched-FPR score separation was also weak.

| checkpoint | AUROC | recall at matched FPR 0.10 |
|---|---:|---:|
| image-F1 best | 0.478 | 0.135 |
| image-F1 last | 0.515 | 0.110 |
| Dice best | 0.553 | 0.145 |

## Decision

Do not commit to a full Nside=512 retrain from this probe. Continue with the Nside=256 v1 tradeoff curve unless a later 512 setup changes the data volume, calibration strategy, or objective enough to demonstrate real positive/null separation.
