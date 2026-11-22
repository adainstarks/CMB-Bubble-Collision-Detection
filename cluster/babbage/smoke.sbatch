#!/bin/bash
#SBATCH --job-name=cmb-smoke
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --exclude=c11,c14
#SBATCH --output=/projects/%u/cmb/logs/smoke-%j.out
# Three-minute check of the whole path on a compute node before anything long is queued:
# the environment's python starts, the GPU is visible, healpy and CAMB work, the pretrained weights load
# from the cache without internet, a tiny test set is made and a tiny network is trained and scored.
set -eo pipefail
source /projects/$USER/cmb/repo/cluster/babbage/env.sh
export HF_HUB_OFFLINE=1
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv
"$CMB_PY" -c "import sys, torch, healpy, camb, segmentation_models_pytorch as smp; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'healpy', healpy.__version__, 'camb', camb.__version__, 'smp', smp.__version__)"
cd "$CMB_REPO"
echo "commit $(cat .git/refs/heads/ml-ceiling-study 2>/dev/null || echo unknown)"   # compute nodes have no git
OUT=$CMB_ROOT/runs/smoke_$SLURM_JOB_ID
"$CMB_PY" scripts/ml/ml_ceiling_study.py make-test --out-dir "$OUT/test" --smoke
"$CMB_PY" scripts/ml/ml_ceiling_study.py make-test --out-dir "$OUT/test_mix" --mix 1.0 --target-snr 3,8 --smoke
"$CMB_PY" scripts/ml/ml_ceiling_study.py train --out-dir "$OUT/run_imagenet" --test-dir "$OUT/test" --test-dir "$OUT/test_mix" --seed 1 --train-mixes 0,1 --smoke --batch-size "${SMOKE_BATCH:-64}" ${SMOKE_TRAIN_FLAGS:-}
"$CMB_PY" scripts/ml/ml_ceiling_study.py summarize --runs-dir "$OUT"
date; echo SMOKE_OK
