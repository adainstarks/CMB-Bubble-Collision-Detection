#!/bin/bash
#SBATCH --job-name=cmb-train
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=c11,c14
#SBATCH --array=0-12%4
#SBATCH --output=/projects/%u/cmb/logs/train-%A_%a.out
# The thirteen training runs of notebook_ml_ceiling.md entry (B):
#   0-4  arm R  ramp only, ImageNet initialisation, seeds 1-5
#   5-7  arm N  ramp only, random initialisation,   seeds 1-3
#   8-12 arm F  family (mixes 0, 0.25, 0.5, 1), ImageNet initialisation, seeds 1-5
set -eo pipefail
source /projects/$USER/cmb/repo/cluster/babbage/env.sh
export HF_HUB_OFFLINE=1
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$CMB_REPO"
echo "commit $(cat .git/refs/heads/ml-ceiling-study 2>/dev/null || echo unknown)"   # compute nodes have no git
i=$SLURM_ARRAY_TASK_ID
if   [ "$i" -le 4 ]; then arm=R; init=imagenet; mixes=0;            seed=$((i + 1))
elif [ "$i" -le 7 ]; then arm=N; init=random;   mixes=0;            seed=$((i - 4))
else                      arm=F; init=imagenet; mixes=0,0.25,0.5,1; seed=$((i - 7))
fi
"$CMB_PY" scripts/ml/ml_ceiling_study.py train --out-dir "$CMB_ROOT/runs/ceiling_v1/${arm}_seed${seed}" \
  --test-dir "$CMB_ROOT/tests/T0" --test-dir "$CMB_ROOT/tests/T35" --test-dir "$CMB_ROOT/tests/T100" \
  --seed "$seed" --init "$init" --train-mixes "$mixes"
date
