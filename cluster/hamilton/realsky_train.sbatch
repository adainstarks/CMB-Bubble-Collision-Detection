#!/bin/bash
#SBATCH --job-name=cmb-realsky-train
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/projects/%u/cmb/logs/realsky-train-%A_%a.out
# The two training runs of notebook_ml_ceiling.md entry (M): 0 = template T (transfer-function ramp),
# 1 = template C (cosine cap, z_crit = 0); random initialisation, seed 1, sizes 2 to 25 deg, Planck mask.
set -eo pipefail
source /projects/$USER/cmb/repo/cluster/babbage/env.sh
export HF_HUB_OFFLINE=1
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$CMB_REPO"
echo "commit $(cat .git/refs/heads/ml-real-sky 2>/dev/null || echo unknown)"   # compute nodes have no git
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then arm=T; tests="--test-dir $CMB_ROOT/tests/T0"; else arm=C; tests=""; fi
"$CMB_PY" scripts/ml/ml_real_sky.py train --out-dir "$CMB_ROOT/runs/${RUNS_SUBDIR:-realsky_v1}/${arm}_seed1" \
  --data-dir "$CMB_ROOT/data" --template "$arm" --seed 1 --init random $tests
date
