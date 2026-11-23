#!/bin/bash
#SBATCH --job-name=cmb-realsky-scan
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/projects/%u/cmb/logs/realsky-scan-%j.out
# The scan of notebook_ml_ceiling.md entry (M): both networks on the four Planck maps and on 400 simulated
# skies with the same mask and preparation; published positions, random positions and injections.
set -eo pipefail
source /projects/$USER/cmb/repo/cluster/babbage/env.sh
export HF_HUB_OFFLINE=1
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$CMB_REPO"
echo "commit $(cat .git/refs/heads/ml-real-sky 2>/dev/null || echo unknown)"   # compute nodes have no git
R=$CMB_ROOT/runs/${RUNS_SUBDIR:-realsky_v1}
"$CMB_PY" scripts/ml/ml_real_sky.py scan --out-dir "$R/scan" --data-dir "$CMB_ROOT/data" \
  --model T:$R/T_seed1/checkpoint_last.pt --model C:$R/C_seed1/checkpoint_last.pt
date
