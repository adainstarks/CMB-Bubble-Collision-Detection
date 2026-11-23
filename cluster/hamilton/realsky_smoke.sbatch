#!/bin/bash
#SBATCH --job-name=cmb-realsky-smoke
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/%u/cmb/logs/realsky-smoke-%j.out
# Checks of notebook_ml_ceiling.md entry (M) before the real runs: self-test, both templates trained for one
# tiny epoch, and a scan with six simulated skies (and the Planck maps listed in MAPS, if any).
set -eo pipefail
source /projects/$USER/cmb/repo/cluster/babbage/env.sh
export HF_HUB_OFFLINE=1
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$CMB_REPO"
echo "commit $(cat .git/refs/heads/ml-real-sky 2>/dev/null || echo unknown)"   # compute nodes have no git
S=$CMB_ROOT/runs/realsky_smoke
"$CMB_PY" scripts/ml/ml_real_sky.py selftest --data-dir "$CMB_ROOT/data" --out-json "$S/selftest.json"
for arm in T C; do
  "$CMB_PY" scripts/ml/ml_real_sky.py train --smoke --out-dir "$S/$arm" --data-dir "$CMB_ROOT/data" --template $arm --seed 1
done
"$CMB_PY" scripts/ml/ml_real_sky.py scan --smoke --out-dir "$S/scan" --data-dir "$CMB_ROOT/data" --maps "${MAPS:-}" \
  --model T:$S/T/checkpoint_last.pt --model C:$S/C/checkpoint_last.pt
date
