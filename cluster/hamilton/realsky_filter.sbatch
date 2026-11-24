#!/bin/bash
#SBATCH --job-name=cmb-realsky-filter
#SBATCH --partition=research
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/%u/cmb/logs/realsky-filter-%j.out
# The filter scan of notebook_ml_ceiling.md entry (P): the matched-filter banks of both templates through the
# tiles, simulated skies, positions, injections and statistics of the network scan of entry (M). CPU only.
# SMOKE=1 runs the few-minute check instead, into runs/realsky_smoke/filter.
set -eo pipefail
source /projects/$USER/cmb/repo/cluster/babbage/env.sh
hostname; date; nproc
cd "$CMB_REPO"
echo "commit $(cat .git/refs/heads/ml-real-sky 2>/dev/null || echo unknown)"   # compute nodes have no git
if [ "${SMOKE:-0}" = "1" ]; then
  "$CMB_PY" scripts/ml/ml_real_sky.py filter-scan --smoke --out-dir "$CMB_ROOT/runs/realsky_smoke/filter" --data-dir "$CMB_ROOT/data" --maps smica
else
  "$CMB_PY" scripts/ml/ml_real_sky.py filter-scan --out-dir "$CMB_ROOT/runs/${RUNS_SUBDIR:-realsky_v1}/filter_scan" --data-dir "$CMB_ROOT/data"
fi
date
