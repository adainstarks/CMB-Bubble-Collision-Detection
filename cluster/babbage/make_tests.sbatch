#!/bin/bash
#SBATCH --job-name=cmb-tests
#SBATCH --partition=research-cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/projects/%u/cmb/logs/tests-%A_%a.out
# The three fixed test sets of notebook_ml_ceiling.md entry (B), with their classical references.
set -eo pipefail
source /projects/$USER/cmb/repo/cluster/babbage/env.sh
hostname; date
cd "$CMB_REPO"
case "$SLURM_ARRAY_TASK_ID" in
  0) "$CMB_PY" scripts/ml/ml_ceiling_study.py make-test --out-dir "$CMB_ROOT/tests/T0"   --mix 0    --n-skies 1500 --seed 20261201 ;;
  1) "$CMB_PY" scripts/ml/ml_ceiling_study.py make-test --out-dir "$CMB_ROOT/tests/T35"  --mix 0.35 --n-skies 1000 --seed 20261202 --target-snr 1,2,3,5,7,10 ;;
  2) "$CMB_PY" scripts/ml/ml_ceiling_study.py make-test --out-dir "$CMB_ROOT/tests/T100" --mix 1.0  --n-skies 1000 --seed 20261203 --target-snr 1,2,3,5,7,10 ;;
esac
date
