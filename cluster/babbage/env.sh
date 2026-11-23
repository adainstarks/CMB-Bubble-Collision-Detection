# Shared settings for the Babbage jobs of the ML ceiling study. Source this file; do not run it.
export CMB_ROOT=/projects/$USER/cmb
export CMB_REPO=$CMB_ROOT/repo
export CMB_PY=/projects/$USER/envs/cmb/bin/python
# pretrained weights are fetched once on the login node; compute nodes have no internet
export TORCH_HOME=$CMB_ROOT/cache/torch
export HF_HOME=$CMB_ROOT/cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg   # healpy's plot functions (used only for the gnomview identity guard) on headless nodes
