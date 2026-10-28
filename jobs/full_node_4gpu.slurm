#!/bin/bash
# Full H100 node (4 GPUs), up to 3 days.
# Comfort mode turns --gres=gpu:4 into: 4 tasks, 22 CPUs/task, exclusive node, all memory.
# Account: submit_job adds the configured one (by hand: sbatch -A p<n>; the filesystem name fs<n> is rejected).
#SBATCH --job-name=full-node
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH -t 72:00:00
#SBATCH --threads-per-core=1   # physical cores only (docs: example_job_scripts.md)

# Lmod is only auto-initialised for interactive shells (~/.bashrc.d/asc-samrc
# checks $PS1), so batch jobs set it up explicitly.
source /cvmfs/software.eessi.io/versions/${EESSI_VERSION:-2025.06}/init/lmod/sh >/dev/null 2>&1
module use /cvmfs/software.asc.ac.at/init/modules
set -euo pipefail          # only after the Lmod init (it reads unset variables)

# Keep large downloads (models, datasets, caches) out of the 50 GB $HOME.
export HF_HOME=$DATA/hf_cache
export TORCH_HOME=$DATA/torch_cache
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
mkdir -p "$HF_HOME" "$TORCH_HOME" "$APPTAINER_CACHEDIR"

nvidia-smi -L

# --- the code to execute goes here ---
