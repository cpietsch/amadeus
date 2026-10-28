#!/bin/bash
# Pull an NGC image into $DATA/containers as a .sif (CPU job; SIF conversion is CPU-heavy).
# Usage: sbatch jobs/pull_container.slurm [image] [name]
#   default image: nvcr.io/nvidia/vllm:26.09-py3   (anonymous NGC pulls work)
#SBATCH --job-name=pull-container
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=zen4_0768
#SBATCH --qos=zen4_0768
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -t 01:00:00

set -euo pipefail

IMAGE=${1:-nvcr.io/nvidia/vllm:26.09-py3}
NAME=${2:-$(basename "${IMAGE/:/_}").sif}
OUT=$DATA/containers

# Keep Apptainer's layer cache and temp files off $HOME (50 GB quota).
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
# Unpack layers on node-local NVMe, not Weka $SCRATCH: many small files made a 7.5 GB vLLM image
# take 30 min there.
export APPTAINER_TMPDIR=$([ -d /local ] && echo /local/$USER/apptainer_tmp || echo $SCRATCH/apptainer_tmp)
mkdir -p "$OUT" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "Host: $(hostname)  image: $IMAGE  ->  $OUT/$NAME"
curl -sS -o /dev/null -w "internet check (nvcr.io): HTTP %{http_code}\n" --max-time 15 https://nvcr.io/v2/ || echo "no outbound internet from compute node"

time apptainer pull --force "$OUT/$NAME" "docker://$IMAGE"
ls -lh "$OUT/$NAME"
