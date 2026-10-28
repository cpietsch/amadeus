#!/bin/bash
# Smoke-test job for the ASC MUSICA cluster (1x H100, devel QoS, 5 min).
# Uses ASC "comfort mode" (default): --gres=gpu:1 gives 1 task, 22 CPUs,
# 25% of node memory on a shared H100 node. See docs-vsc/docs/musica/slurm_modes.md
# Account: submit_job adds the configured one; check yours with
#   sacctmgr show user `id -u` withassoc format=user,defaultaccount,account,qos%40s
#SBATCH --job-name=hello-musica
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=dev_zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH --threads-per-core=1

# Lmod is only auto-initialised for interactive shells (~/.bashrc.d/asc-samrc
# checks $PS1), so batch jobs set it up explicitly.
source /cvmfs/software.eessi.io/versions/${EESSI_VERSION:-2025.06}/init/lmod/sh >/dev/null 2>&1
module use /cvmfs/software.asc.ac.at/init/modules
set -euo pipefail          # only after the Lmod init (it reads unset variables)

echo "=== Job info ==="
echo "Job ID:      ${SLURM_JOB_ID:-n/a}"
echo "Node list:   ${SLURM_JOB_NODELIST:-n/a}"
echo "Host:        $(hostname)"
echo "Date:        $(date -Is)"
echo "Working dir: $(pwd)"

echo
echo "=== CPU / memory ==="
lscpu | grep -E 'Model name|^CPU\(s\)|Socket|NUMA node\(s\)' || true
free -h || true

echo
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "nvidia-smi not found"
fi

echo
echo "=== Modules ==="
module list 2>&1 || echo "module command unavailable"

echo
echo "=== Python / PyTorch (if available) ==="
python3 - <<'EOF' || true
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        x = torch.randn(4096, 4096, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("matmul ok, checksum:", float(y.sum()))
except ImportError:
    print("torch not installed in this environment")
EOF

echo
echo "=== Done ==="
