#!/bin/bash
# Quick GPU-node probe (dev QoS, ~1 min): outbound internet, IB interfaces, /local, apptainer, GPUs.
#SBATCH --job-name=probe-gpu-node
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=dev_zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH --threads-per-core=1
#SBATCH --time=00:03:00

echo "host: $(hostname)  cpus: $(nproc)  mem: $(free -g | awk '/Mem/{print $2}') GB"
echo "--- internet"
for u in https://huggingface.co https://cas-bridge.xethub.hf.co https://registry-1.docker.io/v2/ https://nvcr.io/v2/ https://pypi.org/simple/; do
    curl -sS -o /dev/null -m 10 -w "%{http_code} %{time_total}s  $u\n" "$u" 2>&1 | head -1
done
echo "--- download speed sample (100 MB from HF CDN)"
curl -sS -L -o /dev/null -m 30 -r 0-104857599 -w "%{size_download} bytes in %{time_total}s = %{speed_download} B/s\n" \
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/resolve/main/model-00001-of-00131.safetensors" 2>&1 | tail -1
echo "--- network interfaces"
ip -br addr | grep -vE '^lo|veth'
echo "--- /local, /dev/shm"
df -h /local /dev/shm 2>&1 | tail -2
echo "--- tools"
apptainer --version; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi topo -m 2>/dev/null | head -12
