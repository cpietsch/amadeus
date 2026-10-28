#!/bin/bash
# Start ONE vLLM server process on the node this runs on. Launched per node with srun from a job
# script (jobs/vllm_multinode.slurm). Inputs via environment:
#   SIF MODEL NAME KEY MTP     container, staged weights, served name, API key, MTP on/off (default 0)
#   VLLM_ARGS                  model-specific engine flags (e.g. --moe-backend triton), all roles
#   VLLM_API_ARGS              API-server flags (tool and reasoning parsers), replica and dp-head only
#   VLLM_* env (e.g. VLLM_PLE_CPU_OFFLOAD=1 for Qwen3.8 on the v0.30.0 image): set it in the job's
#                              environment (sbatch --export=ALL,...); srun and apptainer pass it on
#   ROLE                       replica   | independent server, API on all interfaces (clients use
#                                        | the hostname, i.e. Ethernet: see the note at `case`)
#                              dp-head   | first DP_LOCAL DP ranks + API server on 127.0.0.1
#                              dp-worker | DP ranks of this node, headless (engines only)
#   TP                         tensor-parallel size per engine (default 4 = whole node)
#   DP_SIZE DP_LOCAL           DP roles: total DP ranks, ranks on this node (default 2 and 1);
#                              this node's first rank = node index x DP_LOCAL
#   HEAD_IP DP_PORT            DP roles: ibs8 IP of the head node, DP RPC port
#   EP                         DP roles, default 1: --enable-expert-parallel. Needed for block-FP8 MoE
#                              checkpoints: without EP, vLLM shards each expert over DP x TP GPUs, and the
#                              shards may not divide into the 128-wide FP8 blocks (Qwen3.8).
#                              With TP > 1, EP + DP turns on sequence-parallel MoE, which some models'
#                              layers don't support (Qwen3.8's hyper-connections): then TP=1.
set -uo pipefail
IP=$(ip -4 -o addr show ibs8 | awk '{print $4}' | cut -d/ -f1)

# Node index in the job (SLURM_NODEID is relative to the step, i.e. 0 in every 1-node step).
IDX=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -nx "$(hostname -s)" | cut -d: -f1)
IDX=$(( ${IDX:-1} - 1 ))
# Compile caches: replicas share one warm cache per config (identical config, read-only hits).
# DP ranks compile a new config: one cache per node so two nodes never write the same directories.
SUFFIX=""
case $ROLE in dp-*) [ "$IDX" = 0 ] || SUFFIX="-n$IDX" ;; esac
export XDG_CACHE_HOME=$SCRATCH/cache$SUFFIX VLLM_CACHE_ROOT=$SCRATCH/cache$SUFFIX/vllm TRITON_CACHE_DIR=$SCRATCH/cache$SUFFIX/triton
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 VLLM_NO_USAGE_STATS=1
# Cross-node traffic (DP coordination, NCCL bootstrap) on InfiniBand, not the 10.77 Ethernet.
export VLLM_HOST_IP=$IP GLOO_SOCKET_IFNAME=ibs8 NCCL_SOCKET_IFNAME=ibs8
# mlx5_0 is a 25 Gb/s RoCE (Ethernet) port; NCCL should only use the four 200 Gb/s NDR adapters.
export NCCL_IB_HCA='^=mlx5_0'
mkdir -p "$XDG_CACHE_HOME"

TP=${TP:-4}; DP_SIZE=${DP_SIZE:-2}; DP_LOCAL=${DP_LOCAL:-1}
read -ra EXTRA <<< "${VLLM_ARGS:-}"; read -ra API_EXTRA <<< "${VLLM_API_ARGS:-}"
ENGINE=(--served-model-name "$NAME" --tensor-parallel-size "$TP"
        --gpu-memory-utilization 0.85 --max-num-seqs 256 --enable-prefix-caching
        --no-enable-flashinfer-autotune "${EXTRA[@]}")
[ "${MTP:-0}" = 1 ] && ENGINE+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
api() { echo --host "$1" --port 8000 --api-key "$KEY" "${API_EXTRA[@]}"; }
DP=(--data-parallel-size "$DP_SIZE" --data-parallel-size-local "$DP_LOCAL"
    --data-parallel-address "${HEAD_IP:-}" --data-parallel-rpc-port "${DP_PORT:-13345}")
[ "${EP:-1}" = 1 ] && DP+=(--enable-expert-parallel)

# Replicas only carry HTTP, which doesn't need InfiniBand. Their API used to listen on the ibs8 IP
# only, and one node's ibs8 IP was once unreachable from the others (all four IB ports
# share 10.79/16, so the route a peer picks varies by node).
case $ROLE in
    replica)   ARGS=("${ENGINE[@]}" $(api 0.0.0.0)) ;;
    dp-head)   ARGS=("${ENGINE[@]}" $(api 127.0.0.1) "${DP[@]}") ;;
    dp-worker) ARGS=("${ENGINE[@]}" "${DP[@]}" --headless --data-parallel-start-rank $(( IDX * DP_LOCAL ))) ;;
    *) echo "unknown ROLE=$ROLE"; exit 2 ;;
esac
case $ROLE in dp-*) export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET ;; esac   # log IB vs socket transport

echo "$(date +%T) $(hostname) node=$IDX role=$ROLE ip=$IP cpus=$(nproc) gpus=${CUDA_VISIBLE_DEVICES:-?} tp=$TP dp=$DP_SIZE/$DP_LOCAL"
exec apptainer exec --nv -B "$DATA,$SCRATCH,/local" "$SIF" vllm serve "$MODEL" "${ARGS[@]}"
