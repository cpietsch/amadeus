#!/bin/bash
#ASC --vanilla
# vLLM on N whole H100 nodes (N from `sbatch -N`, default 4) with a benchmark sweep: the generic multi-node
# pattern, tested up to 16 nodes (2026-10, Qwen3.8-Flash-Next-FP8 and others). Workload: random 1024 in / 512 out,
# --ignore-eos. One server per node is started by jobs/lib/vllm_node.sh.
# MODES, run in order (default "replicas"):
#   replicas  one independent TP4 server per node, measured twice on the same servers:
#             split   N benchmark clients, client i -> node i (pure server capacity)
#             router  the same N clients all -> one vllm-router (round robin) on node 0, i.e. a
#                     single endpoint; the difference to "split" is what the router costs (skipped
#                     when $ROUTER does not exist)
#   dep       ONE vLLM instance over all nodes: DP = 4N ranks with TP=1 and expert parallelism over
#             4N GPUs, for MoE models (some models' vLLM code cannot do TP>1 + EP + DP; see jobs/lib/vllm_node.sh).
# Model (environment, via sbatch --export=ALL,...):
#   SRC        the weights in $DATA (required), e.g. $DATA/models/<repo name>; staged to /local on every node
#   NAME       served model name (default: the directory name of SRC)
#   SIF        vLLM container (default $DATA/containers/vllm-openai_v0.30.0.sif, from jobs/pull_container.slurm)
#   VLLM_ARGS, VLLM_API_ARGS, MTP: model-specific engine flags, API-server flags (tool/reasoning parsers), MTP
#              speculative decoding 0/1 (default 0). Tested for Qwen3.8-Flash-Next-FP8: VLLM_ARGS="--moe-backend
#              triton", VLLM_API_ARGS="--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser
#              qwen3", MTP=1, and VLLM_PLE_CPU_OFFLOAD=1 (the n-gram embedding offload; v0.30.0 image only,
#              the nightly dropped it). Other variables like this one reach vLLM unchanged: sbatch --export=ALL
#              puts them in the job, srun and apptainer pass the environment on.
#   The smoke request (smoke() below) has no thinking switch: a thinking model may spend its 60 tokens on
#   reasoning and print an empty answer. The Qwen3.8 runs sent "top_p": 0.8, "top_k": 20 and
#   "chat_template_kwargs": {"enable_thinking": false}.
#   ROUTER     vllm-router binary (default $DATA/envs/router/bin/vllm-router; pip vllm-router, venv on $DATA)
# Concurrency: PER_NODE (default "16 64 128") x N in total, plus c=1 (router and dep only; C1=0
# skips it: it takes ~2 min with every node but one idle, and doesn't change with N).
# Vanilla mode: one task per node with all 352 CPUs. Cost: 4 tokens per node-hour; submit_job adds --account.
#   sbatch -N4 --export=ALL,SRC=$DATA/models/<model> jobs/vllm_multinode.slurm                  # replicas + router
#   sbatch -N2 --export=ALL,SRC=$DATA/models/<model>,MODES=dep jobs/vllm_multinode.slurm        # DP=8 x TP1 + EP
#SBATCH --job-name=vllm-mn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=352
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00

set -uo pipefail
# Slurm copies the login node's soft open-file limit (1024) into jobs. The router needs 2 sockets
# per in-flight request: at c=512 it ran out, got "error sending request" from every worker, tripped
# its circuit breaker and answered 1547 of 2048 requests with 503. Hard limit: 524288.
ulimit -n 65536
[ -n "${SRC:-}" ] && [ -d "$SRC" ] || { echo "set SRC to the model directory (sbatch --export=ALL,SRC=...): '${SRC:-}'" >&2; exit 2; }
export SIF=${SIF:-$DATA/containers/vllm-openai_v0.30.0.sif}
[ -e "$SIF" ] || { echo "missing container $SIF" >&2; exit 2; }
export MODEL=/local/$USER/$(basename "$SRC") NAME=${NAME:-$(basename "$SRC")} MTP=${MTP:-0} DP_PORT=13345
export VLLM_ARGS=${VLLM_ARGS:-} VLLM_API_ARGS=${VLLM_API_ARGS:-}
# Servers listen on the cluster network: require a per-job random key (never printed).
export KEY=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n') && export OPENAI_API_KEY=$KEY
ROUTER=${ROUTER:-$DATA/envs/router/bin/vllm-router}   # pip vllm-router (Rust binary), venv on $DATA
MODES=${MODES:-replicas}
PER_NODE=${PER_NODE:-16 64 128}
C1=${C1:-1}
READY_CAP=${READY_CAP:-600}  # s; startup takes ~5 min (warm compile cache)
GRACE=${GRACE:-120}          # s a server may stay unreachable after logging "startup complete"
OUT=results/vllm-$NAME-${SLURM_JOB_NUM_NODES}n-$SLURM_JOB_ID
mkdir -p "$OUT" logs
log() { echo "$(date +%T) $*"; }

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); NN=${#NODES[@]}
STEP=(srun --nodes=1 --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --gres=gpu:4 --mem=0)
IPS=($(srun --ntasks-per-node=1 -l bash -c "ip -4 -o addr show ibs8 | awk '{print \$4}' | cut -d/ -f1" \
       | sort -n | awk '{print $2}'))
log "$NN nodes: ${NODES[*]}, ibs8 ${IPS[*]}, mtp=$MTP, modes: $MODES, per node: $PER_NODE"

# --- stage weights to /local on every node (directories first: parallel `cp --parents` races; .cache is HF
# download metadata, owner-only) ---
t0=$(date +%s)
srun --ntasks-per-node=1 bash -c "mkdir -p $MODEL && cd $SRC && find -L . -path ./.cache -prune -o -type d -print0 \
    | (cd $MODEL && xargs -0 mkdir -p) && find -L . -path ./.cache -prune -o -type f -print0 \
    | xargs -0 -P 16 -I{} cp {} $MODEL/{} && echo \$(hostname) \$(du -sh $MODEL | cut -f1)"
log "staged in $(( $(date +%s) - t0 ))s"

start() {  # $1 = mode; sets PIDS and URLS (the endpoints to benchmark)
    PIDS=(); URLS=()
    local i
    for i in $(seq 0 $((NN - 1))); do
        if [ "$1" = replicas ]; then
            ROLE=replica TP=4 "${STEP[@]}" -w "${NODES[$i]}" bash jobs/lib/vllm_node.sh > "$OUT/$1-server$i.log" 2>&1 &
            URLS+=("http://${NODES[$i]}:8000")    # hostname = Ethernet; replicas need no IB
        else
            ROLE=$([ $i = 0 ] && echo dp-head || echo dp-worker) HEAD_IP=${IPS[0]} TP=1 DP_SIZE=$((4 * NN)) DP_LOCAL=4 \
                "${STEP[@]}" -w "${NODES[$i]}" bash jobs/lib/vllm_node.sh > "$OUT/$1-server$i.log" 2>&1 &
        fi
        PIDS+=($!)
    done
    [ "$1" = dep ] && URLS=("http://127.0.0.1:8000")   # batch step runs on NODES[0]
}
wait_ready() {  # $1 = label, $2 = cap in s. Polls all URLs every 10 s (in parallel, 5 s timeout).
    # In replicas mode a server whose step exits, or that logged "startup complete" but stays
    # unreachable for GRACE s, is dropped (URLS and NS shrink); in dep mode either fails the mode.
    # One 16-node run lost 20 min polling one unreachable server (no curl timeout); in another,
    # apptainer died on one node ("unknown userid") and the whole run stopped.
    local t0=$(date +%s) now i p ready bad pids
    local -A since=()
    while :; do
        now=$(date +%s); ready=(); bad=(); pids=()
        for i in "${!URLS[@]}"; do curl -sf -m 5 -o /dev/null "${URLS[$i]}/health" & pids[$i]=$!; done
        for i in "${!URLS[@]}"; do
            wait "${pids[$i]}" && { ready+=("$i"); continue; }
            if [ "$1" = replicas ] && ! kill -0 "${PIDS[$i]}" 2>/dev/null; then bad+=("$i"); continue; fi
            grep -q "Application startup complete" "$OUT/$1-server$i.log" 2>/dev/null || continue
            since[$i]=${since[$i]:-$now}
            (( now - since[$i] > GRACE )) && bad+=("$i")
        done
        if (( ${#ready[@]} + ${#bad[@]} == ${#URLS[@]} )); then
            if (( ${#bad[@]} )); then
                log "[$1] dropping (exited, or unreachable ${GRACE}s after startup): $(for i in "${bad[@]}"; do printf '%s ' "${URLS[$i]}"; done)"
                [ "$1" = replicas ] && (( ${#ready[@]} > 0 )) || return 1
                URLS=($(for i in "${ready[@]}"; do echo "${URLS[$i]}"; done)); NS=${#URLS[@]}
            fi
            log "[$1] ready after $(( now - t0 ))s, ${#URLS[@]} server(s)"; return 0
        fi
        if [ "$1" != replicas ]; then
            for p in "${PIDS[@]}"; do
                kill -0 $p 2>/dev/null || { log "[$1] a server step exited during startup"; return 1; }
            done
        fi
        (( now - t0 > $2 )) && { log "[$1] not ready after $2 s (${#ready[@]}/${#URLS[@]} up)"; return 1; }
        sleep 10
    done
}
stop() { kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; sleep 10; }
smoke() {  # $1 = base url
    curl -s -m 300 "$1/v1/chat/completions" -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" -d '{
      "model": "'$NAME'", "max_tokens": 60, "temperature": 0.7,
      "messages": [{"role":"user","content":"Name three uses of a supercomputer, comma-separated."}]}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('SMOKE:', d['choices'][0]['message']['content'][:200].replace(chr(10),' '))" \
    || log "smoke test failed on $1"
}
bench() {  # base-url concurrency prompts name
    timeout 15m apptainer exec -B "$DATA,$SCRATCH,/local" "$SIF" \
        vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
            --base-url "$1" --model "$NAME" --tokenizer "$MODEL" \
            --dataset-name random --random-input-len 1024 --random-output-len 512 \
            --num-prompts $3 --max-concurrency $2 --ignore-eos \
            --save-result --result-dir "$OUT" --result-filename "$4.json" \
        > "$OUT/$4.log" 2>&1
}
summary() {  # result json files... -> one line; several files = clients run in parallel (summed)
    python3 - "$@" <<'PY'
import json, sys
rs = [json.load(open(f)) for f in sys.argv[1:]]
dur = max(r["duration"] for r in rs)
out = sum(r["output_throughput"] * r["duration"] for r in rs) / dur
tot = sum(r["total_token_throughput"] * r["duration"] for r in rs) / dur
avg = lambda k: sum(r[k] for r in rs) / len(rs)
print(f"ok={sum(r['completed'] for r in rs)} output={out:.1f} tok/s total={tot:.1f} tok/s "
      f"TTFT mean={avg('mean_ttft_ms'):.0f} ms p99={max(r['p99_ttft_ms'] for r in rs):.0f} ms "
      f"TPOT mean={avg('mean_tpot_ms'):.2f} ms p99={max(r['p99_tpot_ms'] for r in rs):.2f} ms")
PY
}
sweep() {  # $1 = label, $2 = include c=1 (0/1), rest = targets; client i -> target[i mod n]
    local label=$1 one=$2; shift 2
    local T=("$@") C N P i pids files
    for C in $([ "$one" = 1 ] && echo 1) $(for x in $PER_NODE; do echo $(( x * NS )); done); do
        N=$(( C < 8 ? 32 : C * 4 )); P=$(( C == 1 ? 1 : NS )); pids=(); files=()
        for i in $(seq 0 $((P - 1))); do
            bench "${T[$(( i % ${#T[@]} ))]}" $(( C / P )) $(( N / P )) "$label-c$C-$i" &
            pids+=($!); files+=("$OUT/$label-c$C-$i.json")
        done
        wait "${pids[@]}"   # only the clients, not the servers
        log "[$label] c=$C $(summary "${files[@]}" 2>&1)"
    done
}

for MODE in $MODES; do
    log "=== mode $MODE"
    NS=$NN    # servers in use (replicas: wait_ready drops unreachable ones); load is per server
    start "$MODE"
    if ! wait_ready "$MODE" "$READY_CAP"; then
        for i in $(seq 0 $((NN - 1))); do echo "--- $MODE-server$i.log"; grep -v 'NCCL INFO' "$OUT/$MODE-server$i.log" | tail -30; done
        stop; continue
    fi
    grep -hiE 'Available KV cache|GPU KV cache size' "$OUT/$MODE-server"*.log | sort -u | tail -4 | cut -c1-200
    for u in "${URLS[@]}"; do smoke "$u"; done

    if [ "$MODE" = replicas ] && [ ! -x "$ROUTER" ]; then
        sweep split 0 "${URLS[@]}"
        log "[router] skipped: $ROUTER not found"
    elif [ "$MODE" = replicas ]; then
        sweep split 0 "${URLS[@]}"
        "$ROUTER" --host 127.0.0.1 --port 30000 --worker-urls "${URLS[@]}" --policy round_robin \
            --api-key "$KEY" --worker-startup-check-interval 2 --log-level warning > "$OUT/router.log" 2>&1 &
        RPID=$!; t0=$(date +%s)
        until curl -sf http://127.0.0.1:30000/health >/dev/null || [ $(( $(date +%s) - t0 )) -gt 180 ]; do
            kill -0 $RPID 2>/dev/null || break; sleep 2
        done
        if curl -sf http://127.0.0.1:30000/health >/dev/null; then
            log "[router] up after $(( $(date +%s) - t0 ))s, round robin over ${#URLS[@]} servers"
            smoke http://127.0.0.1:30000
            sweep router "$C1" http://127.0.0.1:30000
        else
            log "[router] failed to start"; tail -20 "$OUT/router.log"
        fi
        kill $RPID 2>/dev/null; wait $RPID 2>/dev/null
    else
        sweep "$MODE" "$C1" "${URLS[0]}"
        echo "requests per DP engine (API server stats):"
        grep -oE 'Engine 0[0-9]+: Avg prompt throughput: [1-9][0-9.]*' "$OUT/$MODE-server0.log" | awk '{print $1, $2}' | sort | uniq -c
        echo "NCCL transports:"; grep -hoE 'NET/(IB|Socket) : Using[^;]{0,120}' "$OUT/$MODE"-server*.log | sort | uniq -c | head -4
    fi
    grep -hiE 'SpecDecoding metrics' "$OUT/$MODE-server0.log" | tail -1 | cut -c1-200
    stop
done

srun --ntasks-per-node=1 rm -rf "$MODEL"
log "done, results in $OUT"
