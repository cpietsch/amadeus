"""Job scripts for MUSICA drafted from the runbook's proven patterns (AGENTS.md §5, §6, §8).

draft() writes the cluster-specific parts: resources in the form ASC's submit filter accepts, the Lmod init before
`set -u`, caches on $SCRATCH, known bad GPU nodes excluded, a multi-node launch with retry, vLLM servers with health
checks, and the job's end time. The work itself (`command`) comes from the caller. Every shape here passed
`sbatch --test-only` and a dev-queue run (AGENTS.md §5).
"""
from __future__ import annotations

import math
import re
from datetime import date

from . import config, guard
from . import slurm as S

KINDS = {
    'cpu': 'part of a CPU node: `cores` physical cores and `mem_gb` GB for one program (shared node, billed pro rata)',
    'cpu-node': 'whole CPU nodes (190 physical cores, 768 GB, 0.96 tokens/h each); with nodes > 1 the command runs '
                'once on every node',
    'gpu': '1 to 3 H100 GPUs on a shared node (22 cores and a quarter of the memory per GPU, 1 token/h per GPU); '
           'shared nodes are few, so these jobs can wait hours while whole nodes start at once',
    'gpu-node': 'whole H100 nodes (4 GPUs, 176 physical cores, 4 tokens/h each); with nodes > 1 the command runs once '
                'on every node; with vllm_model, vLLM servers run on every node and the command is their client',
}
# vLLM images and models are read from the user's own $DATA by convention: jobs/pull_container.slurm writes
# $DATA/containers/<name>_<tag>.sif, model downloads go to $DATA/models/<repo name>; vllm_image / vllm_model override
DEFAULT_VLLM_IMAGE = '$DATA/containers/vllm-openai_v0.30.0.sif'
NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
PATH_RE = re.compile(r'^[A-Za-z0-9_./${}~+-]+$')
MODULE_RE = re.compile(r'^[A-Za-z0-9_.+/-]+$')
ARRAY_RE = re.compile(r'^\d+(-\d+)?(:\d+)?(,\d+(-\d+)?(:\d+)?)*(%\d+)?$')
LMOD = ['source /cvmfs/software.eessi.io/versions/${EESSI_VERSION:-2025.06}/init/lmod/sh >/dev/null 2>&1',
        'module use /cvmfs/software.asc.ac.at/init/modules']
MAX_NODES = 64
MAX_RUNS = 8            # runs packed on one GPU node: two per GPU
MAX_ARRAY_ARGS = 1000
PLACEHOLDER = '# --- your commands here ---'

# The job's end, for stopping work before the limit (AGENTS.md §5 "Respect the time limit"); if scontrol does not
# answer, the time limit from now.
END_TIME = r'''END=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'EndTime=[^ ]*' | cut -d= -f2) || true
END=$(date -d "${END:-none}" +%s 2>/dev/null) || END=$(( $(date +%s) + @LIMIT_S@ ))
seconds_left() { echo $(( END - $(date +%s) )); }   # e.g. timeout $(( $(seconds_left) - 60 )) <command>'''

# One step over all nodes (one task per node), with the launch retry of AGENTS.md §5: launches on 8-16 nodes fail now
# and then with "Prolog hung" or socket timeouts; a step that failed within 150 s is launched again after 30 s. Inside,
# `set -e` as in single-node drafts, and the job script's arguments reach node_main.
STEP_RETRY = r'''for attempt in 1 2 3; do
    ts=$(date +%s)
    srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" @STEP_OPTS@--cpu-bind=none \
        bash -c "set -euo pipefail; $(declare -f @FUNCTIONS@); node_main \"\$@\"" bash "$@" && RC=0 || RC=$?
    (( RC == 0 || $(date +%s) - ts > 150 || attempt == 3 )) && break
    log "step failed after $(( $(date +%s) - ts ))s (exit $RC), launch attempt $attempt; retrying in 30 s"
    sleep 30
done'''

# vLLM servers on every node: launched 0.5 s apart, health-checked in parallel, a step that exits is launched once
# more, a server that stays unreachable after "startup complete" or is not up in time is dropped (tested up to 16
# nodes; AGENTS.md §5 "Many vLLM replicas").
VLLM_SERVE = r'''serve_node() {   # the vLLM servers of node $1: @SPN@ per node with @TP@ GPU(s) each
    local i=$1 k gpus C M=$MODEL
    unset VLLM_PORT                  # vLLM's own variable for its internal ports; ours is $PORT
    # OMP_NUM_THREADS=1 (the recipe value): unset, vLLM sizes it per server as CPUs / its own workers and ignores
    # the other servers on the node (on one B200 run, 8 x 176 threads on 352 cores, weights crawled for 30 min;
    # with OMP_NUM_THREADS=1: 19 s). apptainer passes it through.
    export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=1
@STAGE@    for (( k = 0; k < @SPN@; k++ )); do
        gpus=$(seq -s, $(( k * @TP@ )) $(( k * @TP@ + @TP@ - 1 )))
        C=$SCRATCH/cache-$SLURM_JOB_NAME/vllm-n$i-s$k   # compile caches per server: parallel compiles don't collide
        mkdir -p "$C"
        export XDG_CACHE_HOME=$C VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor \
               DG_CACHE_DIR=$C/deep_gemm FLASHINFER_WORKSPACE_BASE=$C
        {   # the exit line lets the job drop a server that dies while others on its node keep the step alive
            CUDA_VISIBLE_DEVICES=$gpus apptainer exec --nv -B "$DATA,$SCRATCH,/local" "$VLLM_SIF" \
                vllm serve "$M" --served-model-name "$SERVED" --host 0.0.0.0 --port $(( PORT + k )) --api-key "$KEY" \
                    --tensor-parallel-size @TP@ "${VLLM_ARGS[@]}"
            echo "vllm server exited (rc $?)"
        } > "$LOGS/server$i-$k.log" 2>&1 &
    done
    wait
}
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
PIDS=(); URLS=(); SRV_NODE=(); SRV_LOG=(); RETRIED=(); LAUNCHED=()
launch() {   # the server step of node $1 (output appended, so a failed first launch stays visible)
    LAUNCHED[$1]=$(date +%s)
    srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gres=gpu:4 --mem=0 -w "${NODES[$1]}" \
        bash -c "set -uo pipefail; $(declare -p VLLM_ARGS); $(declare -f serve_node); serve_node $1" \
        >> "$LOGS/step$1.log" 2>&1 &
    PIDS[$1]=$!
}
stop_servers() { kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; }
trap stop_servers EXIT
T0=$(date +%s)
for i in "${!NODES[@]}"; do
    launch "$i"
    for (( k = 0; k < @SPN@; k++ )); do
        URLS+=("http://${NODES[$i]}:$(( PORT + k ))"); SRV_NODE+=("$i"); SRV_LOG+=("$LOGS/server$i-$k.log")
    done
    sleep 0.5    # staggered: 16 launches at once timed out on two nodes
done
declare -A STATE=() SINCE=()
while :; do      # until every server answers /health or is dropped
    now=$(date +%s); declare -A CHECK=()
    for s in "${!URLS[@]}"; do
        [ -n "${STATE[$s]:-}" ] && continue
        curl -sf -m 5 -o /dev/null "${URLS[$s]}/health" & CHECK[$s]=$!
    done
    for s in "${!CHECK[@]}"; do
        i=${SRV_NODE[$s]}
        if wait "${CHECK[$s]}"; then
            STATE[$s]=$(( now - T0 )); log "server $s (${URLS[$s]}) ready after ${STATE[$s]}s"
        elif ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            if [ -z "${RETRIED[$i]:-}" ]; then
                RETRIED[$i]=1; log "the server step on ${NODES[$i]} exited; launching it once more"
                tail -n 20 "$LOGS/step$i.log" "$LOGS"/server$i-*.log 2>/dev/null; sleep 5; launch "$i"
            else
                STATE[$s]="dropped (step exited)"; log "server $s dropped: step exited"; tail -20 "${SRV_LOG[$s]}" 2>/dev/null
            fi
        elif (( @SPN@ > 1 )) && grep -q '^vllm server exited' "${SRV_LOG[$s]}" 2>/dev/null; then
            STATE[$s]="dropped (exited)"; log "server $s dropped: it exited"; tail -n 20 "${SRV_LOG[$s]}"
        elif (( now - LAUNCHED[i] > LOAD_CAP )) && ! grep -qE "$LOADED" "${SRV_LOG[$s]}" 2>/dev/null; then
            STATE[$s]="dropped (load too slow)"; log "server $s dropped: weights not loaded ${LOAD_CAP}s after its launch"
            tail -n 5 "${SRV_LOG[$s]}" 2>/dev/null
        elif grep -q "Application startup complete" "${SRV_LOG[$s]}" 2>/dev/null; then
            SINCE[$s]=${SINCE[$s]:-$now}
            if (( now - SINCE[$s] > GRACE )); then STATE[$s]="dropped (unreachable)"; log "server $s dropped: unreachable"; fi
        fi
    done
    unset CHECK
    pending=0
    for s in "${!URLS[@]}"; do [ -z "${STATE[$s]:-}" ] && pending=$(( pending + 1 )); done
    (( pending == 0 )) && break
    if (( now - T0 > READY_CAP || END - now < 120 )); then   # leave the client at least 2 min
        for s in "${!URLS[@]}"; do
            [ -z "${STATE[$s]:-}" ] && STATE[$s]="dropped (not ready)" && log "server $s dropped: not ready after $(( now - T0 ))s"
        done
        break
    fi
    sleep 10
done
UP=()
for s in "${!URLS[@]}"; do [[ ${STATE[$s]} =~ ^[0-9]+$ ]] && UP+=("${URLS[$s]}/v1"); done
(( ${#UP[@]} )) || { log "no server came up; logs in $LOGS"; exit 1; }
export VLLM_URLS="${UP[*]}"
log "${#UP[@]} of ${#URLS[@]} servers up after $(( $(date +%s) - T0 ))s: $VLLM_URLS"'''

STAGE = r'''    if [ "$STAGE" = 1 ]; then   # weights to the node's NVMe with parallel copies (173 GiB in about 20 s)
        M=$STAGE_ROOT/$(basename "$MODEL"); local t0=$(date +%s)
        # directories first: parallel `cp --parents` race to create the same parent (".cache: File exists"); find -L: Hugging Face cache snapshots are symlinks into blobs/; .cache (HF download metadata) is
        # owner-only, so another user's copy would fail on it (NOTES.md, Storage)
        if mkdir -p "$M" && ( cd "$MODEL" && find -L . -path ./.cache -prune -o -type d -print0 \
                | ( cd "$M" && xargs -0 mkdir -p ) && find -L . -path ./.cache -prune -o -type f -print0 \
                | xargs -0 -P 16 -I{} cp {} "$M"/{} ) && [ -n "$(ls -A "$M")" ]; then
            echo "$(date +%T) $(hostname) staged $(du -sh "$M" | cut -f1) in $(( $(date +%s) - t0 ))s"
        else
            echo "$(date +%T) $(hostname) staging to $M failed: serving from $MODEL"; M=$MODEL
        fi
    fi
'''

# Several runs packed on one whole GPU node (the pattern of tested training jobs, four per node): run r on GPU r % 4, its own caches and log, started 2 s apart; every run is waited for and the job fails if any failed.
RUNS = r'''RUN_PIDS=()
for (( r = 0; r < @N@; r++ )); do
    RL=$JOB_DIR/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID-r$r.log C=$CACHE/r$r
    mkdir -p "$C"
    log "run $r on GPU $(( r % 4 )): $RL"
    ( set -e; export RUN=$r CUDA_VISIBLE_DEVICES=$(( r % 4 )) OMP_NUM_THREADS=@CORES@ XDG_CACHE_HOME=$C \
          TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor; "run_$r" ) > "$RL" 2>&1 &
    RUN_PIDS[$r]=$!
    sleep 2      # staggered starts: run directories and registry appends one at a time
done
NF=0
for r in "${!RUN_PIDS[@]}"; do
    wait "${RUN_PIDS[$r]}" && RC=0 || RC=$?
    log "run $r finished (exit $RC)"
    (( RC == 0 )) || { NF=$(( NF + 1 )); tail -n 20 "$JOB_DIR/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID-r$r.log"; }
done
log "done: $NF of @N@ runs failed"
exit $(( NF > 0 ))'''

# Tested serve settings for one TP4 server per H100 node (vllm_preset), from production runs in 2026-10: qwen38-fp8 on
# the vLLM nightly image of 2026-10-08, glm53-fp8 on vLLM v0.30.0. The model and image paths follow the $DATA
# convention above; vllm_model and vllm_image replace them, and the flags are kept.
MTP3 = "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
PRESETS = {
    'qwen38-fp8': {
        'model': '$DATA/models/Qwen3.8-Flash-Next-FP8', 'image': '$DATA/containers/vllm-openai_nightly-20261008.sif',
        'args': '--moe-backend triton --gpu-memory-utilization 0.85 --max-num-seqs 256 --enable-prefix-caching '
                '--no-enable-flashinfer-autotune --reasoning-parser qwen3 ' + MTP3,
        'note': 'qwen38-fp8: TP4, MTP 3, 256 sequences, context 262,144 (the model default); ready after 220-280 s '
                'with warm caches, 410 s cold; reasoning_effort accepts only xhigh (default), medium, low.'},
    'glm53-fp8': {
        'model': '$DATA/models/GLM-5.3-Flash', 'image': DEFAULT_VLLM_IMAGE,
        'args': '--language-model-only --kv-cache-dtype bfloat16 --gpu-memory-utilization 0.95 --max-num-seqs 64 '
                '--max-model-len 65536 --no-enable-flashinfer-autotune ' + MTP3
                + ' --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47',
        'note': 'glm53-fp8: TP4, BF16 KV (FP8 KV is Blackwell-only), max-model-len 65,536: size prompt plus max_tokens '
                'to it; KV holds only 363,590 tokens per node (about 26 requests at once); startup 671 s cold; the '
                'reasoning effort defaults to Max (chat_template_kwargs {"reasoning_effort": "low"} for short answers).'},
}

VLLM_CLIENT_DEMO = r'''for u in $VLLM_URLS; do
    curl -sf -m 60 "$u/chat/completions" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "{\"model\": \"$SERVED\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}], \"max_tokens\": 16}" \
        | python3 -c 'import json, sys; print(sys.argv[1], repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))' "$u" \
        || echo "$u: no answer"
done'''


class DraftError(ValueError):
    pass


def _fill(template: str, **values) -> str:
    for k, v in values.items():
        template = template.replace(f'@{k.upper()}@', str(v))
    left = re.findall(r'@[A-Z_]+@', template)
    assert not left, f'unfilled placeholders {left}'
    return template


def _path(value: str, what: str) -> str:
    v = value.strip()
    if not PATH_RE.match(v):
        raise DraftError(f'{what} {value!r}: use a plain path (letters, digits, _ . / - + ~ and $VARS such as $DATA)')
    return '$HOME' + v[1:] if v.startswith('~') else v


def _time(time: str, dev: bool) -> tuple[str, int]:
    mins = S.minutes(str(time).strip())
    if mins is None or mins <= 0:
        raise DraftError(f'time {time!r}: use minutes (30) or [D-]HH:MM:SS (01:30:00)')
    if mins == float('inf') or mins > S.MAX_HOURS * 60:
        raise DraftError(f'time {time!r}: at most {S.MAX_HOURS} h')
    if dev and mins > S.DEV_LIMIT_MIN:
        raise DraftError(f'time {time!r} is over the dev QoS limit of {S.DEV_LIMIT_MIN} min: shorten it, or set '
                         'dev=False for a production run (after a dev-queue test)')
    whole = math.ceil(mins)
    return f'{whole // 60:02d}:{whole % 60:02d}:00', whole


def _validate(kind, name, nodes, gpus, cores, mem_gb, array, dev, vllm_model, vllm_servers_per_node,
              container, python_env) -> None:
    if kind not in KINDS:
        raise DraftError(f'kind {kind!r}: use one of ' + '; '.join(f'{k}: {v}' for k, v in KINDS.items()))
    if not NAME_RE.match(name or ''):
        raise DraftError(f'name {name!r}: letters, digits, _ . - (up to 64), starting with a letter or digit')
    if not 1 <= nodes <= MAX_NODES:
        raise DraftError(f'nodes {nodes}: 1 to {MAX_NODES}')
    if kind in ('cpu', 'gpu') and nodes != 1:
        raise DraftError(f'kind {kind} is part of one node: for several nodes use {kind}-node')
    if dev and nodes > S.DEV_MAX_NODES:
        raise DraftError(f'the dev QoS allows at most {S.DEV_MAX_NODES} nodes: test on 1-2 nodes first, then set dev=False')
    if kind == 'gpu' and not 1 <= gpus <= 3:
        raise DraftError(f'gpus {gpus}: 1 to 3 for kind gpu; 4 GPUs = a whole node (kind gpu-node), which also starts sooner')
    if kind == 'cpu' and not 1 <= cores <= S.CORES[S.CPU]:
        raise DraftError(f'cores {cores}: 1 to {S.CORES[S.CPU]}; for a whole node use kind cpu-node')
    if kind == 'cpu' and mem_gb and not 1 <= mem_gb <= 760:
        raise DraftError(f'mem_gb {mem_gb}: 1 to 760 (a CPU node has 768 GB)')
    if array and not ARRAY_RE.match(array):
        raise DraftError(f'array {array!r}: Slurm syntax such as 0-9, 0-99%4 or 1,3,5')
    if vllm_model and kind != 'gpu-node':
        raise DraftError('vLLM servers need whole GPU nodes: kind gpu-node')
    if vllm_model and vllm_servers_per_node not in (1, 2, 4):
        raise DraftError('vllm_servers_per_node: 1 (TP4), 2 (TP2) or 4 (TP1) per node')
    if vllm_model and array:
        raise DraftError('vLLM servers in an array job: draft one job, then add --array yourself if it is really needed')
    if container and python_env:
        raise DraftError('container and python_env together: pick one (the container brings its own Python)')


def _validate_extras(kind, nodes, mins, dev, array, vllm_model, vllm_servers_per_node, runs, run_minutes,
                     array_args, array_throttle, vllm_preset) -> None:
    if runs:
        if kind != 'gpu-node' or nodes != 1 or vllm_model:
            raise DraftError('runs pack one whole GPU node: kind gpu-node, nodes=1, no vllm_model (for more nodes, '
                             'one job per node or array_args)')
        if len(runs) > MAX_RUNS:
            raise DraftError(f'{len(runs)} runs: at most {MAX_RUNS} per node (4 GPUs, two runs each from 5 runs on)')
        if any(not str(r).strip() for r in runs):
            raise DraftError('runs: every run needs a command')
    if run_minutes:
        if len(run_minutes) != len(runs or []):
            raise DraftError('run_minutes: one estimate per run, in the order of runs')
        if any(not isinstance(m, (int, float)) or m <= 0 for m in run_minutes):
            raise DraftError('run_minutes: positive numbers (minutes per run)')
        if max(run_minutes) > mins:
            raise DraftError(f'the slowest run ({max(run_minutes):g} min) is longer than time ({mins} min): a running '
                             'job\'s limit cannot be raised, so size time for the slowest run plus a margin')
    if array_args:
        if array:
            raise DraftError('array_args sets --array itself (one task per argument string): leave array empty, '
                             'array_throttle caps the tasks at once')
        if vllm_model:
            raise DraftError('vLLM servers in an array job: draft one job, then add --array yourself if it is really needed')
        if len(array_args) > MAX_ARRAY_ARGS:
            raise DraftError(f'array_args: at most {MAX_ARRAY_ARGS} tasks')
        if any('\n' in str(a) for a in array_args):
            raise DraftError('array_args: one line per task')
    if array_throttle and not array_args:
        raise DraftError('array_throttle goes with array_args; with array, write the throttle into it (0-9%2)')
    if array_throttle < 0:
        raise DraftError('array_throttle: tasks at once, 1 or more (0 = the default below the approval limit)')
    if vllm_preset:
        if vllm_preset not in PRESETS:
            raise DraftError(f'vllm_preset {vllm_preset!r}: one of {", ".join(PRESETS)}')
        if vllm_servers_per_node != 1:
            raise DraftError('the vllm presets are tested with one TP4 server per node: vllm_servers_per_node=1')


def _quote_arg(a: str) -> str:
    """Double quotes so $DATA and the like expand when the job starts; \\, " and ` are escaped."""
    return '"' + re.sub(r'([\\"`])', r'\\\1', a.strip()) + '"'


def draft(kind: str, name: str, command: str = '', time: str = '00:10:00', dev: bool = True, nodes: int = 1,
          gpus: int = 1, cores: int = 1, mem_gb: int = 0, array: str = '', description: str = '',
          python_env: str = '', container: str = '', modules: list[str] | None = None, vllm_model: str = '',
          vllm_args: str = '', vllm_image: str = '', vllm_servers_per_node: int = 1, stage_model: bool = True,
          runs: list[str] | None = None, run_minutes: list[float] | None = None,
          array_args: list[str] | None = None, array_throttle: int = 0, vllm_preset: str = '',
          load_cap_s: int = 420) -> dict:
    kind = (kind or '').strip()
    vllm_preset = (vllm_preset or '').strip()
    preset = PRESETS.get(vllm_preset)
    if preset and not vllm_model:
        vllm_model = preset['model']
    _validate(kind, name, nodes, gpus, cores, mem_gb, array, dev, vllm_model, vllm_servers_per_node, container,
              python_env)
    hhmm, mins = _time(time, dev)
    runs = [str(r) for r in runs or []]
    array_args = [str(a) for a in array_args or []]
    _validate_extras(kind, nodes, mins, dev, array, vllm_model, vllm_servers_per_node, runs, run_minutes or [],
                     array_args, array_throttle, vllm_preset)
    if vllm_model and not 60 <= load_cap_s <= 3600:
        raise DraftError(f'load_cap_s {load_cap_s}: 60 to 3600 s from a node\'s launch until its weights are loaded')
    if array_args:
        cap = S.DEV_MAX_NODES if dev else config.approval_nodes() - 1    # nodes at once without an approval
        throttle = array_throttle or max(1, cap // nodes)
        array = f'0-{len(array_args) - 1}' + (f'%{throttle}' if throttle < len(array_args) else '')
    if preset:
        vllm_args = (preset['args'] + ' ' + vllm_args.strip()).strip()
        vllm_image = vllm_image or preset['image']
    modules = modules or []
    for m in modules:
        if not MODULE_RE.match(m):
            raise DraftError(f'module {m!r}: a module name such as CUDA/12.6.0')
    if '\n' in vllm_args:
        raise DraftError('vllm_args: one line of vllm serve options')
    container = _path(container, 'container') if container else ''
    python_env = _path(python_env, 'python_env') if python_env else ''
    model = _path(vllm_model, 'vllm_model') if vllm_model else ''
    image = _path(vllm_image, 'vllm_image') if vllm_image else DEFAULT_VLLM_IMAGE
    gpu_part = kind.startswith('gpu')
    part = S.GPU if gpu_part else S.CPU
    qos = ('dev_' if dev else '') + part
    multi = nodes > 1 and not model
    notes: list[str] = []

    # ---- header
    L = ['#!/bin/bash']
    if kind != 'gpu':
        L.append('#ASC --vanilla')       # plain Slurm: the job gets exactly the resources below
    for line in (description or '').strip().splitlines():
        L.append(f'# {line}'.rstrip())
    shape = (f'kind={kind}' + (f', nodes={nodes}' if nodes > 1 else '') + (', vllm' if model else '')
             + (f', {len(runs)} runs' if runs else '') + (f', {len(array_args)} array_args' if array_args else ''))
    L.append(f'# Drafted by Amadeus (draft_job {shape}) on {date.today()}; check_job before submitting.')
    out = 'logs/%x-%A_%a.out' if array else 'logs/%x-%j.out'
    sb = [f'--job-name={name}', f'--output={out}'] + ([f'--account={config.account()}'] if config.account() else [])
    sb += [f'--partition={part}', f'--qos={qos}']
    if not config.account():
        notes.append('No Slurm account is configured, so the draft has no --account line: add one, or set [project] '
                     f'account in {config.path()} (submit_job adds it then).')
    if kind == 'cpu':
        mem = mem_gb or min(4 * cores, 760)
        sb += ['--nodes=1', '--ntasks=1', f'--cpus-per-task={cores}', '--threads-per-core=1', f'--mem={mem}G']
        notes.append('Vanilla mode for part of a CPU node: in comfort mode a CPU-partition job with --nodes becomes whole '
                     f'nodes of 190 single-core tasks (and is rejected with --cpus-per-task > 1). {cores} cores, {mem} GB; '
                     f'nproc shows {2 * cores} (both hardware threads of each core). `free` shows the whole node\'s '
                     'memory, not --mem (one run showed 755 GB).')
    elif kind == 'cpu-node':
        notes.append('nproc shows 380 per node (190 cores, two hardware threads each); 188 worker processes with '
                     'OMP_NUM_THREADS=1 per node worked.')
        sb += [f'--nodes={nodes}', '--ntasks-per-node=1', f'--cpus-per-task={S.CORES[S.CPU]}', '--threads-per-core=1',
               '--exclusive', '--mem=0']
    elif kind == 'gpu':
        sb += [f'--gres=gpu:{gpus}', '--threads-per-core=1']
        notes.append(f'Comfort mode: --gres=gpu:{gpus} gives {gpus} task(s) of 22 cores and {gpus * 25}% of a shared '
                     'node\'s memory. Shared nodes are few (jobs waited up to 9 h); a whole node (kind gpu-node) '
                     'usually starts at once and can run four such programs side by side (CUDA_VISIBLE_DEVICES=0..3).')
    else:
        sb += [f'--nodes={nodes}', '--ntasks-per-node=1', f'--cpus-per-task={S.THREADS[S.GPU]}', '--gres=gpu:4',
               '--exclusive', '--mem=0']
    sb.append(f'--time={hhmm}')
    if array:
        sb.append(f'--array={array}')
    if gpu_part:
        sb.append('--exclude=' + ','.join(S.BAD_NODES))
    L += [f'#SBATCH {o}' for o in sb]
    L.append('')

    # ---- environment
    if modules:
        L += LMOD + ['module load ' + ' '.join(modules) + ' || { echo "module load ' + ' '.join(modules)
                     + ' failed" >&2; exit 1; }']
    if multi or model:
        L.append('set -uo pipefail                # no -e here: the exit codes of the steps and the client are read below')
    elif runs:
        L.append('set -uo pipefail                # no -e here: the exit codes of the runs are read below')
    else:
        L.append('set -euo pipefail')
    L.append('ulimit -n 65536 2>/dev/null || true    # jobs inherit the soft limit of the login node, 1024 open files')
    L.append('log() { echo "$(date +%T) $*"; }')
    L += _fill(END_TIME, limit_s=mins * 60).splitlines()
    L.append(f'export CACHE=$SCRATCH/cache-{name}        # compile caches (Triton, Inductor, ...), kept for the next job')
    L.append('export XDG_CACHE_HOME=$CACHE TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor '
             'HF_HOME=$DATA/hf_cache')
    L.append('mkdir -p "$CACHE" logs')
    if gpu_part:
        L.append('export HF_HUB_OFFLINE=1       # models from $DATA/models; remove this line if the command downloads from HF')
    funcs = ['log', 'seconds_left']
    # inputs checked before any work starts: a typo fails in seconds with a clear message, not after the allocation
    needs = []
    if python_env:
        needs.append('"$VIRTUAL_ENV/bin/python"')
    if container:
        needs.append('"$SIF"')
    if model:
        needs.append('"$VLLM_SIF"')
        if model.startswith(('/', '$', '.')):       # not a Hugging Face repo id
            needs.append('"$MODEL"')
    if needs:
        L.append('need() {   # fail fast when an input of the job is missing')
        L.append('    local p; for p in "$@"; do [ -e "$p" ] || { log "missing $p: check the path, or fetch it '
                 'first"; exit 2; }; done')
        L.append('}')
    if python_env:
        L.append(f'export VIRTUAL_ENV={python_env} && export PATH="$VIRTUAL_ENV/bin:$PATH"')
    if container:
        binds = '$DATA,$SCRATCH' + (',/local' if gpu_part else '')
        nv = '--nv ' if gpu_part else ''
        L.append(f'export SIF={container}')
        L.append(f'run() {{ apptainer exec {nv}-B "{binds}" "$SIF" "$@"; }}   # run <command>: inside the container')
        funcs.append('run')
        notes.append('Commands run on the host unless prefixed with `run` (apptainer exec in $SIF).')
    if array:
        L.append('export TASK=${SLURM_ARRAY_TASK_ID:-0}')
    if array_args:
        L.append('ARGS_ALL=(   # one argument string per array task: task i gets $ARGS = ARGS_ALL[i], results in $OUT')
        L += [f'    {_quote_arg(a)}' for a in array_args]
        L.append(')')
        L.append(f'export ARGS=${{ARGS_ALL[$TASK]}} OUT=results/{name}-${{SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}-a$TASK')
        L.append('mkdir -p "$OUT"')
    if model:
        L.append(f'export VLLM_SIF={image} MODEL={model}')
    if needs:
        L.append('need ' + ' '.join(needs))
    # nproc honours OMP_NUM_THREADS (it printed "1 cpus" on a 352-CPU node, 2026-10-08)
    L.append('log "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES node(s), ${SLURM_CPUS_ON_NODE:-?} '
             'CPUs here), '
             'revision $(cat REVISION 2>/dev/null || echo unknown)' + (', task $TASK' if array else '')
             + (': $ARGS' if array_args else '') + '"')
    if gpu_part:
        L.append('nvidia-smi -L')
    L.append('')

    body = command.rstrip('\n') if command.strip() else ''

    if model:
        spn = vllm_servers_per_node
        tp = 4 // spn
        L += ['export SERVED=$(basename "$MODEL") PORT=${SERVE_PORT:-8000}',
              '# the servers listen on the cluster network: a random key per job, never printed',
              'export KEY=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d \' \\n\') && export OPENAI_API_KEY=$KEY',
              f'VLLM_ARGS=({vllm_args.strip()})',
              'export LOGS=logs/$SLURM_JOB_NAME-$SLURM_JOB_ID STAGE=' + ('1' if stage_model else '0')
              + ' STAGE_ROOT=${STAGE_ROOT:-/local/$USER}',
              'READY_CAP=${READY_CAP:-900}   # s for all servers to start (16 replicas: 501 s; with warm caches 249 s)',
              'GRACE=${GRACE:-120}           # s a server may stay unreachable after "Application startup complete"',
              # a crawling load (1 layer of 40 per minute on one run) is caught here, not at READY_CAP
              f'LOAD_CAP=${{LOAD_CAP:-{load_cap_s}}}  # s from a node launch (staging included) until each server logs $LOADED',
              'LOADED="Loading weights took|Model loading took|Application startup complete"',
              'mkdir -p "$LOGS"',
              _fill(VLLM_SERVE, spn=spn, tp=tp, stage=STAGE if stage_model else ''),
              '',
              '# --- the client, on the first node: $VLLM_URLS (space-separated OpenAI base URLs), $SERVED (model name),',
              '# $OPENAI_API_KEY. Spread requests over all URLs from one shared queue, and stop a minute before the',
              '# limit: timeout $(( $(seconds_left) - 60 )) <client>. It runs in a subshell with set -e: a failing',
              '# command fails the job, and a bare `wait` there waits for the client\'s own processes, not the servers.',
              '(',
              'set -e',
              body or PLACEHOLDER + '\n' + VLLM_CLIENT_DEMO,
              ')',
              'RC=$?',
              'stop_servers; trap - EXIT']
        if stage_model:
            L.append('srun --ntasks-per-node=1 rm -rf "$STAGE_ROOT/$(basename "$MODEL")"   # the staged weights')
        L += ['log "done (exit $RC)"', 'exit $RC']
        notes.append(f'{spn} vLLM server(s) per node, TP{tp}, image {image}; logs in logs/<name>-<job>/. Model-specific '
                     'flags (MoE backend, MTP, reasoning parser, max length) go into vllm_args; the vLLM templates in '
                     'job_templates show tested settings.')
        if stage_model:
            notes.append('The weights are copied to each node\'s /local first (fast loading); stage_model=False skips it.')
        notes.append(f'A server that has not loaded its weights {load_cap_s} s after its node\'s launch is dropped '
                     '(LOAD_CAP; load_cap_s); one not up after READY_CAP 900 s too. The model\'s generation_config.json '
                     'overrides vLLM\'s sampling defaults (--generation-config vllm turns that off); reasoning_effort '
                     'values are model-specific.')
        if preset:
            notes.append(preset['note'] + f' The job reads the model from {model} and the image from {image} (fetch '
                         'or pull them first, or pass vllm_model / vllm_image; the job exits 2 if they are missing).')
        if dev:
            notes.append('Dev QoS: a vLLM start takes 2-7 min of its 10 (Qwen3.5-4B 110-140 s, Qwen3.8 220-410 s), '
                         'GLM-5.3 cold 11 min and the first DeepSeek-V4.1 compile do not fit. Warm the compile caches '
                         '($SCRATCH/cache-<name>, kept per job name) with a production-QoS run first.')
    elif multi:
        if gpu_part:   # mlx5_0 is a 25 Gb/s RoCE port, mlx5_2..5 the NDR200 adapters (NOTES.md, Hardware)
            L.append("export NCCL_IB_HCA='^=mlx5_0'   # NCCL across nodes: InfiniBand only")
        L += ['export NNODES=$SLURM_JOB_NUM_NODES HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1) END',
              '',
              'node_main() {   # runs once on every node: NODE_RANK 0..NNODES-1, HEAD = the first node',
              'export NODE_RANK=$SLURM_PROCID',
              'export XDG_CACHE_HOME=$CACHE/n$NODE_RANK TRITON_CACHE_DIR=$CACHE/n$NODE_RANK/triton '
              'TORCHINDUCTOR_CACHE_DIR=$CACHE/n$NODE_RANK/inductor   # one cache per node: no write races',
              'mkdir -p "$XDG_CACHE_HOME"',
              body or PLACEHOLDER + '\necho "node $NODE_RANK of $NNODES: $(hostname), ${SLURM_CPUS_ON_NODE:-?} CPUs"',
              '}',
              _fill(STEP_RETRY, step_opts='--gres=gpu:4 ' if gpu_part else '--threads-per-core=1 ',
                    functions=' '.join(funcs + ['node_main'])),
              'log "done (exit $RC)"',
              'exit $RC']
        notes.append('The command runs once per node inside node_main, under set -e, with the job script\'s arguments '
                     'and the exported variables and functions (' + ', '.join(funcs) + '). It is one srun step over '
                     'all nodes: if it fails within 150 s on any node, the whole step is launched again (up to 3 '
                     'times), so make the work safe to restart. For DDP: torchrun --nnodes=$NNODES '
                     '--nproc_per_node=4 --node_rank=$NODE_RANK --rdzv_backend=c10d --rdzv_endpoint=$HEAD:29500 '
                     '--rdzv_id=$SLURM_JOB_ID' + (' (NCCL_IB_HCA is set).' if gpu_part else '.'))
    elif runs:
        L.append('JOB_DIR=$PWD    # the run logs stay here even if the setup changes directory')
        if body:   # in the job's own shell, so its cd, export, source and functions reach the runs
            L += ['# setup before the runs, under set -e (the trap reports a failure or exit)',
                  'trap \'log "setup failed (exit $?)"\' EXIT', 'set -e', body, 'set +e', 'trap - EXIT']
        L += ['', '# --- the runs: run r on GPU r % 4, $RUN = r, its own caches ($CACHE/r<r>) and log',
              '# (logs/<name>-<job>-r<r>.log), each under set -e; the job fails if any run failed']
        for r, cmd in enumerate(runs):
            L += [f'run_{r}() {{', cmd.rstrip('\n'), '}']
        L.append(_fill(RUNS, n=len(runs), cores=S.CORES[S.GPU] // len(runs)))
        per_gpu = '' if len(runs) <= 4 else ' Runs 5-8 share GPUs with runs 1-4 (two per GPU).'
        notes.append(f'{len(runs)} runs side by side on one node, OMP_NUM_THREADS={S.CORES[S.GPU] // len(runs)} each '
                     f'(the 176 cores shared out).{per_gpu} The node is billed until the slowest run ends, and a running '
                     'job\'s time limit cannot be raised: size time for the slowest run.')
        if run_minutes and max(run_minutes) > 1.5 * min(run_minutes):
            notes.append(f'run_minutes {min(run_minutes):g}-{max(run_minutes):g}: the GPUs of the short runs idle (and '
                         'are billed) until the longest ends; group runs of similar length per job.')
    else:
        L += [body or PLACEHOLDER, 'log "done"']
    if mins > 60 and not model:
        notes.append('Long job: write results as they arrive, and stop loops or clients before the limit (seconds_left).')
    if not dev:
        notes.append('Production QoS: run the same script on the dev QoS first (dev=True: up to 10 min and 2 nodes).')

    script = '\n'.join(L) + '\n'
    check = guard.evaluate(script, [])
    return {'script': script, 'suggested_path': f'jobs/{name}.slurm',
            'check': {k: check[k] for k in ('errors', 'warnings', 'needs_approval', 'estimate')},
            'notes': notes,
            'next': 'Fill in the commands, then check_job(script_text=...) for the rules and sbatch --test-only without '
                    f'pushing; save it in your project (e.g. jobs/{name}.slurm), push_files, then submit_job.'}
