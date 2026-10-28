"""Pure log helpers for job_log, job_details and wait_for_job: cleaning (ANSI codes, tqdm carriage returns, binary
junk), progress bars, known-benign noise, failure diagnosis from known signatures plus sacct rows, and milestone
matching. No SSH here: callers fetch the text and pass it in.

The signatures come from real MUSICA jobs (agent session reports, AGENTS.md §8; fixtures in tests/fixtures/). To add one, append a row to SIGNATURES (or NOISE for harmless lines); earlier rows win when the
category is chosen, so put specific root causes before generic ones. Generic rows (generic=True: a missing path,
module or command) rank after every specific one and never match WARNING/INFO lines; launch rows (Prolog hung,
socket timeouts) decide the category only when the job failed fast with nothing more specific in the log."""
from __future__ import annotations

import re

from . import slurm as S

INFRA, SCRIPT, PROGRAM, BENIGN, UNKNOWN = 'infrastructure', 'script', 'program', 'benign', 'unknown'

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-Z\\-_]')
JUNK = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]')
NODE = re.compile(r'\bn\d{4}-\d{3}\b')
# vLLM and torch line prefixes: '(Worker_TP0 pid=123) ', '[rank0]: ' (torchrun; '[rank0]:[W...' for C++ warnings),
# 'ERROR 10-07 19:00:42 [core.py:1366] '
PREFIX = re.compile(r'^(?:\((?P<tag>[\w.-]+)(?: pid=\d+)?\) )?(?:\[rank\d+\]: ?)?'
                    r'(?:(?P<level>INFO|WARNING|ERROR|DEBUG|CRITICAL) \d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)? \[[^\]]*\] ?)?')
QUIET_LEVEL = re.compile(r'^(?:WARNING|WARN|INFO|DEBUG)\b:?')     # also apptainer's and Python's 'WARNING: ...'
FAST_FAIL_S = 150          # a launch failure ends the job (or step) within this (AGENTS.md §5 Patterns)
# a tqdm or vLLM progress bar: ' 14%|█▍   | 185/1250 [09:12<52:57,  2.98s/it, loss=0.4]' or
# 'Loading safetensors checkpoint shards:  2% Completed | 2/131 [00:00<00:36,  3.56it/s]'
_NUM = r'\d+(?:\.\d+)?[kMGTP]?(?:i?B)?'
BAR = re.compile(r'(?:(?P<pct>\d{1,3})%(?:\|[^|\n]*\||\s+Completed\s+\|)\s*)?'
                 rf'(?P<n>{_NUM})/(?P<total>{_NUM})\s*'
                 r'\[(?P<elapsed>[\d:]+)(?:<(?P<eta>[\d:?]+))?(?:,\s*(?P<rate>[^,\]]*))?(?:,\s*(?P<post>[^\]]*))?\]')
LOADING = re.compile(r'(?i)load|weight|shard|checkpoint|fetch|download|\.(?:safetensors|bin|gguf|pth?)\b')
BYTES = re.compile(r'i?B$|i?B/s$')     # a total or rate in bytes: a download or copy (hf's per-file bars)
WARMUP = re.compile(r'(?i)captur|compil|warm|elaborat|\bjit\b')


def _sig(name, category, pattern, meaning, action='', **opt) -> dict:
    """opt: transient (a resubmit or retry usually works), node (name the node near the line), level ('warning'
    does not decide the category), completed_ok (benign when the job ended COMPLETED 0:0), generic (ranks after the
    specific rows, skipped on WARNING/INFO lines), launch (a warning unless the job failed fast), after=(regex,
    category, meaning, action): used instead when a line matching regex came earlier in the log,
    scope='exception' (matched against whole Python tracebacks; exc= must match the exception line)."""
    return {'name': name, 'category': category, 'rx': re.compile(pattern), 'meaning': meaning, 'action': action,
            **opt}


SIGNATURES = [
    # --- cluster side
    _sig('apptainer_unknown_userid', INFRA, r'unknown userid \d+',
         'apptainer cannot look up our user on this node (node-side account problem; seen on n3010-019)',
         'resubmit with --exclude=<node> and tell the human (support ticket to ASC)', node=True),
    _sig('socket_timeout', INFRA, r'Socket timed out on send/recv operation',
         'Slurm step launch timed out (slow prolog or many launches at once, 8-16 nodes)',
         'relaunch the step after 30 s (draft_job multi-node scripts retry by themselves); resubmit if the script '
         'has no retry; launch per-node steps 0.5 s apart', transient=True, level='warning', launch=True),
    _sig('prolog_hung', INFRA, r'Prolog hung on node (?P<node>\S+)',
         'Slurm prolog slow on these nodes at launch (harmless when the step still ran)',
         'if the step failed fast: retry after 30 s or resubmit unchanged; if it ran, ignore', transient=True,
         completed_ok=True, level='warning', launch=True),
    _sig('cert_expired', INFRA, r'Permission denied \(publickey,keyboard-interactive',
         'the MUSICA certificate expired or is missing (our access, not the job)',
         'start_login and show the link to the human at once'),
    # --- job script, environment, resources
    _sig('unbound_variable', SCRIPT, r'(?P<what>\S+): unbound variable',
         'bash set -u abort: an unset variable, or a declared but empty associative array',
         'initialise it (declare -A X=()), use ${X:-}, init Lmod before set -u; dry-run the script'),
    _sig('module_not_found', SCRIPT, r'module: command not found',
         'Lmod is not initialised in the batch shell',
         'source the EESSI Lmod init before `module` (draft_job skeleton)'),
    _sig('python_h_missing', SCRIPT, r'Python\.h: No such file or directory',
         'the venv sits on the system Python, which has no headers (Triton/torch.compile need them)',
         'build the venv with uv on a standalone Python (AGENTS §6)'),
    _sig('no_python_module', SCRIPT, r'No module named',
         'the Python environment lacks a package (wrong venv or container, or PYTHONPATH)',
         'check the python/venv path the script uses; install the package into that env', generic=True),
    _sig('cp_parents_race', SCRIPT, r'cannot make directory .*: File exists',
         'parallel `cp --parents` raced creating the same directory while staging',
         'create the directories first, then copy files in parallel (current draft_job staging does)'),
    _sig('command_not_found', SCRIPT, r'(?P<what>\S+): command not found',
         'a command is not on PATH in the job (module, venv or container not loaded)',
         'load the module/venv first or use the full path; GPU modules exist only on GPU nodes', generic=True),
    _sig('apptainer_fatal', SCRIPT, r'^FATAL:\s',
         'apptainer could not start the container', 'check the SIF path, -B binds ($DATA,$SCRATCH) and '
         'APPTAINER_TMPDIR; the line says which', node=True),
    _sig('file_not_found', SCRIPT, r"(?i)no such file or directory|FileNotFoundError|can't open file",
         'a path does not exist on the node', 'check the path against $DATA/$SCRATCH and the job\'s working '
         'directory; push_files first; /local is per node', generic=True),
    _sig('disk_full', SCRIPT, r'No space left on device',
         'a filesystem is full (/local on the login node has 11 GB; /tmp; quotas)',
         'write to $SCRATCH or $DATA; check df -h $SCRATCH and mmlsquota'),
    _sig('quota', SCRIPT, r'Disk quota exceeded', 'a storage quota is full',
         'free space in $DATA/$SCRATCH (deleting needs the human) or ask the human'),
    _sig('host_oom', SCRIPT, r'oom_kill event|OOM Killed|Exceeded (?:step|job) memory limit',
         'host memory exhausted: the OOM killer stopped a task',
         'more memory (whole node; vanilla --mem=0) or fewer workers/smaller batches'),
    _sig('node_config', SCRIPT, r'Requested node configuration is not available',
         'the request does not fit a node as comfort mode reads it (AGENTS §8)',
         'part of a CPU node: #ASC --vanilla with -N1 -n1 -c N --threads-per-core=1; GPU nodes: at most 176 '
         'cores with --threads-per-core=1'),
    _sig('invalid_account', SCRIPT, r'Invalid account or account/partition combination',
         'wrong --account (the filesystem name, e.g. fs<number>, is not the account)',
         "use the project's Slurm account ([project] account in the config; submit_job adds it when missing)"),
    _sig('time_raise_denied', SCRIPT, r'Access/permission denied for job',
         "a running job's time limit cannot be raised (only while pending)",
         'stop at a checkpoint and resubmit the rest; estimate --time first (AGENTS §5 Costs)'),
    _sig('time_limit', SCRIPT, r'(?i)CANCELLED AT \S+ DUE TO TIME LIMIT',
         'the job reached its --time (outputs may be partial; or it hung: see the last progress)',
         'check the outputs; set --time from a measured rate x 1.3; a running job\'s limit cannot be raised'),
    _sig('open_files', SCRIPT, r'error sending request|503 Service Unavailable',
         'router/server ran out of open files (soft limit 1024 inherited from the login node)',
         'ulimit -n 65536 in the job script'),
    _sig('ssh_node_refused', SCRIPT, r'Permission denied \(publickey,gssapi',
         'ssh to compute nodes is refused for us (AGENTS §2)',
         'node_exec or srun --jobid=N --overlap; inside a job use srun steps'),
    _sig('staging_fallback', SCRIPT, r'staging to \S+ failed: serving from',
         'copying the weights to /local failed, so vLLM loaded them from $DATA (slower)',
         "see the cp error above (often HF .cache metadata of a model downloaded by another user)", level='warning'),
    # --- our program or model configuration
    _sig('cuda_oom', PROGRAM, r'CUDA out of memory|OutOfMemoryError|CUDA error: out of memory',
         'GPU memory exhausted', 'smaller batch or sequence length; vLLM: lower --gpu-memory-utilization, '
         'max-num-seqs or max-model-len'),
    _sig('vllm_free_memory', PROGRAM, r'Free memory on device \S+ .* is less than desired GPU memory utilization',
         'vLLM: less GPU memory free at startup than gpu_memory_utilization asks for',
         'lower --gpu-memory-utilization (e.g. 0.95); check for another process on the GPU'),
    _sig('vllm_kv_too_small', PROGRAM, r'KV cache is needed',
         'vLLM: the weights leave too little KV cache for max-model-len (0.36 GiB free on one run)',
         'lower --max-model-len or max-num-seqs, raise gpu-memory-utilization, more GPUs (TP) or FP8 KV'),
    _sig('vllm_no_server', PROGRAM, r'no server came up',
         'no vLLM server became ready', 'read the server logs (logs/<name>-<job>/); see exceptions'),
    _sig('vllm_server_dropped', PROGRAM, r'server \S+ dropped: (?P<what>.+)',
         "the draft's health check dropped a vLLM server", 'read that server\'s log; if the same node '
         'repeats (unreachable IP, n3010-027), --exclude it', node=True),
    _sig('vllm_engine_init', PROGRAM, r'Engine core initialization failed',
         'vLLM engine failed to start; the root cause is the first worker error above',
         'read exceptions[0]; fix the model or parallel settings (e.g. NotImplementedError for DP x TP + EP)'),
    _sig('vllm_engine_dead', PROGRAM, r'EngineDeadError',
         'vLLM engine died; the root cause is the first error above it', 'see exceptions[0]',
         after=(r'\[shutdown\]|shutdown triggered', BENIGN, "EngineDeadError during the job's own server shutdown "
                '(harmless)', '')),
    _sig('shm_broadcast_wait', BENIGN, r'No available shared memory broadcast block found',
         'vLLM waited for a worker (normal during compile and graph capture)', '',
         after=(r'Application startup complete', PROGRAM, 'a vLLM worker stalled after startup (possible engine '
                'hang, n3013-013)', 'health-check this server; move its requests to the others')),
    _sig('shm_broadcast_timeout', PROGRAM, r'shm_broadcast', 'vLLM engine worker hang (shm_broadcast TimeoutError, '
         'n3013-013', 'health checks during the run; move its requests to the other servers; '
         'exclude the node if it repeats', scope='exception', exc=r'^TimeoutError'),
    _sig('unsloth_assert', PROGRAM, r'(?i)unsloth', 'an assertion inside Unsloth (a stride assertion was seen at '
         'batch size 1, 2026-10-10)', 'check batch size and padding; reproduce on the workstation',
         scope='exception', exc=r'^AssertionError'),
    _sig('causal_conv1d_fallback', PROGRAM,
         r'(?i)causal[_-]?conv1d.*(?:fall ?back|slow|not (?:installed|available))|(?:fall ?back|slow).*causal[_-]?conv1d',
         'causal_conv1d kernels missing: slow PyTorch fallback (training much slower)',
         'install causal-conv1d matching torch/CUDA in the env', level='warning'),
    _sig('segfault', PROGRAM, r'Segmentation fault|SIGSEGV', 'a native crash (segmentation fault)',
         'check library versions (CUDA, torch, container); reproduce small'),
    _sig('nccl', PROGRAM, r'NCCL error|nccl\w*Error|Watchdog caught collective operation timeout',
         'NCCL communication failed: usually another rank crashed or hung first, otherwise the network',
         "find the first error on the other ranks; multi-node: NCCL_IB_HCA='^=mlx5_0'; if no rank failed, "
         'resubmit with the node excluded'),
    # --- not failures
    _sig('job_cancelled', BENIGN, r'\*\*\* JOB \d+ ON \S+ CANCELLED AT \S+(?: DUE to SIGNAL \w+)? \*\*\*',
         'the job was cancelled (scancel or cancel_job)'),
]

# harmless lines: (regex, reason). filter_benign hides them; diagnose counts them.
NOISE = [(re.compile(p), why) for p, why in [
    (r'Could not find a suitable TLS CA certificate bundle|Exception in thread .*_report_usage_worker',
     'vLLM usage-stats upload (TLS CA path); harmless, VLLM_NO_USAGE_STATS=1 turns it off'),
    (r'resource_tracker: There appear to be \d+ leaked (?:shared_memory|semaphore) objects|'
     r"warnings\.warn\('resource_tracker:", 'leaked shared_memory warning at shutdown'),
    (r'^srun: forcing job termination|^srun: Job step aborted: Waiting up to \d+ seconds|'
     r'\*\*\* STEP \d+\.\d+ ON \S+ CANCELLED AT \S+ (?i:DUE TO) SIGNAL Killed|^srun: error: \S+: task \d+: Killed$',
     'srun stopping a step the script ended itself (normal server shutdown)'),
    (r'^srun: Step created for StepId=', 'srun step start notice'),
    (r'Failed to import from vllm\._C', "vLLM client import warning (vllm bench serve without the compiled ops)"),
    (r'^(?:sbatch: )?DEBUG\b|applying job settings for >>', 'DEBUG lines (router start-up, ASC submit filter)'),
    (r'warn xattr\{.*\} (?:ignoring ENOTSUP|destination filesystem does not support xattrs)',
     'apptainer build on a filesystem without xattrs'),
    (r'(?i)^(?:WARNING|INFO):?\s.*\b(?:chdir|change directory)\b', 'apptainer chdir warning'),
    (r'squashfuse_ll\b.*\btime', 'apptainer squashfuse_ll timeout notice'),
    (r'^\[notice\] ', 'pip upgrade notice'),
    (r'\bNCCL INFO ', 'NCCL INFO lines'),
    (r'^\[?W\d{4} \d\d:\d\d:\d\d', 'PyTorch C++ warning line (W....)'),
    (r'\[TileLang:tilelang\.cache\.kernel_cache:WARNING\]', 'TileLang kernel cache notice'),
    (r'Still waiting to acquire lock on .*\.lock|You are sending unauthenticated requests to the HF Hub|'
     r'`hf-cli` skill is not installed', 'Hugging Face download notices'),
]]
# whole tracebacks that are harmless: (regex over the block, reason)
BENIGN_TRACEBACKS = [(re.compile(r'_report_usage_worker|usage_lib\.py|TLS CA certificate bundle'), NOISE[0][1])]

# exit codes of srun tasks and batch scripts: code -> (category, meaning)
EXIT_CODES = {
    1: (PROGRAM, 'general error: see the first error or exception'),
    2: (SCRIPT, 'bad usage or shell syntax error'),
    124: (SCRIPT, 'stopped by `timeout` in the script (an inner time limit): results may be partial'),
    126: (SCRIPT, 'command not executable (permissions or wrong file)'),
    127: (SCRIPT, 'command not found (PATH, module or container)'),
    134: (PROGRAM, 'SIGABRT: abort, often a failed C/C++/CUDA assertion'),
    137: (UNKNOWN, 'SIGKILL: OOM killer, or Slurm killed it (time limit, cancel, step end)'),
    139: (PROGRAM, 'SIGSEGV: segmentation fault in native code'),
    140: (INFRA, 'on MUSICA seen when a multi-node launch failed (Prolog hung, socket timeouts): retry'),
    143: (UNKNOWN, 'SIGTERM: scancel, the time limit, or the script stopping it'),
    255: (UNKNOWN, 'ssh/srun failure or exit(-1); apptainer FATAL also gives 255'),
}
SIGNALS = {6: 'SIGABRT', 9: 'SIGKILL', 11: 'SIGSEGV', 15: 'SIGTERM'}
STATES = {   # sacct State -> (category, meaning, action, transient)
    'NODE_FAIL': (INFRA, "a node failed on ASC's side (within a minute at launch; later in a run it can also be memory)",
                  "resubmit unchanged (billed only for the minutes used); check sinfo -h -N -n <nodes> -o '%N %t %E' "
                  'and --exclude a node that is down', True),
    'BOOT_FAIL': (INFRA, 'a node failed to boot', 'resubmit', True),
    'PREEMPTED': (INFRA, 'the job was preempted', 'resubmit', True),
    'TIMEOUT': (SCRIPT, 'the job reached its --time (outputs may be partial)',
                'check the outputs and last progress; set --time from a measured rate x 1.3', False),
    'OUT_OF_MEMORY': (SCRIPT, 'the job ran out of host memory', 'more memory or fewer workers', False),
    'DEADLINE': (SCRIPT, 'the job reached its --deadline', 'resubmit with a later deadline', False),
    'CANCELLED': (BENIGN, 'the job was cancelled (scancel or cancel_job)', '', False),
}
LAUNCH = {s['name'] for s in SIGNATURES if s.get('launch')}
TASK_EXIT = re.compile(r'srun: error: (?P<node>\S+): task (?P<task>[\d,-]+): Exited with exit code (?P<rc>\d+)')
VLLM_MILESTONES = [r'staged \S+ in \d+s', r'ready after \d+s', r'servers up after \d+s', r'done \(exit \d+\)']
MAX_FINDINGS, MAX_LINE = 8, 200


def _text(text) -> str:
    if isinstance(text, bytes):
        return text.decode('utf-8', 'replace')
    return text or ''


def _bar_key(line: str):
    m = None
    for m in BAR.finditer(line):
        pass
    if not m:
        return None
    return _label(line[:m.start()]), m.group('total')


def _label(head: str) -> str:
    tag, body = _split(head)
    body = body.strip().rstrip(':|').strip()[-80:]
    return f'({tag}) {body}'.strip() if tag else body


def _split(line: str) -> tuple[str, str]:
    """-> (process tag, line without vLLM/torch prefixes)."""
    m = PREFIX.match(line)
    return (m.group('tag') or ''), line[m.end():]


def _quiet(line: str) -> bool:
    """A WARNING/INFO/DEBUG line (vLLM's prefix or a leading 'WARNING:')."""
    m = PREFIX.match(line)
    return m.group('level') in ('INFO', 'WARNING', 'DEBUG') or bool(QUIET_LEVEL.match(line[m.end():]))


def clean(text, progress: bool = False) -> str:
    """Strip ANSI codes and binary junk; keep the last state of each \\r-updated line and collapse runs of the
    same progress bar to its last state. progress=True keeps every \\r update as its own line (like tr '\\r' '\\n')."""
    text = ANSI.sub('', _text(text)).replace('\r\n', '\n')
    out: list[str] = []
    for line in text.split('\n'):
        parts = [p for p in line.split('\r') if p.strip()]
        for part in (parts if progress else parts[-1:]) or ['']:
            out.append(JUNK.sub('', part).rstrip())
    if progress:
        return '\n'.join(out)
    kept: list[str] = []
    prev = None
    for line in out:
        key = _bar_key(line) if '/' in line and '[' in line else None
        if key and key == prev:
            kept[-1] = line
        else:
            kept.append(line)
        prev = key
    return '\n'.join(kept)


def _num(s: str | None):
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def progress(text, limit: int = 8) -> list[dict]:
    """Latest state of each progress bar (one per label and total), most recently updated last, at most `limit`.
    kind: loading (weights, shards, downloads), warmup (graph capture, compiling) or work."""
    bars: dict = {}
    for seg in re.split(r'[\r\n]', ANSI.sub('', _text(text))):
        m = None
        for m in BAR.finditer(seg):
            pass
        if not m:
            continue
        label = _label(JUNK.sub('', seg[:m.start()]))
        n, total = _num(m.group('n')), _num(m.group('total'))
        pct = _num(m.group('pct'))
        if pct is None and isinstance(n, (int, float)) and isinstance(total, (int, float)) and total:
            pct = round(100 * n / total, 1)
        rate = (m.group('rate') or '').strip()
        bar = {'label': label, 'n': n, 'total': total, 'percent': pct,
               'rate': None if not rate or rate.startswith('?') else rate,
               'elapsed': m.group('elapsed'), 'eta': None if (m.group('eta') or '?') == '?' else m.group('eta'),
               'kind': 'loading' if LOADING.search(label) or BYTES.search(m.group('total')) or BYTES.search(rate)
                       else 'warmup' if WARMUP.search(label) else 'work'}
        if m.group('post'):
            bar['postfix'] = m.group('post').strip()[:80]
        key = (label, m.group('total'))
        bars.pop(key, None)
        bars[key] = {k: v for k, v in bar.items() if v not in (None, '')}
    return list(bars.values())[-limit:] if limit else list(bars.values())


def benign_reason(line: str) -> str | None:
    body = _split(line)[1]
    for rx, why in NOISE:
        if rx.search(body):
            return why
    return None


def benign(line: str) -> bool:
    """True for a known-harmless line (see NOISE)."""
    return benign_reason(line) is not None


def _tracebacks(plain: list[tuple[str, str]]) -> list[dict]:
    """Python tracebacks, followed per process tag (vLLM workers interleave): {'idx': line numbers, 'text',
    'exc': the exception line or None, 'at': its line number}."""
    done, open_ = [], {}
    for i, (tag, body) in enumerate(plain):
        cur = open_.get(tag)
        if body.startswith('Traceback (most recent call last)'):
            if cur:
                done.append(cur)
            open_[tag] = {'idx': [i], 'body': [body], 'exc': None, 'at': None}
        elif cur is not None:
            cur['idx'].append(i)
            cur['body'].append(body)
            if body.strip() and not body[:1].isspace():
                cur['exc'], cur['at'] = body.strip(), i
                done.append(open_.pop(tag))
    done += open_.values()
    for tb in done:
        tb['text'] = '\n'.join(tb.pop('body'))
        tb['benign'] = next((why for rx, why in BENIGN_TRACEBACKS if rx.search(tb['text'])), None)
    return sorted(done, key=lambda t: t['idx'][0])


def filter_benign(text) -> tuple[str, dict[str, int]]:
    """Cleaned text without known-benign lines and harmless tracebacks -> (text, {reason: lines hidden})."""
    lines = clean(text).split('\n')
    plain = [_split(l) for l in lines]
    drop: dict[int, str] = {}
    for tb in _tracebacks(plain):
        if tb['benign']:
            drop.update({i: tb['benign'] for i in tb['idx']})
    hidden: dict[str, int] = {}
    kept = []
    for i, line in enumerate(lines):
        why = drop.get(i) or benign_reason(line)
        if why:
            hidden[why] = hidden.get(why, 0) + 1
        else:
            kept.append(line)
    return '\n'.join(kept), hidden


def _node_near(lines: list[str], i: int) -> str | None:
    for j in [i] + [k for d in (1, 2, 3) for k in (i + d, i - d)]:
        if 0 <= j < len(lines):
            m = NODE.search(lines[j])
            if m:
                return m.group(0)
    return None


def _get(row: dict, *keys):
    for k in keys:
        if row.get(k) not in (None, ''):
            return row[k]
    return None


def _exit(code: str | None) -> tuple[int | None, int | None]:
    """sacct ExitCode 'rc:signal' -> (rc, signal)."""
    try:
        rc, _, sig = str(code).partition(':')
        return int(rc), int(sig or 0)
    except ValueError:
        return None, None


def _limit_s(row: dict) -> float | None:
    raw = _get(row, 'time_limit_s')
    if raw is not None:
        return float(raw)
    raw = _get(row, 'TimelimitRaw')
    if raw is not None and str(raw).isdigit():
        return int(raw) * 60.0
    m = S.minutes(_get(row, 'Timelimit', 'time_limit'))
    return m * 60 if m else None


def _hit(f: dict, sig: dict | None, line: str, i: int, node: str | None, rank: int, **over) -> None:
    if sig is not None:
        base = {'name': sig['name'], 'category': sig['category'], 'meaning': sig['meaning'],
                'action': sig['action'], 'transient': sig.get('transient', False), 'level': sig.get('level', 'error')}
    else:
        base = {}
    base.update(over)
    key = base['name']
    if key in f:
        f[key]['count'] += 1
        if node and node not in f[key]['nodes']:
            f[key]['nodes'].append(node)
        return
    f[key] = {**base, 'line': line.strip()[:MAX_LINE], 'at': i, 'count': 1, 'nodes': [node] if node else [],
              'rank': rank}


def _state_findings(rows: list[dict], f: dict) -> tuple[bool, bool]:
    """Adds sacct-based findings -> (every row ended COMPLETED 0:0, some row ended otherwise)."""
    ok, failed = bool(rows), False
    for row in rows:
        state = (str(_get(row, 'State', 'state') or '').split() or [''])[0]
        rc, sig = _exit(_get(row, 'ExitCode', 'exit_code'))
        elapsed = _num(str(_get(row, 'ElapsedRaw', 'elapsed_s') or '')) or 0
        nodes = _get(row, 'NodeList', 'nodelist') or ''
        if state != 'COMPLETED' or rc not in (0, None) or sig:
            ok = False
            failed = failed or state in S.TERMINAL
        if state in STATES:
            cat, meaning, action, transient = STATES[state]
            if state == 'NODE_FAIL' and isinstance(elapsed, (int, float)) and elapsed <= 120:
                meaning = f"a node failed on ASC's side {int(elapsed)} s after the start "
            rank = 0 if cat == INFRA else 5 if cat != BENIGN else 1000
            _hit(f, None, f'{state} {_get(row, "ExitCode", "exit_code") or ""}'.strip(), -1, nodes or None, rank,
                 name=f'state_{state.lower()}', category=cat, meaning=meaning, action=action, transient=transient,
                 level='error' if cat != BENIGN else 'info')
        elif state == 'FAILED' and rc:
            cat, meaning = EXIT_CODES.get(rc, (UNKNOWN, f'exit code {rc}'))
            _hit(f, None, f'FAILED {rc}:{sig or 0}', -1, None, 900, name=f'exit_{rc}', category=cat,
                 meaning=f'batch script exit {rc}: {meaning}', action='', level='error')
        if sig:   # CANCELLED 0:15, TIMEOUT, ...: the state already explains the signal
            _hit(f, None, f'{state} {rc}:{sig}', -1, None, 900, name=f'signal_{sig}', category=UNKNOWN,
                 meaning=f'killed by signal {sig} ({SIGNALS.get(sig, "?")})', action='',
                 level='info' if state in STATES else 'error')
    return ok, failed


def _fast(rows: list[dict]) -> bool:
    """Every row ended within FAST_FAIL_S (unknown elapsed counts as fast)."""
    times = [_num(str(_get(r, 'ElapsedRaw', 'elapsed_s') or '')) for r in rows]
    return all(not isinstance(t, (int, float)) or t < FAST_FAIL_S for t in times)


def diagnose(log_text, sacct_rows: list[dict] | None = None) -> dict:
    """Classify a failed or suspicious job from its log text (cleaned here) and optional sacct rows (raw sacct keys
    like State/ExitCode/ElapsedRaw/NodeList/TimelimitRaw, or job_details' state/exit_code/elapsed_s/nodelist).
    -> {category, summary, findings: [{name, category, meaning, action, line, count, nodes, transient}],
        exceptions: [first real Python exception lines], benign_hidden: {reason: count}, note}."""
    lines = clean(log_text).split('\n')
    plain = [_split(l) for l in lines]
    bodies = [b for _, b in plain]
    f: dict[str, dict] = {}
    ok, failed = _state_findings(sacct_rows or [], f)

    hidden: dict[str, int] = {}
    skip: set[int] = set()
    tbs = _tracebacks(plain)
    for tb in tbs:
        if tb['benign']:
            skip.update(tb['idx'])
            hidden[tb['benign']] = hidden.get(tb['benign'], 0) + len(tb['idx'])
    first_after = {}
    line_sigs = [(k, s) for k, s in enumerate(SIGNATURES) if s.get('scope') != 'exception']
    for i, body in enumerate(bodies):
        if i in skip or not body.strip():
            continue
        why = benign_reason(lines[i])
        if why:
            hidden[why] = hidden.get(why, 0) + 1
            continue
        quiet = None
        for k, sig in line_sigs:
            if not sig['rx'].search(body):
                continue
            if sig.get('generic'):
                quiet = _quiet(lines[i]) if quiet is None else quiet
                if quiet:
                    continue
            over = {}
            if 'after' in sig:
                rx, cat, meaning, action = sig['after']
                if rx not in first_after:
                    first_after[rx] = next((j for j, b in enumerate(bodies) if re.search(rx, b)), None)
                if first_after[rx] is not None and first_after[rx] < i:
                    over = {'name': sig['name'] + '_late' if cat != BENIGN else sig['name'] + '_shutdown',
                            'category': cat, 'meaning': meaning, 'action': action,
                            'level': 'info' if cat == BENIGN else 'error'}
            if sig.get('completed_ok') and ok:
                over.update(category=BENIGN, level='info', action='', transient=False)
            if sig['category'] == BENIGN and not over:
                over = {'level': 'info'}
            m = sig['rx'].search(body)
            node = m.groupdict().get('node') or (_node_near(lines, i) if sig.get('node') else None)
            _hit(f, sig, body, i, node, (400 if sig.get('generic') else 10) + k, **over)
            break
        m = TASK_EXIT.search(body)
        if m:
            rc = int(m.group('rc'))
            cat, meaning = EXIT_CODES.get(rc, (UNKNOWN, f'exit code {rc}'))
            _hit(f, None, body, i, m.group('node'), 900, name=f'task_exit_{rc}', category=cat,
                 meaning=f'srun task exit {rc}: {meaning}', action='', transient=rc == 140, level='error')

    exceptions = []
    exc_sigs = [(k, s) for k, s in enumerate(SIGNATURES) if s.get('scope') == 'exception']
    for tb in tbs:
        if tb['benign'] or not tb['exc']:
            continue
        if tb['exc'] not in exceptions:
            exceptions.append(tb['exc'])
        sig = next(((k, s) for k, s in exc_sigs if s['rx'].search(tb['text'])
                    and re.search(s['exc'], tb['exc'])), None)
        if sig:
            _hit(f, sig[1], tb['exc'], tb['at'], _node_near(lines, tb['at']), 10 + sig[0])
        elif not any(v['at'] == tb['at'] for v in f.values()):
            _hit(f, None, tb['exc'], tb['at'], None, 500, name='python_exception', category=PROGRAM,
                 meaning='Python exception (the first one is usually the root cause)', action='see exceptions',
                 level='error')

    # launch warnings (Prolog hung, socket timeouts) decide only when the job failed fast with nothing more specific:
    # a step that ran for minutes and then failed had another cause
    specific = [v for v in f.values() if v['level'] == 'error' and v['category'] != BENIGN and v['rank'] < 900]
    if failed and not specific and _fast(sacct_rows or []):
        for v in f.values():
            if v['name'] in LAUNCH and v['level'] == 'warning':
                v['level'] = 'error'
    found = sorted(f.values(), key=lambda v: (v['level'] != 'error', v['category'] == BENIGN, v['rank'], v['at']))
    errors = [v for v in found if v['level'] == 'error' and v['category'] != BENIGN]
    if errors:
        category = errors[0]['category']
        if category == UNKNOWN:
            category = next((v['category'] for v in errors if v['category'] != UNKNOWN), UNKNOWN)
    elif failed:
        warn = [v for v in found if v['level'] == 'warning']
        explained = any(v['name'].startswith('state_') and v['category'] == BENIGN for v in found)   # cancelled
        category = warn[0]['category'] if warn else BENIGN if explained else UNKNOWN
    else:
        category = BENIGN
    states = [v for v in found if v['name'].startswith('state_')]      # 'cancelled' over its 'signal 15'
    top = (errors or [v for v in found if v['level'] == 'warning'] or states or found or [None])[0]
    if top:
        where = f' on {",".join(top["nodes"][:3])}' if top['nodes'] else ''
        summary = f'{category}: {top["meaning"]}{where}' + (f' -> {top["action"]}' if top['action'] else '')
    else:
        summary = f'{category}: no known failure signature' + (' (see the log tail)' if failed else '')
    res = {'category': category, 'summary': summary[:400]}
    if found:
        res['findings'] = [{k: v for k, v in x.items() if k not in ('rank', 'at') and v not in ([], False, '')}
                           for x in found[:MAX_FINDINGS]]
    if exceptions:
        res['exceptions'] = [e[:MAX_LINE] for e in exceptions[:5]]
    if hidden:
        res['benign_hidden'] = hidden
    if ok and errors:
        note = 'COMPLETED, but the log shows errors: check the outputs (a wrapper can hide exit codes)'
        row = sacct_rows[0]
        limit, elapsed = _limit_s(row), _num(str(_get(row, 'ElapsedRaw', 'elapsed_s') or ''))
        if limit and isinstance(elapsed, (int, float)) and elapsed < 0.2 * limit:
            note += f'; it used {int(elapsed)} s of a {int(limit)} s limit'
        res['note'] = note
    return res


def find_milestones(text, patterns: list[str]) -> list[dict]:
    """For each regex: whether it matched the cleaned log, the first matching line, its line number and the count.
    An invalid regex gives {'pattern', 'error'}."""
    lines = clean(text).split('\n')
    out = []
    for p in patterns:
        try:
            rx = re.compile(p)
        except re.error as e:
            out.append({'pattern': p, 'error': f'invalid regex: {e}'})
            continue
        hits = [i for i, l in enumerate(lines) if rx.search(l)]
        item = {'pattern': p, 'matched': bool(hits)}
        if hits:
            item.update(line=lines[hits[0]].strip()[:MAX_LINE], line_no=hits[0] + 1, count=len(hits))
        out.append(item)
    return out


def missing_note(state: str = '', elapsed_s: float | None = None, reason: str = '') -> str:
    """What to print for a log file that does not exist yet: during the node prolog it is normal."""
    if state.startswith('PENDING'):
        return 'not created yet (the job is pending)'
    if state.startswith('RUNNING') and (reason == 'Prolog' or (elapsed_s is not None and elapsed_s < 300)):
        return 'not created yet (node prolog; normal in the first minute or two)'
    return 'not there'
