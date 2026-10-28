"""Checks a job and login-node commands against the runbook (AGENTS.md §0, §5, §8, §9).

evaluate(): the static check of a job script plus sbatch options, before submission.
check_actual(): the same limits applied to Slurm's own reading of a submitted (held) job, the authoritative check.
approval_gaps(): what a given approval text does not cover (a 16-node job needs its node count named).
referenced_paths(), vllm_cache_dirs(): paths for check_job to test on the cluster; asc_rewrites(): what ASC's
submit filter printed in sbatch's output.
command_policy(): what run_command must refuse or have approved. These are guards against mistakes, not a security
boundary: an agent determined to get around them can.

Results: errors (must not be submitted as is), warnings (likely mistakes), needs_approval (a human has to approve),
inject (sbatch options the server adds: account, known bad nodes).
"""
from __future__ import annotations

import functools
import os
import re
import shlex
import subprocess

from . import config
from . import slurm as S

# approval limits: config [limits] approval_nodes, approval_tokens, headsup_nodes (a job of headsup_nodes or more needs
# its own approval naming the node count; AGENTS.md §0 rule 4)
HEADS_UP = 'heads-up'        # marks the approval item that a generic or budget approval does not cover
VLLM_MEM_MAX = 0.975         # 0.98 failed at startup on H100: 91.1 of 93.09 GiB free
PROTECTED_ENV = ('DATA', 'SCRATCH', 'HOME', 'USER', 'TMPDIR')
PATH_OPTS = ('output', 'error', 'input', 'chdir', 'export-file')

LMOD_RE = re.compile(r'^\s*(source|\.)\s+\S*/init/lmod/')
SET_RE = re.compile(r'\bset\s+([-+])([a-zA-Z]+)')
SET_O_RE = re.compile(r'\bset\s+([-+])o\s+(nounset|errexit)\b')
BARE_RC_RE = re.compile(r'^\s*[A-Za-z_]\w*=\$\?\s*(#.*)?$')
SAME_LINE_RC_RE = re.compile(r';\s*[A-Za-z_]\w*=\$\?')
DECLARE_RE = re.compile(r'^\s*(?:declare|typeset|local)\s+((?:-\w+\s+)+)(.*)$')


def lint_script(text: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    lines = text.splitlines()
    if not lines or not lines[0].startswith('#!'):
        errors.append('the script has no shebang line (#!/bin/bash): sbatch rejects it')
        lines = [''] + lines
    flags = ''.join(t[1:] for t in lines[0].split()[1:] if t.startswith('-') and not t.startswith('--'))
    nounset, errexit = 'u' in flags, 'e' in flags
    lmod_seen, in_quote, prev, rc_warned, decl_warned = False, False, '', False, False
    scopes: list[tuple[bool, bool]] = []      # `set` inside a `(` ... `)` subshell ends with it
    for raw in lines[1:]:
        part = raw
        if in_quote:                     # inside a multi-line single-quoted string (e.g. a bash -c '...' body)
            if raw.count("'") % 2 == 1:
                in_quote = False
                part = raw.split("'", 1)[1] if "'" in raw else ''
            else:
                continue
        if part.lstrip().startswith('#') or not part.strip():
            continue
        if part.count("'") % 2 == 1:
            in_quote = True
            part = part.rsplit("'", 1)[0]
        bare = part.split('#', 1)[0].strip()
        if bare == '(':
            scopes.append((nounset, errexit))
            continue
        if bare.startswith(')') and scopes:
            nounset, errexit = scopes.pop()
            part = bare[1:]
        if not lmod_seen and LMOD_RE.match(part):
            lmod_seen = True
            if nounset:
                errors.append('nounset (`set -u`) is on at the Lmod init: the EESSI init reads unset variables and the '
                              'job ends silently (AGENTS.md §8). Init Lmod first, then `set -euo pipefail`')
        for sign, letters in SET_RE.findall(part):
            if 'u' in letters:
                nounset = sign == '-'
            if 'e' in letters:
                errexit = sign == '-'
        for sign, name in SET_O_RE.findall(part):
            if name == 'nounset':
                nounset = sign == '-'
            else:
                errexit = sign == '-'
        m = DECLARE_RE.match(part)
        if m and nounset and not decl_warned and re.search(r'[aA]', m.group(1)):
            names = [t for t in m.group(2).split() if re.fullmatch(r'[A-Za-z_]\w*', t)]
            if names:
                warnings.append(f'`declare {m.group(1).strip()} {" ".join(names)}` without `=()` under `set -u`: bash '
                                'treats the empty array as unset, so `${#X[@]}` aborts the script ("unbound variable"). '
                                'Write `declare -A X=()` (AGENTS.md §8)')
                decl_warned = True
        if errexit and not rc_warned:
            standalone = BARE_RC_RE.match(part) and not prev.rstrip().endswith('||')
            if standalone or SAME_LINE_RC_RE.search(part):
                warnings.append(f'`{part.strip()}` under `set -e`: a failing command ends the script before `$?` is '
                                'read, so a retry loop around it never retries. Use `cmd && RC=0 || RC=$?` '
                                '(AGENTS.md §5 Patterns)')
                rc_warned = True
        prev = raw
    if any(l.strip() == '#ASC --vanilla' for l in lines[2:]):
        warnings.append('`#ASC --vanilla` is not on line 2; put it right after the shebang')
    return errors, warnings


def _limits(part: str | None, qos: str | None, time_str: str | None, est: dict, in_use: dict) -> tuple[list, list]:
    """The time, dev-QoS and approval rules shared by the static and the Slurm-side check."""
    errors: list[str] = []
    approval: list[str] = []
    mins = S.minutes(time_str)
    nodes = est['nodes']
    if mins is None:
        errors.append('no --time: set a tight limit (tight limits are backfilled sooner; billing covers only the time used)')
    elif mins == float('inf'):
        errors.append(f'--time {time_str} means no time limit: set a tight one')
    elif mins > S.MAX_HOURS * 60:
        errors.append(f'--time {time_str} is over the {S.MAX_HOURS} h maximum')
    if qos and qos.startswith('dev_'):
        if mins is not None and mins > S.DEV_LIMIT_MIN:
            errors.append(f'dev QoS allows at most {S.DEV_LIMIT_MIN} min (--time {time_str})')
        if nodes > S.DEV_MAX_NODES:
            errors.append(f'dev QoS allows at most {S.DEV_MAX_NODES} nodes ({nodes} requested)')
    used_n, used_t = int(in_use.get('nodes', 0)), float(in_use.get('tokens', 0.0))
    tokens = est['max_tokens']
    APPROVAL_NODES, APPROVAL_TOKENS = config.approval_nodes(), config.approval_tokens()
    HEADS_UP_NODES = config.headsup_nodes()
    if part == S.B200:
        approval.append('B200 partition (private ISTA nodes, no published price)')
    if nodes >= APPROVAL_NODES:
        approval.append(f'{nodes} nodes (≥ {APPROVAL_NODES} needs approval)')
    elif est['nodes_at_once'] >= APPROVAL_NODES:
        approval.append(f'the array runs {est["nodes_at_once"]} nodes at once')
    elif est['nodes_at_once'] + used_n >= APPROVAL_NODES:
        approval.append(f'{used_n} nodes already running or pending plus {est["nodes_at_once"]} = '
                        f'{used_n + est["nodes_at_once"]} nodes at once (jobs at the same time count together)')
    if est['nodes_at_once'] >= HEADS_UP_NODES:
        cost = f'up to {tokens:g} tokens' if tokens is not None else 'an unknown number of tokens'
        approval.append(f'{HEADS_UP}: {est["nodes_at_once"]} nodes at once need their own approval that names the node '
                        f'count; a budget or general approval does not cover it. Tell the human first: '
                        f'{est["nodes_at_once"]} nodes, {cost}, time limit {time_str or "?"} and the expected wall '
                        f'time, and check that a {APPROVAL_NODES}-node run of the same script has completed (AGENTS.md §0 rule 4)')
    if tokens is not None and tokens > APPROVAL_TOKENS:
        approval.append(f'up to {tokens:g} tokens at the time limit (> {APPROVAL_TOKENS:g})')
    elif tokens is not None and tokens + used_t > APPROVAL_TOKENS:
        approval.append(f'{used_t:g} tokens still committed by running or pending jobs plus up to {tokens:g} = '
                        f'{used_t + tokens:.1f} (> {APPROVAL_TOKENS:g}; a series counts together)')
    if qos and qos.startswith(('long_', 'fast_', 'high_')):
        approval.append(f'QoS {qos} (long or priority QoS)')
    return errors, approval


def approval_gaps(needs_approval: list[str], approval: str) -> list[str]:
    """The approval items that `approval` (the human's words) does not cover: all of them without text; a heads-up
    item (≥ headsup_nodes, 16 by default) unless the text names its node count ("16 nodes", "-N16", "16-node")."""
    if not (approval or '').strip():
        return list(needs_approval)
    gaps = []
    for item in needs_approval:
        m = re.match(rf'{HEADS_UP}: (\d+) nodes', item)
        if m and not _names_nodes(approval, m.group(1)):
            gaps.append(item)
    return gaps


# "16 node-hours", "16 nodes x 1h" are budgets, not a node count
NOT_A_COUNT = r'(?!\s*-?\s*(?:h(?:ou)?rs?|h)\b)(?!\s*[x×*]\s*\d)'
# "heads-up before 16 nodes", "4 nodes, not 16 nodes", "up to 16 N": a condition or a limit, not this job's approval
NEGATED = re.compile(r'\b(?:before|not|no|under|below|than|without|except|up\s+to|max(?:imum)?|budget(?:\s+of)?)'
                     r'\W*(?:a\W+|the\W+)?$', re.I)


def _names_nodes(text: str, n: str) -> bool:
    """Whether `text` approves a job of `n` nodes by naming the count ("16 nodes", "16-node", "-N16", "nodes=16")."""
    pat = rf'(?<!\d){n}\s*-?\s*(?:nodes?|N)\b{NOT_A_COUNT}|(?:-N\s*|\bnodes?\s*[=:]?\s*){n}(?!\d){NOT_A_COUNT}'
    return any(not NEGATED.search(text[max(0, m.start() - 20):m.start()]) for m in re.finditer(pat, text, re.I))


def evaluate(script_text: str, cli_tokens: list[str], in_use: dict | None = None,
             expected_minutes: float | None = None) -> dict:
    """in_use: {'nodes': ..., 'tokens': ...} held by the user's running and pending jobs (rule 11).
    expected_minutes: the caller's runtime estimate (else one stated in the script's header comments)."""
    in_use = in_use or {'nodes': 0, 'tokens': 0.0}
    sopts, serr = S.script_options(script_text)
    copts, cerr = S.parse_args(cli_tokens, cli=True)
    opts = {**sopts, **copts}
    lint_err, warnings = lint_script(script_text)
    errors = serr + cerr + lint_err
    if (syn := syntax_error(script_text)):
        errors.append(f'bash -n: {syn}')
    inject: list[str] = []
    lines = script_text.splitlines()

    acct, want, fs = opts.get('account'), config.account(), config.filesystem()
    if not want:
        if acct is None:
            warnings.append('no --account and no Slurm account configured: ' + config.account_error())
    elif acct is None:
        inject.append(f'--account={want}')
        opts['account'] = want
    elif acct != want:
        errors.append(f'account {acct!r} is wrong: use {want}'
                      + (f' ({fs} is the filesystem, not the account)' if fs and acct == fs else ''))

    part = opts.get('partition')
    qos = opts.get('qos')
    if not part:
        errors.append(f'no partition: set one of {sorted(S.PARTITIONS)}')
    elif part not in S.PARTITIONS:
        errors.append(f'unknown partition {part!r}: use one of {sorted(S.PARTITIONS)}')
    if part in S.QOS:
        if not qos:
            errors.append(f'no QoS: set --qos to one of {sorted(S.QOS[part])} (dev_* for tests of up to 10 min)')
        elif qos not in S.QOS[part]:
            errors.append(f'QoS {qos!r} does not match partition {part}: use one of {sorted(S.QOS[part])}')

    mins = S.minutes(opts.get('time'))
    if mins is not None and 24 * 60 < mins <= S.MAX_HOURS * 60:
        warnings.append(f'--time {opts["time"]} is over 24 h; is that needed?')

    vanilla = len(lines) > 1 and lines[1].strip() == '#ASC --vanilla'
    if part == S.CPU and not vanilla:
        # ASC's comfort mode on the CPU partition (sbatch --test-only, 2026-10-14): with --nodes, whole nodes of 190
        # single-core tasks (-N1 -n1 -c8 rejected); without, a shared node with --cpus-per-task set to 1
        fix = ('For several cores for one program add `#ASC --vanilla` on line 2 with --nodes=1 --ntasks=1 '
               '--cpus-per-task=<cores> --threads-per-core=1 --mem=<GB>G; draft_job(kind="cpu") writes it')
        if opts.get('cpus-per-task', '1') not in ('', '1'):
            if 'nodes' in opts:
                errors.append('--nodes on the CPU partition in comfort mode means whole nodes of 190 single-core tasks, '
                              'so --cpus-per-task > 1 is rejected ("Requested node configuration is not available"). '
                              + fix)
            else:
                errors.append('comfort mode on the CPU partition sets --cpus-per-task=1 on a shared node: the job would '
                              'get one core. ' + fix)
        elif 'nodes' in opts and 'exclusive' not in opts:
            warnings.append('--nodes on the CPU partition in comfort mode means whole nodes (190 tasks each, billed in '
                            'full at 0.96 tokens/h); for part of a node drop --nodes or add `#ASC --vanilla` on line 2')

    if part in S.CORES and opts.get('cpus-per-task', '').isdigit():
        cpt = int(opts['cpus-per-task'])
        tpn = int(opts['ntasks-per-node']) if opts.get('ntasks-per-node', '').isdigit() else 1
        per_node = cpt * tpn
        tpc1 = opts.get('threads-per-core') == '1'
        if part == S.CPU and not vanilla:
            cap, why = S.CORES[S.CPU], (' (ASC comfort mode turns a CPU-partition request into single-core tasks; '
                                        'add `#ASC --vanilla` on line 2 for plain Slurm)')
        else:
            cap, why = (S.CORES[part] if tpc1 else S.THREADS[part]), ''
        if per_node > cap:
            errors.append(f'{per_node} CPUs per node{" with --threads-per-core=1" if tpc1 else ""} exceed the {cap} that '
                          f'{part} nodes give Slurm: "Requested node configuration is not available" (AGENTS.md §8). '
                          f'Use at most --cpus-per-task={cap // tpn}{why}')

    if part == S.GPU and 'exclude' not in opts and 'nodelist' not in opts:
        inject.append('--exclude=' + ','.join(S.BAD_NODES))
    elif part == S.GPU and 'exclude' in opts:
        excluded = set(S.expand_hostlist(opts['exclude']))
        missing = [n for n in S.BAD_NODES if n not in excluded]
        if missing:
            warnings.append(f'--exclude lacks the known bad nodes {",".join(missing)} (NOTES.md, Hardware)')
    for k in ('output', 'error'):
        v = opts.get(k, '')
        if '/' in v and '%' in v.rsplit('/', 1)[0]:
            warnings.append(f'--{k} {v}: the directory part has a % pattern; Slurm does not create directories, so '
                            'the job fails without a log unless it exists')

    for k in PATH_OPTS:
        v = opts.get(k, '')
        if '$' in v or v.startswith('~'):
            errors.append(f'--{k} {v}: sbatch expands neither $VARIABLES nor ~ in #SBATCH lines or sbatch_args, so this '
                          'is a literal path. Use one relative to the submit directory (logs/%x-%j.out) or an absolute '
                          f'one (/data/{config.filesystem() or "<filesystem>"}/<user>/...)')
    for src in (sopts, copts):
        bad = [k for k in _export_keys(src.get('export', '')) if k in PROTECTED_ENV or k.startswith('SLURM_')]
        if bad:
            errors.append(f'--export sets {", ".join(bad)}: that replaces the cluster\'s own variable for the whole job '
                          '($DATA/envs/... became a relative path). Rename the variable (e.g. SET=...)')
    top = S.array_max_index(opts.get('array'))
    if top is not None and top >= S.MAX_ARRAY_SIZE:
        errors.append(f'--array index {top} is not below MaxArraySize {S.MAX_ARRAY_SIZE}: sbatch rejects it')
    rule_err, rule_warn = _script_rules(script_text, opts, vanilla)
    errors += rule_err
    warnings += rule_warn
    if (w := time_vs_expected(opts.get('time'), expected_minutes or stated_minutes(script_text))):
        warnings.append(w)

    est = S.estimate(opts)
    lim_err, approval = _limits(part, qos, opts.get('time'), est, in_use)
    errors += lim_err
    busy = int((in_use.get('dev_nodes') or {}).get(qos, 0)) if qos else 0
    if busy and busy + est['nodes_at_once'] > S.DEV_MAX_NODES:
        # observed 2026-10-14: a second dev job pended with QOSMaxNodePerUserLimit while a 2-node dev job ran
        warnings.append(f'{qos} runs at most {S.DEV_MAX_NODES} nodes per user at once, across jobs: with your {busy} '
                        f'node(s) there this job waits (QOSMaxNodePerUserLimit) until they end')
    return {'options': opts, 'errors': errors, 'warnings': warnings, 'needs_approval': approval,
            'inject': inject, 'estimate': est}


def check_actual(job: dict, in_use: dict | None = None) -> dict:
    """The limits applied to Slurm's reading of a submitted job (a parse_scontrol() record)."""
    in_use = in_use or {'nodes': 0, 'tokens': 0.0}
    nums = [int(x) for x in re.findall(r'\d+', job.get('NumNodes', '') or '')]
    opts = {'partition': job.get('Partition', ''), 'qos': job.get('QOS', ''), 'time': job.get('TimeLimit', ''),
            'nodes': str(max(nums) if nums else 1)}
    arr = job.get('ArrayTaskId', '')
    if arr and arr not in ('(null)', 'N/A'):
        thr = job.get('ArrayTaskThrottle', '')
        opts['array'] = arr if '%' in arr or not thr.isdigit() or thr == '0' else f'{arr}%{thr}'
    m = re.search(r'gres/gpu(?::[a-z0-9]+)?=(\d+)', job.get('TRES', '') or '')
    if m and opts['nodes'] == '1':
        opts['gres'] = f'gpu:{m.group(1)}'
    est = S.estimate(opts)
    errors, approval = _limits(opts['partition'], opts['qos'], opts['time'], est, in_use)
    if opts['partition'] and opts['partition'] not in S.PARTITIONS:
        errors.append(f'Slurm put the job on unknown partition {opts["partition"]!r}')
    elif opts['partition'] in S.QOS and opts['qos'] not in S.QOS[opts['partition']]:
        errors.append(f'Slurm reads QoS {opts["qos"]!r} for partition {opts["partition"]}')
    return {'errors': errors, 'needs_approval': approval, 'estimate': est,
            'slurm': {k: job.get(k) for k in ('JobId', 'Partition', 'QOS', 'NumNodes', 'TimeLimit', 'ArrayTaskId')
                      if job.get(k)}}


# ---------------------------------------------------------------- script rules beyond the line-by-line lint

CMD = r'(?:^|[;&|(){]|\bdo\b|\bthen\b|\belse\b|\btime\b|\bexec\b)\s*'
LOOP_RE = re.compile(CMD + r'(?:for|while|until)\b')
DONE_RE = re.compile(CMD + r'done\b')
FUNC_RE = re.compile(r'^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)')
OPEN_RE = re.compile(r'(?:^|[\s;&|])\{(?=\s|$)')
CLOSE_RE = re.compile(r'(?:^|[\s;&])\}(?=[\s;&|>)]|$)')
HEREDOC_RE = re.compile(r'(?<!<)<<-?(?!<)\s*([\'"]?)([A-Za-z_]\w*)\1')     # not <<< (a here-string)
ARITH_RE = re.compile(r'\$?\(\((?:[^()]|\([^()]*\))*\)\)')
SRUN_RE = re.compile(CMD + r'srun\b(.*)')
# EESSI's init/bash sets up Lmod too
LMOD_INIT = re.compile(r'init/lmod/|lmod\.sh\b|/etc/profile\b|/init/(?:bash|sh)\b|lmod/lmod/init/')
SERVER_RE = re.compile(r'\bvllm\s+serve\b|\bllama-server\b|\bvllm-router\b|vllm\.entrypoints')
CACHE_VARS = ('XDG_CACHE_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'HF_HOME',
              'APPTAINER_CACHEDIR', 'DG_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE')
COMPILE_CACHES = ('VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'XDG_CACHE_HOME')


def _code_lines(text: str) -> list[str]:
    """The script's command lines: comments (also the #SBATCH header) and here-document bodies removed, backslash
    continuations joined. A quoted string over several lines (a `bash -c '...'` body) still gives one line each."""
    out: list[str] = []
    cur: list[str] = []
    quote, heredoc = '', None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if heredoc is not None:
            j = text.find('\n', i)
            j = n if j < 0 else j
            if text[i:j].strip() == heredoc:
                heredoc = None
            i = j + 1
            continue
        if quote:
            if c == '\\' and quote == '"' and i + 1 < n:
                cur.append(text[i:i + 2])
                i += 2
                continue
            if c == quote:
                quote = ''
            if c == '\n':
                out.append(''.join(cur))
                cur = []
            else:
                cur.append(c)
            i += 1
            continue
        if c == '\\' and i + 1 < n:
            cur.append(' ' if text[i + 1] == '\n' else text[i:i + 2])
            i += 2
            continue
        if c in '\'"':
            quote = c
        elif c == '#' and (not cur or cur[-1] in ' \t;&|('):
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        elif c == '\n':
            line = ''.join(cur)
            out.append(line)
            cur = []
            if (m := HEREDOC_RE.search(ARITH_RE.sub('', line))):     # `$(( 1 << k ))` is a shift
                heredoc = m.group(2)
            i += 1
            continue
        cur.append(c)
        i += 1
    out.append(''.join(cur))
    return [l.rstrip() for l in out if l.strip()]


def _structure(lines: list[str]) -> list[tuple[int, str | None]]:
    """Per command line: the loop depth it runs at (a line opening a loop counts as inside it) and the function it
    belongs to (None at the top level)."""
    out = []
    depth, braces, func, func_at = 0, 0, None, 0
    for line in lines:
        opens, closes = len(LOOP_RE.findall(line)), len(DONE_RE.findall(line))
        m = FUNC_RE.match(line)
        if m and func is None:
            func, func_at = m.group(1), braces
        out.append((depth + (1 if opens else 0), func))
        depth = max(0, depth + opens - closes)
        braces = max(0, braces + len(OPEN_RE.findall(line)) - len(CLOSE_RE.findall(line)))
        if func is not None and braces <= func_at:
            func = None
    return out


def _loops(lines: list[str], info: list[tuple[int, str | None]]) -> list[list[str]]:
    """The outermost loops, each as its list of lines (head to `done`)."""
    loops, cur = [], None
    for line, (d, _) in zip(lines, info):
        if d >= 1 and cur is None:
            cur = []
        if cur is not None:
            cur.append(line)
            depth = sum(len(LOOP_RE.findall(l)) - len(DONE_RE.findall(l)) for l in cur)
            if depth <= 0:
                loops.append(cur)
                cur = None
    if cur:
        loops.append(cur)
    return loops


def _words(line: str) -> list[str]:
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def _assignments(lines: list[str]) -> list[tuple[str, str, str]]:
    """(name, value, line) for plain assignments, also after `export`/`readonly`/`declare -x` and env prefixes."""
    out = []
    for line in lines:
        for m in re.finditer(r'(?:^|[\s;&(])(?:export\s+|readonly\s+|declare\s+-x\s+)?([A-Za-z_]\w*)=('
                             r'"[^"]*"|\'[^\']*\'|\$\([^)]*\)|[^\s;&|)]*)', line):
            out.append((m.group(1), m.group(2).strip('"\''), line))
    return out


def _export_keys(value: str) -> list[str]:
    """Variables an --export value sets (ALL,NAME=value,...; NAME alone only passes the existing one on)."""
    return [p.split('=', 1)[0].strip() for p in value.split(',') if '=' in p]


def _int(v: str | None) -> int:
    return int(v) if v and v.isdigit() else 0


@functools.lru_cache(maxsize=1)
def _bash_ok() -> bool:
    """A local bash of version 4 or later (macOS's 3.2 parses some of the cluster's bash 5 syntax differently)."""
    try:
        r = subprocess.run(['bash', '-c', 'echo ${BASH_VERSINFO[0]}'], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().isdigit() and int(r.stdout) >= 4
    except (OSError, subprocess.TimeoutExpired):
        return False


def syntax_error(text: str) -> str | None:
    """`bash -n` on the script, locally (no cluster): the first error line, or None (also without a usable bash).
    With extglob on: under -n a `shopt -s extglob` never runs, and extglob only widens what parses."""
    if not _bash_ok():
        return None
    try:
        r = subprocess.run(['bash', '-O', 'extglob', '-n'], input=text, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode == 0:
        return None
    msg = next((l for l in r.stderr.splitlines() if l.strip()), 'syntax error')
    return re.sub(r'^(?:bash|/\S+/bash|environment): ', '', msg)


def _script_rules(text: str, opts: dict, vanilla: bool) -> tuple[list[str], list[str]]:
    """Rules on the script body together with its options (from pitfalls seen in real agent sessions, AGENTS.md §8)."""
    errors: list[str] = []
    warnings: list[str] = []
    lines = _code_lines(text)
    info = _structure(lines)
    joined = '\n'.join(lines)
    part, qos = opts.get('partition'), opts.get('qos') or ''
    assigns = _assignments(lines)
    srun_lines = [m.group(1) for l in lines if (m := SRUN_RE.search(l))]

    # comfort mode, CPU partition: -n N without --nodes gives N one-core tasks and the batch script one CPU
    ntasks = max(_int(opts.get('ntasks')), _int(opts.get('ntasks-per-node')))
    if part == S.CPU and not vanilla and ntasks > 1 and 'nodes' not in opts and not srun_lines:
        warnings.append(f'comfort mode gives {ntasks} single-core tasks, and the batch script itself 1 CPU: without '
                        '`srun` a multi-process program runs on one core. For one program with '
                        f'{ntasks} cores use `#ASC --vanilla` on line 2 with --ntasks=1 --cpus-per-task={ntasks} '
                        '--threads-per-core=1 (draft_job kind="cpu")')

    # comfort mode, GPU partition
    if part == S.GPU and not vanilla:
        gpn = S.gpus_per_node(opts)
        mem = next((k for k in ('mem', 'mem-per-cpu', 'mem-per-gpu') if k in opts), None)
        if mem:
            errors.append(f'--{mem} on the GPU partition in comfort mode is rejected ("gpu partitions only support '
                          'full node/half node memory via gpu count"): drop it, comfort mode sets memory from the GPU '
                          'count; or add `#ASC --vanilla` on line 2 (whole nodes: --mem=0)')
        if 'nodes' in opts and gpn is not None and 1 <= gpn <= 3:
            errors.append(f'--nodes with --gres=gpu:{gpn} is rejected in comfort mode ("Cannot specify number of GPUs '
                          'when requesting full nodes"): 1-3 GPUs: --gres=gpu:N without --nodes; whole nodes: '
                          '--nodes=N --gres=gpu:4')
        if gpn == 4:
            layout = [k for k in ('ntasks', 'ntasks-per-node', 'cpus-per-task') if k in opts]
            fix = ('for one process per node with all GPUs and CPUs add `#ASC --vanilla` on line 2 with '
                   '--ntasks-per-node=1 --cpus-per-task=352 --gres=gpu:4 --exclusive --mem=0 (draft_job kind="gpu-node")')
            if layout:
                warnings.append(f'comfort mode on whole GPU nodes replaces {", ".join("--" + k for k in layout)} with 4 '
                                f'tasks x 22 CPUs per node (one GPU each); {fix}')
            plain = [r for r in srun_lines if not re.search(r'(?:^|\s)(?:-n\s*\d|--ntasks|--overlap|--jobid)', r)]
            if plain:
                warnings.append(f'`srun` without -n in comfort mode on whole GPU nodes starts 4 copies per node, one GPU '
                                f'each; {fix}, or give srun -n')

    # overwritten cluster variables in the script
    over = sorted({n for n, _, l in assigns if n in ('DATA', 'SCRATCH', 'HOME')
                   and re.match(rf'^\s*(?:export\s+|readonly\s+|declare\s+-x\s+)?{n}=\S*\s*$', l)})
    if over:
        warnings.append(f'the script assigns {", ".join(over)}: every later ${over[0]}/... path then points elsewhere '
                        '. Use another name (e.g. SET)')

    # caches
    home = sorted({n for n, v, _ in assigns if n in CACHE_VARS and re.match(r'(~|\$HOME\b|\$\{HOME\}|/home/)', v)})
    if home:
        warnings.append(f'{", ".join(home)} under $HOME (50 GB quota, ASC asks not to): use $SCRATCH (AGENTS.md §6)')
    if re.search(r'\bDG_JIT_CACHE_DIR\b', joined):
        warnings.append('DG_JIT_CACHE_DIR is not read by DeepGEMM; its variable is DG_CACHE_DIR (AGENTS.md §6)')
    loops = _loops(lines, info)
    for body in loops:
        b = '\n'.join(body)
        if 'CUDA_VISIBLE_DEVICES=' in b and re.search(r'&\s*$', b, re.M):
            fixed = sorted({n for n, _, l in assigns if n in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')
                            and l not in body})
            if fixed and not any(re.search(rf'\b{n}=', b) for n in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                                                                   'XDG_CACHE_HOME')):
                warnings.append(f'parallel runs per GPU share one {" and ".join(fixed)}: parallel compiles collide; set '
                                'one cache directory per run inside the loop (AGENTS.md §5 Patterns)')
                break

    # declared multi-node launches
    nodes = S.node_count(opts)
    for body in loops:
        b = '\n'.join(body)
        if (re.search(r'\bsrun\b[^\n]*(?:\s-w\s|--nodelist)', b) and re.search(r'&\s*$', b, re.M)
                and not re.search(r'\bsleep\b', b)):
            warnings.append('a loop launches `srun -w <node> ... &` steps without a pause: 16 launches at once timed out '
                            'on two nodes. Add `sleep 0.5` between launches and relaunch a failed step once')
            break
    if nodes >= 4 and srun_lines:
        in_loop = any(d >= 1 and SRUN_RE.search(l) for l, (d, _) in zip(lines, info))
        if not in_loop and 'RC=$?' not in joined.replace(' ', ''):
            warnings.append(f'{nodes} nodes and no retry around `srun`: launches on 8-16 nodes fail now and then '
                            '("Prolog hung", socket timeouts). Retry a step that failed within 150 s (AGENTS.md §5 '
                            'Patterns; draft_job writes it)')

    # Lmod
    lmod_at = next((k for k, l in enumerate(lines) if re.search(LMOD_INIT, l)), None)
    mod_at = next((k for k, l in enumerate(lines) if re.match(r'\s*(?:module|ml)\s+\S', l)), None)
    if mod_at is not None and (lmod_at is None or mod_at < lmod_at):
        warnings.append('`module` before the Lmod init: batch shells have no `module` command ("module: command not '
                        'found"). Source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/sh first (AGENTS.md §6)')

    # hf download --include A B
    for l in lines:
        if re.search(r'\b(?:hf|huggingface-cli)\s+download\b', l):
            w = _words(l)
            for k, t in enumerate(w[:-2]):
                if t == '--include' and _hf_repo_before(w, k) and not w[k + 2].startswith('-') and w[k + 2] not in ('&&', '||', '|', ';', '&') \
                        and not re.match(r'^[0-9]*[<>]', w[k + 2]):
                    warnings.append('`hf download --include A B` fetches only B (it reads B as a file name and then '
                                    'ignores --include): give one --include per pattern (AGENTS.md §8)')
                    break
            else:
                continue
            break

    # apptainer pull/build on Weka
    if re.search(r'\b(?:apptainer|singularity)\s+(?:pull|build)\b', joined):
        tmp = [v for n, v, _ in assigns if n in ('APPTAINER_TMPDIR', 'SINGULARITY_TMPDIR')]
        if not any(re.search(r'/local\b|/dev/shm|\$TMPDIR|\$\{TMPDIR|/tmp\b', v) for v in tmp):
            warnings.append('`apptainer pull/build` without APPTAINER_TMPDIR on /local (or /dev/shm): unpacking on Weka '
                            'took 1783 s for a 7.5 GB image. export APPTAINER_TMPDIR=/local/$USER/apptainer_tmp '
                            '(AGENTS.md §5 skeleton)')

    # servers and waits
    curl_warned = False
    for body in loops:
        for l in body:
            rest = l.split('curl', 1)[1] if re.search(r'\bcurl\b', l) else None
            # localhost refuses at once while a server is down; a remote host that drops packets hangs curl
            if rest is not None and not re.search(r'(?:^|\s)(?:--max-time\b|-[a-zA-Z]*m)', rest) \
                    and not re.search(r'//(?:127\.0\.0\.1|localhost)\b', rest):
                warnings.append(f'`curl` in a loop without -m/--max-time: one hung request blocks the loop and its '
                                f'deadline (`{l.strip()[:60]}`); add -m 5 for health checks')
                curl_warned = True
                break
        if curl_warned:
            break
    if SERVER_RE.search(joined):
        e, w = _server_rules(lines, info, joined, part, qos, opts)
        errors += e
        warnings += w
    return errors, warnings


HF_FLAGS = {'--quiet', '-q', '--force-download', '--resume-download', '--dry-run', '--help'}


def _hf_repo_before(w: list[str], k: int) -> bool:
    """Whether the repo id (the first positional after `download`) comes before w[k]: `hf download --include A REPO`
    is valid, REPO being the positional."""
    start = next((i for i, t in enumerate(w) if t == 'download'), k)
    for i in range(start + 1, k):
        prev = w[i - 1]
        takes_value = prev.startswith('-') and '=' not in prev and prev not in HF_FLAGS and i - 1 > start
        if not w[i].startswith('-') and not takes_value:
            return True
    return False


def _server_rules(lines, info, joined, part, qos, opts) -> tuple[list[str], list[str]]:
    """vLLM, llama.cpp and router scripts."""
    errors: list[str] = []
    warnings: list[str] = []
    vllm = re.search(r'\bvllm\s+serve\b|vllm\.entrypoints', joined)
    if vllm and part in (S.GPU, None):
        for m in re.finditer(r'--gpu-memory-utilization[=\s]+["\']?([0-9]*\.?[0-9]+)', joined):
            if float(m.group(1)) > VLLM_MEM_MAX:
                errors.append(f'--gpu-memory-utilization {m.group(1)} is above {VLLM_MEM_MAX} on H100: only 91.1 of '
                                f'93.09 GiB are free at startup, so vLLM refuses to start (0.98 did)')
                break
    serve_lines = [k for k, l in enumerate(lines) if re.search(r'\bvllm\s+serve\b', l)]
    several = 'CUDA_VISIBLE_DEVICES' in joined and (len(serve_lines) > 1 or any(info[k][0] for k in serve_lines))
    if several and 'OMP_NUM_THREADS' not in joined:
        warnings.append('several vLLM servers per node without OMP_NUM_THREADS: each sets it to CPUs / its own workers, '
                        'unaware of the others, so loading oversubscribes the CPUs (>30 min instead of '
                        '19 s on one run). export OMP_NUM_THREADS=1')
    if re.search(r'\bvllm-router\b|--worker-urls', joined) and not re.search(r'\bulimit\s+-\w*n', joined):
        warnings.append('a router without `ulimit -n 65536`: jobs inherit the login node\'s 1024 open files, and at high '
                        'concurrency the router answered 503 (AGENTS.md §8)')
    if (vllm and not re.search(r'--api-key|VLLM_API_KEY', joined)
            and not re.search(r'--host[=\s]+["\']?(?:127\.0\.0\.1|localhost)', joined)):
        warnings.append('`vllm serve` listens on all interfaces (no --host 127.0.0.1) without --api-key: anyone on the '
                        'cluster network can use it. Pass a random per-job --api-key (AGENTS.md §5 Patterns)')
    if re.search(r'\bllama-server\b[^\n]*--host[=\s]+["\']?0\.0\.0\.0', joined) and '--api-key' not in joined:
        warnings.append('`llama-server --host 0.0.0.0` without --api-key: pass a random per-job key')
    if vllm and qos.startswith('dev_') and not vllm_cache_dirs('\n'.join(lines), opts.get('job-name')):
        warnings.append(COLD_START)
    # a bare `wait` at the top level while a backgrounded server still runs waits for the server, not the work
    killers = {f for l, (_, f) in zip(lines, info) if f and re.search(r'\bkill\b', l)}
    bg = False
    for l, (_, f) in zip(lines, info):
        if f is not None:
            continue
        if SERVER_RE.search(l) and re.search(r'(?<!&)&\s*$', l):
            bg = True
        elif bg and (re.search(r'\bkill\b', l) or any(re.search(rf'(?:^|[;&|]\s*){re.escape(k)}\b', l) for k in killers)):
            bg = False
        elif bg and re.fullmatch(r'\s*wait\s*', l):
            warnings.append('a bare `wait` while a server runs in the background waits for the server too, so the job '
                            'hangs to its time limit: wait for the work\'s PIDs (`wait $PID`), or stop the server first')
            break
    return errors, warnings


COLD_START = ('vLLM on the dev QoS without a compile cache kept between jobs ($SCRATCH/cache-...): a cold first start '
              '(compile, graph capture) took 5-11 min (GLM-5.3: 671 s), which may not fit 10 min. Set '
              'VLLM_CACHE_ROOT/TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR under $SCRATCH and warm them once on the normal QoS')


def vllm_cache_dirs(script_text: str, job_name: str | None = None) -> list[str]:
    """The compile cache directories a script keeps between jobs (under $SCRATCH or $DATA; $SLURM_JOB_NAME replaced
    by `job_name`). Paths with other job-specific variables ($SLURM_JOB_ID, a loop index) are left out: they are
    always cold. check_job can test these on the login node: if none exists yet, COLD_START applies."""
    lines = _code_lines(script_text)
    values = {}
    for n, v, _ in _assignments(lines):
        values.setdefault(n, v)
    out = []
    for n, v, _ in _assignments(lines):
        if n not in COMPILE_CACHES:
            continue
        r = _resolve(v, values, job_name)
        if r and re.match(r'(\$DATA|\$SCRATCH|/data/|/scratch/)', r) and '$' not in r.split('/', 1)[-1]:
            out.append(r)
    return list(dict.fromkeys(out))


BASE_VARS = ('DATA', 'SCRATCH', 'HOME')


def _resolve(value: str, values: dict, job_name: str | None = None, depth: int = 0) -> str | None:
    """A shell value with $VAR / ${VAR} / ${VAR:-default} replaced from the script's own assignments; $DATA, $SCRATCH
    and $HOME stay as they are (the cluster expands them). None when it uses $(...)."""
    if '$(' in value or '`' in value or depth > 6:
        return None

    def sub(m):
        name, default = m.group(1) or m.group(3), m.group(2)
        if name in BASE_VARS:
            return '$' + name
        if name == 'SLURM_JOB_NAME' and job_name:
            return job_name
        if name in values and values[name] != value:
            r = _resolve(values[name], values, job_name, depth + 1)
            if r is not None and (default is None or not re.search(rf'\$\{{?{name}\b', values[name])):
                return r
        if default is not None:
            r = _resolve(default, values, job_name, depth + 1)
            return r if r is not None else m.group(0)
        return m.group(0)
    return re.sub(r'\$\{([A-Za-z_]\w*)(?::?-([^}]*))?\}|\$([A-Za-z_]\w*)', sub, value)


ASSET_RE = re.compile(r'\.sif$|/models?/|/containers/|/envs?/|/venvs?\b|/conda\b|/bin/(?:python3?|torchrun|activate|vllm)'
                      r'$|/datasets?/|/opt/|\.(?:jsonl|parquet|safetensors|gguf|npz|csv|arrow|tar|pt|bin)$|/data/|/input')
SKIP_RE = re.compile(r'cache|/logs?\b|/tmp\b|/results?/|/out\b|/outputs?\b|/runs?/|\.log$|\.out$|\.err$', re.I)
OUT_FLAGS = re.compile(r'^--?(?:out|output|output[-_]dir|out[-_]dir|save[-_]?\w*|local-dir|log[-_]?\w*|cache[-_]?\w*|'
                       r'dest|prefix|o|O)$')         # -O: wget's output file
CONDA_CREATE = re.compile(r'\b(?:conda|mamba|micromamba)\s+(?:env\s+)?create\b')


def referenced_paths(script_text: str) -> list[str]:
    """The paths a job reads that must exist before it starts: models, containers (.sif), envs (venv, conda, a
    python interpreter) and input data, as absolute paths or based on $DATA/$SCRATCH/$HOME (in that form, for the
    cluster to expand). Conservative: paths the script creates or writes (mkdir, redirections, cp/mv targets,
    `apptainer pull`, --out/--local-dir), caches, logs and results, /local, and paths with variables the script
    does not set (job ids, loop indices) are left out."""
    lines = _code_lines(script_text)
    first: dict[str, str] = {}
    for n, v, _ in _assignments(lines):
        first.setdefault(n, v)
    # a variable is read from its latest assignment before the line (the first one if none came before: functions
    # defined above their variables), so a staging `M=$MODEL ... M=/local/...` does not make $MODEL "created"
    at: list[dict] = []
    values = dict(first)
    for l in lines:
        for n, v, _ in _assignments([l]):
            if n in values and values[n] != v:      # X=$X/sub extends the value before it
                v = re.sub(rf'\$\{{{n}\}}|\${n}\b', lambda _m, prev=values[n]: prev, v)
            values[n] = v
        at.append(dict(values))
    created: set[str] = set()

    def res(tok: str, vals: dict) -> str | None:
        r = _resolve(tok.strip('"\');)'), vals)
        return r.rstrip('/') if r else None

    for l, vals in zip(lines, at):
        w = _words(l)
        res_ = functools.partial(res, vals=vals)
        for k, t in enumerate(w):
            nxt = w[k + 1] if k + 1 < len(w) else ''
            if nxt and (t in ('>', '>>', '1>', '2>', '&>', '2>>', 'tee') or t == '-a' and k and w[k - 1] == 'tee'):
                created.add(res_(nxt) or '')
            elif re.match(r'^[0-9&]?>>?\S', t):
                created.add(res_(re.sub(r'^[0-9&]?>>?', '', t)) or '')
            elif OUT_FLAGS.match(t.split('=', 1)[0]) and ('=' in t or nxt):
                created.add(res_(t.split('=', 1)[1] if '=' in t else nxt) or '')
        if 'venv' in w or 'virtualenv' in w:          # python -m venv X, uv venv X: the script makes the env
            k = w.index('venv') if 'venv' in w else w.index('virtualenv')
            created.update(res_(t) or '' for t in w[k + 1:] if not t.startswith('-'))
        for k, t in enumerate(w):                     # the arguments of mkdir/touch (not the rest of the line)
            if t in ('mkdir', 'touch') or t == 'ln' and k == 0:
                for a in w[k + 1:]:
                    if a in SEPARATORS or a in ('then', 'do', '{', '}'):
                        break
                    if not a.startswith('-'):
                        created.add(res_(a) or '')
        if CONDA_CREATE.search(l):                     # conda env create -p X
            created.update(res_(w[k + 1]) or '' for k, t in enumerate(w[:-1]) if t == '-p')
        if 'git' in w and 'clone' in w:                 # git clone URL DIR
            args = [t for t in w[w.index('clone') + 1:] if not t.startswith('-')]
            if len(args) >= 2:
                created.add(res_(args[1]) or '')
        for m in re.finditer(r'\b(?:cp|mv|rsync|ln|pull|build)\b([^;&|]*)', l):
            args = [t for t in _words(m.group(1)) if not t.startswith('-')]
            if args:
                created.add(res_(args[0] if m.group(0).startswith(('pull', 'build')) else args[-1]) or '')
    created.discard('')

    def is_created(p: str) -> bool:
        return any(p == c or p.startswith(c + '/') or c.startswith(p + '.') for c in created)

    candidates = []
    for l, vals in zip(lines, at):
        for n, v, _ in _assignments([l]):
            candidates.append((n, v, vals))
    for l, vals in zip(lines, at):
        for t in _words(l):
            for piece in re.split(r'[=,:]', t) if not t.startswith(('$', '/', '~')) else [t]:
                candidates.append(('', piece, vals))
    out = []
    for name, raw, vals in candidates:
        if not re.match(r'["\']?(\$\{?(DATA|SCRATCH|HOME)\b|/data/|/scratch/|/home/|~/|\$\{?[A-Za-z_])', raw):
            continue
        p = res(raw, vals)
        if not p or not re.match(r'(\$DATA|\$SCRATCH|\$HOME|/data/|/scratch/|/home/|~/)', p):
            continue
        p = p.replace('${', '$')
        if '$' in p[1:].lstrip('{').split('/', 1)[-1] or '*' in p or '%' in p or p.count('/') < 1:
            continue
        if re.fullmatch(r'\$(DATA|SCRATCH|HOME)(/[^/]+)?|/(data|scratch)/fs\d+/[^/]+', p):
            continue
        if re.search(r'(OUT|LOG|RES|CACHE|TMP|STAGE|DEST|SAVE)', name) or SKIP_RE.search(p):
            continue
        if not (ASSET_RE.search(p) or re.search(r'(MODEL|SIF|IMAGE|CONTAINER|ENV|PY|RECORDS|INPUT|DATASET|TOKENIZER|'
                                                 r'ADAPTER|CKPT|CHECKPOINT|WEIGHTS|SRC)', name)):
            continue
        if is_created(p):
            continue
        out.append(p)
    return list(dict.fromkeys(out))


STATED_RE = re.compile(r'(?:expected(?:\s+(?:run\s*time|runtime|wall\s*time|duration))?\s*[:=]?\s*(?:~|about\s+)?'
                       r'(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(min|minutes|h|hours)\b)|'
                       r'(?:~|about\s+)?(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(min|minutes|h|hours)\s+expected\b',
                       re.I)


def stated_minutes(script_text: str) -> float | None:
    """A runtime the script's header comments state ("expected runtime: 25 min", "~35-40 min expected"); the upper
    end of a range. Only the leading comment block counts."""
    for line in script_text.splitlines()[1:]:
        s = line.strip()
        if s and not s.startswith('#'):
            break
        if s.startswith(('#SBATCH', '#ASC')):
            continue
        m = STATED_RE.search(s)
        if m:
            a, b, unit = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(4), m.group(5), m.group(6))
            v = float(b or a)
            return v * 60 if unit.lower().startswith('h') else v
    return None


def time_vs_expected(time_str: str | None, expected: float | None) -> str | None:
    """A warning when --time is far above the expected runtime: more than max(2 x expected, expected + 15 min)."""
    mins = S.minutes(time_str)
    if not expected or mins is None or mins == float('inf'):
        return None
    if mins > max(2 * expected, expected + 15):
        return (f'--time {time_str} is {mins / expected:.1f}x the expected {expected:g} min: tight limits are '
                f'backfilled sooner and billing covers only the time used; about {max(expected * 1.5, expected + 10):.0f} '
                'min would do')
    return None


def asc_rewrites(sbatch_output: str, opts: dict | None = None) -> dict:
    """What ASC's submit filter did, from sbatch's output (also --test-only): {'mode', 'settings', 'exclusive',
    'changed' (requested -> set, for options the job gave), 'accepted' (True/False/None)}."""
    out: dict = {'mode': None, 'settings': {}, 'exclusive': False, 'changed': [], 'accepted': None}
    for line in sbatch_output.splitlines():
        if (m := re.search(r'applying job settings for >> (.+?) <<', line)):
            out['mode'] = m.group(1).strip()
        elif (m := re.search(r'sbatch: (?:error: )?setting --([\w-]+)=(\S+)', line)):
            out['settings'][m.group(1)] = m.group(2)
        elif 'enabled exclusive node access' in line:
            out['exclusive'] = True
    if re.search(r'Submitted batch job|to start at', sbatch_output):
        out['accepted'] = True
    if re.search(r'allocation failure|Batch job submission failed', sbatch_output):
        out['accepted'] = False
    for k, v in out['settings'].items():
        mine = (opts or {}).get(k)
        if mine not in (None, '') and mine != v:
            out['changed'].append(f'--{k} {mine} -> {v}')
    return out


# ---------------------------------------------------------------- run_command

SLURM_TOOLS = {'sbatch': 'submit_job', 'scancel': 'cancel_job', 'salloc': 'a batch job (submit_job)',
               'srun': 'a batch job (submit_job)', 'sacctmgr': 'the human (account administration)',
               'strigger': 'the human', 'sattach': 'job_log'}
SCONTROL_CHANGES = {'update', 'hold', 'uhold', 'release', 'requeue', 'requeuehold', 'suspend', 'resume', 'top',
                    'notify', 'reboot', 'create', 'delete', 'reconfigure', 'shutdown', 'takeover'}
# sacctmgr is allowed only in its read-only forms: `sacctmgr [options] show|list ENTITY ...` (sacctmgr takes a command
# abbreviated to 3 letters: `sho`, `lis`; `sh` is ambiguous with shutdown), help and version. Everything else (add,
# create, modify, update, delete, remove, archive, load, dump, clear, reconfigure, shutdown, and a bare `sacctmgr`,
# which reads its commands from stdin) is account administration, the human's. `show runawayjobs` offers to fix the
# jobs it lists (and does so with -i), so it is refused too, in every spelling Slurm accepts: RunawayJobs abbreviated
# to 2 letters or more, its aliases OrphanJobs and LostJobs to 1 or more. sacctmgr parses options with getopt_long,
# which also takes them after the command word (`sacctmgr -i show -n runawayjobs`), and none of its options takes a
# value: every word that starts with '-' (up to `--`) is an option wherever it stands.
SACCTMGR_READ = ('show', 'list')
SACCTMGR_FIXES = (('runawayjobs', 2), ('orphanjobs', 1), ('lostjobs', 1))   # (entity, shortest abbreviation)
SACCTMGR_INFO = {'-h', '--help', '-V', '--version', '--usage', 'help', 'version'}
DESTRUCTIVE = {'rm', 'rmdir', 'mv', 'truncate', 'shred', 'unlink'}
WRAPPERS = {'sudo', 'command', 'exec', 'nohup', 'time', 'env', 'nice', 'ionice', 'stdbuf', 'timeout', 'xargs',
            'builtin', 'then', 'do', 'else', 'elif', '!', '{', 'if', 'while', 'until'}
# wrapper options that take a value (`timeout -s 9 5 kill 1`, `nice -n 5`, `env -u X`, `xargs -n 1`): the value is not
# the command word
WRAPPER_VALUE_OPTS = {
    'timeout': {'-s', '--signal', '-k', '--kill-after'}, 'nice': {'-n', '--adjustment'},
    'ionice': {'-c', '-n', '-p', '-P', '-u', '--class', '--classdata', '--pid'}, 'env': {'-u', '--unset', '-C', '--chdir'},
    'stdbuf': {'-i', '-o', '-e', '--input', '--output', '--error'}, 'sudo': {'-u', '-g', '-C', '-h', '-p', '-U'},
    'xargs': {'-n', '-P', '-I', '-L', '-d', '-s', '-a', '-E', '--max-args', '--max-procs', '--max-lines',
              '--delimiter', '--arg-file', '--eof', '--replace', '--max-chars'}}


def sacctmgr_command(rest: list[str]) -> tuple[bool, list[str]]:
    """(whether `sacctmgr REST` only reads, its command words). Read-only: show/list (see SACCTMGR_READ) of any entity
    but the runaway-jobs one (SACCTMGR_FIXES), help, version, or only -h/-V and other options. Option words are
    skipped wherever they stand (getopt_long permutes them); after `--` every word is a command word."""
    words, info, opts_done = [], False, False
    for t in rest:
        if not opts_done and t == '--':
            opts_done = True
        elif not opts_done and t.startswith('-') and t != '-':
            info = info or t in SACCTMGR_INFO
        else:
            words.append(t.lower())
    if not words:
        return info, words
    cmd = words[0]
    if cmd in SACCTMGR_INFO:
        return True, words
    if len(cmd) < 3 or not any(full.startswith(cmd) for full in SACCTMGR_READ):
        return False, words
    entity = words[1] if len(words) > 1 else ''
    fixes = any(len(entity) >= n and full.startswith(entity) for full, n in SACCTMGR_FIXES)
    return not fixes, words


DURATION = re.compile(r'^\d+(?:\.\d+)?[smhd]?$')
SHELLS = ('bash', 'sh', 'zsh', 'dash')
SHELL_C = re.compile(r'^-[a-zA-Z]*c[a-zA-Z]*$')       # -c, -lc, -ec, -xc


def command_start(seg: list[str]) -> tuple[int, list[str]]:
    """(index of the command word, wrappers before it) in a simple command's words: assignments, wrappers and the
    wrappers' options and values (timeout's duration too) are skipped."""
    i, wrappers = 0, []
    while i < len(seg):
        t = seg[i]
        if re.match(r'^[A-Za-z_]\w*=', t):
            i += 1
            continue
        if t not in WRAPPERS:
            break
        wrappers.append(t)
        i += 1
        takes = WRAPPER_VALUE_OPTS.get(t, set())
        while i < len(seg) and seg[i].startswith('-') and seg[i] != '-':
            i += 2 if seg[i] in takes else 1
            if seg[i - 1] == '--':
                break
        if t == 'timeout' and i < len(seg) and DURATION.match(seg[i]) or i < len(seg) and seg[i].isdigit():
            i += 1
    return i, wrappers


def shell_body(rest: list[str]) -> str | None:
    """The command string of `bash -c BODY` (also -lc, -ec), from the words after the shell's name."""
    k = next((j for j, t in enumerate(rest) if SHELL_C.match(t)), None)
    return rest[k + 1] if k is not None and k + 1 < len(rest) else None


ACCESS = re.compile(r'(~|\$HOME|\$\{HOME\}|/home/[^/\s]+)/\.(ssh|bashrc|bash_profile|bash_login|profile|step|'
                    r'config/wireguard)\b|authorized_keys|\.config/wireguard')
SEPARATORS = {';', '&&', '||', '|', '&', '(', ')', ';;', '|&', '}', '\n'}


def _segments(command: str) -> list[list[str]]:
    lex = shlex.shlex(command.replace('\n', ' ; '), posix=True, punctuation_chars=';&|()<>')
    lex.whitespace_split = True
    lex.commenters = ''
    segs, cur = [], []
    try:
        for tok in lex:
            if tok in SEPARATORS:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(tok)
    except ValueError:                   # unbalanced quotes: fall back to whitespace
        return [command.split()]
    if cur:
        segs.append(cur)
    return segs


def command_policy(command: str, _depth: int = 0) -> dict:
    """What run_command must refuse (with the tool to use instead) or have approved by the human."""
    if ACCESS.search(command):
        return {'refuse': 'access configuration and secrets (~/.ssh, ~/.bashrc, ~/.profile, ~/.step, the WireGuard '
                          'config) are the human\'s (AGENTS.md §0 rule 6–7, §9)', 'needs_approval': []}
    approval: list[str] = []
    inner = re.findall(r'\$\(([^()]*)\)', command) + re.findall(r'`([^`]*)`', command)
    for sub in inner:
        r = command_policy(sub, _depth + 1) if _depth < 4 else {'refuse': None, 'needs_approval': []}
        if r['refuse']:
            return r
        approval += r['needs_approval']
    for seg in _segments(command):
        i, _ = command_start(seg)
        if i >= len(seg):
            continue
        word, rest = os.path.basename(seg[i]), seg[i + 1:]
        body = None
        if word in SHELLS:
            body = shell_body(rest) or ''
        elif word == 'eval':
            body = ' '.join(rest)
        elif word == 'ssh':                                  # ssh <node> <command>, e.g. to a compute node
            args = [t for t in rest if not t.startswith('-')]
            body = ' '.join(args[1:])
        if body is not None:
            if body and _depth < 4:
                r = command_policy(body, _depth + 1)
                if r['refuse']:
                    return r
                approval += r['needs_approval']
            continue
        if word == 'sacctmgr':
            read_only, words = sacctmgr_command(rest)
            if read_only:
                continue
            what = f'`sacctmgr {" ".join(words[:2])}`' if words else 'a bare `sacctmgr` (reads commands from stdin)'
            return {'refuse': f'{what} can change Slurm state; it is for {SLURM_TOOLS[word]}. Read-only forms '
                              'are allowed: `sacctmgr [-n] [-P] show|list ENTITY ...`', 'needs_approval': []}
        if word in SLURM_TOOLS:
            return {'refuse': f'`{word}` changes Slurm state; use {SLURM_TOOLS[word]} instead (it applies the '
                              'runbook\'s checks and records the job)', 'needs_approval': []}
        if word == 'scontrol':
            sub = next((t.lower() for t in rest if not t.startswith('-')), '')
            if sub in SCONTROL_CHANGES:
                return {'refuse': f'`scontrol {sub}` changes Slurm state; use update_job or cancel_job instead',
                        'needs_approval': []}
        if word in DESTRUCTIVE:
            approval.append(f'`{" ".join(seg[i:])[:120]}` deletes or moves files')
        elif word == 'find' and any(t in ('-delete', '-exec', '-execdir') for t in rest):
            approval.append('`find` with -delete/-exec can delete files')
        elif word == 'rsync' and any(t.startswith('--delete') or t == '--remove-source-files' for t in rest):
            approval.append('`rsync --delete` deletes files')
        elif word == 'git' and rest[:1] and rest[0] in ('clean', 'reset') and (rest[0] == 'clean' or '--hard' in rest):
            approval.append(f'`git {rest[0]}` can discard files')
    return {'refuse': None, 'needs_approval': list(dict.fromkeys(approval))}
