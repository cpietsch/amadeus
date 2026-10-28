"""Slurm option parsing (as sbatch reads them), time limits, hostlists and MUSICA token (credit) estimates."""
from __future__ import annotations

import math
import re
import shlex

GPU = 'zen4_0768_h100x4'
CPU = 'zen4_0768'
B200 = 'zen5_2304_b200x8'
# tokens per node-hour (price sheet v1.4); B200 has no published price
RATES = {GPU: 4.0, CPU: 0.96}
PARTITIONS = {GPU, CPU, B200}
QOS = {GPU: {GPU, 'dev_' + GPU}, CPU: {CPU, 'dev_' + CPU}, B200: {B200}}
GPUS_PER_NODE = {GPU: 4, B200: 8}
DEV_LIMIT_MIN, DEV_MAX_NODES, MAX_HOURS = 10, 2, 72
# usable physical cores per node with --threads-per-core=1 (GPU nodes: CoreSpecCount=16; CPU nodes: 190; B200 nodes:
# CPUTot 384, CPUEfctv 352, as on the H100 nodes, n3024-003 on 2026-10-08)
CORES = {GPU: 176, CPU: 190, B200: 176}
THREADS = {GPU: 352, CPU: 380, B200: 352}
MAX_ARRAY_SIZE = 100000            # scontrol show config: array indices must be below it
BAD_NODES = ['n3010-019', 'n3010-027', 'n3012-034']    # NOTES.md, Hardware (known bad GPU nodes)
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED',
            'BOOT_FAIL', 'DEADLINE', 'REVOKED'}

# sbatch's options and whether they take a value: r = required, o = optional (only attached with '='), n = none.
# An option missing here is reported as unknown rather than guessed, because a wrong guess can hide a node count.
OPTIONS = {k: 'r' for k in (
    'account acctg-freq array batch bb bbf begin chdir cluster-constraint clusters comment constraint container '
    'container-id core-spec cores-per-socket cpu-freq cpus-per-gpu cpus-per-task deadline delay-boot dependency '
    'distribution error exclude export export-file extra extra-node-info gid gpu-bind gpu-freq gpus gpus-per-node '
    'gpus-per-socket gpus-per-task gres gres-flags hint input job-name kill-on-invalid-dep licenses mail-type mail-user '
    'mcs-label mem mem-bind mem-per-cpu mem-per-gpu mincpus network nodefile nodelist nodes ntasks ntasks-per-core '
    'ntasks-per-gpu ntasks-per-node ntasks-per-socket open-mode output partition power prefer priority profile qos '
    'reservation segment signal sockets-per-node switches tasks-per-node thread-spec threads-per-core time time-min '
    'tmp tres-bind tres-per-task uid wait-all-nodes wckey wrap').split()}
OPTIONS.update({k: 'o' for k in 'exclusive get-user-env nice no-kill oom-kill-step propagate resv-ports'.split()})
OPTIONS.update({k: 'n' for k in (
    'contiguous help hold ignore-pbs no-requeue overcommit oversubscribe parsable quiet reboot requeue spread-job '
    'test-only usage use-min-nodes verbose version wait').split()})
SHORT = {'A': 'account', 'a': 'array', 'b': 'begin', 'B': 'extra-node-info', 'c': 'cpus-per-task', 'C': 'constraint',
         'd': 'dependency', 'D': 'chdir', 'e': 'error', 'F': 'nodefile', 'G': 'gpus', 'H': 'hold', 'h': 'help',
         'i': 'input', 'J': 'job-name', 'k': 'no-kill', 'L': 'licenses', 'm': 'distribution', 'M': 'clusters',
         'n': 'ntasks', 'N': 'nodes', 'o': 'output', 'O': 'overcommit', 'p': 'partition', 'q': 'qos', 'Q': 'quiet',
         's': 'oversubscribe', 'S': 'core-spec', 't': 'time', 'V': 'version', 'v': 'verbose', 'w': 'nodelist',
         'W': 'wait', 'x': 'exclude'}
ALIASES = {'tasks-per-node': 'ntasks-per-node'}


def _long(name: str) -> tuple[str | None, str | None]:
    """Exact name or unique prefix (as getopt_long accepts) -> (canonical name, error)."""
    if name in OPTIONS:
        return ALIASES.get(name, name), None
    hits = [o for o in OPTIONS if o.startswith(name)] if name else []
    if len(hits) == 1:
        return ALIASES.get(hits[0], hits[0]), None
    if hits:
        return None, f'--{name} is ambiguous ({", ".join("--" + h for h in sorted(hits)[:6])}): write the full option name'
    return None, f'--{name} is not an sbatch option this server knows, so it cannot check the job: remove it or report it'


def parse_args(tokens: list[str], cli: bool = False) -> tuple[dict[str, str], list[str]]:
    """sbatch options -> ({canonical long name: value ('' for flags)}, errors).

    cli=True: the tokens are sbatch_args placed before the script on the command line, so any token that is not an
    option or an option's value is an error (sbatch would take it as the job script), as is an item with whitespace."""
    opts: dict[str, str] = {}
    errors: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if cli and re.search(r'\s', t):
            errors.append(f'{t!r} contains whitespace: pass each option and its value as separate items')
            i += 1
            continue
        if t.startswith('--') and len(t) > 2:
            name, eq, val = t[2:].partition('=')
            full, err = _long(name)
            if err:
                errors.append(err)
            else:
                kind = OPTIONS.get(full) or OPTIONS.get(name)
                if kind == 'r':
                    if eq:
                        opts[full] = val
                    elif i + 1 < len(tokens):
                        opts[full] = tokens[i + 1]
                        i += 1
                    else:
                        errors.append(f'--{full} needs a value')
                elif kind == 'o':
                    opts[full] = val if eq else ''
                else:
                    if eq:
                        errors.append(f'--{full} takes no value')
                    opts[full] = ''
        elif t.startswith('-') and len(t) > 1 and t != '--':
            j = 1
            while j < len(t):
                full = SHORT.get(t[j])
                if full is None:
                    errors.append(f'-{t[j]} (in {t!r}) is not an sbatch option this server knows')
                    break
                kind = OPTIONS[full]
                rest = t[j + 1:]
                if kind == 'n':
                    opts[full] = ''
                    j += 1
                    continue
                if kind == 'o':
                    opts[full] = rest
                elif rest:
                    opts[full] = rest.lstrip('=')
                elif i + 1 < len(tokens):
                    opts[full] = tokens[i + 1]
                    i += 1
                else:
                    errors.append(f'-{t[j]} needs a value')
                break
        elif cli:
            errors.append(f'{t!r} is not an sbatch option: sbatch would take it as the job script. Put arguments for '
                          'the script in script_args')
        i += 1
    return opts, errors


def script_options(text: str) -> tuple[dict[str, str], list[str]]:
    """#SBATCH directives as sbatch reads them: only at column 0, up to the first command line."""
    tokens: list[str] = []
    errors: list[str] = []
    pbs = False
    lines = text.splitlines()
    for line in lines[1:] if lines and lines[0].startswith('#!') else lines:
        if line.startswith('#SBATCH'):
            try:
                tokens += shlex.split(line[len('#SBATCH'):], comments=True)
            except ValueError:
                tokens += line[len('#SBATCH'):].split('#')[0].split()
        elif line.startswith(('#PBS', '#BSUB')):
            pbs = True
        elif not line.strip() or line.lstrip().startswith('#'):
            continue
        else:
            break
    opts, errs = parse_args(tokens)
    errors += errs
    if pbs and 'ignore-pbs' not in opts:
        errors.append('the script has #PBS/#BSUB directives, which sbatch also applies; this server does not read '
                      'them: convert them to #SBATCH or add #SBATCH --ignore-pbs')
    return opts, errors


def minutes(limit: str | None) -> float | None:
    """Slurm time limit -> minutes; None if missing or unreadable. 0 and UNLIMITED mean no limit (inf).
    Formats: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S."""
    if not limit:
        return None
    s = limit.strip()
    if s.upper() in ('UNLIMITED', 'INFINITE'):
        return float('inf')
    try:
        if '-' in s:
            d, s = s.split('-', 1)
            parts = [int(x) for x in s.split(':')]
            h, m, sec = (parts + [0, 0])[:3]
            v = int(d) * 1440 + h * 60 + m + sec / 60
        else:
            parts = [int(x) for x in s.split(':')]
            if len(parts) == 1:
                v = float(parts[0])
            elif len(parts) == 2:
                v = parts[0] + parts[1] / 60
            else:
                v = parts[0] * 60 + parts[1] + parts[2] / 60
    except ValueError:
        return None
    return float('inf') if v == 0 else v


def expand_hostlist(spec: str) -> list[str]:
    """Slurm hostlist: 'n3010-[019,027],n3012-034' -> ['n3010-019', 'n3010-027', 'n3012-034'] (padding kept)."""
    items, depth, cur = [], 0, ''
    for ch in spec or '':
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            items.append(cur)
            cur = ''
        else:
            cur += ch
    items.append(cur)
    out: list[str] = []

    def expand(item: str) -> list[str]:
        m = re.search(r'\[([^\]]*)\]', item)
        if not m:
            return [item] if item else []
        head, tail = item[:m.start()], item[m.end():]
        res = []
        for part in m.group(1).split(','):
            if '-' in part:
                a, b = part.split('-', 1)
                width = len(a)
                res += [head + str(k).zfill(width) + t for k in range(int(a), int(b) + 1) for t in expand(tail) or ['']]
            elif part:
                res += [head + part + t for t in expand(tail) or ['']]
        return res
    for item in items:
        out += expand(item.strip())
    return out


def _total_gpus(opts: dict[str, str]) -> int | None:
    g = opts.get('gpus')
    if g:
        m = re.search(r'(\d+)\s*$', g)
        return int(m.group(1)) if m else None
    return None


def gpus_per_node(opts: dict[str, str]) -> int | None:
    m = re.search(r'gpu(?::[a-z0-9]+)?:(\d+)', opts.get('gres', ''))
    if m:
        return int(m.group(1))
    if opts.get('gpus-per-node'):
        m = re.search(r'(\d+)\s*$', opts['gpus-per-node'])
        return int(m.group(1)) if m else None
    return None


def node_count(opts: dict[str, str]) -> int:
    """Nodes the job can get: the largest of --nodes (every element of a size list or range), --gpus spread over the
    partition's GPUs per node, --ntasks / --ntasks-per-node, and the hosts in --nodelist."""
    n = 1
    for part in (opts.get('nodes') or '').split(','):
        nums = [int(x) for x in re.findall(r'\d+', part.split(':')[0])]
        if nums:
            n = max(n, max(nums))
    g = _total_gpus(opts)
    per = GPUS_PER_NODE.get(opts.get('partition', ''), 4)
    if g:
        n = max(n, math.ceil(g / per))
    nt, tpn = opts.get('ntasks', ''), opts.get('ntasks-per-node', '')
    if nt.isdigit() and tpn.isdigit() and int(tpn) > 0:
        n = max(n, math.ceil(int(nt) / int(tpn)))
    if opts.get('nodelist'):
        try:
            n = max(n, len(set(expand_hostlist(opts['nodelist']))))
        except ValueError:
            pass
    return n


def max_nodes(spec: str | None) -> int:
    return node_count({'nodes': spec or ''})


def array_size(spec: str | None) -> tuple[int, int]:
    """--array spec -> (tasks, tasks running at once)."""
    if not spec:
        return 1, 1
    body, _, limit = spec.partition('%')
    n = 0
    for part in body.split(','):
        rng, _, step = part.strip().partition(':')
        try:
            if '-' in rng:
                a, b = rng.split('-', 1)
                n += len(range(int(a), int(b) + 1, int(step or 1)))
            elif rng:
                n += 1
        except ValueError:
            n += 1
    n = max(n, 1)
    lim = int(limit) if limit.strip().isdigit() and int(limit) > 0 else n
    return n, min(n, lim)


def array_max_index(spec: str | None) -> int | None:
    """The largest index in an --array spec (None if there is none or it is unreadable)."""
    body = (spec or '').partition('%')[0]
    nums = [int(x) for x in re.findall(r'\d+', re.sub(r':\d+', '', body))]
    return max(nums) if nums else None


def node_share(opts: dict[str, str]) -> float:
    """Billed share of a node: 1–3 GPUs on one GPU node pay per GPU; everything else is counted as whole nodes."""
    g = gpus_per_node(opts)
    if g is None:
        g = _total_gpus(opts)
    if (opts.get('partition') == GPU and g is not None and 1 <= g < 4 and node_count(opts) == 1
            and 'exclusive' not in opts):
        return g / 4
    return 1.0


def estimate(opts: dict[str, str]) -> dict:
    """Upper bound of a job's cost at its time limit (billing covers only the time used)."""
    part = opts.get('partition')
    mins = minutes(opts.get('time'))
    nodes = node_count(opts)
    tasks, at_once = array_size(opts.get('array'))
    share = node_share(opts)
    rate = RATES.get(part)
    out = {'nodes': nodes, 'array_tasks': tasks, 'nodes_at_once': nodes * at_once, 'node_share': share,
           'time_limit_min': None if mins == float('inf') else mins, 'rate_per_node_hour': rate}
    if rate is not None and mins is not None and mins != float('inf'):
        out['max_tokens'] = round(rate * share * nodes * tasks * mins / 60, 2)
    else:
        out['max_tokens'] = None
    return out


def credits_from_tres(partition: str, elapsed_s: int, alloc_tres: str) -> float | None:
    """Tokens of one finished allocation, as in AGENTS.md §5 Monitoring (gres/np_* = billed node share × 100)."""
    m = re.search(r'gres/np_[a-z0-9_]+=(\d+)', alloc_tres or '')
    rate = RATES.get(partition)
    if not m or rate is None:
        return None
    return round(rate * int(m.group(1)) / 100 * elapsed_s / 3600, 3)


def parse_scontrol(line: str) -> dict[str, str]:
    """One line of `scontrol show job -o` -> {Key: value}. A value runs up to the next ' Key=' (values like
    TRES=cpu=88,node=1 keep their inner '=')."""
    keys = list(re.finditer(r'(?:(?<=\s)|^)([A-Z][A-Za-z0-9_:/]*)=', line))
    out = {}
    for k, m in enumerate(keys):
        end = keys[k + 1].start() if k + 1 < len(keys) else len(line)
        out[m.group(1)] = line[m.end():end].strip()
    return out
