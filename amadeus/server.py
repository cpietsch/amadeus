"""Amadeus: MCP server for the ASC MUSICA supercomputer, with the runbook's rules (AGENTS.md) built into its tools.

Run over stdio: uv run --project <repo> amadeus
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import fcntl
import functools
import inspect
import os
import re
import shlex
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import auth, config, drafts, files, guard, issues, ledger, logs, remote, sound, templates, transport
from . import slurm as S

ROOT = Path(__file__).resolve().parents[1]
MARK = '__amadeus_section__'
JOB_ID = re.compile(r'^\d+(_\d+)?$')
INTERACTIVE = re.compile(r'^\s*(vim?|nano|emacs|top|htop|watch|less|more|tmux|screen|ssh\s+-t)\b')
# ssh from the login node to a compute node (n3010-014): refused by MUSICA (2026-10-08, 2026-10-13)
COMPUTE_SSH = re.compile(r'(^|[;&|(])\s*(timeout\s+\S+\s+)?ssh\s[^;&|]*\bn\d{4}-\d{3}\b')
# sbatch failed in a way that may still have created the job
UNCERTAIN = re.compile(r'Socket timed out|Unable to contact slurm controller|Connection refused|'
                       r'Transport endpoint|Zero Bytes were transmitted', re.I)
MAX_LOG_FILES = 20
MAX_SCRIPT_TEXT = 200_000
NO_CWD = '__amadeus_no_cwd__'
NO_VAL_TASK = '4294967294'          # what Slurm puts for %a in jobs that are not arrays

INSTRUCTIONS = """Amadeus: interface to the ASC MUSICA supercomputer (Vienna). @PROJECT@

- Start with access_status. If the certificate is missing or under 60 min and you are answering a message from the
  human (not a cron, /loop or stop-hook prompt), call start_login and show the link at once: it must be approved in
  a browser within about 60 s. With nobody there, start no code; tell the human when they next write. Never pass
  the link to anyone else. Tool results then carry `certificate`; `certificate_warning` (submit_job, wait_for_job,
  list_jobs) means jobs end after the expiry: offer an early renewal before the human goes offline (running jobs are
  not affected). No reminder crons unless the human asks (then CronList first; delete it when told to stop).
- The login node is for light work only (4 cores): run_command for small checks, ls, du, git. Anything heavy
  (downloads, builds, compute) goes into a job: check_job, then submit_job. Slurm-changing commands (sbatch, scancel,
  scontrol update, srun, salloc) are refused in run_command: use submit_job, cancel_job, update_job.
- To look at the node of your running job (nvidia-smi, ps, top -b, py-spy), use node_exec: ssh to compute nodes is
  refused on MUSICA. A running job's time limit cannot be raised (only while pending): set --time with margin.
- submit_job enforces the runbook: account, partition and matching QoS, a time limit, dev-QoS limits (10 min, 2
  nodes), CPU limits per node. It submits on hold, checks Slurm's own reading of the job, then releases it. It needs
  the human's explicit approval, passed in `approval` in their words, for @NODES@ or more nodes at once (your running
  and pending jobs count together), more than @TOKENS@ tokens (including what your other jobs can still spend), the
  B200 partition, or long/priority QoS. A budget the human approved for an experiment can be quoted as the approval
  for that experiment's jobs, except @HEADSUP@ or more nodes: those need their own heads-up and an approval naming the
  node count. Never invent an approval.
- Writing a job script: draft_job gives one in the form MUSICA accepts (CPU or GPU, part of a node or whole nodes,
  multi-node with launch retry, vLLM servers with a client); job_templates lists tested scripts (Amadeus's jobs/ and the project's template
  directories).
  check_job(script_text=...) checks a draft and runs sbatch --test-only without pushing it first; it also tests that
  the models, containers and envs the script reads exist (missing_paths) and shows ASC's rewrites (asc). Read its
  warnings: they are the runbook's §8 pitfalls.
- Test ladder: dev QoS first, then 1 node, then 2 and 4, before 16.
- After a dropped connection never resubmit blindly: submit_job checks for the job itself; otherwise list_jobs first.
- Wait with wait_for_job instead of sleeping or watcher loops: several jobs, milestones with deadlines, a quiet-log
  alarm and expected outputs; pass next_cursor back to get only new log lines. After a restart just call it again.
  When a run is clearly broken or crawling, cancel it and resubmit; don't salvage it.
- Logs: job_log returns them cleaned (ANSI, \r progress updates); progress=True gives the progress bars (weight
  loading is not work), hide_benign=True drops known-harmless errors, file= reads per-run logs. job_details diagnoses
  an ended job that failed: infrastructure (cluster side: resubmit), script, program; don't grep for 'Error' by hand.
- @BUDGET@
- Destructive actions need the human: deleting or moving data (run_command asks for `approval`), cancelling jobs the
  agent did not submit, raising a pending job's time or nodes beyond the limits (update_job), kill in node_exec.
- push_files warns when pending or running jobs use the target: pending jobs run the pushed code.
- Files: push_files/fetch_files read and write on the machine given by `host` ('' = the machine the server runs on,
  the default; other machines from [files] hosts are reached over SSH).
- If a tool misbehaves, returns something wrong, or a tool you need is missing, call report_issue (what happened,
  what you expected, how to reproduce, your project). Unexpected errors are filed automatically. A maintainer session
  fixes open issues; list_issues shows their status.
- Full runbook: resource amadeus://runbook; platform facts: amadeus://notes@NOTES@."""


def instructions() -> str:
    """The server instructions with the configured project, approval limits and budget filled in."""
    b = config.budget()
    prices = f'H100 node {S.RATES[S.GPU]:g}/h, CPU node {S.RATES[S.CPU]:g}/h (ASC price sheet)'
    budget = (f'Budget: {b:,.0f} tokens in total; {prices}.' if b is not None else
              f'Prices in tokens: {prices}; the project\'s budget is in the ASC projects portal.')
    notes = '; project notes: amadeus://project' if config.notes() else ''
    return _fill(INSTRUCTIONS, PROJECT=config.summary(), NODES=str(config.approval_nodes()),
                 TOKENS=f'{config.approval_tokens():g}', HEADSUP=str(config.headsup_nodes()), BUDGET=budget,
                 NOTES=notes)


def _fill(template: str, **values: str) -> str:
    for k, v in values.items():
        template = template.replace(f'@{k}@', v)
    return template


def _cwd(cwd: str) -> str:
    return cwd.strip() or config.workdir()


mcp = MCPServer('amadeus', title='Amadeus (MUSICA cluster)', instructions=instructions(), version='0.3.0')

RO = ToolAnnotations(read_only_hint=True, open_world_hint=True)
RW = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True)


class ClusterError(ToolError):
    """An expected operational failure (SSH down, a Slurm command failed): reported to the caller, not filed."""


def _ssh_note(r: remote.Result) -> str:
    """The note for a failed command; for a failed SSH connection, whether the certificate or the tunnel is to blame."""
    note = (auth.ssh_hint(r.err) if r.ssh_failed and not r.started else None) or r.summary(2000).get('note', '')
    return '; '.join(filter(None, (note, r.recovery)))


def _fail(what: str, r: remote.Result) -> ClusterError:
    note = _ssh_note(r)
    err = r.err.strip() or r.out.strip()[-500:]
    return ClusterError(f'{what} failed (exit {r.code}): {err}' + (f'; {note}' if note else ''))


def _client(ctx: Context | None) -> str:
    try:
        info = ctx.session.client_params.client_info
        return f'{info.name} {info.version}'
    except Exception:
        return ''


def _filed(name: str, kwargs: dict, exc: Exception) -> Exception:
    if isinstance(exc, ToolError):          # deliberate failures (ClusterError included) go to the caller as they are
        return exc
    ctx = next((v for v in kwargs.values() if isinstance(v, Context)), None)
    args = {k: v for k, v in kwargs.items() if not isinstance(v, Context)}
    iid = issues.auto_report(name, args, exc, _client(ctx))
    note = f' (filed automatically as issue {iid}; a maintainer session will fix it)' if iid else ''
    # a ToolError's message reaches the client; any other exception would arrive as a bare "Error executing tool"
    return ToolError(f'internal error: {type(exc).__name__}: {exc}{note}')


NO_CERT_FIELD = {'access_status', 'start_login'}   # they report the certificate themselves


def _with_cert(name: str, res):
    """Add `certificate` to a dict result when under 60 min are left or it is gone (findings 0, 9). Cached in auth
    (re-read at most every 5 min, no SSH); any failure here is swallowed, never the tool's."""
    if name in NO_CERT_FIELD or not isinstance(res, dict) or 'certificate' in res:
        return res
    try:
        if (w := auth.expiry_warning()):
            res['certificate'] = w
    except Exception:
        pass
    return res


def tool(annotations: ToolAnnotations):
    """mcp.tool plus the issue channel: unexpected exceptions are filed as issues before they reach the client.
    Dict results get the certificate warning (_with_cert)."""
    def deco(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapped(*a, **k):
                try:
                    res = await fn(*a, **k)
                except Exception as e:
                    sound.play('error')
                    raise _filed(fn.__name__, k, e) from e
                if sound.failed(res):
                    sound.play('error')
                return await asyncio.to_thread(_with_cert, fn.__name__, res)
        else:
            @functools.wraps(fn)
            def wrapped(*a, **k):
                try:
                    res = fn(*a, **k)
                except Exception as e:
                    sound.play('error')
                    raise _filed(fn.__name__, k, e) from e
                if sound.failed(res):
                    sound.play('error')
                return _with_cert(fn.__name__, res)
        mcp.tool(annotations=annotations)(wrapped)
        return fn
    return deco


def _check_id(job_id: str) -> str:
    job_id = str(job_id).strip()
    if not JOB_ID.match(job_id):
        raise ToolError('job_id must look like 1234567 or 1234567_3')
    return job_id


# ---------------------------------------------------------------- access

@tool(RW)
def access_status(probe: bool = True) -> dict:
    """Tunnel, certificate expiry and an SSH test. Call first in every session. (Starts the tunnel if it is down; with
    wireproxy, a stall before login restarts it, at most once a minute, also from other tools that run commands on
    the login node, not yet from the copying in push_files/fetch_files: `recovery`.)
    The first call of a server run also carries `banner` (Amadeus' ASCII art): show it to the human once."""
    global _greeted
    res = auth.status(probe)
    if not _greeted:
        _greeted = True
        res = {'banner': sound.BANNER.strip('\n')} | res
    return res


_greeted = False


@tool(RW)
def start_login() -> dict:
    """Start a certificate login (or an early renewal while the current one is still valid) and return the device link.
    Show the link to the human immediately; it expires after about 60 s. Only one login runs at a time. Call it only
    when answering a message from the human, not from a cron, /loop or stop-hook turn (AGENTS.md §3)."""
    return auth.start_login()


# ---------------------------------------------------------------- commands

@tool(DESTRUCTIVE)
def run_command(command: str, timeout_s: int = 120, lmod: bool = False, approval: str = '') -> dict:
    """Run a shell command on the login node (bash, non-interactive). `$DATA`, `$SCRATCH`, `$USER` expand there.
    Light work only: no compute, builds or big downloads (use a job). lmod=True initialises Lmod first (for `module`).
    Refused: Slurm-changing commands (use submit_job, cancel_job, update_job; sacctmgr only in its read-only
    show/list forms, e.g. `sacctmgr -nP show qos format=Name,MaxWall`), ssh to compute nodes and srun (use node_exec)
    and changes to SSH/shell/VPN config.
    Deleting or moving files (rm, mv, rsync --delete, ...) needs the human's approval in `approval`, in their words.
    The command is not repeated after an SSH failure; when `recovery` says the tunnel was restarted after a stall
    (nothing ran remotely), run it again."""
    if INTERACTIVE.match(command):
        return {'error': 'interactive commands hang over a batch SSH session; use a job, or a non-interactive form'}
    if COMPUTE_SSH.search(command):
        return {'error': 'ssh to compute nodes is refused on MUSICA (publickey); use node_exec(job_id, command) to run '
                         'a command on the node of your running job'}
    policy = guard.command_policy(command)
    if policy.get('refuse'):
        words = _words(command)
        if words and 'srun' in words[1]:
            return {'error': policy['refuse'], 'note': 'to look at the node of a running job (nvidia-smi, ps, py-spy) '
                                                       'use node_exec(job_id, command)'}
        return {'error': policy['refuse']}
    if policy.get('needs_approval') and not approval.strip():
        return {'error': 'needs the human\'s approval: ' + '; '.join(policy['needs_approval']),
                'note': 'ask the human and pass their words in `approval`'}
    if policy.get('needs_approval'):
        ledger.append({'event': 'command', 'command': command, 'approval': approval,
                       'reasons': policy['needs_approval']})
    r = remote.run(command, timeout=min(max(timeout_s, 5), 1800), lmod=lmod)
    out = r.summary()
    if r.ssh_failed and not r.started and (hint := auth.ssh_hint(r.err)):
        out['note'] = hint
    return out


# ---------------------------------------------------------------- jobs: status

SQ_KEYS = ['job_id', 'name', 'state', 'time_used', 'time_limit', 'nodes', 'partition', 'qos', 'reason', 'start',
           'submitted']
SQ_FMT = '%i|%j|%T|%M|%l|%D|%P|%q|%R|%S|%V'


def _squeue_rows(extra: str = '', expand: bool = False) -> list[dict]:
    r = remote.run(f'squeue -u $USER -h {"-r " if expand else ""}-o {shlex.quote(SQ_FMT)} {extra}', timeout=60,
                   retry=True)
    if not r.ok:
        raise _fail('squeue', r)
    return _parse_squeue(r.out)


def _parse_squeue(text: str) -> list[dict]:
    rows = []
    for l in text.splitlines():
        v = l.split('|')
        if len(v) == len(SQ_KEYS):
            rows.append(dict(zip(SQ_KEYS, v)))
    return rows


def _elapsed_min(s: str) -> float:
    """squeue %M (M:SS, H:MM:SS, D-H:MM:SS) -> minutes."""
    try:
        days = 0
        if '-' in s:
            d, s = s.split('-', 1)
            days = int(d)
        parts = [int(x) for x in s.split(':')]
        if len(parts) == 2:
            return days * 1440 + parts[0] + parts[1] / 60
        if len(parts) == 3:
            return days * 1440 + parts[0] * 60 + parts[1] + parts[2] / 60
    except ValueError:
        pass
    return 0.0


def _in_use(rows: list[dict] | None = None) -> dict:
    """Nodes and remaining maximum tokens held by the user's running and pending jobs (rule 11).

    rows: squeue -r rows (one per array task). Pending jobs count with the larger of Slurm's node count and the node
    count recorded at submission. Jobs submitted through the server in the last 3 minutes that squeue does not show
    yet count with their estimate (squeue can miss a fresh job), unless sacct says they have already ended."""
    rows = _squeue_rows(expand=True) if rows is None else rows
    known = ledger.submitted()
    nodes, tokens, seen, dev = 0, 0.0, set(), {}
    for row in rows:
        if row['state'] not in ('RUNNING', 'PENDING', 'CONFIGURING', 'RESIZING', 'SUSPENDED', 'REQUEUED'):
            continue
        master = row['job_id'].split('_')[0]
        seen.add(master)
        n = int(row['nodes']) if row['nodes'].isdigit() else 1
        est = (known.get(master) or {}).get('estimate') or {}
        if row['state'] == 'PENDING' and est.get('nodes'):
            n = max(n, int(est['nodes']))
        nodes += n
        if row.get('qos', '').startswith('dev_'):
            dev[row['qos']] = dev.get(row['qos'], 0) + n
        limit = S.minutes(row['time_limit'])
        limit = S.MAX_HOURS * 60 if limit is None or limit == float('inf') else limit
        rate = S.RATES.get(row['partition'], 0.0)
        tokens += rate * n * max(0.0, limit - _elapsed_min(row['time_used'])) / 60
    now = time.time()
    fresh = {jid: e for jid, e in known.items() if jid not in seen and now - e.get('epoch', 0) <= 180}
    for jid in _ended(list(fresh)):
        fresh.pop(jid, None)
    for jid, e in fresh.items():
        est = e.get('estimate') or {}
        nodes += int(est.get('nodes_at_once') or est.get('nodes') or 0)
        tokens += float(est.get('max_tokens') or 0)
    out = {'nodes': nodes, 'tokens': round(tokens, 2)}
    if dev:
        out['dev_nodes'] = dev
    return out


def _ended(job_ids: list[str]) -> set[str]:
    """Which of these jobs sacct shows as ended (a job that finished within seconds is gone from squeue too).
    On any failure: none, so they keep counting."""
    if not job_ids:
        return set()
    r = remote.run(f'sacct -X -n -P -j {",".join(job_ids)} -o JobIDRaw,State', timeout=45, retry=True)
    if not r.ok:
        return set()
    out = set()
    for line in r.out.splitlines():
        v = line.split('|')
        if len(v) >= 2 and v[1].split()[0] in S.TERMINAL | {'COMPLETING'}:
            out.add(v[0].split('_')[0])
    return out


def _cluster_epoch(stamp: str) -> float | None:
    """A Slurm time as squeue prints it (Vienna local time, no zone) -> epoch; None for N/A, Unknown and the like."""
    try:
        from zoneinfo import ZoneInfo
        t = datetime.fromisoformat(stamp.strip())
        return (t if t.tzinfo else t.replace(tzinfo=ZoneInfo(auth.CLUSTER_TZ))).timestamp()
    except Exception:
        return None


def _squeue_ends(rows: list[dict], now: float) -> dict[str, float]:
    """When each job reaches its time limit at the latest: running from the time used, pending from Slurm's start
    estimate (or now)."""
    ends = {}
    for row in rows:
        limit = S.minutes(row.get('time_limit'))
        if limit is None or limit == float('inf'):
            continue
        if row.get('state') == 'PENDING':
            start = max(_cluster_epoch(row.get('start', '')) or now, now)
            ends[row['job_id']] = start + limit * 60
        else:
            ends[row['job_id']] = now + max(0.0, limit - _elapsed_min(row.get('time_used', ''))) * 60
    return ends


def _outlasting(ends: dict[str, float]) -> dict:
    """{'certificate_warning': ...} when jobs end after the certificate expires (findings 0, 9, 55); never raises."""
    try:
        note = auth.outlasting(ends)
    except Exception:
        note = None
    return {'certificate_warning': note} if note else {}


@tool(RO)
def list_jobs() -> dict:
    """My running and pending jobs (arrays as one row), with the purpose recorded at submission (if submitted through
    this server), and what they hold together: nodes at once and the tokens they can still spend at their limits.
    For pending jobs `start` is Slurm's estimate (unreliable) and `reason` why it waits."""
    r = remote.run(f'squeue -u $USER -h -o {shlex.quote(SQ_FMT)}; echo {MARK}; '
                   f'squeue -u $USER -h -r -o {shlex.quote(SQ_FMT)}', timeout=60, retry=True)
    if not r.ok:
        raise _fail('squeue', r)
    collapsed, expanded = (r.out.split(MARK + '\n') + [''])[:2]
    rows = _parse_squeue(collapsed)
    known = ledger.submitted()
    for row in rows:
        e = known.get(row['job_id'].split('_')[0])
        if e:
            row['purpose'] = e.get('purpose')
    return {'jobs': rows, 'in_use': _in_use(_parse_squeue(expanded)), **_outlasting(_squeue_ends(rows, time.time()))}


SACCT_FMT = ('JobID,JobIDRaw,JobName,User,Partition,QOS,State,ExitCode,Submit,Start,End,ElapsedRaw,NNodes,NodeList,'
             'AllocTRES,StdOut,StdErr,WorkDir,TimelimitRaw')
# SubmitLine can hold '|' and is long: read on its own, last column, one line per job
SUBMIT_CMD = 'sacct -j {} -X -n -P -o JobIDRaw,SubmitLine 2>/dev/null | head -n 1 | head -c 4000'
SUBMIT_MAX = 500
SECRET = re.compile(r'(\b\w*(?:TOKEN|KEY|SECRET|PASSW\w*|PASS)\w*=)[^,\s\'"]+', re.I)


def _parse_sacct(text: str) -> list[dict]:
    keys = SACCT_FMT.split(',')
    rows = []
    for l in text.splitlines():
        v = l.split('|')
        if len(v) == len(keys):
            rows.append(dict(zip(keys, v)))
    return rows


def _sacct(job_id: str, submit_line: bool = False) -> list[dict]:
    """sacct rows of a job (every array task and requeue); submit_line=True also reads the job's sbatch command line
    (in the same SSH call) into the first row's 'SubmitLine'."""
    cmd = f'sacct -j {shlex.quote(job_id)} -X -D --array -n -P -o {SACCT_FMT}'
    if submit_line:
        cmd = f'{cmd} || exit $?; echo {MARK}; {SUBMIT_CMD.format(shlex.quote(job_id))}'
    r = remote.run(cmd, timeout=60, retry=True)
    if not r.ok:
        raise _fail('sacct', r)
    head, _, tail = r.out.partition(MARK + '\n')
    rows = _parse_sacct(head)
    if rows and tail.strip():
        rows[0]['SubmitLine'] = tail.strip().split('|', 1)[-1]
    return rows


def _submit_line(s: str) -> str:
    """The sbatch command line, clipped, with values of secret-looking variables (--export HF_TOKEN=...) hidden."""
    s = SECRET.sub(r'\1***', s.strip())
    return s if len(s) <= SUBMIT_MAX else s[:SUBMIT_MAX] + ' [...]'


def _first_node(nodelist: str) -> str:
    if not nodelist or nodelist in ('None assigned', '(null)'):
        return ''
    try:
        return S.expand_hostlist(nodelist)[0]
    except Exception:
        m = re.match(r'^([^\[,]+)(?:\[(\d+))?', nodelist)
        return (m.group(1) + (m.group(2) or '')) if m else ''


def _resolve(pattern: str, row: dict) -> str:
    """A Slurm filename pattern (%x %j %A %a %u %N %n %%, with zero padding like %4a) for one sacct row."""
    if '\\' in pattern:                          # a backslash switches the replacement off
        p = pattern.replace('\\', '')
    else:
        jid = row['JobID']
        master, _, task = jid.partition('_')
        task = task if task.isdigit() else ''
        vals = {'x': row['JobName'], 'j': row.get('JobIDRaw') or master, 'J': row.get('JobIDRaw') or master,
                'A': master, 'a': task or NO_VAL_TASK, 'u': row.get('User', ''), 'N': _first_node(row['NodeList']),
                'n': '0', '%': '%'}

        def sub(m):
            width, key = m.group(1), m.group(2)
            v = vals.get(key, m.group(0))
            return v.zfill(int(width)) if width and v.isdigit() else v
        p = re.sub(r'%(\d*)([xjJAauNn%])', sub, pattern)
    if not p.startswith('/') and row.get('WorkDir'):
        p = row['WorkDir'].rstrip('/') + '/' + p
    return p


def _log_paths(rows: list[dict], keys=('StdOut', 'StdErr')) -> list[str]:
    paths = []
    for row in rows:
        for k in keys:
            if row.get(k):
                p = _resolve(row[k], row)
                if p not in paths:
                    paths.append(p)
    return paths


def _missing_notes(rows: list[dict], paths_of) -> dict[str, str]:
    """{path: why it may not exist yet} for each row's paths (paths_of(row) -> list): pending, or in the prolog."""
    notes = {}
    for row in rows:
        el = int(row['ElapsedRaw']) if str(row.get('ElapsedRaw', '')).isdigit() else None
        for p in paths_of(row):
            notes.setdefault(p, logs.missing_note(_state(row), el))
    return notes


LOG_FILE = '__amadeus_file__'
LOG_BYTES = 400_000         # per file and call: a \r-only tqdm stream is one "line" of any length
PROGRESS_BYTES = 65_536
# before grep: \r updates as lines and no ANSI colour codes, so a pattern matches what the cleaned log shows
UNWRAP = r"""tr '\r' '\n' < "$f" | sed -e 's/\x1b\[[0-9;?]*[ -\/]*[@-~]//g'"""


def _read_logs(paths: list[str], lines: int = 100, grep: str = '', raw: bool = False,
               nbytes: int = 0) -> tuple[list[tuple[str, str | None]], str]:
    """([(path, text or None when missing)], error) for up to MAX_LOG_FILES files; paths with * ? [ are globs.
    Every path and pattern is passed quoted, never interpolated into shell code. grep: an ERE, grep -a (binary
    logs); nbytes: the last bytes instead of lines. Text is cleaned (logs.clean) unless raw."""
    paths = paths[:MAX_LOG_FILES]
    if not paths:
        return [], ''
    n = int(lines)
    if nbytes:
        show = f'tail -c {int(nbytes)} -- "$f"'
    elif grep and raw:
        show = f'grep -a -E -- {shlex.quote(grep)} "$f" | tail -n {n} | tail -c {LOG_BYTES}'
    elif grep:
        show = f'{UNWRAP} | grep -a -E -- {shlex.quote(grep)} | tail -n {n} | tail -c {LOG_BYTES}'
    else:
        show = f'tail -n {n} -- "$f" | tail -c {LOG_BYTES}'
    show += '; echo'            # the next header starts on its own line
    parts = []
    for p in paths:
        if re.search(r'[*?\[]', p):     # unquoted $p: pathname expansion only (IFS empty), never command substitution
            parts.append(f'(p={shlex.quote(p)}; IFS=; shopt -s nullglob; k=0; for f in $p; do [ -f "$f" ] || continue; '
                         f'k=$((k+1)); [ $k -gt {MAX_LOG_FILES} ] && break; printf "{LOG_FILE} %s\\n" "$f"; {show}; '
                         f'done; [ $k = 0 ] && printf "{LOG_FILE}! %s\\n" "$p")')
        else:
            parts.append(f'f={shlex.quote(p)}; if [ -f "$f" ]; then printf "{LOG_FILE} %s\\n" "$f"; {show}; '
                         f'else printf "{LOG_FILE}! %s\\n" "$f"; fi')
    r = remote.run('\n'.join(parts + ['true']), timeout=60, retry=True)
    if r.ssh_failed or r.timed_out:
        return [], f'(log tails unavailable: {_ssh_note(r) or r.err.strip()})'
    secs: list[list] = []
    for line in r.out.split('\n'):
        if line.startswith((LOG_FILE + ' ', LOG_FILE + '! ')):
            secs.append([line.split(' ', 1)[1], None if line.startswith(LOG_FILE + '!') else []])
        elif secs and secs[-1][1] is not None:
            secs[-1][1].append(line)
    out = []
    for path, body in secs[:MAX_LOG_FILES]:
        if body is None:
            out.append((path, None))
        else:
            text = '\n'.join(body).rstrip('\n')
            out.append((path, text if raw or nbytes else logs.clean(text)))
    return out, r.err.strip()


def _format_logs(secs: list[tuple[str, str | None]], notes: dict | None = None, last: int = 0,
                 limit: int = 20_000) -> str:
    """'=== path' sections; a missing file with the reason it may not exist yet; last: lines per file."""
    parts = []
    for path, text in secs:
        if text is None:
            parts.append(f'=== {path} ({(notes or {}).get(path, "not there")})')
        else:
            if last:
                text = '\n'.join(text.split('\n')[-last:])
            parts.append(f'=== {path}\n{text}' if text else f'=== {path} (empty)')
    return remote.clip('\n'.join(parts), limit)


def _tail(paths: list[str], lines: int, grep: str = '', notes: dict | None = None) -> str:
    """Cleaned tails of up to MAX_LOG_FILES files, as '=== path' sections."""
    more = ''
    if len(paths) > MAX_LOG_FILES:
        more = f'\n[{len(paths) - MAX_LOG_FILES} more files not shown; use job_log with a task id]'
    secs, err = _read_logs(paths, lines, grep)
    if not secs and err.startswith('(log tails unavailable'):
        return err
    return _format_logs(secs, notes) + (('\n' + err) if err else '') + more


def _compact(d: dict) -> dict:
    """A logs.diagnose() result trimmed for a tool result."""
    out = {k: d[k] for k in ('category', 'summary', 'note') if d.get(k)}
    if d.get('findings'):
        out['findings'] = [{k: f[k] for k in ('name', 'meaning', 'action', 'transient', 'line', 'nodes') if f.get(k)}
                           for f in d['findings'][:3]]
    if d.get('exceptions'):
        out['exceptions'] = d['exceptions'][:2]
    if d.get('benign_hidden'):
        out['benign_hidden'] = d['benign_hidden']
    return out


def _limit(raw: str) -> str:
    """TimelimitRaw (minutes) -> D-HH:MM:SS / HH:MM:SS; as given otherwise (UNLIMITED, Partition_Limit)."""
    if not str(raw).isdigit():
        return raw
    m = int(raw)
    d, h, mm = m // 1440, m % 1440 // 60, m % 60
    return (f'{d}-' if d else '') + f'{h:02d}:{mm:02d}:00'


def _pending_info(job_id: str) -> list[dict]:
    """Reason, start estimate and dependency of a pending job (squeue), [] on failure."""
    r = remote.run(f"squeue -j {shlex.quote(job_id)} -h -r -o '%i|%r|%S|%E'", timeout=60, retry=True)
    out = []
    for l in r.out.splitlines() if r.ok else []:
        v = l.split('|')
        if len(v) == 4:
            out.append({'job_id': v[0], 'reason': v[1], 'start': v[2]}
                       | ({'dependency': v[3]} if v[3] not in ('', '(null)') else {}))
    return out


DIAG_LINES = 200            # log lines per file read for the diagnosis of an ended job


def _details(job_id: str, log_lines: int = 30) -> dict:
    rows = _sacct(job_id, submit_line=True)
    if not rows:
        return {'error': f'no job {job_id} in sacct'}
    total = 0.0
    out_rows = []
    for row in rows:
        try:
            elapsed = int(row['ElapsedRaw'] or 0)
        except ValueError:
            elapsed = 0
        tokens = S.credits_from_tres(row['Partition'], elapsed, row['AllocTRES'])
        total += tokens or 0
        out_rows.append({'job_id': row['JobID'], 'name': row['JobName'], 'partition': row['Partition'],
                         'qos': row['QOS'], 'state': row['State'], 'exit_code': row['ExitCode'],
                         'submit': row['Submit'], 'start': row['Start'], 'end': row['End'],
                         'elapsed_s': elapsed, 'time_limit': _limit(row.get('TimelimitRaw', '')),
                         'nodes': row['NNodes'], 'nodelist': row['NodeList'], 'tokens': tokens})
    res = {'rows': out_rows, 'tokens_total': round(total, 3)}
    if rows[0].get('SubmitLine'):
        res['submit_line'] = _submit_line(rows[0]['SubmitLine'])
    if any(r['State'].startswith('PENDING') for r in rows):
        if (p := _pending_info(job_id)):
            res['pending'] = p
    e = ledger.submitted().get(job_id.split('_')[0])
    if e:
        res['purpose'] = e.get('purpose')
    paths = _log_paths(rows)
    notes = _missing_notes(rows, _log_paths_of)
    ended = all(_state(r) in S.TERMINAL for r in rows)
    if ended:
        # one read serves the diagnosis (more lines) and the tails shown
        n = max(min(log_lines, 500), DIAG_LINES)
        secs, err = _read_logs(paths, n)
        if secs:
            d = logs.diagnose(_format_logs(secs, limit=200_000), rows)
            if any(_state(r) != 'COMPLETED' for r in rows) or d.get('note') or d.get('category') != logs.BENIGN:
                res['diagnosis'] = _compact(d)
        if log_lines > 0:
            more = (f'\n[{len(paths) - MAX_LOG_FILES} more files not shown; use job_log with a task id]'
                    if len(paths) > MAX_LOG_FILES else '')
            res['logs'] = (_format_logs(secs, notes, last=min(log_lines, 500)) + (('\n' + err) if err else '') + more
                           if secs else err)
    elif log_lines > 0:
        res['logs'] = _tail(paths, min(log_lines, 500), notes=notes)
    return res


def _log_paths_of(row: dict) -> list[str]:
    return _log_paths([row])


@tool(RO)
def job_details(job_id: str, log_lines: int = 30) -> dict:
    """State, exit code, times, time limit, nodes and cost of a job (every array task and requeue attempt), its sbatch
    command line, the reason (and dependency) a pending job waits, and the cleaned tails of its output files (at most
    20 files). An ended job that failed, or completed with errors in its log, gets a `diagnosis`: category
    (infrastructure = cluster side, resubmit; script; program; benign), summary, known signatures with their meaning and action
    (transient = a resubmit usually works), and the first real exceptions; harmless noise is not counted."""
    return _details(_check_id(job_id), log_lines)


@tool(RO)
def job_log(job_id: str, stream: str = 'out', lines: int = 100, grep: str = '', file: str = '',
            progress: bool = False, hide_benign: bool = False, raw: bool = False) -> dict:
    """Tail of a job's stdout (stream='out'), stderr ('err') or of `file`: a path or glob relative to the job's
    working directory (%j %x %A %a allowed, e.g. 'logs/%x-%j-r*.log' or 'results/run-%j/train.log'; at most 20
    files). For array jobs pass the task id (123_4) to see one task.
    The text is cleaned: ANSI colours and binary bytes removed, \\r progress updates collapsed to their last state
    (raw=True: as written). grep: an extended regex, matched per line (also on \\r updates and in binary logs).
    progress=True: instead of text, the latest state of each progress bar per file (label, n, total, percent, rate,
    elapsed, eta; kind loading / warmup / work: 'Loading weights 723/723' is not training).
    hide_benign=True: drop known-harmless lines and tracebacks (vLLM usage-stats TLS error, shutdown noise, srun's
    'forcing job termination', ...); counted in benign_hidden. A log missing while the job is pending or in its
    first minutes is normal (node prolog)."""
    job_id = _check_id(job_id)
    if stream not in ('out', 'err'):
        return {'error': "stream must be 'out' or 'err'"}
    file = _relative(file, 'file') if str(file or '').strip() else ''
    rows = _sacct(job_id)
    if not rows:
        return {'error': f'no job {job_id} in sacct'}
    if file:
        def paths_of(row):
            return [_resolve(file, row)]
    else:
        def paths_of(row):
            return _log_paths([row], ('StdErr',) if stream == 'err' else ('StdOut',))
    paths = list(dict.fromkeys(p for row in rows for p in paths_of(row)))
    notes = _missing_notes(rows, paths_of)
    res: dict = {'paths': paths[:MAX_LOG_FILES]}
    if len(paths) > MAX_LOG_FILES:
        res['note'] = f'{len(paths) - MAX_LOG_FILES} more files not shown; pass a task id (123_4)'
    if progress:
        secs, err = _read_logs(paths, nbytes=PROGRESS_BYTES)
        bars = {p: b for p, t in secs if t is not None and (b := logs.progress(t))}
        res['progress'] = bars
        missing = {p: notes.get(p, 'not there') for p, t in secs if t is None}
        if missing:
            res['missing'] = missing
        if not bars and not err:
            res['note'] = f'no progress bars in the last {PROGRESS_BYTES // 1024} kB of these files'
        if err:
            res['error'] = err
        return res
    secs, err = _read_logs(paths, min(max(lines, 1), 2000), grep, raw=raw)
    if hide_benign:
        hidden: dict[str, int] = {}
        kept = []
        for p, t in secs:
            if t is not None:
                t, h = logs.filter_benign(t)
                for k, v in h.items():
                    hidden[k] = hidden.get(k, 0) + v
            kept.append((p, t))
        secs = kept
        if hidden:
            res['benign_hidden'] = hidden
    res['text'] = (_format_logs(secs, notes) + (('\n' + err) if err else '')) if secs else err
    return res


# ---------------------------------------------------------------- jobs: waiting

UNTIL = ('end', 'all_end', 'any_end', 'any_fail', 'start')
WAIT_MAX_JOBS = 50
WAIT_MAX_ROWS = 8           # jobs / array tasks whose logs are watched in one call
WAIT_MAX_MILESTONES = 10
WAIT_MAX_EXPECT = 20
WAIT_READ = 8000            # new log bytes read per file when returning
WAIT_FIRST = 3000           # without a cursor: the last bytes of each log
WAIT_TEXT = 4000            # cleaned characters per file in the result
WAIT_LINES = 40
WAIT_SEEN = 200
_ALARMS_SENT: dict[str, set[str]] = {}    # job id -> alarm keys wait_for_job returned (this process)
SALVAGE = ("Cancel, don't salvage: if the run is broken or crawling, cancel_job it, fix the script and resubmit; "
           "never pause the batch script or raise limits to rescue partial results")


def _state(row: dict) -> str:
    return (row.get('State') or '').split(' ')[0]


def _relative(path: str, what: str) -> str:
    path = str(path).strip()
    if not path or path.startswith(('/', '~', '$')) or '..' in Path(path).parts:
        raise ToolError(f'{what} must be a path relative to the job\'s working directory, without ..: {path!r}')
    return path


def _wait_milestones(milestones) -> list[dict]:
    out = []
    for m in milestones or []:
        m = {'pattern': m} if isinstance(m, str) else m
        if not isinstance(m, dict) or not str(m.get('pattern') or '').strip():
            raise ToolError("each milestone needs a 'pattern' (a regex), optionally 'within_s' and 'file'")
        p = str(m['pattern'])
        try:
            re.compile(p)
        except re.error as e:
            raise ToolError(f'milestone {p!r}: invalid regex: {e}') from None
        f = str(m.get('file') or 'out').strip()
        if f not in ('out', 'err'):
            f = _relative(f, 'milestone file')
        try:
            within = float(m['within_s']) if m.get('within_s') not in (None, '', 0) else None
        except (TypeError, ValueError):
            raise ToolError(f'milestone {p!r}: within_s must be a number of seconds') from None
        out.append({'pattern': p, 'file': f, 'within_s': within})
    if len(out) > WAIT_MAX_MILESTONES:
        raise ToolError(f'at most {WAIT_MAX_MILESTONES} milestones')
    return out


def _probe_script(targets: list[tuple]) -> str:
    """Facts about job files in one shell script; every path and pattern is passed quoted."""
    parts = ['echo "T $(date +%s)"']
    for i, t in enumerate(targets):
        path = shlex.quote(t[1])
        if t[0] == 'stat':
            parts.append(f'f={path}; if [ -f "$f" ]; then echo "S {i} $(stat -c "%s %Y" -- "$f")"; fi')
        elif t[0] == 'grep':        # Perl regex (\d works); an ERE if this grep has no -P
            pat = shlex.quote(t[2])
            # on the text job_log shows (\r split, ANSI codes removed), so a pattern that matches there matches here
            parts.append(f'f={path}; if [ -f "$f" ]; then m=$({UNWRAP} | grep -a -n -m1 -P -- {pat} 2>/dev/null); '
                         f'[ $? = 2 ] && m=$({UNWRAP} | grep -a -n -m1 -E -- {pat} 2>/dev/null); '
                         f'[ -n "$m" ] && echo "G {i} $(printf %s "$m" | head -c 2000 | base64 -w0)"; fi')
        elif t[0] == 'glob':        # unquoted $p: pathname expansion only, never command substitution
            parts.append(f'(p={path}; IFS=; shopt -s nullglob; for f in $p; do [ -e "$f" ] && '
                         f'echo "E {i} $(stat -c "%s %Y" -- "$f") $(printf %s "$f" | base64 -w0)"; done | head -n 20)')
    return '\n'.join(parts + ['true'])


def _read_script(files: list[tuple[str, int | None]]) -> str:
    """New bytes of each file since its offset (None: the last WAIT_FIRST bytes), at most WAIT_READ, base64."""
    parts = []
    for i, (path, off) in enumerate(files):
        o = -1 if off is None else int(off)
        parts.append(f'f={shlex.quote(path)}; if [ -f "$f" ]; then s=$(stat -c %s -- "$f"); o={o}; '
                     f'[ "$o" -gt "$s" ] && o=0; [ "$o" -lt 0 ] && o=$((s-{WAIT_FIRST})); [ "$o" -lt 0 ] && o=0; '
                     f'b=$((s-{WAIT_READ})); [ "$b" -lt "$o" ] && b=$o; '
                     f'echo "R {i} $s $b $o $(tail -c +$((b+1)) -- "$f" | head -c $((s-b)) | base64 -w0)"; fi')
    return '\n'.join(parts + ['true'])


def _b64(s: str) -> bytes:
    try:
        return base64.b64decode(s)
    except (ValueError, TypeError):
        return b''


class _Watch:
    """One wait_for_job call: the jobs' sacct rows, file facts from the probes, and the events the caller has seen."""

    def __init__(self, ids, until, milestones, quiet_s, expect, cursor):
        self.ids, self.until, self.ms, self.quiet_s, self.expect = ids, until, milestones, quiet_s, expect
        cursor = cursor if isinstance(cursor, dict) else {}
        self.cursor = bool(cursor)
        self.off = {}
        for k, v in (cursor.get('off') or {}).items():
            if isinstance(v, int) or str(v).isdigit():
                self.off[str(k)] = int(v)
        self.seen = [str(k) for k in cursor.get('seen') or []]
        # without a cursor (a re-call after a compaction): alarms this server already returned for these jobs come
        # back as info, so an old MILESTONE or a lasting LATE does not end every call at once
        self.pre = set() if self.cursor else {k for j in ids for k in _ALARMS_SENT.get(j.split('_')[0], ())}
        self.rows: dict[str, list[dict]] = {}
        self.stat: dict[str, tuple[int, int]] = {}
        self.hits: dict[tuple, tuple[int, str]] = {}
        self.found: dict[tuple, list[tuple[int, int, str]]] = {}
        self.now = time.time()
        self.targets: list[tuple] = []

    # -- polling
    def poll_cmd(self) -> str:
        ids = ','.join(self.ids)
        # epoch times: the server and the cluster may sit in different time zones
        return (f'SLURM_TIME_FORMAT=%s sacct -j {ids} -X --array -n -P -o {SACCT_FMT}; echo "{MARK} $?"\n'
                + _probe_script(self.targets))

    def take(self, out: str) -> bool:
        """Parse a poll's output; False when sacct failed."""
        head, sep, probe = out.partition(MARK + ' ')
        if not sep or not probe.startswith('0'):
            return False
        keys = SACCT_FMT.split(',')
        self.rows = {}
        for l in head.splitlines():
            v = l.split('|')
            if len(v) != len(keys):
                continue
            row = dict(zip(keys, v))
            for jid in self.ids:
                if row['JobID'] == jid or row['JobID'].split('_')[0] == jid or row['JobIDRaw'] == jid:
                    self.rows.setdefault(jid, []).append(row)
                    break
        self.apply(probe.split('\n', 1)[1] if '\n' in probe else '', self.targets)
        return True

    def apply(self, out: str, targets: list[tuple]) -> None:
        for l in out.splitlines():
            v = l.split(' ')
            try:
                if v[0] == 'T':
                    self.now = int(v[1])
                    continue
                t = targets[int(v[1])]
                if v[0] == 'S':
                    self.stat[t[1]] = (int(v[2]), int(v[3]))
                elif v[0] == 'G':
                    n, _, line = _b64(v[2]).decode(errors='replace').partition(':')
                    self.hits[t[3]] = (int(n), logs.clean(line).strip()[:logs.MAX_LINE])
                elif v[0] == 'E':
                    self.found.setdefault(t[2], []).append((int(v[2]), int(v[3]),
                                                            _b64(v[4]).decode(errors='replace')))
            except (IndexError, ValueError):
                continue
        for t in targets:
            if t[0] == 'glob':
                self.found.setdefault(t[2], [])

    def catch_up(self) -> list[tuple]:
        """Targets the last poll could not know yet (a job just started or ended); then next poll's targets."""
        now = self.want()
        new = [t for t in now if t not in self.targets]
        self.targets = now
        return new

    def want(self) -> list[tuple]:
        out = []
        for row in self.watched():
            rid = row['JobID']
            for p in _log_paths([row]):
                out.append(('stat', p))
            for k, m in enumerate(self.ms):
                if (rid, k) not in self.hits:
                    out.append(('grep', self.file(m['file'], row), m['pattern'], (rid, k)))
        for jid in self.ids:
            rows = self.rows.get(jid, [])
            if self.expect and rows and all(_state(r) in S.TERMINAL for r in rows):
                for row in rows[:WAIT_MAX_ROWS]:
                    for k, e in enumerate(self.expect):
                        if (row['JobID'], k) not in self.found:
                            out.append(('glob', _resolve(e, row), (row['JobID'], k)))
        return list(dict.fromkeys(out))

    @staticmethod
    def file(f: str, row: dict) -> str:
        if f in ('out', 'err'):
            paths = _log_paths([row], ('StdErr',) if f == 'err' else ('StdOut',))
            return paths[0] if paths else ''
        return _resolve(f, row)

    def watched(self) -> list[dict]:
        """Started rows (running first), spread over the jobs, at most WAIT_MAX_ROWS."""
        per = max(1, WAIT_MAX_ROWS // len(self.ids))
        out = []
        for jid in self.ids:
            rows = [r for r in self.rows.get(jid, []) if _state(r) in STARTED or
                    (_state(r) in S.TERMINAL and r.get('Start', '') not in ('', 'None', 'Unknown'))]
            rows.sort(key=lambda r: _state(r) in S.TERMINAL)
            out += rows[:per]
        return out[:WAIT_MAX_ROWS]

    # -- what happened
    def job(self, jid: str) -> dict:
        rows = self.rows.get(jid, [])
        states = [_state(r) for r in rows]
        d = {'job_id': jid, 'state': ', '.join(f'{s} ×{states.count(s)}' if states.count(s) > 1 else s
                                               for s in dict.fromkeys(states)) or 'unknown'}
        d['ended'] = bool(rows) and all(s in S.TERMINAL for s in states)
        d['started'] = d['ended'] or any(s in STARTED for s in states)
        d['failed'] = d['ended'] and any(s != 'COMPLETED' for s in states)
        el = [int(r['ElapsedRaw']) for r in rows if r.get('ElapsedRaw', '').isdigit()]
        if el:
            d['elapsed_s'] = max(el)
        if d['ended']:
            d['exit_code'] = ', '.join(dict.fromkeys(r['ExitCode'] for r in rows))
        return d

    def outputs(self, jid: str) -> dict | None:
        rows = [r for r in self.rows.get(jid, []) if _state(r) in S.TERMINAL][:WAIT_MAX_ROWS]
        if not self.expect or not rows:
            return None
        files, missing = [], []
        for row in rows:
            start = _epoch(row.get('Start', ''))
            for k, e in enumerate(self.expect):
                hits = self.found.get((row['JobID'], k))
                if hits is None:
                    return None             # not probed yet
                fresh = [h for h in hits if start is None or h[1] >= start - 60]
                if not fresh:
                    missing.append(_resolve(e, row) + (' (only older files)' if hits else ''))
                for size, mtime, path in fresh[:5]:
                    files.append({'path': path, 'bytes': size} | ({'empty': True} if not size else {}))
        return {'ok': not missing, 'files': files[:20], 'missing': missing[:20]}

    def events(self) -> list[tuple[str, str, str]]:
        """(key, text, kind); kind: start, end, fail, alarm (always returns), info."""
        ev = []
        for jid in self.ids:
            j = self.job(jid)
            if j['ended']:
                kind, word = ('fail', 'FAILED') if j['failed'] else ('end', 'ENDED')
                if j['failed'] and str(j['state']).startswith('CANCELLED'):
                    word = 'CANCELLED'   # still kind 'fail' for until=any_fail, but no salvage hint
                ev.append((f'{word} {jid}', f'{word} {jid}: {j["state"]} (exit {j["exit_code"]}, '
                                             f'{j.get("elapsed_s", 0)} s)', kind))
                o = self.outputs(jid)
                if o and not o['ok'] and not j['failed']:   # a failed job's FAILED event already says it
                    ev.append((f'OUTPUTS_MISSING {jid}', f'OUTPUTS_MISSING {jid}: {j["state"]} but missing '
                                                         f'{", ".join(o["missing"][:5])}', 'alarm'))
            elif j['started']:
                ev.append((f'STARTED {jid}', f'STARTED {jid}: {j["state"]}', 'start'))
        for row in self.watched():
            rid, st = row['JobID'], _state(row)
            el = int(row['ElapsedRaw']) if row.get('ElapsedRaw', '').isdigit() else 0
            for k, m in enumerate(self.ms):
                hit = self.hits.get((rid, k))
                if hit:
                    ev.append((f'MILESTONE {rid} {m["pattern"]}',
                               f'MILESTONE {rid}: /{m["pattern"]}/ at line {hit[0]}: {hit[1]}', 'alarm'))
                elif m['within_s'] and el > m['within_s']:
                    ev.append((f'LATE {rid} {m["pattern"]}',
                               f'LATE {rid}: /{m["pattern"]}/ not in {m["file"]} after {el} s '
                               f'(due within {m["within_s"]:g} s)',
                               'alarm' if st not in S.TERMINAL else 'info'))
            if self.quiet_s and st == 'RUNNING' and el >= self.quiet_s:
                mt = [self.stat[p][1] for p in _log_paths([row]) if p in self.stat]
                quiet = int(self.now - max(mt)) if mt else el
                if quiet >= self.quiet_s:
                    ev.append((f'QUIET {rid} {max(mt) if mt else 0}',
                               f'QUIET {quiet} s: {rid} is RUNNING but its logs have not grown for {quiet} s', 'alarm'))
        return ev

    def fresh(self) -> list[tuple[str, str, str]]:
        return [(k, t + ' (reported before)', 'info') if kind == 'alarm' and k in self.pre else (k, t, kind)
                for k, t, kind in self.events() if k not in self.seen]

    def unwatched(self) -> list[str]:
        """Started jobs none of whose rows fit into WAIT_MAX_ROWS: no milestone, LATE, QUIET or log lines."""
        rows = {r['JobID'] for r in self.watched()}
        return [j for j in self.ids if self.job(j)['started'] and not self.job(j)['ended']
                and not any(r['JobID'] in rows for r in self.rows.get(j, []))]

    def due(self, fresh) -> bool:
        kinds = {e[2] for e in fresh}
        if 'alarm' in kinds or self.all_ended():
            return True
        want = {'start': {'start', 'end', 'fail'}, 'any_end': {'end', 'fail'}, 'any_fail': {'fail'}}
        return bool(kinds & want.get(self.until, set()))

    def all_ended(self) -> bool:
        return all(self.job(j)['ended'] for j in self.ids)

    # -- the result
    def read(self) -> tuple[list[tuple[str, int | None]], str]:
        files = []
        for row in self.watched():
            for p in _log_paths([row]):
                if (p, self.off.get(p)) not in files:
                    files.append((p, self.off.get(p)))
        return files, _read_script(files)

    def take_read(self, files, out: str) -> dict[str, str]:
        texts, budget = {}, 3 * WAIT_TEXT
        for l in out.splitlines():
            v = l.split(' ')
            try:
                path, off = files[int(v[1])]
                _size, b, o = int(v[2]), int(v[3]), int(v[4])
            except (IndexError, ValueError):
                continue
            chunk = _b64(v[5]) if len(v) > 5 else b''
            nl = chunk.rfind(b'\n')
            # a partial last line (still being written) is shown now and read again next time
            self.off[path] = b + nl + 1 if 0 <= nl < len(chunk) - 1 else b + len(chunk)
            text = '\n'.join(logs.clean(chunk).strip('\n').split('\n')[-WAIT_LINES:])
            if off is not None and b > o:
                text = f'[... {b - o} bytes skipped ...]\n' + text
            if text.strip() and budget > 0:
                texts[path] = remote.clip(text, min(WAIT_TEXT, budget))
                budget -= len(texts[path])
        return texts

    def cursor_out(self, fresh) -> dict:
        for k, _, kind in fresh:
            if kind == 'alarm':
                _ALARMS_SENT.setdefault(k.split(' ')[1].split('_')[0], set()).add(k)
        while len(_ALARMS_SENT) > WAIT_SEEN:
            _ALARMS_SENT.pop(next(iter(_ALARMS_SENT)))
        seen = self.seen + [e[0] for e in fresh if e[0] not in self.seen]
        return {'off': dict(self.off), 'seen': seen[-WAIT_SEEN:]}


def _epoch(stamp: str) -> int | None:
    """A sacct time printed with SLURM_TIME_FORMAT=%s."""
    return int(stamp) if str(stamp).isdigit() else None


def _diagnosis(rows: list[dict]) -> dict:
    return _compact(logs.diagnose(_tail(_log_paths(rows), DIAG_LINES), rows))


@tool(RO)
async def wait_for_job(ctx: Context, job_id: str = '', job_ids: list[str] | None = None, until: str = 'end',
                       milestones: list[dict] | None = None, quiet_alarm_s: int = 0, expect: list[str] | None = None,
                       log_cursor: dict | None = None, max_wait_s: int = 900, poll_s: int = 30) -> dict:
    """Wait for one job (job_id) or several (job_ids) and return at the first event that matters, with the new log
    lines since the last call. Returns at the latest after max_wait_s (10 to 3600).

    until: 'end' / 'all_end' = all jobs ended (default), 'any_end' = the first job ends, 'any_fail' = the first job
    ends other than COMPLETED (or all ended), 'start' = a job starts (or ends).
    Alarms always return at once:
    - milestones: [{pattern, within_s, file}]: a regex (Perl/Python style) grepped in the job's 'out' (default), 'err'
      or a path relative to its working directory (%j %x %a allowed). A match returns MILESTONE; no match within_s
      seconds after the job's start returns LATE (e.g. vLLM 'ready after \\d+s' within 600).
    - quiet_alarm_s: QUIET when a RUNNING job's logs have not grown for that long (0 = off). vLLM compile and graph
      capture can be silent for about 3 min: use 300 for big runs, 120-180 while testing.
    - expect: output paths relative to the working directory (%j %x %a, globs): OUTPUTS_MISSING when a job ended
      without files written since its start (a COMPLETED state can hide a failed run).
    Events: STARTED, ENDED, FAILED, MILESTONE, LATE, QUIET, OUTPUTS_MISSING; failed jobs come with a diagnosis.
    finished: True = all ended, False = still running (an event or max_wait_s), None = cluster or job not found.
    Pass next_cursor back as log_cursor: you then get only new log lines and events you have not seen yet.
    State lives in Slurm: after a restart, a dropped connection or a compaction just call again with the same ids;
    alarms this server already returned for these jobs then come back as info and do not end the call at once.
    Never build sleep/sacct watcher loops or background monitors. Each poll is one SSH command for all jobs."""
    ids = [job_id] if str(job_id or '').strip() else []
    ids = list(dict.fromkeys(_check_id(i) for i in ids + list(job_ids or [])))
    if not ids:
        return {'error': 'give job_id or job_ids'}
    if len(ids) > WAIT_MAX_JOBS:
        return {'error': f'at most {WAIT_MAX_JOBS} jobs per call'}
    until = (until or 'end').strip()
    if until not in UNTIL:
        return {'error': f'until must be one of {", ".join(UNTIL)}'}
    exp = [_relative(e, 'expect') for e in expect or []]
    if len(exp) > WAIT_MAX_EXPECT:
        return {'error': f'at most {WAIT_MAX_EXPECT} expected outputs'}
    w = _Watch(ids, until, _wait_milestones(milestones), max(0, int(quiet_alarm_s or 0)), exp, log_cursor)
    t0 = time.monotonic()
    max_wait_s = min(max(max_wait_s, 10), 3600)
    poll_s = min(max(poll_s, 5), 300)
    fails, unseen, last_err = 0, 0, ''
    while True:
        r = await asyncio.to_thread(remote.run, w.poll_cmd(), 60, False, None, True)
        elapsed = time.monotonic() - t0
        if not r.ok or not w.take(r.out):
            fails += 1
            last_err = _ssh_note(r) or r.err.strip()[-500:]
            if fails >= 3:
                return {'finished': None, 'waited_s': round(elapsed), 'error': f'cannot reach the cluster: {last_err}',
                        'last_states': {j: w.job(j)['state'] for j in ids}}
        else:
            fails = 0
            new = w.catch_up()
            if new:
                p = await asyncio.to_thread(remote.run, _probe_script(new), 60, False, None, True)
                if p.ok:
                    w.apply(p.out, new)
                w.catch_up()
            missing = [j for j in ids if j not in w.rows]
            unseen = unseen + 1 if missing else 0
            if missing and unseen >= 3 and elapsed > 30:
                return {'finished': None, 'waited_s': round(elapsed), 'error': f'no job {", ".join(missing)} in sacct'}
            fresh = [] if missing else w.fresh()
            if not missing and w.due(fresh):
                return await asyncio.to_thread(_wait_result, w, fresh, elapsed)
        states = '; '.join(f'{j}: {w.job(j)["state"]}' for j in ids)
        await ctx.report_progress(min(elapsed, max_wait_s), max_wait_s, states[:300])
        remaining = max_wait_s - elapsed
        if remaining <= 0:
            res = (await asyncio.to_thread(_wait_result, w, w.fresh(), elapsed) if w.rows
                   else {'finished': False, 'waited_s': round(elapsed)})
            res['note'] = 'still running or pending; call wait_for_job again (with log_cursor=next_cursor)'
            return res
        await asyncio.sleep(min(poll_s, remaining))


def _watch_ends(w: _Watch) -> dict[str, float]:
    """Latest ends of the jobs still running or pending, from the time limit recorded at submission."""
    known, ends = ledger.submitted(), {}
    for jid in w.ids:
        limit = ((known.get(jid.split('_')[0]) or {}).get('estimate') or {}).get('time_limit_min')
        if not limit or w.job(jid)['ended']:
            continue
        starts = [t for r in w.rows.get(jid, []) if (t := _epoch(r.get('Start', '')))]
        ends[jid] = (max(starts) if starts else max(w.now, time.time())) + float(limit) * 60
    return ends


def _wait_result(w: _Watch, fresh: list, elapsed: float) -> dict:
    jobs = []
    for jid in w.ids:
        j = w.job(jid)
        jobs.append({k: v for k, v in j.items() if k not in ('ended', 'started', 'failed')})
        o = w.outputs(jid)
        if o:
            jobs[-1]['outputs'] = o
            if j['failed'] and o['ok']:
                jobs[-1]['note'] = f'{j["state"]} but the expected outputs are there: check them before rerunning'
    res = {'finished': w.all_ended(), 'waited_s': round(elapsed)}
    if fresh:
        res['events'] = [e[1] for e in fresh]
    res['jobs'] = jobs
    if (skipped := w.unwatched()):
        res['unwatched'] = {'job_ids': skipped[:20], 'note': f'at most {WAIT_MAX_ROWS} running jobs or tasks are watched '
                            'per call: these get no milestone, LATE or QUIET checks and no log lines; wait on them in '
                            'a separate call'}
    if w.ms:
        res['milestones'] = [{'job_id': row['JobID'], 'pattern': m['pattern'], 'matched': (row['JobID'], k) in w.hits}
                             | ({'line': w.hits[(row['JobID'], k)][1]} if (row['JobID'], k) in w.hits else {})
                             for row in w.watched() for k, m in enumerate(w.ms)][:20]
    failed = [j for j in w.ids if w.job(j)['failed']][:3]
    if len(w.ids) == 1 and res['finished']:
        try:                        # one job: its details, as before (their diagnosis goes under the job id)
            det = _details(w.ids[0], 0 if w.cursor else 30)
            if 'diagnosis' in det:
                res['diagnosis'] = {w.ids[0]: det.pop('diagnosis')}
            res.update(det)
        except ClusterError as e:
            res['note'] = f'finished, but the details could not be read: {e}'
    files, script = w.read()
    if files:
        r = remote.run(script, 60, False, None, True)
        if r.ok:
            texts = w.take_read(files, r.out)
            if texts:
                res['new_lines'] = texts
    for jid in failed:
        if jid in res.get('diagnosis', {}):
            continue
        try:
            res.setdefault('diagnosis', {})[jid] = _diagnosis(w.rows[jid])
        except ClusterError:
            pass
    broken = [j for j in failed if not str(w.job(j)['state']).startswith('CANCELLED')]
    if broken or any(e[0].startswith(('LATE', 'QUIET')) for e in fresh):
        res['hint'] = SALVAGE
    res.update(_outlasting(_watch_ends(w)))
    res['next_cursor'] = w.cursor_out(fresh)
    return res


# ---------------------------------------------------------------- jobs: inspect a running node

NODE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9-]*$')
NODE_EXEC_MAX_S = 120
NODE_EXEC_MAX_CMD = 4000
SIGNALS = {'kill', 'pkill', 'killall', 'skill', 'snice'}
FUSER_KILL = re.compile(r'(?:^|[\s;&|(])fuser\b[^;&|]*\s-\w*k')
DETACH = {'nohup', 'setsid', 'disown', 'daemonize', 'at', 'batch', 'screen', 'tmux'}
# interactive programs, except the batch forms that end by themselves (top -b, watch is never fine)
NODE_INTERACTIVE = re.compile(r'^\s*(vim?|nano|emacs|htop|watch|less|more|tmux|screen|ssh|top(?!.*\s-\w*b))\b')
STARTED = {'RUNNING', 'SUSPENDED', 'COMPLETING', 'CONFIGURING', 'RESIZING', 'STOPPED', 'SIGNALING'}


def _words(command: str, depth: int = 0) -> tuple[list[str], list[str]] | None:
    """(tokens, command words) of a shell command: the first word of every simple command plus the wrappers before
    it (nohup, timeout, env, ...). None if it cannot be read (unbalanced quotes)."""
    lex = shlex.shlex(command.replace('\n', ' ; '), posix=True, punctuation_chars=';&|()<>')
    lex.whitespace_split = True
    lex.commenters = ''
    try:
        toks = list(lex)
    except ValueError:
        return None
    segs, cur = [], []
    for t in toks:
        if t in guard.SEPARATORS:
            segs.append(cur)
            cur = []
        else:
            cur.append(t)
    segs.append(cur)
    firsts = []
    # nested commands: bash -c '...' (also -lc, -ec), $(...), `...`
    bodies = re.findall(r'\$\(([^()]*)\)', command) + re.findall(r'`([^`]*)`', command)
    for seg in segs:
        i, wrappers = guard.command_start(seg)
        firsts += wrappers
        if i < len(seg):
            firsts.append(os.path.basename(seg[i]))
            if firsts[-1] in guard.SHELLS and (b := guard.shell_body(seg[i + 1:])):
                bodies.append(b)
    for b in bodies if depth < 4 else []:
        inner = _words(b, depth + 1)
        if inner is None:
            return None
        toks, firsts = toks + inner[0], firsts + inner[1]
    return toks, firsts


def node_exec_policy(command: str) -> dict:
    """What node_exec refuses, and what needs the human's approval, on top of run_command's policy."""
    if not command.strip():
        return {'refuse': 'command is empty'}
    if len(command) > NODE_EXEC_MAX_CMD:
        return {'refuse': f'command is over {NODE_EXEC_MAX_CMD} characters: put longer scripts in a file in the job\'s '
                          'directory and run that'}
    if NODE_INTERACTIVE.match(command):
        return {'refuse': 'interactive programs hang without a terminal; use batch forms (top -b -n1, nvidia-smi, ps)'}
    words = _words(command)
    if words is None:
        return {'refuse': 'cannot read the command (unbalanced quotes)'}
    toks, firsts = words
    if '&' in toks or DETACH & set(firsts):
        # 10.txt: a detached `nohup srun --overlap ... &` kept a B200 job alive after its cause was known
        return {'refuse': 'background or detached processes are not allowed: node_exec is for short diagnostics that '
                          'end within the timeout'}
    if 'srun' in firsts:
        return {'refuse': 'no srun inside node_exec: the command already runs on the node'}
    policy = guard.command_policy(command)
    if policy.get('refuse'):
        return {'refuse': policy['refuse']}
    approval = list(policy.get('needs_approval') or [])
    if SIGNALS & set(firsts) or FUSER_KILL.search(command):
        approval.append('it sends signals to processes (kill, pkill, fuser -k, ...): that can end or stall the job')
    return {'refuse': None, 'needs_approval': approval}


def _running_job(job_id: str) -> tuple[dict | None, str]:
    """The user's own running job (or array task) as `scontrol show job` reads it, or (None, why not)."""
    r = remote.run(f'printf "%s\\n" "$USER"; echo {MARK}; scontrol show job -o {shlex.quote(job_id)}', timeout=60,
                   retry=True)
    user, _, text = r.out.partition(MARK + '\n')
    if not r.ok and not text.strip():
        if r.ssh_failed or r.timed_out:
            raise _fail('scontrol show job', r)
        return None, f'no job {job_id} in Slurm (finished jobs drop out after a few minutes): {r.err.strip()[:300]}'
    master, _, task = job_id.partition('_')
    jobs = [S.parse_scontrol(l) for l in text.splitlines() if l.strip()]
    if task:
        jobs = [j for j in jobs if j.get('ArrayJobId') == master and j.get('ArrayTaskId') == task]
    else:
        jobs = [j for j in jobs if j.get('JobId') == job_id or j.get('ArrayJobId') == job_id]
    running = [j for j in jobs if j.get('JobState') == 'RUNNING']
    if not jobs:
        return None, f'no job {job_id} in Slurm'
    if len(running) > 1:
        return None, (f'{job_id} is an array with {len(running)} running tasks: pass one, e.g. '
                      f'{master}_{running[0].get("ArrayTaskId", "0")}')
    job = running[0] if running else jobs[0]
    owner = job.get('UserId', '').split('(')[0]
    if owner != user.strip():
        return None, f'{job_id} belongs to {owner or "someone else"}: node_exec only runs inside your own jobs'
    if job.get('JobState') != 'RUNNING':
        return None, f'{job_id} is {job.get("JobState") or "not running"}: node_exec needs a RUNNING job'
    return job, ''


@tool(DESTRUCTIVE)
def node_exec(job_id: str, command: str, node: str = '', timeout_s: int = 60, approval: str = '') -> dict:
    """Run a short diagnostic command on the node of one of your RUNNING jobs (ssh to compute nodes is refused on
    MUSICA; this uses `srun --jobid=<id> --overlap` from the login node and shares the job's resources).
    Typical: `nvidia-smi`, `top -b -n1 | head -30`, `free -g`, `ps -u $USER -o pid,pcpu,rss,etime,args --forest`,
    `$DATA/envs/tools/bin/py-spy dump --pid <pid>`, `tail` of a node-local log, `df -h /local`.
    node: one of the job's nodes (default: its first). timeout_s: 5 to 120. No background processes, no srun.
    The same rules as run_command apply; deleting files or signalling processes (kill, pkill, fuser -k) needs the
    human's approval in `approval`, in their words. The check goes by command word (also behind timeout, nice, env,
    xargs, bash -c); killing processes by any other means (e.g. os.kill in a python -c) needs that approval too."""
    job_id = _check_id(job_id)
    policy = node_exec_policy(command)
    if policy['refuse']:
        return {'error': policy['refuse']}
    if node and not NODE_NAME.match(node):
        return {'error': 'node must be a node name such as n3010-014'}
    if policy['needs_approval'] and not approval.strip():
        return {'error': 'needs the human\'s approval: ' + '; '.join(policy['needs_approval']),
                'note': 'ask the human and pass their words in `approval`'}
    job, why = _running_job(job_id)
    if job is None:
        return {'error': why}
    nodes = S.expand_hostlist(job.get('NodeList', '')) if job.get('NodeList') not in (None, '', '(null)') else []
    if not nodes:
        return {'error': f'{job_id} has no nodes listed yet'}
    node = node or nodes[0]
    if node not in nodes:
        return {'error': f'{node} is not one of the job\'s nodes ({job.get("NodeList")})', 'nodes': nodes[:50]}
    t = min(max(int(timeout_s), 5), NODE_EXEC_MAX_S)
    if policy['needs_approval']:
        ledger.append({'event': 'node_exec', 'job_id': job_id, 'node': node, 'command': command, 'approval': approval,
                       'reasons': policy['needs_approval']})
    # the outer timeout also covers a step that waits for resources; the inner one ends the command itself
    cmd = (f'timeout {t + 30} srun --jobid={shlex.quote(job["JobId"])} --overlap -N1 -n1 --cpus-per-task=1 '
           f'--nodelist={shlex.quote(node)} timeout {t} bash -c {shlex.quote(command)}')
    r = remote.run(cmd, timeout=t + 45)
    res = {'job_id': job_id, 'node': node, **r.summary(12_000)}
    if r.code == 124:
        res['note'] = f'timed out after {t} s (the command or the step launch); use shorter forms (head, -n1)'
    elif r.code != 0 and re.search(r'srun: error|Unable to create step|Job/step already completing', r.err):
        res['note'] = 'srun could not start a step in the job (it may be ending); see stderr'
    return res


# sinfo %T: lowercase state names, combined forms ('idle+planned', 'mixed+drain') and suffix marks ('*' not responding,
# '~' powered off, '#' powering up, '!' '%' powering down, '$' maintenance reservation, '@' '^' reboot, '-' planned)
_STATE_MARKS = '*~#!%$@^-'
_OTHER_STATES = ('drain', 'drng', 'down', 'fail', 'futr', 'future', 'inval', 'unk', 'reboot', 'boot', 'npc', 'perfctr',
                 'block')


def _node_class(state: str) -> str:
    """allocated, idle (free for new jobs), planned (held free by the backfill scheduler for a job next in line) or
    other (down, drained, failing, rebooting, not responding, reserved, maintenance), from a sinfo %T state."""
    words, marks = [], set()
    for part in state.strip().lower().split('+'):
        w = part.rstrip(_STATE_MARKS)
        marks |= set(part[len(w):])
        words.append(w)
    if not words or not words[0] or any(w.startswith(_OTHER_STATES) for w in words):
        return 'other'
    if words[0] in ('allocated', 'alloc', 'mixed', 'mix', 'completing', 'comp'):
        return 'allocated'
    if {'planned', 'plnd'} & set(words) or '-' in marks:
        return 'planned'
    # powered-off ('~', 'powered_down') or powering ('#', '!', '%') idle nodes boot on demand: free; reserved,
    # maintenance, not responding or waiting for a reboot: not free for an ordinary job
    free = words[0] == 'idle' or words[0].startswith('pow')
    if free and not {'reserved', 'resv', 'maint'} & set(words) and not marks & set('*$@^'):
        return 'idle'
    return 'other'


@tool(RO)
def cluster_load() -> dict:
    """Nodes per partition: allocated, idle (free now), planned (held free by the scheduler for jobs next in line:
    not available to new jobs, except a short one backfilled before their planned start; plain sinfo %A/%F counts them
    as idle), other (down, drained, reserved, maintenance, not responding) and total, plus `states` (node count per raw
    Slurm state); pending and running jobs per partition; my fair-share and the priority factors of my pending jobs
    (low fair-share means long waits; idle GPU nodes can take CPU work at 4×)."""
    acct = f'-A {shlex.quote(config.account())} ' if config.account() else ''
    cmd = (f"sinfo -h -o '%R|%F'; echo {MARK}; squeue -h -t PD -o '%P' | sort | uniq -c; echo {MARK}; "
           f"squeue -h -t R -o '%P' | sort | uniq -c; echo {MARK}; "
           f"sshare -h -P {acct}-u $USER -o Account,User,NormShares,EffectvUsage,FairShare 2>/dev/null; "
           f"echo {MARK}; sprio -h -u $USER -o '%i|%Y|%A|%F|%J|%Q' 2>/dev/null; "
           f"echo {MARK}; sinfo -h -o '%R|%T|%D'")
    r = remote.run(cmd, timeout=60, retry=True)
    if not r.ok and not r.out:
        raise _fail('cluster_load', r)
    sec = r.out.split(MARK + '\n')
    sec += [''] * (6 - len(sec))

    def num(x):
        return int(x) if x.strip().isdigit() else x
    nodes = {}
    for l in sec[0].splitlines():
        if '|' in l:
            p, f = l.split('|', 1)
            a, i, o, t = (f.split('/') + ['?'] * 4)[:4]
            # Slurm's idle count includes planned nodes; without the state breakdown below planned is unknown
            nodes[p] = {'allocated': num(a), 'idle': num(i), 'planned': None, 'other': num(o), 'total': num(t)}
    by_state = {}
    for l in sec[5].splitlines():
        parts = l.split('|')
        if len(parts) == 3 and parts[2].strip().isdigit():
            st = by_state.setdefault(parts[0], {})
            st[parts[1].strip()] = st.get(parts[1].strip(), 0) + int(parts[2])
    for p, st in by_state.items():
        n = {'allocated': 0, 'idle': 0, 'planned': 0, 'other': 0}
        for state, k in st.items():
            n[_node_class(state)] += k
        nodes[p] = {**n, 'total': sum(st.values()), 'states': st}

    def counts(text):
        d = {}
        for l in text.splitlines():
            parts = l.split()
            if len(parts) == 2 and parts[0].isdigit():
                d[parts[1]] = int(parts[0])
        return d
    fair = [dict(zip(['account', 'user', 'norm_shares', 'effective_usage', 'fairshare'], l.split('|')))
            for l in sec[3].splitlines() if l.strip()]
    prio = [dict(zip(['job_id', 'priority', 'age', 'fairshare', 'jobsize', 'qos'], l.split('|')))
            for l in sec[4].splitlines() if l.strip()]
    res = {'nodes': nodes, 'pending_jobs': counts(sec[1]), 'running_jobs': counts(sec[2]), 'my_fairshare': fair,
           'my_pending_priority': prio}
    if any(n.get('planned') for n in nodes.values()):
        res['planned_note'] = ('planned nodes are held free by the backfill scheduler for higher-priority jobs next in '
                               'line: not free for new jobs; only a job whose --time ends before their planned start '
                               'can be backfilled onto them')
    return res


@tool(RO)
def budget(since: str = '2026-10-01') -> dict:
    """Tokens (credits) used by the whole project since a date, per user, from sacct (CPU nodes included; B200 as
    node-hours, no published price) and what is left of the configured budget. The projects portal is authoritative.
    Without a configured account: only the user's own jobs."""
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', since):
        return {'error': 'since must be YYYY-MM-DD'}
    awk = ("{s=0; if (match($4,/gres\\/np_[a-z0-9_]+=[0-9]+/)) {x=substr($4,RSTART,RLENGTH); sub(/.*=/,\"\",x); s=x/100} "
           "h=s*$3/3600; if ($2==\"zen4_0768_h100x4\") c[$1]+=4*h; else if ($2==\"zen4_0768\") c[$1]+=0.96*h; "
           "else b[$1]+=h} END {for (u in c) printf \"T %s %.2f\\n\", u, c[u]; for (u in b) printf \"B %s %.3f\\n\", u, b[u]}")
    who = f'-a -A {shlex.quote(config.account())}' if config.account() else '-u "$USER"'
    cmd = (f'set -o pipefail; sacct {who} -S {since} -X -n -P -o User,Partition,ElapsedRaw,AllocTRES | '
           f"awk -F'|' {shlex.quote(awk)}"
           # sacct shows some users as numeric uids; name them where the login node can
           ' | while read -r k u v; do n=$(getent passwd "$u" | cut -d: -f1); echo "$k ${n:-$u} $v"; done')
    r = remote.run(cmd, timeout=120, retry=True)
    if not r.ok:
        raise _fail('sacct (budget)', r)
    users, other = {}, {}
    for l in r.out.splitlines():
        parts = l.split()
        if len(parts) == 3:
            (users if parts[0] == 'T' else other)[parts[1]] = float(parts[2])
    total = round(sum(users.values()), 1)
    res = {'since': since, 'project': config.project() or None, 'account': config.account() or None,
           'tokens_by_user': users, 'tokens_total': total}
    if not config.account():
        res['account_note'] = 'only your own jobs: ' + config.account_error()
    b = config.budget()
    if b is not None:
        res.update(budget=b, remaining_approx=round(b - total, 1))
    else:
        res['note'] = f'no budget configured ([project] budget_tokens in {config.path()})'
    res['other_partitions_node_hours'] = other
    return res


# ---------------------------------------------------------------- jobs: submit, update, cancel

@contextmanager
def _submit_lock():
    """One submission at a time across all server processes, so jobs submitted together are counted together."""
    p = ledger.path().parent / 'submit.lock'
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _script_text_error(script: str) -> str | None:
    """An error when `script` (a path on the cluster) is the text of a job script instead; None for a path."""
    s = script.strip()
    if not ('\n' in s or s.startswith(('#!', '#SBATCH'))):
        return None
    first = s.splitlines()[0]
    first = first if len(first) <= 60 else first[:60] + '...'
    return (f'`script` is the path of a job script on the cluster, relative to cwd, not the script itself (got text '
            f'starting {first!r}). To check a draft, use check_job(script_text=...); to submit it, save it in the '
            'project, push it (push_files) and pass its path.')


def _prepare(script: str, cwd: str, sbatch_args: list[str], text: str | None = None,
             expected_minutes: float | None = None) -> dict:
    """Read the script on the cluster (or take `text`), the user's queue and the cluster time; apply the rules."""
    if text is None and (err := _script_text_error(script)):
        return {'error': err}               # before any SSH call: `cat -- <script text>` would echo it all back
    if (err := files.check_remote(cwd)):
        return {'error': err}
    if text is None:
        go, read = f'cd {remote.q(cwd)} || exit 3', f'cat -- {shlex.quote(script)} || exit 4; '
    else:   # a draft may be checked before its project is pushed: then from the home directory
        go, read = f'{{ cd {remote.q(cwd)} 2>/dev/null || {{ echo {NO_CWD}; cd; }}; }}', ''
    cmd = (f'{go}; pwd -P; echo {MARK}; {read}echo {MARK}; '
           f'squeue -u $USER -h -r -o {shlex.quote(SQ_FMT)}; echo {MARK}; date +%Y-%m-%dT%H:%M:%S')
    r = remote.run(cmd, timeout=60, retry=True)
    if r.code == 3:
        return {'error': f'cannot cd to {cwd} on the cluster: {r.err.strip()}'}
    if r.code == 4:
        return {'error': f'cannot read {script} in {cwd}: {r.err.strip()}'}
    if not r.ok:
        raise _fail('reading the job script', r)
    real, read, queue, now = (r.out.split(MARK + '\n') + ['', '', ''])[:4]
    text = read if text is None else text
    missing = NO_CWD in real
    real = real.replace(NO_CWD, '').strip()
    in_use = _in_use(_parse_squeue(queue))
    res = guard.evaluate(text, sbatch_args, in_use, expected_minutes=expected_minutes or None)
    res.update({'workdir': real.strip(), 'in_use': in_use, 'cluster_time': now.strip(), 'text': text})
    if missing:
        res['warnings'].append(f'{cwd} does not exist on the cluster yet: checked from the home directory; push the '
                               'project before submitting')
    elif res['workdir'].startswith('/home/'):
        res['warnings'].append('the job runs from $HOME; ASC asks to run jobs from $DATA (a directory there, or a symlink to one in ~)')
    dirs = set()
    for k in ('output', 'error'):
        v = res['options'].get(k)
        if v and '/' in v:
            d = v.rsplit('/', 1)[0]
            if '%' not in d:
                dirs.add(d)
    res['log_dirs'] = sorted(dirs)
    return res


PROBE = '__amadeus_probe__'
MAX_PROBE = 40


def _plain(path: str) -> bool:
    """A path remote.q can pass safely: no variables or substitutions beyond a leading ~ / $DATA-like prefix."""
    return not re.search(r'[$`\n]', re.sub(r'^(~|\$\{?[A-Z_]+\}?)', '', path))


def _asset_probe(prep: dict, refuse: bool = False) -> tuple[str, list[str], list[str]]:
    """Shell lines that test on the cluster the paths the job reads (models, containers, envs, inputs: findings 29,
    63) and, for vLLM on a dev QoS, whether one of its compile caches exists; prefixed to a command that runs anyway.
    refuse=True: exit 5 when a path is missing (before sbatch). -> (shell, paths, caches)."""
    text, opts = prep.get('text') or '', prep['options']
    paths = [p for p in guard.referenced_paths(text) if _plain(p)][:MAX_PROBE]
    caches = []
    if str(opts.get('qos', '')).startswith('dev_') and re.search(r'\bvllm\s+serve\b', text):
        caches = [c for c in guard.vllm_cache_dirs(text, opts.get('job-name')) if _plain(c)][:MAX_PROBE]
    tests = [f'[ -e {remote.q(p)} ] || {{ echo "{PROBE} m {i}"; m=1; }}' for i, p in enumerate(paths)]
    tests += [f'[ -d {remote.q(c)} ] && echo "{PROBE} c {i}"' for i, c in enumerate(caches)]
    if not tests:
        return '', paths, caches
    return 'm=0; ' + '; '.join(tests) + ('; [ "$m" = 0 ] || exit 5; ' if refuse else '; '), paths, caches


def _asset_result(out: str, paths: list[str], caches: list[str]) -> tuple[list[str], bool]:
    """(missing paths, cold: a vLLM dev job none of whose compile caches exists) from a probe's output."""
    missing, warm = [], False
    for l in out.splitlines():
        v = l.split(' ')
        if len(v) == 3 and v[0] == PROBE and v[2].isdigit():
            if v[1] == 'm' and int(v[2]) < len(paths):
                missing.append(paths[int(v[2])])
            elif v[1] == 'c':
                warm = True
    return missing, bool(caches) and not warm


def _missing_msg(missing: list[str]) -> str:
    return (f'the job reads {", ".join(missing[:10])}, which do{"es" if len(missing) == 1 else ""} not exist on the '
            'cluster (a typo, or not fetched yet; a job died in seconds this way): fix the path or fetch it first '
            'in a job' + ('; what the project has installed: amadeus://project' if config.notes() else ''))


def _sbatch_cmd(cwd: str, script: str, prep: dict, sbatch_args: list[str], script_args: list[str], *,
                test_only: bool = False, hold: bool = False) -> str:
    mk = ''
    if prep['log_dirs'] and not test_only:
        mk = f'mkdir -p -- {" ".join(shlex.quote(d) for d in prep["log_dirs"])} && '
    flags = ['--test-only'] if test_only else ['--parsable'] + (['--hold'] if hold else [])
    argv = ['sbatch', *flags, *prep['inject'], *sbatch_args, script, *script_args]
    return f'cd {remote.q(cwd)} && {mk}{" ".join(shlex.quote(a) for a in argv)}'


def _public(prep: dict) -> dict:
    keep = ('errors', 'warnings', 'needs_approval', 'inject', 'estimate', 'workdir', 'in_use')
    out = {k: prep[k] for k in keep if k in prep}
    out['options'] = {k: v for k, v in prep['options'].items()
                      if k in ('job-name', 'partition', 'qos', 'time', 'nodes', 'gres', 'gpus', 'array', 'exclude',
                               'account', 'cpus-per-task', 'ntasks', 'ntasks-per-node', 'output', 'error',
                               'dependency', 'nodelist')}
    return out


def _scontrol_job(job_id: str) -> dict | None:
    r = remote.run(f'scontrol show job -o {shlex.quote(job_id)}', timeout=60, retry=True)
    if not r.ok:
        return None
    for line in r.out.splitlines():
        if not line.strip():
            continue
        job = S.parse_scontrol(line)
        if job.get('JobId') == job_id or job.get('ArrayJobId') == job_id:
            return job
    return None


def _find_uncertain(name: str, since: str, wait_s: int = 60) -> list[dict]:
    """After an sbatch whose outcome is unknown: jobs with this name submitted since `since` (cluster time) that the
    ledger does not know, polled for up to wait_s (a fresh job can take a moment to show)."""
    known = set(ledger.submitted())
    t0 = time.monotonic()
    while True:
        r = remote.run(f'sacct -u $USER -X -n -P --name={shlex.quote(name)} -S now-30minutes '
                       f'-o JobIDRaw,JobID,State,Submit', timeout=45, retry=True)
        found = []
        if r.ok:
            for l in r.out.splitlines():
                v = l.split('|')
                if len(v) == 4 and v[3] >= since and v[0] not in known and '_' not in v[1]:
                    found.append({'job_id': v[0], 'state': v[2], 'submitted': v[3]})
        if found or time.monotonic() - t0 >= wait_s:
            return found
        time.sleep(10)


@tool(RO)
def check_job(script: str = '', cwd: str = '', sbatch_args: list[str] | None = None,
              script_args: list[str] | None = None, script_text: str = '', expected_minutes: float = 0) -> dict:
    """Check a job before submitting: runbook rules (account, partition/QoS, time, dev limits, CPU limits, script
    pitfalls), cost estimate at the time limit, whether a human must approve it (counting your running and pending
    jobs), and `sbatch --test-only`. `script` is a path relative to `cwd` on the cluster (default: the configured
    workdir), never the script's text (refused without touching the cluster); or pass the script itself as `script_text` (e.g. a draft_job result) to check it without pushing it first (`cwd` is still
    where it would run). Pass each sbatch option and its value as separate list items. The test-only start time is
    unreliable. Also tested on the cluster: that the models, containers, envs and inputs the script reads exist
    (`missing_paths`; submit_job refuses them unless the job has a --dependency), and for vLLM on the dev QoS whether
    a compile cache is warm. `asc`: what ASC's submit filter changed in the request (comfort mode).
    expected_minutes: your runtime estimate; a --time far above it is flagged."""
    sbatch_args, script_args = sbatch_args or [], script_args or []
    if (err := config.account_error()):
        return {'error': err}
    cwd = _cwd(cwd)
    if bool(script.strip()) == bool(script_text.strip()):
        return {'error': 'pass either script (a path on the cluster) or script_text (the script itself)'}
    if len(script_text) > MAX_SCRIPT_TEXT:
        return {'error': f'script_text is over {MAX_SCRIPT_TEXT // 1000} kB'}
    text = script_text if script_text.strip() else None
    prep = _prepare(script, cwd, sbatch_args, text, expected_minutes)
    if 'error' in prep:
        return prep
    out = _public(prep)
    probe, paths, caches = _asset_probe(prep)
    if prep['errors']:
        out['ready'] = False
        if probe:
            _assets_into(out, remote.run(probe + 'true', timeout=60, retry=True).out, paths, caches, prep)
        return out
    if text is None:
        cmd, stdin = _sbatch_cmd(cwd, script, prep, sbatch_args, script_args, test_only=True), None
    else:   # the text goes to a temporary file on the login node, which is removed again
        argv = ['sbatch', '--test-only', *prep['inject'], *sbatch_args]
        cmd = (f'{{ cd {remote.q(cwd)} 2>/dev/null || cd; }} && f=$(mktemp "${{TMPDIR:-/tmp}}/amadeus-XXXXXX.slurm") && cat > "$f" && '
               f'{{ {" ".join(shlex.quote(a) for a in argv)} "$f" {" ".join(shlex.quote(a) for a in script_args)}; '
               f'rc=$?; rm -f "$f"; exit $rc; }}')
        stdin = text.encode()
    r = remote.run(probe + cmd, timeout=90, retry=True, stdin=stdin)
    refused = _assets_into(out, r.out, paths, caches, prep)
    lines = [l for l in (r.out + r.err).splitlines()
             if 'DEBUG' not in l and 'vanilla' not in l and not l.startswith(PROBE)]
    out['test_only'] = {'exit_code': r.code, 'output': '\n'.join(lines[-15:])}
    asc = guard.asc_rewrites(r.out + r.err, prep['options'])
    if asc['mode'] or asc['changed']:
        out['asc'] = {k: asc[k] for k in ('mode', 'changed', 'exclusive') if asc[k]}
    out['ready'] = r.ok and not refused
    return out


def _assets_into(out: dict, probe_out: str, paths: list[str], caches: list[str], prep: dict) -> bool:
    """Add the probe's findings to a check result's warnings; True when submit_job would refuse the job."""
    missing, cold = _asset_result(probe_out, paths, caches)
    warnings = out.setdefault('warnings', [])
    if cold and guard.COLD_START not in warnings:
        warnings.append(guard.COLD_START)
    if not missing:
        return False
    out['missing_paths'] = missing
    dep = bool(prep['options'].get('dependency'))
    warnings.append(_missing_msg(missing) + ('. The job has a --dependency: fine if an earlier job creates them' if dep
                                             else '. allow_missing: fine if the job creates them before use'
                                             if prep.get('allow_missing') else
                                             '. submit_job refuses this (allow_missing=True if the job fetches or '
                                             'creates them itself)'))
    return not dep and not prep.get('allow_missing')


@tool(ToolAnnotations(read_only_hint=True, open_world_hint=False))
def draft_job(kind: str, name: str, command: str = '', time: str = '00:10:00', dev: bool = True, nodes: int = 1,
              gpus: int = 1, cores: int = 1, mem_gb: int = 0, array: str = '', description: str = '',
              python_env: str = '', container: str = '', modules: list[str] | None = None, vllm_model: str = '',
              vllm_args: str = '', vllm_image: str = '', vllm_servers_per_node: int = 1,
              stage_model: bool = True, runs: list[str] | None = None, run_minutes: list[float] | None = None,
              array_args: list[str] | None = None, array_throttle: int = 0, vllm_preset: str = '',
              load_cap_s: int = 420) -> dict:
    """Draft a Slurm job script for MUSICA from the runbook's tested patterns; nothing is sent to the cluster.
    kind: cpu (part of a CPU node: `cores`, `mem_gb`), cpu-node (whole CPU nodes), gpu (1-3 GPUs on a shared node:
    `gpus`; shared nodes can take hours to start), gpu-node (whole H100 nodes; with vllm_model: vLLM servers on every
    node, `vllm_servers_per_node` 1/2/4 = TP4/TP2/TP1, and `command` as their client on the first node).
    command: the work (shell lines); with nodes > 1 it runs once on every node ($NODE_RANK, $NNODES, $HEAD).
    time: minutes or HH:MM:SS; dev=True uses the dev QoS (at most 10 min, 2 nodes): test there first.
    array: e.g. 0-9%2 ($TASK in the script). python_env: a venv path (e.g. $DATA/envs/train); container: a .sif
    (prefix commands with `run`); modules: Lmod modules to load.
    runs (gpu-node, 1 node): up to 8 commands in parallel on one whole node, run r on GPU r % 4 ($RUN, own caches,
    log logs/<name>-<job>-r<r>.log; the job fails if any run failed; `command` runs first as setup); run_minutes:
    one estimate per run (refused if the slowest exceeds `time`).
    array_args: one argument string per array task (sets --array=0-(n-1)%throttle; the task gets $ARGS and
    $OUT=results/<name>-<arrayjob>-a<task>); array_throttle: tasks at once (0 = within the approval limits); not
    together with `array`.
    vllm_preset: tested serve settings for some models, 'qwen38-fp8' or 'glm53-fp8' (flags for one TP4 server per
    node; model and image default to $DATA/models/<repo name> and $DATA/containers/<image>, which you must download
    first, or set vllm_model/vllm_image; vllm_args are appended, later flags win); load_cap_s: drop a vLLM server whose weights are not loaded this many seconds
    after its node's launch (60-3600).
    Drafts check at job start that the env, container and model exist (exit 2 otherwise), set HF_HUB_OFFLINE=1 on
    GPUs and OMP_NUM_THREADS=1 for vLLM. Returns the script, the static check (estimate, approvals needed) and notes;
    then check_job(script_text=...) for sbatch --test-only and the path checks on the cluster."""
    try:
        return drafts.draft(kind, name, command=command, time=time, dev=dev, nodes=nodes, gpus=gpus, cores=cores,
                            mem_gb=mem_gb, array=array, description=description, python_env=python_env,
                            container=container, modules=modules, vllm_model=vllm_model, vllm_args=vllm_args,
                            vllm_image=vllm_image, vllm_servers_per_node=vllm_servers_per_node,
                            stage_model=stage_model, runs=runs, run_minutes=run_minutes, array_args=array_args,
                            array_throttle=array_throttle, vllm_preset=vllm_preset, load_cap_s=load_cap_s)
    except drafts.DraftError as e:
        return {'error': str(e)}


@tool(ToolAnnotations(read_only_hint=True, open_world_hint=False))
def job_templates(name: str = '') -> dict:
    """Tested job scripts to copy from: Amadeus's jobs/ and the project's template directories (config).
    Without `name`: the list, each with a summary from its comments and its resources. With `name` (as listed, or
    the file name): the script and the helper scripts it uses. Prefer draft_job for a new job; use these for
    patterns it does not cover (vLLM multi-node benchmarks, DDP training, CPU-node arrays, container pulls)."""
    if name.strip():
        return templates.get(name)
    return {'templates': templates.index(),
            'note': 'job_templates(name) returns a script; adapt paths and the account-independent parts, then check_job'}


def _approval_note(gaps: list[str], approval: str) -> str:
    if not approval.strip():
        return 'needs the human\'s approval: ask them (with the estimate) and pass their words in `approval`'
    if any(g.startswith(guard.HEADS_UP) for g in gaps):
        return (f'the approval does not cover a job of {config.headsup_nodes()}+ nodes: it needs its own heads-up to the '
                'human (node count, maximum tokens, wall time) and their approval naming the node count')
    return 'the approval does not cover: ' + '; '.join(gaps)


def _verify_and_release(job_id: str, prep: dict, approval: str, keep_held: bool) -> dict:
    """The authoritative check: Slurm's own reading of the held job against the approval rules, then release."""
    job = _scontrol_job(job_id)
    if job is None:
        c = remote.run(f'scancel {job_id}', timeout=60)
        state = 'cancelled' if c.ok else 'still held (cancel failed: cancel it with cancel_job)'
        ledger.append({'event': 'cancel', 'job_id': job_id, 'reason': 'could not verify the held job'} if c.ok else
                      {'event': 'note', 'job_id': job_id, 'note': 'held, unverified, cancel failed'})
        return {'submitted': False, 'job_id': job_id, 'note': f'the job could not be read back for checking; {state}'}
    check = guard.check_actual(job, prep['in_use'])
    if check['errors'] or guard.approval_gaps(check['needs_approval'], approval):
        c = remote.run(f'scancel {job_id}', timeout=60)
        if c.ok:
            ledger.append({'event': 'cancel', 'job_id': job_id, 'reason': 'Slurm read the job differently: '
                           + '; '.join(check['errors'] + check['needs_approval'])})
        return {'submitted': False, 'job_id': job_id, 'slurm_view': check,
                'note': ('cancelled: ' if c.ok else 'NOT cancelled (it stays held; use cancel_job): ')
                + 'Slurm reads this job differently from its script and options, and as Slurm reads it '
                + ('it breaks the rules' if check['errors'] else 'it needs the human\'s approval')}
    if keep_held:
        return {'submitted': True, 'job_id': job_id, 'held': True, 'slurm_view': check['estimate']}
    rel = remote.run(f'scontrol release {job_id}', timeout=60, retry=True)
    if not rel.ok:
        return {'submitted': True, 'job_id': job_id, 'held': True, 'slurm_view': check['estimate'],
                'note': f'checked, but the release failed ({rel.err.strip()}): run update_job(job_id, release=True)'}
    return {'submitted': True, 'job_id': job_id, 'held': False, 'slurm_view': check['estimate']}


@tool(RW)
def submit_job(script: str, purpose: str, cwd: str = '', sbatch_args: list[str] | None = None,
               script_args: list[str] | None = None, approval: str = '', expected_minutes: float = 0,
               allow_missing: bool = False) -> dict:
    """Submit a batch job after the runbook checks (see check_job) and record it in the local ledger.
    `script`: the path of the job script on the cluster, relative to `cwd` (default: the configured workdir); script
    text is refused (check a draft with check_job(script_text=...), then save it in the project, push_files, submit).
    The job is submitted on hold, checked against Slurm's own reading of it, and released (or cancelled if Slurm
    reads it differently and it then breaks the rules). `purpose`: one line on what the job is for. `approval`: the
    human's explicit approval in their words (who, what, limit); required above the approval limits ([limits] in the
    config; by default ≥ 4 nodes at once, > 20 tokens including your other jobs), for B200 and long/priority QoS; a job
    of 16 or more nodes (headsup_nodes) needs an approval that names its node count, even inside an approved budget. Refused when a model, container, env or input the script reads does
    not exist on the cluster (unless the job has a --dependency, or allow_missing=True for a job that fetches or
    creates them itself before use: then a warning). Pass each sbatch option and its value as separate items.
    expected_minutes: your runtime estimate (a --time far above it is flagged)."""
    sbatch_args, script_args = sbatch_args or [], script_args or []
    if (err := config.account_error()):
        return {'error': err}
    cwd = _cwd(cwd)
    if not purpose.strip():
        return {'error': 'purpose is required (one line: what the job is for)'}
    with _submit_lock():
        prep = _prepare(script, cwd, sbatch_args, expected_minutes=expected_minutes)
        if 'error' in prep:
            return prep
        out = _public(prep)
        if prep['errors']:
            return {'submitted': False, **out, 'note': 'fix the errors first'}
        if (gaps := guard.approval_gaps(prep['needs_approval'], approval)):
            return {'submitted': False, **out, 'not_covered': gaps, 'note': _approval_note(gaps, approval)}
        keep_held = 'hold' in prep['options']
        prep['allow_missing'] = bool(allow_missing)
        probe, paths, caches = _asset_probe(prep, refuse=not prep['options'].get('dependency') and not allow_missing)
        cmd = probe + _sbatch_cmd(cwd, script, prep, sbatch_args, script_args, hold=True)
        r = remote.run(cmd, timeout=120, retry=False)
        m = re.search(r'^(\d+)(?:;\S+)?\s*$', r.out, re.M)
        if _assets_into(out, r.out, paths, caches, prep) and r.code == 5 and not m:
            out['warnings'] = [w for w in out['warnings'] if not w.startswith('the job reads ')]
            return {'submitted': False, **out, 'errors': out['errors'] + [_missing_msg(out['missing_paths'])],
                    'note': 'nothing was submitted: fix or fetch the missing paths first'}
        entry = {'event': 'submit', 'purpose': purpose, 'cwd': cwd, 'workdir': prep['workdir'], 'script': script,
                 'sbatch_args': prep['inject'] + sbatch_args, 'script_args': script_args, 'estimate': prep['estimate'],
                 'approval': approval or None, 'needs_approval': prep['needs_approval'] or None, 'epoch': time.time()}
        if not m:
            res = {'submitted': False, **out, 'sbatch': r.summary(4000)}
            if r.ssh_failed or r.timed_out or UNCERTAIN.search(r.err):
                # the job may exist anyway (AGENTS.md rule 8): look before anything is resubmitted
                name = prep['options'].get('job-name') or os.path.basename(script)
                found = _find_uncertain(name, prep['cluster_time'])
                if len(found) == 1:
                    job_id = found[0]['job_id']
                    ledger.append({**entry, 'job_id': job_id, 'adopted': 'found after an uncertain sbatch'})
                    v = _verify_and_release(job_id, prep, approval, keep_held)
                    return {**out, **v, 'note': 'sbatch\'s answer was lost, but the job was found and adopted. '
                            + v.get('note', '')}
                res['uncertain'] = {'jobs_found': found,
                                    'note': ('several jobs with this name appeared: check them with job_details before '
                                             'anything else' if found else
                                             'no new job with this name appeared within 60 s: it is safe to resubmit')}
            return res
        job_id = m.group(1)
        ledger.append({**entry, 'job_id': job_id})
        v = _verify_and_release(job_id, prep, approval, keep_held)
    notes = [l for l in r.err.splitlines() if l.strip() and 'DEBUG' not in l and 'vanilla' not in l]
    asc = guard.asc_rewrites(r.out + r.err, prep['options'])
    if asc['changed']:
        out['asc'] = {k: asc[k] for k in ('mode', 'changed', 'exclusive') if asc[k]}
    res = {**out, **v, 'sbatch_messages': notes[-10:]}
    if v.get('submitted'):
        res['next'] = f'wait_for_job("{job_id}") or job_details("{job_id}")'
        if (limit := (prep.get('estimate') or {}).get('time_limit_min')):
            res.update(_outlasting({job_id: time.time() + float(limit) * 60}))   # the earliest it can end at its limit
        if (limit := S.minutes(prep['options'].get('time'))) and 60 < limit < float('inf'):
            res['time_limit_note'] = ('final once the job starts: MUSICA denies raising a running job\'s limit (update_job '
                                      'can raise it only while pending); billing covers only the time used')
    return res


def _time_raise_refused(job: dict, time_limit: str) -> dict | None:
    """MUSICA lets users raise TimeLimit only while a job is pending ('Access/permission denied for job N'
    while running); refuse that before scontrol. Lowering stays allowed."""
    state = job.get('JobState', '')
    old, new = S.minutes(job.get('TimeLimit')), S.minutes(time_limit)
    if state not in STARTED or old is None or new is None or new <= old:
        return None
    return {'updated': False, 'state': state, 'time_limit': job.get('TimeLimit'),
            'error': f'{job.get("JobId")} is {state}: MUSICA lets users raise a time limit only while the job is pending '
                     '(running jobs get "Access/permission denied"); a longer limit for a running job needs ASC support '
                     'through the human',
            'options': ['let it run and stop the work before the limit (read EndTime from scontrol in the job), then '
                        'submit the rest as a new job that resumes from a checkpoint',
                        'cancel it now and resubmit with a longer limit (time used so far is billed and lost unless '
                        'the work checkpoints)',
                        'lowering the limit is allowed']}


@tool(DESTRUCTIVE)
def update_job(job_id: str, time_limit: str = '', nodes: str = '', release: bool = False, hold: bool = False,
               approval: str = '') -> dict:
    """Change a job: a new time limit (e.g. '02:00:00') or node count (pending jobs only), or release / hold it.
    A time limit can be raised only while the job is pending: MUSICA denies raising a running job's limit (lowering
    works). The new limits are re-checked like a submission (≥ 4 nodes at once, > 20 tokens need `approval`, the
    human's words). Jobs not submitted through this server need `approval` for any change."""
    job_id = _check_id(job_id)
    if not (time_limit or nodes or release or hold):
        return {'error': 'nothing to change: give time_limit, nodes, release or hold'}
    if time_limit and S.minutes(time_limit) is None:
        return {'error': f'cannot read time_limit {time_limit!r} (use e.g. 90, 01:30:00 or 1-00:00:00)'}
    if nodes and not re.match(r'^\d+(-\d+)?$', nodes):
        return {'error': 'nodes must look like 4 or 2-4'}
    ours = job_id.split('_')[0] in ledger.submitted()
    if not ours and not approval.strip():
        return {'updated': False, 'note': f'{job_id} was not submitted through this server: ask the human and pass '
                                          'their words in `approval`'}
    job = _scontrol_job(job_id)
    if job is None:
        return {'error': f'cannot read job {job_id} (finished, or the cluster is unreachable)'}
    if time_limit and (refused := _time_raise_refused(job, time_limit)):
        return refused
    after = dict(job)
    if time_limit:
        after['TimeLimit'] = time_limit
    if nodes:
        after['NumNodes'] = nodes
    rows = _squeue_rows(expand=True)
    in_use = _in_use([r for r in rows if r['job_id'].split('_')[0] != job_id.split('_')[0]])
    check = guard.check_actual(after, in_use)
    if check['errors']:
        return {'updated': False, 'errors': check['errors'], 'estimate': check['estimate']}
    if (time_limit or nodes) and (gaps := guard.approval_gaps(check['needs_approval'], approval)):
        return {'updated': False, 'needs_approval': check['needs_approval'], 'estimate': check['estimate'],
                'note': _approval_note(gaps, approval)}
    cmds = []
    fields = (f' TimeLimit={shlex.quote(time_limit)}' if time_limit else '') + (f' NumNodes={shlex.quote(nodes)}' if nodes else '')
    if fields:
        cmds.append(f'scontrol update JobId={job_id}{fields}')
    if hold:
        cmds.append(f'scontrol hold {job_id}')
    if release:
        cmds.append(f'scontrol release {job_id}')
    r = remote.run(' && '.join(cmds), timeout=60)
    if r.ok:
        ledger.append({'event': 'update', 'job_id': job_id, 'time_limit': time_limit or None, 'nodes': nodes or None,
                       'release': release or None, 'hold': hold or None, 'approval': approval or None,
                       'estimate': check['estimate']})
    return {'updated': r.ok, 'estimate': check['estimate'], **r.summary(3000)}


@tool(DESTRUCTIVE)
def cancel_job(job_id: str, reason: str, approval: str = '') -> dict:
    """Cancel a job. Jobs submitted through this server can be cancelled directly; any other job needs the human's
    approval in `approval` (AGENTS.md §9)."""
    job_id = _check_id(job_id)
    ours = job_id.split('_')[0] in ledger.submitted()
    if not ours and not approval.strip():
        return {'cancelled': False, 'note': f'{job_id} was not submitted through this server: ask the human and pass '
                                            'their words in `approval`'}
    r = remote.run(f'scancel {job_id}', timeout=60)
    if not r.ok:
        return {'cancelled': False, **r.summary(3000)}
    ledger.append({'event': 'cancel', 'job_id': job_id, 'reason': reason, 'approval': approval or None})
    s = remote.run(f'sleep 2; sacct -j {job_id} -X -n -o JobID,State', timeout=60, retry=True)
    return {'cancelled': True, 'state': s.out.strip() if s.ok else 'unknown (could not read sacct)'}


@tool(RO)
def ledger_entries(last: int = 20) -> dict:
    """The last submissions, updates, cancellations and approved commands made through this server
    (local ledger, ~/.local/state/amadeus/jobs.jsonl by default; AMADEUS_LEDGER overrides)."""
    return {'path': str(ledger.path()), 'entries': ledger.entries()[-max(1, min(last, 500)):]}


# ---------------------------------------------------------------- files

@tool(DESTRUCTIVE)
def push_files(local_dir: str, remote_dir: str, mode: str = 'tracked', exclude: list[str] | None = None,
               host: str | None = None) -> dict:
    """Copy a directory to the cluster (tar over SSH, retried; staged, so a failed push changes nothing; overwrites
    files of the same name, never deletes others). mode='tracked': git-tracked and untracked-but-not-ignored files plus
    a REVISION file, for code repositories; mode='all': everything except .git, venvs, caches and *.sif.
    `host`: the machine local_dir is on ('' = the machine the server runs on; default: the server's client machine,
    see the result's 'host'). local_dir must be inside the local root there (the home directory, outside hidden directories); remote_dir under $DATA, $SCRATCH or ~/
    (e.g. '$DATA/<repo>'). Code only: big data is fetched in jobs.
    Warns (does not refuse) when your pending or running jobs work in or around remote_dir: Slurm keeps only a copy
    of the batch script, so pending jobs run the pushed code and running ones may read changed files mid-run."""
    jobs = _jobs_using(remote_dir)
    res = files.push(local_dir, remote_dir, mode, exclude, host)
    warnings = []
    if isinstance(jobs, str):
        warnings.append(jobs)
    elif jobs:
        res['jobs_using_target'] = jobs[:20]
        states = {j['state'] for j in jobs}
        warnings.append(f'{len(jobs)} of your jobs work in or around {remote_dir} ('
                        + ', '.join(f'{n} {s.lower()}' for s, n in
                                    ((s, sum(j['state'] == s for j in jobs)) for s in sorted(states)))
                        + '): pending jobs will run the pushed code when they start, running ones may read changed '
                        'helpers, configs or REVISION mid-run. For long chains push to a per-revision directory')
    if str(res.get('revision', '')).endswith('-dirty'):
        warnings.append(f'revision {res["revision"]}: job logs will not identify the exact code; commit first if '
                        'your project allows')
    if mode == 'tracked' and 'error' not in res:
        ignored = _ignored_sources(local_dir, files.default_host() if host is None else host)
        if ignored:
            res['ignored_sources'] = ignored
            warnings.append('source files caught by .gitignore were not pushed (see ignored_sources): check '
                            '.gitignore; anchor patterns to the root, e.g. /worlds/')
    if warnings:
        res['warnings'] = res.get('warnings', []) + warnings
    return res


# squeue's WorkDir (%Z) and command (%o), resolved on the cluster; the target resolved the same way
_JOBS_USING = r"""realpath -m -- @PATH@; echo @MARK@
squeue -u "$USER" -h -o '%i|%T|%j|%Z|%o' | while IFS='|' read -r i t j z o; do
  printf '%s|%s|%s|%s|%s
' "$i" "$t" "$j" "$(realpath -m -- "$z" 2>/dev/null || echo "$z")" \
    "$(dirname -- "$(realpath -m -- "$o" 2>/dev/null || echo "$o")")"
done
"""


def _jobs_using(remote_dir: str) -> list[dict] | str:
    """The user's pending and running jobs whose WorkDir or script directory is at, under or above remote_dir.
    A string when the queue could not be read (a warning, the push goes ahead)."""
    norm, err = files.normalize_remote(remote_dir)
    if err:
        return []                                  # files.push reports the error
    cmd = _JOBS_USING.replace('@PATH@', remote.q(norm)).replace('@MARK@', MARK)
    r = remote.run(cmd, timeout=60, retry=True)
    target, _, rows = r.out.partition(MARK + '\n')
    target = target.strip().rstrip('/')
    if not r.ok or not target:
        return f'could not check the queue for jobs using {remote_dir}: {(r.err.strip() or "no output")[:200]}'

    def related(d: str) -> bool:
        d = d.rstrip('/')
        return bool(d) and d != '/' and (d == target or d.startswith(target + '/') or target.startswith(d + '/'))
    out = []
    for line in rows.splitlines():
        v = line.split('|')
        if len(v) != 5 or v[1] not in ('PENDING', 'RUNNING', 'CONFIGURING', 'SUSPENDED', 'REQUEUED', 'RESIZING'):
            continue
        if related(v[3]) or related(v[4]):
            out.append({'job_id': v[0], 'state': v[1], 'name': v[2], 'workdir': v[3]})
    return out


SOURCE_SUFFIXES = ('.py', '.sh', '.slurm', '.sbatch', '.yaml', '.yml', '.toml')
GENERATED = re.compile(r'(^|/)(\.?venv|__pycache__|node_modules|results|data|logs|outputs?|runs|wandb|checkpoints?|'
                       r'build|dist|\.?cache|\.[^/]*_cache|[^/]*\.egg-info)(/|$)')
_IGNORED = r"""git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0
git ls-files -z --others --ignored --exclude-standard 2>/dev/null | head -c 1000000
printf '\n@@TRACKED@@\n'
git ls-files -z 2>/dev/null | head -c 2000000
"""


def _ignored_sources(local_dir: str, host: str) -> list[str]:
    """Source files that .gitignore keeps out of a tracked push ('worlds/' also matches pkg/worlds/): ignored
    *.py/*.sh/*.slurm/... below a subdirectory that has tracked files, outside generated directories. At most 20;
    [] on any failure (this is only a hint)."""
    snippet = files._host_prelude(host) + files._fill(files._SOURCE, PATH=shlex.quote(local_dir)) + _IGNORED
    try:
        code, out, _, _ = files._run(files._host_argv(host, snippet), 60)
    except Exception:
        return []
    ignored, sep, tracked = out.partition(b'\n@@TRACKED@@\n')
    if code != 0 or not sep:
        return []
    dirs = set()                                 # every directory (not the root) above a tracked file
    for n in tracked.split(b'\0'):
        d = os.path.dirname(os.fsdecode(n))
        while d:
            dirs.add(d)
            d = os.path.dirname(d)
    found = []
    for n in ignored.split(b'\0'):
        name = os.fsdecode(n)
        if (name.endswith(SOURCE_SUFFIXES) and not GENERATED.search(name) and os.path.dirname(name)
                and any(name.startswith(d + '/') for d in dirs) and not files._excluded(name, files.SECRET_PATTERNS)):
            found.append(name)
    return sorted(found)[:20]


@tool(DESTRUCTIVE)
def fetch_files(remote_path: str, local_dir: str, max_mb: float = 500, host: str | None = None) -> dict:
    """Copy a file or directory from the cluster into local_dir on `host` (tar over SSH, retried; overwrites files of
    the same name). Refuses more than max_mb (fetch subdirectories of big results). `host` as in push_files."""
    return files.fetch(remote_path, local_dir, max_mb, host)


# ---------------------------------------------------------------- issues

@tool(RW)
def report_issue(title: str, what_happened: str, ctx: Context, expected: str = '', reproduce: str = '',
                 kind: str = 'bug', severity: str = 'medium', tool_name: str = '', project: str = '') -> dict:
    """Report a problem with this server or the cluster interface, for the maintainer session that fixes them.
    kind: bug | wrong-result | missing-feature | docs | cluster | other. severity: high (blocks work) | medium | low.
    Give the tool name, the arguments you used, what came back, what you expected, and your project (repository).
    No secrets. Returns the issue id; list_issues shows its status later."""
    return issues.report(title, what_happened, expected, reproduce, kind, severity, tool_name, project, _client(ctx))


@tool(RO)
def list_issues(status: str = 'open') -> dict:
    """Issues filed through report_issue or automatically. status: open | in-progress | needs-human | fixed |
    wontfix | duplicate | all."""
    return {'directory': str(issues.directory()), 'issues': issues.listing(status)}


# ---------------------------------------------------------------- resources

@mcp.resource('amadeus://runbook', name='runbook', mime_type='text/markdown',
              description='AGENTS.md: how agents operate MUSICA (rules, patterns, troubleshooting)')
def runbook() -> str:
    return (ROOT / 'AGENTS.md').read_text()


@mcp.resource('amadeus://notes', name='platform-notes', mime_type='text/markdown',
              description='NOTES.md: MUSICA hardware, Slurm, storage, software and access')
def notes() -> str:
    return (ROOT / 'NOTES.md').read_text()


@mcp.resource('amadeus://project', name='project-notes', mime_type='text/markdown',
              description="The configured project's notes ([project] notes in the config): budget approvals, what "
                          'is installed, status')
def project_notes() -> str:
    p = config.notes()
    if p is None:
        return f'No project notes configured: set [project] notes in {config.path()} to a Markdown file.'
    try:
        return p.read_text()
    except OSError as e:
        return f'The project notes {p} cannot be read: {e}'


# ---------------------------------------------------------------- entry point

def main() -> None:
    ap = argparse.ArgumentParser(description='Amadeus, the MCP server for the MUSICA cluster')
    ap.parse_args()
    sound.greet()
    transport.run_stdio(mcp)


if __name__ == '__main__':
    main()
