"""Commands on the MUSICA login node over SSH: the rules of scripts/mx in Python.

Batch mode, a hard timeout, optional Lmod init, retries only for idempotent commands on transient SSH failures.
A stall of the userspace tunnel before login (wireproxy mode) restarts the tunnel (auth.heal_tunnel). Every command
first prints START_MARK to stderr, so an exit 255 after it is known to come from a command that ran (its own, or a
connection lost mid-run), never from a failed login.
`AMADEUS_SSH` replaces the SSH command (tests use a local shim); otherwise the ssh config comes from network.py.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass

LMOD_INIT = ('source /cvmfs/software.eessi.io/versions/${EESSI_VERSION:-2025.06}/init/lmod/sh >/dev/null 2>&1; '
             'module use /cvmfs/software.asc.ac.at/init/modules;')
# SSH failures worth another try: the tunnel stalls now and then for 10 to 60 s (AGENTS.md §8). Besides the banner
# exchange, OpenSSH reports a stall as a keepalive timeout (ServerAliveInterval 15 x CountMax 4) or a remote close.
TRANSIENT = re.compile(r'banner exchange|timed out|Connection reset|Broken pipe|port 65535|Connection closed|'
                       r'kex_exchange_identification|ssh_exchange_identification|Connection refused|'
                       r'Timeout, server .* not responding|closed by remote host|client_loop: send disconnect', re.I)
MAX_OUT = 30_000
# Printed to stderr by the remote shell before the command; removed from Result.err. Seen = the command started.
START_MARK = '__amadeus_command_started__'


def ssh_base() -> list[str]:
    if (cmd := os.environ.get('AMADEUS_SSH')):
        return shlex.split(cmd)
    from . import network
    return ['ssh', '-o', 'BatchMode=yes'] + network.ssh_args() + ['musica']


def retry_delay() -> float:
    return float(os.environ.get('AMADEUS_RETRY_DELAY', '20'))


@dataclass
class Result:
    code: int
    out: str
    err: str
    seconds: float
    attempts: int = 1
    recovery: str = ''      # what was done about the connection (a tunnel restart), for the caller to pass on
    started: bool = False   # the remote shell printed START_MARK: the command ran (exit 255 is then not a login failure)

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def timed_out(self) -> bool:
        return self.code == 124

    @property
    def ssh_failed(self) -> bool:
        return self.code == 255

    def summary(self, limit: int = MAX_OUT) -> dict:
        d = {'exit_code': self.code, 'stdout': clip(self.out, limit), 'stderr': clip(self.err, limit // 3),
             'seconds': round(self.seconds, 1)}
        if self.attempts > 1:
            d['attempts'] = self.attempts
        if self.recovery:
            d['recovery'] = self.recovery
        if self.timed_out:
            d['note'] = 'timed out (raise timeout_s, or move the work into a job)'
        elif self.ssh_failed and self.started:
            d['note'] = ('exit 255 after the command started on the login node: its own exit code, or the connection '
                         'was lost mid-run')
        elif self.ssh_failed:
            d['note'] = 'SSH failed: check access_status (tunnel, certificate)'
        return d


def clip(text: str, limit: int = MAX_OUT) -> str:
    """Keep the tail of long output; that is where errors are."""
    if len(text) <= limit:
        return text
    return f'[... {len(text) - limit} characters cut ...]\n' + text[-limit:]


def _clean_err(err: str) -> tuple[str, bool]:
    """stderr without ssh's pseudo-terminal notice and START_MARK; and whether START_MARK was there."""
    lines = err.splitlines()
    kept = [l for l in lines if not l.startswith('Pseudo-terminal will not be allocated') and l.strip() != START_MARK]
    started = any(l.strip() == START_MARK for l in lines)
    return '\n'.join(kept), started


def _heal(res: Result) -> str:
    """Restart a stalled wireproxy tunnel (auth.heal_tunnel); '' when nothing was done. Never raises.

    Only for a failure before the command started: no START_MARK and nothing on stdout. An exit 255 of a command
    that ran (rsync or ssh from the login node failing, `exit 255`) must not restart a healthy tunnel."""
    if not res.ssh_failed or res.started or res.out.strip():
        return ''
    try:
        from . import auth   # auth imports this module
        return auth.heal_tunnel(res.err) or ''
    except Exception:
        return ''


def run(cmd: str, timeout: float = 120, lmod: bool = False, stdin: bytes | None = None, retry: bool = False) -> Result:
    """Run `cmd` in bash on the login node. retry=True only for commands that are safe to repeat: up to 3 attempts in
    all, again after a transient SSH failure (exit 255), and once after a timeout (exit 124).

    An SSH failure before login in wireproxy mode (a tunnel stall: no START_MARK, nothing on stdout, a stall message
    only ssh's own connection setup prints) restarts the tunnel, at most once a minute across
    calls; a retry=True command then runs again at once (one more attempt even after the last), a retry=False one is
    not repeated. Either way `recovery` says so."""
    if lmod:
        cmd = f'{LMOD_INIT} {cmd}'
    cmd = f'echo {START_MARK} >&2; {cmd}'
    attempts = 3 if retry else 1
    timeout_retries = 1
    recovery = ''
    t0 = time.monotonic()
    res = Result(255, '', '', 0)
    i = -1
    while i + 1 < attempts:
        i += 1
        try:
            p = subprocess.run(ssh_base() + [cmd], input=stdin, capture_output=True, timeout=timeout)
            err, started = _clean_err(p.stderr.decode(errors='replace'))
            res = Result(p.returncode, p.stdout.decode(errors='replace'), err, time.monotonic() - t0, i + 1,
                         started=started)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b'').decode(errors='replace')
            err, started = _clean_err((e.stderr or b'').decode(errors='replace'))
            res = Result(124, out, err, time.monotonic() - t0, i + 1, started=started)
        if not recovery and (recovery := _heal(res)):
            if not retry:
                recovery += '; the command was not repeated (nothing ran remotely): run it again'
                break
            recovery += '; the command was run again'
            attempts = max(attempts, i + 2)   # one attempt through the new tunnel, at once
            continue
        if i == attempts - 1:
            break  # no sleep after the last attempt
        if res.ssh_failed and TRANSIENT.search(res.err):
            pass
        elif res.timed_out and timeout_retries:
            timeout_retries -= 1
        else:
            break
        time.sleep(retry_delay())
    res.recovery = recovery
    return res


def q(s: str) -> str:
    """Quote for the remote shell, but let a leading ~/ or $VAR expand there."""
    m = re.match(r'^(~|\$[A-Z_]+|\$\{[A-Z_]+\})(?:/(.*))?$', s)
    if m:
        head, rest = m.group(1), m.group(2)
        # the slash stays unquoted: `~'/x'` would not expand the tilde
        return head + ('/' + shlex.quote(rest) if rest else ('/' if rest == '' else ''))
    return shlex.quote(s)
