"""Tunnel and certificate: wraps scripts/musica_auth.sh and scripts/vpn.sh (AGENTS.md §2–3).

`AMADEUS_AUTH_SCRIPT` replaces scripts/musica_auth.sh and `AMADEUS_STEP_DIR` replaces ~/.step (the directory
where musica_auth.sh lets step write asc-login.log / asc-renew.log; tests use a sandbox for both).
"""
from __future__ import annotations

import fcntl
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import network, remote

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r'https://auth\.asc\.ac\.at/device\?code=\d+')
DEAD_RE = re.compile(r'login failed|expired', re.I)
STEP_LOGS = {'--wait': ['asc-login.log'], '--renew': ['asc-renew.log']}
MTIME_SLACK = 2.0  # coarse file timestamps, and /proc start times count from a whole-second boot time
LOCK_WAIT = 150    # longer than certificate() (60 s) plus a spawn


def auth_script() -> Path:
    return Path(os.environ.get('AMADEUS_AUTH_SCRIPT') or ROOT / 'scripts' / 'musica_auth.sh')


def step_dir() -> Path:
    return Path(os.environ.get('AMADEUS_STEP_DIR') or Path.home() / '.step')


def _wrapper_log() -> Path:
    return step_dir() / 'amadeus-login.log'


def _pidfile() -> Path:
    return step_dir() / 'amadeus-login.pid'


vpn_mode = network.vpn_mode
tunnel_running = network.wireproxy_running


def tunnel() -> dict:
    if vpn_mode() == 'system':
        return {'mode': 'system', 'state': 'not managed here (the operating system\'s VPN or network); the SSH test '
                                           'shows whether the login node is reachable'}
    return {'mode': 'wireproxy', 'state': 'running' if tunnel_running() else 'stopped'}


def certificate(timeout: float = 60) -> dict:
    """Expiry and minutes left, without starting a login (musica_auth.sh --check). Refreshes the cached expiry."""
    try:
        p = subprocess.run([str(auth_script()), '--check'], capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL, env=network.script_env())
    except subprocess.TimeoutExpired:
        return _remember({'valid': False, 'check_failed': True, 'detail': 'musica_auth.sh --check timed out'})
    except OSError as e:
        return _remember({'valid': False, 'check_failed': True, 'detail': f'cannot run musica_auth.sh: {e}'})
    m = re.search(r'valid until (\S+) \((-?\d+) min left\)', p.stdout)
    if not m:
        return _remember({'valid': False, 'detail': (p.stdout + p.stderr).strip() or 'no certificate'})
    left = int(m.group(2))
    return _remember({'valid': left > 0, 'valid_until': with_offset(m.group(1)), 'minutes_left': left,
                      'renew_soon': left < WARN_MIN})


def with_offset(stamp: str) -> str:
    """ssh-keygen prints local time without a zone; add this process's UTC offset, which may differ from the
    human's (the dev container runs in UTC)."""
    try:
        t = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    return (t if t.tzinfo else t.astimezone()).isoformat()


# ---------------------------------------------------------------- cached expiry and warnings (findings 0, 9, 65)

WARN_MIN = 60            # musica_auth.sh --check fails below this, and the runbook asks for a renewal
CERT_TTL = 300           # tool results re-read the expiry at most every 5 min (no SSH, only the local agent)
CERT_TTL_BAD = 30        # without a valid certificate: soon again, so the warning goes once the human renewed
CERT_QUICK_S = 10        # a --check from the tool wrapper never holds a tool longer than this
CLUSTER_TZ = 'Europe/Vienna'   # Slurm, job logs and the human (CEST until 2026-10-25, then CET)
_cert_mu = threading.Lock()
_cert: dict = {}         # {'at': monotonic time of the read, 'cert': certificate(), 'expires': epoch or None}


def _remember(cert: dict) -> dict:
    # the expiry as the script counted it (minutes left), so a clock or zone mix-up cannot shift it
    exp = time.time() + cert['minutes_left'] * 60 if isinstance(cert.get('minutes_left'), int) else None
    _cert.clear()
    _cert.update(at=time.monotonic(), cert=cert, expires=exp)
    return cert


def forget_certificate() -> None:
    """The next cached read asks musica_auth.sh again (after an SSH auth failure or a login)."""
    _cert.clear()


def cached_certificate() -> dict | None:
    """certificate(), re-read at most every CERT_TTL s, with minutes_left counted from the cached expiry. None when
    unknown (the check failed or another thread is reading it). Never raises."""
    try:
        snap = dict(_cert)
        cert = snap.get('cert', {})
        # short while invalid or close to expiry: a renewal (also by another server process) shows within 30 s
        fine = cert.get('valid') and not cert.get('renew_soon') and (cert.get('minutes_left') or 0) >= WARN_MIN
        ttl = CERT_TTL if fine else CERT_TTL_BAD
        if not snap or time.monotonic() - snap['at'] > ttl:
            if _cert_mu.acquire(blocking=False):
                try:
                    certificate(CERT_QUICK_S)
                finally:
                    _cert_mu.release()
            snap = dict(_cert)
        cert = snap.get('cert')
        if not cert or cert.get('check_failed'):
            return None
        if snap.get('expires') is None:
            return dict(cert)
        left = round((snap['expires'] - time.time()) / 60)
        return {**cert, 'valid': snap['expires'] > time.time(), 'minutes_left': left, 'renew_soon': left < WARN_MIN,
                'expires_epoch': snap['expires']}
    except Exception:
        return None


def clock(epoch: float, with_date: bool | None = None) -> str:
    """Vienna time with its UTC offset ('21:48+02:00'); the date too when it is not today there."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(CLUSTER_TZ)
    except Exception:
        tz = None
    t = datetime.fromtimestamp(epoch, tz).astimezone(tz)
    hm = t.isoformat(timespec='minutes')
    if with_date is None:
        with_date = t.date() != datetime.now(t.tzinfo).date()
    return hm.replace('T', ' ') if with_date else hm[11:]


# AGENTS.md §3: a login only when the human is there; a cron, /loop or stop-hook turn does not show that
WHEN_HUMAN = ('if you are answering a message from the human (not a cron, /loop or stop-hook prompt), call start_login '
              'and put the link first in your reply; otherwise tell them when they next write, and start no code '
              '(AGENTS.md §3)')


def expiry_warning() -> str | None:
    """For every tool result: a line when under WARN_MIN minutes are left or the certificate is gone."""
    c = cached_certificate()
    if c is None:
        return None
    if not c.get('valid'):
        when = f' at {clock(c["expires_epoch"])}' if c.get('expires_epoch') else ''
        return (f'expired{when} or missing (running jobs are not affected; SSH, fetches and submissions are): '
                + WHEN_HUMAN)
    if c['minutes_left'] < WARN_MIN:
        return f'expires {clock(c["expires_epoch"])} ({c["minutes_left"]} min): ' + WHEN_HUMAN
    return None


def outlasting(ends: dict[str, float]) -> str | None:
    """A note when jobs end (epoch, at the latest by their time limit) after the certificate expires."""
    c = cached_certificate()
    if not c or not c.get('valid') or not c.get('expires_epoch'):
        return None
    exp = c['expires_epoch']
    late = sorted(((t, j) for j, t in ends.items() if t and t > exp))
    if not late:
        return None
    jobs = ', '.join(f'{j} (ends by {clock(t)})' for t, j in late[:5]) + (f' and {len(late) - 5} more'
                                                                           if len(late) > 5 else '')
    return (f'{jobs}: results land after the certificate expires at {clock(exp)}. Running jobs are not affected, '
            'but nothing can be checked, fetched or submitted until it is renewed: offer the human an early renewal '
            '(start_login) before they go offline')


AUTH_DENIED = re.compile(r'Permission denied \(publickey(?!,gssapi)')   # gssapi: compute-node ssh, not ours
PRE_AUTH = re.compile(r'banner exchange|port 65535|kex_exchange_identification|ssh_exchange_identification|'
                      r'Connection timed out|Connection refused', re.I)
# The part of PRE_AUTH only ssh's own connection setup through the proxy prints (the proxy shows as UNKNOWN port
# 65535): a stall that a tunnel restart clears. A bare 'Connection timed out/refused' can also come from a command
# that ran (rsync or ssh from the login node to another host), so it never restarts the tunnel.
STALL = re.compile(r'banner exchange|kex_exchange_identification|ssh_exchange_identification|UNKNOWN port 65535',
                   re.I)


def ssh_hint(err: str) -> str | None:
    """What a failed SSH connection (exit 255) means (finding 36): an expired certificate or a tunnel stall. The
    port-65535 timeout also shows while the certificate is gone, so the cached expiry decides."""
    if AUTH_DENIED.search(err or ''):
        forget_certificate()
        return 'certificate expired or missing (a link must be approved within about 60 s): ' + WHEN_HUMAN
    if PRE_AUTH.search(err or ''):
        network.forget_probe()            # the network path is decided again on the next call
        c = cached_certificate()
        if c is not None and not c.get('valid'):
            return 'certificate expired or missing (this timeout shows then too): ' + WHEN_HUMAN
        return ('tunnel stall or login-node hiccup before login (nothing ran remotely). With wireproxy, Amadeus '
                'restarts the tunnel itself in commands on the login node (at most once a minute; `recovery` says '
                'so) and repeats those safe to repeat; file transfers (push_files/fetch_files) do not restart it '
                'yet. Otherwise retry in a minute. If it persists, access_status shows the tunnel')
    return None


# ---------------------------------------------------------------- healing a stalled tunnel

RESTART_MIN_S = 60       # at most one tunnel restart a minute, across threads and server processes


def _restart_stamp() -> Path:
    return network.state_dir() / 'tunnel-restart'


def heal_tunnel(err: str) -> str | None:
    """After an SSH failure before login (STALL): restart wireproxy and say so, or None when nothing was done.

    A stalled userspace tunnel does not recover by waiting, a restart clears it at once. Only in wireproxy mode,
    only while the certificate is valid (an expired one shows the same timeout), and at most once per
    RESTART_MIN_S across threads and processes (a timestamp file in the state dir, under a lock), so parallel calls
    that all hit the stall restart it once. The caller makes sure the command did not start (remote._heal) and decides
    whether it may be repeated."""
    if not STALL.search(err or '') or vpn_mode() != 'wireproxy':
        return None
    c = cached_certificate()
    if not c or not c.get('valid'):
        return None
    d = network.state_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'tunnel-restart.lock', 'a+') as f:
        deadline = time.monotonic() + 60
        while True:   # a parallel restart takes a few seconds: wait for it, then see its timestamp
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return None
                time.sleep(0.1)
        try:
            try:
                last = float(_restart_stamp().read_text().split()[0])
            except (OSError, ValueError, IndexError):
                last = 0.0
            if 0 <= time.time() - last < RESTART_MIN_S:
                return None
            _restart_stamp().write_text(f'{time.time():.3f}\n')
            detail = _restart_tunnel()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    network.forget_probe()
    return f'tunnel restarted after a stall ({detail})'


def wireproxy_pids() -> list[int]:
    """This user's wireproxy processes running our config (`wireproxy -c <WIREPROXY_CONF>`), by argv, not pattern."""
    conf = os.path.realpath(network.WIREPROXY_CONF.expanduser())
    me, uid = os.getpid(), os.getuid()

    def ours(argv: list[str]) -> bool:
        return bool(argv) and os.path.basename(argv[0]) == 'wireproxy' and any(
            a == '-c' and os.path.realpath(os.path.expanduser(argv[i + 1])) == conf for i, a in enumerate(argv[:-1]))

    pids = []
    if not PROC:
        try:
            ps = subprocess.run(['ps', '-U', str(uid), '-o', 'pid=,args='], capture_output=True, text=True,
                                timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        for line in ps.splitlines():
            pid_s, _, args = line.strip().partition(' ')
            if pid_s.isdigit() and int(pid_s) != me and ours(args.split()):
                pids.append(int(pid_s))
        return pids
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            if os.stat(f'/proc/{d}').st_uid != uid:
                continue
        except OSError:
            continue
        if ours(_argv(int(d))) and _live_state(int(d)):
            pids.append(int(d))
    return pids


def _tunnel_up() -> bool:
    """scripts/vpn.sh up (next to the auth script): starts wireproxy unless it runs."""
    vpn = auth_script().parent / 'vpn.sh'
    if not vpn.exists():
        return False
    try:
        return subprocess.run([str(vpn), 'up'], capture_output=True, timeout=30, stdin=subprocess.DEVNULL,
                              env=network.script_env()).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _restart_tunnel() -> str:
    """Stop wireproxy by PID (TERM, KILL after 5 s) and start it again with vpn.sh up; what happened, in words."""
    pids = wireproxy_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 5
    while any(_live_state(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    for pid in pids:
        if _live_state(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    stopped = f'stopped pid {", ".join(map(str, pids))}' if pids else 'it was not running'
    return f'{stopped}; ' + ('started again' if _tunnel_up() else 'starting it again failed: see access_status')


# ---------------------------------------------------------------- which login runs

@dataclass
class _Login:
    pid: int
    spawn: float      # epoch seconds; only log content written after this belongs to the login
    mode: str | None  # '--wait', '--renew', or None (unknown: look at both step logs)
    ours: bool        # started by a amadeus server (pid file, wrapper log), else found running outside it


_mu = threading.Lock()
_proc: subprocess.Popen | None = None  # the login this process started; poll() reaps it
_proc_login: _Login | None = None


PROC = os.path.isdir('/proc/self')        # Linux; elsewhere (macOS) `ps` stands in


def _ps(pid: int, field: str) -> str:
    try:
        return subprocess.run(['ps', '-o', f'{field}=', '-p', str(pid)], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ''


def _stat_fields(pid: int) -> list[str] | None:
    if not PROC:
        state = _ps(pid, 'stat')
        return [state[:1]] if state else None
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()  # [0] is the state
    except (OSError, IndexError):
        return None


def _argv(pid: int) -> list[str]:
    if not PROC:
        return _ps(pid, 'args').split()
    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return []
    return [a.decode(errors='replace') for a in raw.split(b'\0') if a]


def _environ(pid: int) -> dict[str, str]:
    if not PROC:
        return {}              # not readable without /proc; same user is taken as same HOME
    try:
        raw = Path(f'/proc/{pid}/environ').read_bytes()
    except OSError:
        return {}
    pairs = (a.decode(errors='replace').partition('=') for a in raw.split(b'\0') if a)
    return {k: v for k, _, v in pairs}


def _start_epoch(pid: int) -> float | None:
    if not PROC:
        return None
    f = _stat_fields(pid)
    try:
        btime = next(int(l.split()[1]) for l in Path('/proc/stat').read_text().splitlines() if l.startswith('btime '))
        return btime + int(f[19]) / os.sysconf('SC_CLK_TCK')  # field 22: start time in clock ticks after boot
    except (OSError, ValueError, IndexError, TypeError, StopIteration):
        return None


def _live_state(pid: int) -> bool:
    """Exists and is not a zombie (os.kill(pid, 0) succeeds on zombies). A zombie child of ours is reaped."""
    if pid <= 0:  # waitpid(0) or waitpid(-1) would reap any child
        return False
    f = _stat_fields(pid)
    if not f:
        return False
    if f[0] in ('Z', 'X'):
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:  # ChildProcessError: not our child, its own parent reaps it
            pass
        return False
    return True


def _is_login_argv(argv: list[str]) -> bool:
    """`musica_auth.sh --renew|--wait` (not --check) or `step ssh login …`."""
    if not argv:
        return False
    if any(os.path.basename(a) == 'musica_auth.sh' for a in argv[:2]):
        return '--renew' in argv or '--wait' in argv
    return os.path.basename(argv[0]) == 'step' and argv[1:3] == ['ssh', 'login']


def _read_pidfile() -> _Login | None:
    try:
        parts = _pidfile().read_text().split()  # "<pid> <spawn epoch> <mode>"; older files hold only the pid
        pid = int(parts[0])
        spawn = float(parts[1]) if len(parts) > 1 else _pidfile().stat().st_mtime
    except (OSError, ValueError, IndexError):
        return None
    if pid <= 0:
        return None
    return _Login(pid, spawn, parts[2] if len(parts) > 2 else None, True)


def _own_login() -> _Login | None:
    """The login a amadeus server started, while it runs. A zombie or a reused pid does not count."""
    global _proc
    with _mu:
        if _proc is not None:
            if _proc.poll() is None:
                return _proc_login
            _proc = None
    rec = _read_pidfile()
    if rec and _live_state(rec.pid) and any('musica_auth.sh' in a for a in _argv(rec.pid)):
        return rec
    return None


def _foreign_login() -> _Login | None:
    """A login started outside the server for the same HOME: a human's `musica_auth.sh --renew` (AGENTS.md §3), or a
    `step ssh login` left by a plain musica_auth.sh. Starting a second one would replace its agent socket."""
    home, me, uid = os.environ.get('HOME', str(Path.home())), os.getpid(), os.getuid()
    if not PROC:
        try:
            ps = subprocess.run(['ps', '-U', str(uid), '-o', 'pid=,args='], capture_output=True, text=True,
                                timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        for line in ps.splitlines():
            pid_s, _, args = line.strip().partition(' ')
            argv = args.split()
            if pid_s.isdigit() and int(pid_s) != me and _is_login_argv(argv):
                mode = '--renew' if '--renew' in argv else '--wait' if '--wait' in argv else None
                return _Login(int(pid_s), 0.0, mode, False)
        return None
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) == me:
            continue
        pid = int(d)
        try:
            if os.stat(f'/proc/{d}').st_uid != uid:
                continue
        except OSError:
            continue
        argv = _argv(pid)
        if not _is_login_argv(argv) or not _live_state(pid) or _environ(pid).get('HOME', home) != home:
            continue
        mode = '--renew' if '--renew' in argv else '--wait' if '--wait' in argv else None
        return _Login(pid, _start_epoch(pid) or 0.0, mode, False)
    return None


def _running(login: _Login) -> bool:
    with _mu:
        p = _proc
    if p is not None and p.pid == login.pid:
        return p.poll() is None
    if not _live_state(login.pid):
        return False
    argv = _argv(login.pid)
    return any('musica_auth.sh' in a for a in argv) if login.ours else _is_login_argv(argv)


def login_running() -> int | None:
    login = _own_login() or _foreign_login()
    return login.pid if login else None


# ---------------------------------------------------------------- logs and links

def _fresh_texts(login: _Login) -> list[tuple[Path, float, str]]:
    """The login's wrapper log (if ours) and step logs, only if written since it started."""
    paths = ([_wrapper_log()] if login.ours else []) + \
        [step_dir() / n for n in STEP_LOGS.get(login.mode, STEP_LOGS['--wait'] + STEP_LOGS['--renew'])]
    out = []
    for p in paths:
        try:
            mtime = p.stat().st_mtime
            if mtime >= login.spawn - MTIME_SLACK:
                out.append((p, mtime, p.read_text(errors='replace')))
        except OSError:
            continue
    return out


def _live_link(text: str) -> str | None:
    """The last device link, unless the output says after it that the grant expired or the login failed."""
    links = list(LINK_RE.finditer(text))
    if not links or 'login failed' in text.lower() or DEAD_RE.search(text, links[-1].end()):
        return None
    return links[-1].group()


def _find_link(login: _Login) -> str | None:
    found = [(mtime, link) for _, mtime, text in _fresh_texts(login) if (link := _live_link(text))]
    return max(found)[1] if found else None


def _outcome(text: str) -> str | None:
    """How a finished login ended, from the wrapper's output."""
    low = text.lower()
    if 'expired' in low:
        return 'device code expired before it was approved'
    if 'login failed' in low:
        return 'login failed'
    m = re.search(r'^certificate valid until (\S+)', text, re.M)
    return f'certificate obtained (valid until {m.group(1)})' if m else None


def _log_tail(login: _Login) -> str:
    """Recent output with every device link masked: the only link to show is the live one in 'link'."""
    text = '\n'.join(f'[{p.name}]\n{t.strip()}' for p, _, t in _fresh_texts(login) if t.strip())
    return remote.clip(LINK_RE.sub('<device link>', text), 1500)


# ---------------------------------------------------------------- start

@contextmanager
def _login_lock(timeout: float = LOCK_WAIT):
    """Exclusive across threads and processes (several sessions, each with its own server), next to the pid file."""
    step_dir().mkdir(parents=True, exist_ok=True)
    with open(step_dir() / 'amadeus-login.lock', 'a+') as f:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(0.1)
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _previous_outcome() -> str | None:
    rec = _read_pidfile()
    if rec is None:
        return None
    try:
        text = _wrapper_log().read_text(errors='replace')
    except OSError:
        text = ''
    ago = max(0, round((time.time() - rec.spawn) / 60))
    return f'{_outcome(text) or "ended, outcome unknown"}; it started {ago} min ago'


def _spawn(mode: str) -> _Login:
    global _proc, _proc_login
    spawn = time.time()
    with _wrapper_log().open('w') as log:  # close_fds (default) keeps the lock fd out of the child
        p = subprocess.Popen([str(auth_script()), mode], stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, env=network.script_env())
    login = _Login(p.pid, spawn, mode, True)
    tmp = _pidfile().with_suffix('.tmp')
    tmp.write_text(f'{p.pid} {spawn:.3f} {mode}\n')
    os.replace(tmp, _pidfile())
    with _mu:
        _proc, _proc_login = p, login
    return login


INSTALL_HINT = {
    'step': 'the step CLI (smallstep): macOS `brew install step`; Linux: the release binary from '
            'github.com/smallstep/cli/releases into ~/.local/bin (no root needed)',
    'ssh': 'OpenSSH client (ssh, ssh-add, ssh-keygen)',
    'nc': 'OpenBSD netcat for the ProxyCommand (Debian/Ubuntu: apt install netcat-openbsd); macOS has it',
    'wireproxy': 'wireproxy from github.com/windtf/wireproxy/releases into ~/.local/bin',
    'pgrep': 'pgrep/pkill (Debian/Ubuntu: apt install procps)',
    'gnu-tools': 'GNU tar, find, stat, realpath for push_files/fetch_files (brew install coreutils findutils gnu-tar)',
}


def missing_tools() -> list[str]:
    """Local programs the access path needs and this process cannot find (INSTALL.md). A container rebuild can
    remove system packages (2026-10-15: step and nc were gone and logins failed with a buried message)."""
    need = ['step', 'ssh', 'ssh-add', 'ssh-keygen', 'pgrep']
    if re.match(r'^\S*\bnc(\s|$)', network.proxy_command()):
        need.append('nc')
    if vpn_mode() == 'wireproxy':
        need.append('wireproxy')
    missing = []
    for t in need:
        found = shutil.which(t) or (t == 'wireproxy' and (Path.home() / '.local/bin/wireproxy').exists())
        if not found:
            missing.append(t)
    if sys.platform == 'darwin' and not all(any(Path(p, t).exists() for p in (
            f'/opt/homebrew/opt/{pkg}/libexec/gnubin', f'/usr/local/opt/{pkg}/libexec/gnubin'))
            for pkg, t in (('coreutils', 'realpath'), ('findutils', 'find'), ('gnu-tar', 'tar'))):
        missing.append('gnu-tools')   # files.GNU_SHIM puts these first on PATH
    return missing


def _missing_note(missing: list[str]) -> str:
    hints = sorted({INSTALL_HINT['ssh' if t.startswith('ssh-') else t] for t in missing})
    return 'missing on this machine: ' + ', '.join(missing) + '. Install: ' + '; '.join(hints) + ' (INSTALL.md)'


def start_login(wait_s: float = 20) -> dict:
    """Start a login (or an early renewal) in the background and return the device link.

    One login at a time, across threads and server processes, and none while a human's musica_auth.sh login runs:
    while one runs, its link is returned instead of a new code (the link expires after about 60 s; AGENTS.md §3).
    A link is shown only while its login runs and only from output written since that login started."""
    out: dict = {}
    if 'step' in (missing := missing_tools()):
        return {'login': 'not started', 'missing_tools': missing, 'error': _missing_note(missing)}
    if problem := network.setup_problem():
        return {'login': 'not started', 'error': problem}
    with _login_lock() as locked:
        if not locked:
            return {'login': 'busy', 'detail': 'another start_login is still checking the certificate; call again in a '
                                               'minute (it will not start a second login)'}
        login = _own_login() or _foreign_login()
        if login:
            out['login'] = 'already running'
            if not login.ours:
                out['started_by'] = (f'outside this server (pid {login.pid}, e.g. a human running musica_auth.sh); '
                                     'no second login was started')
        else:
            previous = _previous_outcome()
            mode = '--renew' if certificate().get('valid') else '--wait'
            login = _spawn(mode)
            out['login'] = 'started'
            if previous:
                out['previous_login'] = previous
                if not previous.startswith('certificate obtained'):
                    out['note'] = f'The previous login is over ({previous}). A new login with a new code was started.'
    if login.mode:
        out['mode'] = ('early renewal (current certificate stays valid until approval)' if login.mode == '--renew'
                       else 'new login')
    out['started_s_ago'] = round(max(0.0, time.time() - login.spawn)) if login.spawn else None

    deadline = time.monotonic() + max(0.0, wait_s)
    while True:  # the lock is released: other callers get 'already running' and poll the same logs
        running = _running(login)
        link = _find_link(login) if running else None
        if link or not running or time.monotonic() >= deadline:
            break
        time.sleep(0.5)

    out['log_tail'] = _log_tail(login)
    if link:
        out['link'] = link
        out['instructions'] = ('Show this link to the human now: it must be approved in a browser (with the second '
                               'factor) within about 60 s. Then call access_status to confirm.')
    elif not running:
        text = _wrapper_log().read_text(errors='replace') if login.ours and _wrapper_log().exists() else ''
        result = _outcome(text) or 'the login process ended'
        out['result'] = result
        if result.startswith('certificate obtained'):
            forget_certificate()          # tool results drop the expiry warning at once
        out['next'] = ('call access_status to confirm' if result.startswith('certificate obtained') else
                       "no live link. Don't loop: when the human is at a browser, call start_login again for a new code")
    else:
        out['next'] = ('no device link yet; call start_login again in a few seconds (it returns this login\'s link and '
                       'does not start another login)')
    return out


def status(probe: bool = True) -> dict:
    if probe and vpn_mode() == 'wireproxy' and not tunnel_running():
        _tunnel_up()       # the login path starts it too (musica_auth.sh); a check should not need a login first
    out = {'tunnel': tunnel(), 'certificate': certificate()}
    out['ssh_config'] = ('~/.ssh/config (your own `musica` entry)' if network.uses_own_config()
                         else str(network.config_path()))
    if problem := network.setup_problem():
        out['fix_first'] = problem
    if missing := missing_tools():
        out['missing_tools'] = missing
        out['fix_first'] = '; '.join(filter(None, (out.get('fix_first'), _missing_note(missing))))
    login = _own_login() or _foreign_login()
    if login:
        out['login_in_progress'] = True if login.ours else 'started outside this server'
    if probe and out['certificate'].get('valid'):
        r = remote.run('hostname; date -Is', timeout=45, retry=True)
        out['ssh'] = 'ok: ' + r.out.strip().replace('\n', ', ') if r.ok else r.summary(2000)
        if r.recovery:
            out['tunnel']['recovery'] = r.recovery
    if not out['certificate'].get('valid'):
        out['next'] = 'certificate missing or expired: ' + WHEN_HUMAN
    elif out['certificate'].get('renew_soon'):
        out['next'] = 'certificate expires within the hour: offer the human an early renewal (start_login)'
    return out
