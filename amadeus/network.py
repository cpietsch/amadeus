"""The way to MUSICA: which network path (system VPN or wireproxy) and which SSH settings, without touching ~/.ssh.

Network (`vpn_mode`): `MUSICA_VPN` or `[cluster] vpn` = system | wireproxy | auto (default). Auto: wireproxy when it
already runs; else the system's own path (the WireGuard app, a university network) when the login node answers on
port 22 directly; else wireproxy when ~/.config/wireguard/wireproxy.conf exists; else system (unreachable: the SSH
test says so). So a machine with the WireGuard app never starts a second tunnel with the same key (INSTALL.md §1).

SSH (`ssh_args`): Amadeus writes its own config, `$XDG_STATE_HOME/amadeus/ssh_config` (default ~/.local/state), from
`[cluster] user` and the network mode, with the ASC host CA from `asc_known_hosts` next to this file and a dedicated
ssh-agent socket beside it, and runs `ssh -F` on it. The user's ~/.ssh/config and known_hosts are not read. An
existing `musica` entry in ~/.ssh/config (installs from before) wins: then plain `ssh musica`, as before.
"""
from __future__ import annotations

import os
import pwd
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import config

HOSTS = {'musica': 'musica.vie.asc.ac.at', 'musica-vie': 'musica.vie.asc.ac.at',
         'musica-lnz': 'musica.lnz.asc.ac.at', 'musica-inn': 'musica.inn.asc.ac.at'}
PROBE = ('musica.vie.asc.ac.at', 22)
PROBE_TIMEOUT = 3.0
PROBE_TTL = 300          # a direct path that answered is trusted 5 min
PROBE_TTL_BAD = 60       # one that did not is tried again after a minute (the probe costs up to 3 s then)
KNOWN_HOSTS = Path(__file__).with_name('asc_known_hosts')
WIREPROXY_CONF = Path('~/.config/wireguard/wireproxy.conf')
SOCKS = '127.0.0.1:1080'

_probe: dict = {}        # {'at': monotonic, 'ok': bool}
_write_mu = threading.Lock()
_own: dict = {}          # {'at': monotonic, 'home': str, 'g': ssh -G output or None}


def state_dir() -> Path:
    return Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local' / 'state') / 'amadeus'


def config_path() -> Path:
    return state_dir() / 'ssh_config'


def agent_socket() -> Path:
    return state_dir() / 'agent.sock'


def user() -> str:
    """The ASC username: `[cluster] user` (AMADEUS_USER), else the User of an own `musica` entry in ~/.ssh/config."""
    if u := config.cluster_user():
        return u
    g = _own_entry()
    return _field(g, 'user') if g else ''


# ---------------------------------------------------------------- the user's own ~/.ssh/config entry

def _field(g: str, key: str) -> str:
    return next((l.split(' ', 1)[1] for l in g.splitlines() if l.startswith(key + ' ')), '')


def _own_entry() -> str | None:
    """`ssh -G musica` when ~/.ssh/config maps the name `musica` to a host (an install from before), else None."""
    home = str(Path.home())
    if _own and _own['home'] == home and time.monotonic() - _own['at'] < 60:
        return _own['g']
    try:
        real_home = pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        real_home = None
    # ssh takes ~ from the passwd entry, not $HOME: with another HOME (tests, sandboxes) name that HOME's file
    own = Path(home) / '.ssh' / 'config'
    cmd = ['ssh', '-G', 'musica'] if os.path.realpath(home) == os.path.realpath(real_home or '') else \
        (['ssh', '-F', str(own), '-G', 'musica'] if own.is_file() else None)
    try:
        g = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                           stdin=subprocess.DEVNULL).stdout if cmd else ''
    except (OSError, subprocess.TimeoutExpired):
        g = ''
    found = g if _field(g, 'hostname') not in ('', 'musica') else None
    _own.update(at=time.monotonic(), home=home, g=found)
    return found


def uses_own_config() -> bool:
    return _own_entry() is not None


def proxy_command() -> str:
    """The ProxyCommand in effect: ours (wireproxy mode) or the one in the user's own entry."""
    g = _own_entry()
    if g is not None:
        return '' if _field(g, 'proxycommand') == 'none' else _field(g, 'proxycommand')
    return f'nc -X 5 -x {SOCKS} %h %p' if vpn_mode() == 'wireproxy' else ''


# ---------------------------------------------------------------- network

def wireproxy_running() -> bool:
    return subprocess.run(['pgrep', '-f', 'wireproxy -c'], capture_output=True).returncode == 0


def direct_reachable() -> bool:
    """Does the login node answer on port 22 without wireproxy (system VPN or partner network)? Cached PROBE_TTL s."""
    if _probe and time.monotonic() - _probe['at'] < (PROBE_TTL if _probe['ok'] else PROBE_TTL_BAD):
        return _probe['ok']
    try:
        with socket.create_connection(PROBE, PROBE_TIMEOUT):
            ok = True
    except OSError:
        ok = False
    _probe.update(at=time.monotonic(), ok=ok)
    if ok:
        _remember_system()
    return ok


def _system_marker() -> Path:
    return state_dir() / 'system-path-seen'


def _remember_system() -> None:
    """This machine reached the login node without wireproxy once (a WireGuard app, a university network): from
    now on auto mode never chooses wireproxy here, so a stall or an app not yet connected never starts a second
    tunnel with the same key (the two would stall each other). `[cluster] vpn = "wireproxy"` overrides it."""
    try:
        state_dir().mkdir(parents=True, exist_ok=True)
        _system_marker().touch()
    except OSError:
        pass


def forget_probe() -> None:
    _probe.clear()


def vpn_mode() -> str:
    """'wireproxy' (userspace WireGuard started by scripts/vpn.sh) or 'system' (the operating system's VPN or network;
    not managed here)."""
    mode = (os.environ.get('MUSICA_VPN') or config.cluster_vpn()).strip().lower()
    if mode in ('wireproxy', 'system'):
        return mode
    own = _own_entry()
    if own is not None:   # the user's own entry decides: through the SOCKS proxy or not
        return 'wireproxy' if SOCKS in _field(own, 'proxycommand') else 'system'
    if direct_reachable() or _system_marker().exists():
        return 'system'
    if wireproxy_running() or WIREPROXY_CONF.expanduser().exists():
        return 'wireproxy'
    return 'system'


# ---------------------------------------------------------------- our ssh config

def render(mode: str, login: str) -> str:
    names = ' '.join(HOSTS)
    lines = ['# Written by Amadeus (amadeus/network.py) on every start; edits are overwritten.',
             '# Use it by hand: ssh -F <this file> musica', '']
    for alias, host in HOSTS.items():
        lines += [f'Host {alias}', f'    HostName {host}']
    lines += [f'Host {names}']
    if login:
        lines.append(f'    User {login}')
    lines += [f'    IdentityAgent "{agent_socket()}"',
              f'    UserKnownHostsFile "{KNOWN_HOSTS}"',
              '    GlobalKnownHostsFile /dev/null',
              '    StrictHostKeyChecking yes',
              '    BatchMode yes',
              '    ControlMaster no',
              '    ServerAliveInterval 15',
              '    ServerAliveCountMax 4',
              '    ConnectTimeout 20']
    if mode == 'wireproxy':
        lines.append(f'    ProxyCommand nc -X 5 -x {SOCKS} %h %p')
    return '\n'.join(lines) + '\n'


def ensure_config() -> Path:
    """Write our ssh config if it differs from what it should be now (atomic; mode 600)."""
    p = config_path()
    text = render(vpn_mode(), config.cluster_user())
    try:
        if p.read_text() == text:
            return p
    except OSError:
        pass
    with _write_mu:   # tools run in threads; other server processes write the same text through their own temp file
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix='.ssh_config.')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(text)
            os.chmod(tmp, 0o600)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return p


def ssh_args() -> list[str]:
    """Options before the host name: `-F <ours>`, or none when the user's own ~/.ssh/config has `musica`."""
    return [] if uses_own_config() else ['-F', str(ensure_config())]


def script_env() -> dict[str, str]:
    """For scripts/musica_auth.sh and vpn.sh: the network mode decided here and the ssh config to read."""
    env = dict(os.environ, MUSICA_VPN=vpn_mode(), MUSICA_ASC_USER=user())
    env['MUSICA_SSH_CONFIG'] = '' if uses_own_config() else str(ensure_config())
    return env


def setup_problem() -> str | None:
    """What is missing for SSH, in words, or None."""
    if not uses_own_config() and not config.cluster_user():
        return (f'no ASC username configured: set [cluster] user in {config.path()} (or AMADEUS_USER); see '
                'config.example.toml')
    return None


def main() -> None:
    """`python -m amadeus.network`: shell assignments for scripts/lib/access.sh (the scripts run by hand)."""
    vpn = vpn_mode()
    cfg = '' if uses_own_config() else str(ensure_config())
    print(f'MUSICA_VPN={shlex.quote(vpn)}; MUSICA_SSH_CONFIG={shlex.quote(cfg)}; '
          f'MUSICA_ASC_USER={shlex.quote(user())}')
    if problem := setup_problem():
        print(f'echo {shlex.quote(problem)} >&2', file=sys.stdout)


if __name__ == '__main__':
    main()
