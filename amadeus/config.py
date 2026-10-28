"""The project this server works for: ~/.config/amadeus/config.toml (AMADEUS_CONFIG replaces the path).

The file names the ASC project (account, filesystem, budget, working directory, template directories), the approval
limits and the file tools' hosts; config.example.toml in the repository shows every key. Environment variables win
over the file: AMADEUS_ACCOUNT, AMADEUS_PROJECT, AMADEUS_FILESYSTEM, AMADEUS_BUDGET, AMADEUS_WORKDIR,
AMADEUS_TEMPLATE_DIRS (os.pathsep-separated), AMADEUS_USER, AMADEUS_LOCAL_ROOT, AMADEUS_HOSTS (comma-separated),
AMADEUS_DEFAULT_HOST; set but empty counts as unset, except for the two lists (empty there means none). The file is
read again when it changes, the environment on every call; only the server's instructions (project line, approval
limits) are built once, at start, so they change on a restart.

Without an account the server still answers read-only tools (budget then shows the user's own usage); check_job and
submit_job refuse (account_error()).
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULT_PATH = Path.home() / '.config' / 'amadeus' / 'config.toml'
DEFAULT_WORKDIR = '$DATA'          # ASC asks to run jobs from $DATA, not $HOME
DEFAULT_LOCAL_ROOT = '~'
# approval thresholds (AGENTS.md §0 rule 4, §9)
APPROVAL_NODES = 4
APPROVAL_TOKENS = 20.0
HEADS_UP_NODES = 16

_cache: dict = {'key': None, 'data': {}, 'error': None}


def path() -> Path:
    return Path(os.environ.get('AMADEUS_CONFIG') or DEFAULT_PATH).expanduser()


def _file() -> dict:
    """The parsed file ({} if there is none), cached until its path or modification time changes."""
    p = path()
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(p), None, None)
    if _cache['key'] != key:
        data, error = {}, None
        if key[1] is not None:
            try:
                data = tomllib.loads(p.read_text())
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
                error = f'{p}: {e}'
        _cache.update(key=key, data=data, error=error)
    return _cache['data']


def error() -> str | None:
    """Why the config file could not be read (None when it is fine or absent)."""
    _file()
    return _cache['error']


def _get(section: str, key: str, env: str | None = None, default=None):
    # an env variable that is set but empty counts as unset (wrappers often export X=${X:-})
    if env is not None and (v := os.environ.get(env, '').strip()):
        return v
    sec = _file().get(section)
    v = sec.get(key) if isinstance(sec, dict) else None
    return default if v is None else v


def _str(section: str, key: str, env: str | None = None, default: str = '') -> str:
    v = _get(section, key, env, default)
    return str(v).strip() if v is not None else default


def _num(section: str, key: str, default: float, env: str | None = None) -> float:
    v = _get(section, key, env, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _paths(v) -> list[str]:
    if isinstance(v, str):
        v = v.split(os.pathsep)
    return [str(p).strip() for p in (v or []) if str(p).strip()]


# ---------------------------------------------------------------- [project]

def project() -> str:
    """The ASC project name, shown to agents ('' when not configured)."""
    return _str('project', 'name', 'AMADEUS_PROJECT')


def account() -> str:
    """The Slurm account ('' when not configured: check_job and submit_job refuse then)."""
    return _str('project', 'account', 'AMADEUS_ACCOUNT')


def filesystem() -> str:
    """The project's filesystem name: /data/<filesystem>/<user>, /scratch/<filesystem>/<user>."""
    return _str('project', 'filesystem', 'AMADEUS_FILESYSTEM')


def budget() -> float | None:
    """The project's total budget in tokens (None when not configured)."""
    v = _get('project', 'budget_tokens', 'AMADEUS_BUDGET')
    try:
        return float(v) if v not in (None, '') else None
    except (TypeError, ValueError):
        return None


def ends() -> str:
    return _str('project', 'ends')


def workdir() -> str:
    """The default working directory on the cluster (check_job, submit_job)."""
    return _str('project', 'workdir', 'AMADEUS_WORKDIR') or DEFAULT_WORKDIR


def notes() -> Path | None:
    """The project notes file served as amadeus://project, if configured."""
    v = _str('project', 'notes')
    return Path(v).expanduser() if v else None


def template_dirs() -> list[Path]:
    """Extra directories of tested job scripts besides the repository's jobs/ (AMADEUS_TEMPLATE_DIRS replaces them)."""
    env = os.environ.get('AMADEUS_TEMPLATE_DIRS')
    v = env if env is not None else _get('project', 'template_dirs', default=[])
    return [Path(p).expanduser() for p in _paths(v)]


# ---------------------------------------------------------------- [cluster]

def cluster_user() -> str:
    """The ASC username Amadeus logs in as (its own ssh config, amadeus/network.py)."""
    return _str('cluster', 'user', 'AMADEUS_USER')


def cluster_vpn() -> str:
    """The network path: auto (default), system or wireproxy (MUSICA_VPN wins; amadeus/network.py)."""
    return _str('cluster', 'vpn', default='auto') or 'auto'


# ---------------------------------------------------------------- [limits]

def approval_nodes() -> int:
    return int(_num('limits', 'approval_nodes', APPROVAL_NODES))


def approval_tokens() -> float:
    return _num('limits', 'approval_tokens', APPROVAL_TOKENS)


def headsup_nodes() -> int:
    return int(_num('limits', 'headsup_nodes', HEADS_UP_NODES))


# ---------------------------------------------------------------- [files]

def local_root() -> str:
    """The local root of push_files/fetch_files on the machine the server runs on."""
    return _str('files', 'local_root', 'AMADEUS_LOCAL_ROOT') or DEFAULT_LOCAL_ROOT


def hosts() -> list[str]:
    """Extra machines for the file tools, by SSH name (unchecked; files.hosts() filters them)."""
    env = os.environ.get('AMADEUS_HOSTS')
    v = env.split(',') if env is not None else _get('files', 'hosts', default=[])
    if isinstance(v, str):
        v = v.split(',')
    return [str(h).strip() for h in v or [] if str(h).strip()]


def default_host() -> str:
    return _str('files', 'default_host', 'AMADEUS_DEFAULT_HOST')


# ---------------------------------------------------------------- messages

def account_error() -> str | None:
    """The refusal of check_job/submit_job when no Slurm account is configured, else None."""
    if account():
        return None
    why = f' ({error()})' if error() else ''
    return (f'no Slurm account configured{why}: set [project] account in {path()} (or AMADEUS_ACCOUNT), see '
            'config.example.toml and INSTALL.md in the Amadeus repository. Read-only tools work without it')


def summary() -> str:
    """One line for the server instructions: the configured project, or how to configure one."""
    if error() and not account():
        return (f'The config file could not be read ({error()}): read-only tools work, check_job and submit_job '
                'refuse until it is fixed (see INSTALL.md).')
    if not account():
        return (f'No project is configured yet ({path()}; see INSTALL.md): read-only tools work, check_job and '
                'submit_job refuse until [project] account is set.')
    parts = [f'project {project()}' if project() else '', f'Slurm account {account()}',
             f'filesystem {filesystem()}' if filesystem() else '']
    line = 'Configured for ' + ', '.join(p for p in parts if p) + f'; default working directory {workdir()}'
    if ends():
        line += f'; the project ends {ends()}'
    return line + '.'
