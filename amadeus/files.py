"""Moving files between a client's machine and the cluster: tar over SSH with retries (AGENTS.md §4, §4a).

"Local" is a host: '' is the machine this server runs on; the names in [files] hosts of the config (AMADEUS_HOSTS;
e.g. a GPU workstation) are reached over SSH, so files can move between the cluster and that machine. Bytes go host --tar--> this process --ssh--> cluster (and back),
relayed in chunks so a push stops at MAX_PUSH_MB. All host-side work is a bash snippet, run with `bash -c` here or
through the host's SSH command (AMADEUS_HOST_SSH_<NAME>, default `ssh -o BatchMode=yes <name>`).

Local paths must resolve inside the host's local root ([files] local_root or AMADEUS_LOCAL_ROOT on this machine,
default the home directory; the home directory on other hosts) with no hidden component (.ssh, .config, .git, ...) below it;
fetches never write into this server's own code, state, scripts or instructions. Remote paths are normalised and
checked here, then checked again on the cluster after resolving symlinks: below $DATA, $SCRATCH or a non-hidden
directory in $HOME.
"""
from __future__ import annotations

import contextlib
import fnmatch
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import config, remote

DEFAULT_EXCLUDES = ['.git', '.venv', 'venv', '__pycache__', 'node_modules', '*.sif', '.mypy_cache', '.pytest_cache']
# never pushed, in either mode (rule 7: never copy secrets); reported as skipped_secrets
SECRET_PATTERNS = ['.env', '.env.*', '*.pem', '*.key', '*.p12', 'id_rsa*', 'id_ecdsa*', 'id_ed25519*', '.netrc',
                   '.git-credentials']
MAX_PUSH_MB = 2000
CHUNK = 1 << 20
REPO = Path(__file__).resolve().parent.parent
# never fetched into on this machine: the server's code and ledger, scripts that run with the cluster credentials,
# the dependency lock `uv run` installs from, and the instructions agents follow
PROTECTED = ('amadeus', 'state', 'scripts', 'tests', 'pyproject.toml', 'uv.lock', 'AGENTS.md', 'CLAUDE.md',
             'NOTES.md')
HOST_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._@-]*$')
SAY = 'amadeus: '

# remote roots: a variable the cluster expands, or the project's storage written out
_VAR_HEAD = re.compile(r'^(~|\$HOME|\$\{HOME\}|\$DATA|\$\{DATA\}|\$SCRATCH|\$\{SCRATCH\})(?=/|$)(.*)$', re.S)
_ABS_HEAD_RX = r'^(/(?:data|scratch)/{fs}/[^/]+)(?=/|$)(.*)$'


def _abs_head(path: str):
    """/data/<filesystem>/<user>/... or /scratch/...: the configured filesystem, else any fs<digits>."""
    fs = config.filesystem()
    return re.match(_ABS_HEAD_RX.format(fs=re.escape(fs) if fs else r'fs\d+'), path, re.S)


# ---------------------------------------------------------------- remote paths

def normalize_remote(path: str) -> tuple[str | None, str | None]:
    """(normalised path, None) or (None, reason). Collapses '//' and '/./', refuses '..', $HOME itself and hidden
    components under $HOME. The cluster repeats the check on the resolved path (symlinks), see _cluster_guard."""
    if not isinstance(path, str) or not path.strip():
        return None, 'remote path is empty'
    if any(ord(c) < 32 for c in path):
        return None, f'remote path {path!r} contains control characters'
    must = f'remote path {path!r} must start with $DATA, $SCRATCH or ~/'
    if '..' in path.split('/'):
        return None, f'remote path {path!r} is not allowed: no ".." components'
    m = _VAR_HEAD.match(path)
    if m:
        head, rest = m.group(1).replace('{', '').replace('}', ''), m.group(2)
    elif path.startswith('/'):
        flat = '/' + '/'.join(p for p in path.split('/') if p not in ('', '.'))
        m = _abs_head(flat)
        if not m:
            return None, must
        head, rest = m.group(1), m.group(2)
    else:
        return None, must
    parts = [p for p in rest.split('/') if p not in ('', '.')]
    if head in ('~', '$HOME'):
        if not parts:
            return None, f'remote path {path!r} is not allowed: not $HOME itself'
        if any(p.startswith('.') for p in parts):
            return None, f'remote path {path!r} is not allowed: no hidden files or directories in $HOME'
    return head + ''.join('/' + p for p in parts), None


def check_remote(path: str) -> str | None:
    """Static pre-check of a remote path; None if allowed, else the reason."""
    return normalize_remote(path)[1]


def _below_root(norm: str) -> bool:
    m = _VAR_HEAD.match(norm)
    return bool(m and m.group(2).strip('/')) or bool(_abs_head(norm))


_CLUSTER_GUARD = r'''want=@WANT@
r=$(realpath @MODE@ -- @PATH@ 2>/dev/null) || { echo "amadeus: $want does not exist on the cluster" >&2; exit 3; }
ok=
for b in "${DATA:-}" "${SCRATCH:-}"; do
  [ -n "$b" ] || continue
  b=$(realpath -m -- "$b")
  case $r in "$b"/?*) ok=1 ;; esac
done
if [ -z "$ok" ]; then
  h=$(realpath -m -- "$HOME")
  case $r in "$h"/?*) case /${r#"$h"/}/ in */.*) ;; *) ok=1 ;; esac ;; esac
fi
[ -n "$ok" ] || { echo "amadeus: $want resolves to $r, which is not below \$DATA, \$SCRATCH or a non-hidden directory in \$HOME" >&2; exit 4; }
'''


def _cluster_guard(norm: str, must_exist: bool) -> str:
    """Shell that sets $r to the resolved path, or exits 3 (missing) / 4 (not allowed). Catches symlinks into ~/.ssh."""
    return _fill(_CLUSTER_GUARD, WANT=shlex.quote(norm), MODE='-e' if must_exist else '-m', PATH=remote.q(norm))


# ---------------------------------------------------------------- hosts

def hosts() -> list[str]:
    """Hosts the file tools may use: '' (this server's machine) and the names in [files] hosts (AMADEUS_HOSTS)."""
    names = ['']
    for h in config.hosts():
        if HOST_NAME.match(h) and h not in names:
            names.append(h)
    return names


def default_host() -> str:
    return config.default_host()


def _host_error(host) -> str | None:
    if not isinstance(host, str) or host not in hosts():
        known = ', '.join(repr(h) for h in hosts())
        return f'unknown host {host!r}: use one of {known} ("" is the machine the server runs on; more in [files] hosts of the config)'
    return None


# macOS: the snippets need GNU tar, find, stat and realpath -m; prefer Homebrew's (brew install coreutils findutils
# gnu-tar) over the BSD tools. A no-op elsewhere.
GNU_SHIM = r'''if [ "$(uname -s)" = Darwin ]; then
  for d in /opt/homebrew/opt/{coreutils,findutils,gnu-tar}/libexec/gnubin /usr/local/opt/{coreutils,findutils,gnu-tar}/libexec/gnubin; do
    [ -d "$d" ] && PATH=$d:$PATH
  done
  export PATH
fi
'''


def _host_argv(host: str, snippet: str) -> list[str]:
    snippet = GNU_SHIM + snippet
    if host == '':
        return ['bash', '-c', snippet]
    var = 'AMADEUS_HOST_SSH_' + re.sub(r'[^A-Za-z0-9]', '_', host).upper()
    return shlex.split(os.environ.get(var, f'ssh -o BatchMode=yes {host}')) + ['bash -c ' + shlex.quote(snippet)]


_HOST_PRELUDE = r'''export LC_ALL=C
say() { printf 'amadeus: %s\n' "$*" >&2; }
root=$(realpath -m -- @ROOT@) || { say "cannot resolve the local root"; exit 4; }
root=${root%/}
protected=(@PROTECTED@)
resolve() {
  case $1 in
    '~') realpath -m -- "$HOME" ;;
    '~/'*) realpath -m -- "$HOME/${1#'~/'}" ;;
    /*) realpath -m -- "$1" ;;
    *) say "local path $1 must be absolute or start with ~/"; return 1 ;;
  esac
}
allowed() {  # allowed PATH [overlap]: inside the root, no hidden component below it; with overlap, clear of the server's files
  case $1/ in "$root"/*) ;; *) say "$1 is outside $root, the tree the file tools may use on this machine"; return 1 ;; esac
  case ${1#"$root"}/ in */.*) say "$1 has a hidden component (.ssh, .config, .git, ...) below $root"; return 1 ;; esac
  [ -n "${2:-}" ] || return 0
  local x
  for x in "${protected[@]}"; do
    case $1/ in "$x"/*) say "$1 is inside the MCP server's own $x"; return 1 ;; esac
    case $x/ in "$1"/*) say "$1 would contain the MCP server's own $x"; return 1 ;; esac
  done
}
'''


def _host_prelude(host: str) -> str:
    if host == '':
        root = shlex.quote(os.path.expanduser(config.local_root()))
    else:
        root = '"$HOME"'
    protected = ' '.join(shlex.quote(os.path.realpath(REPO / p)) for p in PROTECTED) if host == '' else ''
    return _fill(_HOST_PRELUDE, ROOT=root, PROTECTED=protected)


def _fill(template: str, **values: str) -> str:
    for k, v in values.items():
        template = template.replace(f'@{k}@', v)
    return template


def _local_path_error(path) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return 'local path is empty'
    if any(ord(c) < 32 for c in path):
        return f'local path {path!r} contains control characters'
    return None


# ---------------------------------------------------------------- running and relaying

def _transient(code: int, err: str) -> bool:
    return code == 255 and bool(remote.TRANSIENT.search(err))


def _said(err: str) -> list[str]:
    return [l[len(SAY):] for l in err.splitlines() if l.startswith(SAY)]


def _run(argv: list[str], timeout: float) -> tuple[int, bytes, str, int]:
    """Run a short command (retried on transient SSH failures); (exit code, stdout, stderr, attempts)."""
    for attempt in range(1, 4):
        try:
            p = subprocess.run(argv, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
            code, out, err = p.returncode, p.stdout, p.stderr.decode(errors='replace')
        except subprocess.TimeoutExpired:
            code, out, err = 124, b'', f'timed out after {timeout:g} s'
        if attempt == 3 or not _transient(code, err):
            return code, out, err, attempt
        time.sleep(remote.retry_delay())
    raise AssertionError('unreachable')


def _host_failure(host: str, code: int, err: str, what: str) -> dict:
    said = _said(err)
    if said:
        return {'error': '; '.join(said), 'host': host}
    where = f'host {host!r}' if host else 'this machine'
    msg = f'{what} on {where} failed (exit {code})'
    if code == 255 and host:
        msg += f': SSH to {host} failed (the server reaches it with AMADEUS_HOST_SSH_' \
               f'{re.sub(r"[^A-Za-z0-9]", "_", host).upper()} or `ssh {host}`)'
    return {'error': msg, 'stderr': remote.clip(err, 2000), 'host': host}


@dataclass
class _Relay:
    prod_rc: int
    cons_rc: int
    prod_err: str
    cons_err: str
    cons_out: str
    sent: int
    over: bool
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.prod_rc == 0 and self.cons_rc == 0 and not self.over and not self.timed_out

    @property
    def transient(self) -> bool:
        return _transient(self.cons_rc, self.cons_err) or _transient(self.prod_rc, self.prod_err)


def _relay(producer: list[str], consumer: list[str], timeout: float, feed: bytes | None = None,
           limit: float | None = None) -> _Relay:
    """producer | consumer through this process, in chunks. Stops past `limit` bytes. The last chunk is held back
    until the producer has exited 0, so a failed producer leaves the consumer with a truncated gzip stream (which
    tar refuses) instead of a complete archive of fewer files."""
    with tempfile.TemporaryFile() as pe, tempfile.TemporaryFile() as ce, tempfile.TemporaryFile() as co:
        a = subprocess.Popen(producer, stdin=subprocess.PIPE if feed is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=pe)
        try:
            b = subprocess.Popen(consumer, stdin=subprocess.PIPE, stdout=co, stderr=ce)
        except OSError:
            a.kill()
            a.wait()
            raise
        expired = threading.Event()

        def kill_both():
            expired.set()
            for p in (a, b):
                with contextlib.suppress(OSError):
                    p.kill()
        timer = threading.Timer(timeout, kill_both)
        timer.daemon = True
        timer.start()
        if feed is not None:
            def feed_producer():
                try:
                    a.stdin.write(feed)
                except OSError:
                    pass
                finally:
                    with contextlib.suppress(OSError):
                        a.stdin.close()
            threading.Thread(target=feed_producer, daemon=True).start()
        sent, over, gone, pending = 0, False, False, b''
        try:
            while True:
                chunk = a.stdout.read1(CHUNK)
                if not chunk:
                    break
                sent += len(chunk)
                if limit is not None and sent > limit:
                    over = True
                    break
                if pending:
                    try:
                        b.stdin.write(pending)
                    except (OSError, ValueError):
                        gone = True
                        break
                pending = chunk
            if over or gone:
                a.kill()
            elif a.wait() == 0 and pending:
                try:
                    b.stdin.write(pending)
                except (OSError, ValueError):
                    pass
        finally:
            with contextlib.suppress(OSError, ValueError):
                b.stdin.close()
            with contextlib.suppress(OSError, ValueError):
                a.stdout.close()
            rb = b.wait()
            ra = a.wait()
            timer.cancel()
        pe.seek(0), ce.seek(0), co.seek(0)
        return _Relay(ra, rb, pe.read().decode(errors='replace'), ce.read().decode(errors='replace'),
                      co.read().decode(errors='replace'), sent, over, expired.is_set())


def _transfer(producer: list[str], consumer: list[str], timeout: float, feed: bytes | None = None,
              limit: float | None = None) -> tuple[_Relay, int]:
    """Up to 3 attempts; only an SSH exit 255 with a transient message is retried, with no sleep after the last."""
    for attempt in range(1, 4):
        r = _relay(producer, consumer, timeout, feed, limit)
        if r.ok or r.over or r.timed_out or attempt == 3 or not r.transient:
            return r, attempt
        time.sleep(remote.retry_delay())
    raise AssertionError('unreachable')


def _transfer_failure(r: _Relay, attempts: int, timeout: float) -> dict:
    err = (r.prod_err + '\n' + r.cons_err).strip()
    said = _said(err)
    if r.timed_out:
        msg = f'timed out after {timeout:g} s (raise the timeout, or move big data inside the cluster)'
    elif said:
        msg = '; '.join(said)
    else:
        msg = remote.clip(err, 3000) or 'transfer failed'
    out = {'error': msg, 'exit_codes': [r.prod_rc, r.cons_rc], 'attempts': attempts}
    if said and err:
        out['stderr'] = remote.clip(err, 3000)
    if 255 in (r.prod_rc, r.cons_rc):
        out['note'] = 'SSH failed: check access_status (tunnel, certificate)'
    return out


# ---------------------------------------------------------------- push

def _patterns(exclude) -> list[str]:
    if isinstance(exclude, str):
        exclude = [exclude]
    out = []
    for p in exclude or []:
        p = str(p).strip()
        while p.startswith('./'):
            p = p[2:]
        p = p.strip('/')
        if p:
            out.append(p)
    return out


def _excluded(rel: str, patterns: list[str]) -> bool:
    """fnmatch against the relative path, each of its parent directories ('data', 'data/raw') and each component."""
    parts = rel.split('/')
    candidates = parts + ['/'.join(parts[:i]) for i in range(2, len(parts) + 1)]
    return any(fnmatch.fnmatchcase(c, p) for p in patterns for c in candidates)


def _prune_expr(patterns: list[str]) -> str:
    """find(1) pruning for `mode='all'`: skips excluded trees early. Python filters again with _excluded, so only
    patterns that find and fnmatch read the same way are used here."""
    terms = []
    for p in patterns:
        if '\\' in p or '[' in p:
            continue
        terms.append(f'-path {shlex.quote("./" + p)}' if '/' in p else f'-name {shlex.quote(p)}')
    return ' -o '.join(terms) or '-false'


_SOURCE = r'''src=$(resolve @PATH@) || exit 4
allowed "$src" || exit 4
[ -d "$src" ] || { say "$src is not a directory"; exit 5; }
cd -- "$src" || exit 5
'''

_LIST_TRACKED = r'''printf 'src\t%s\0' "$src"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { say "$src is not in a git repository; use mode=\"all\""; exit 6; }
printf 'rev\t%s\0' "$(git rev-parse --verify -q --short HEAD 2>/dev/null || echo unknown)"
printf 'dirty\t%s\0' "$(git status --porcelain --untracked-files=no -- . 2>/dev/null | wc -l)"
git ls-files -z --cached | xargs -0 -r stat --printf 't\t%F\t%s\t%n\0' -- 2>/dev/null
[ "${PIPESTATUS[0]}" = 0 ] || { say "git ls-files failed in $src"; exit 7; }
git ls-files -z --others --exclude-standard | xargs -0 -r stat --printf 'u\t%F\t%s\t%n\0' -- 2>/dev/null
[ "${PIPESTATUS[0]}" = 0 ] || { say "git ls-files failed in $src"; exit 7; }
exit 0
'''

_LIST_ALL = r'''printf 'src\t%s\0' "$src"
find . -mindepth 1 \( @PRUNE@ \) -prune -o \( -type f -o -type l \) -printf 'u\t%y\t%s\t%P\0' || { say "find failed in $src"; exit 7; }
'''

_PUSH_PRODUCER = r'''exec tar czf - --null --verbatim-files-from --no-recursion -T -
'''

_PUSH_CONSUMER = r'''mkdir -p -- "$r" || exit 5
s=$(mktemp -d "$r/.push.XXXXXX") || exit 5
trap 'rm -rf -- "$s"' EXIT
trap 'exit 1' HUP INT TERM PIPE
chmod --reference="$r" -- "$s" 2>/dev/null
tar xzf - -C "$s" --no-same-owner || { echo "amadeus: the upload did not arrive complete; $want is unchanged" >&2; exit 6; }
@REVISION@
cp -a -- "$s/." "$r/" || { @PARTIAL@ echo "amadeus: copying into $want failed part-way" >&2; exit 7; }
printf '%s\n' "$r"
'''

_FILE_KINDS = {'f', 'l', 'regular file', 'regular empty file', 'symbolic link'}


def push(local_dir: str, remote_dir: str, mode: str = 'tracked', exclude: list[str] | None = None,
         host: str | None = None, timeout: float = 600) -> dict:
    """Copy local_dir on `host` into remote_dir on the cluster. mode 'tracked': git-tracked plus untracked-but-not-
    ignored files and a REVISION file (the commit, '-dirty' if tracked files are modified or untracked files go
    along); 'all': every file minus DEFAULT_EXCLUDES. `exclude`: fnmatch patterns against the relative path, its
    parent directories and each component ('data/raw', 'raw', '*.ckpt'). Key files and .env files are never sent
    (SECRET_PATTERNS). The upload lands in a staging directory and is copied over remote_dir only when complete.
    Never deletes on the remote side."""
    host = default_host() if host is None else host
    if (err := _host_error(host)):
        return {'error': err, 'host': host}
    if mode not in ('tracked', 'all'):
        return {'error': "mode must be 'tracked' or 'all'", 'host': host}
    if (err := _local_path_error(local_dir)):
        return {'error': err, 'host': host}
    norm, err = normalize_remote(remote_dir)
    if err:
        return {'error': err, 'host': host}
    if not _below_root(norm):
        return {'error': f'remote_dir {remote_dir!r} must name a directory below $DATA, $SCRATCH or ~/', 'host': host}
    patterns = _patterns(exclude)
    if mode == 'all':
        patterns = DEFAULT_EXCLUDES + patterns
    prelude = _host_prelude(host)
    source = _fill(_SOURCE, PATH=shlex.quote(local_dir))
    lister = _LIST_TRACKED if mode == 'tracked' else _fill(_LIST_ALL, PRUNE=_prune_expr(patterns))
    code, out, err, _ = _run(_host_argv(host, prelude + source + lister), timeout)
    if code != 0:
        return _host_failure(host, code, err, 'listing the files')
    src, rev, dirty, files, secrets = '', 'unknown', False, {}, []
    for rec in out.split(b'\0'):
        if not rec:
            continue
        kind, _, rest = rec.partition(b'\t')
        if kind == b'src':
            src = os.fsdecode(rest)
        elif kind == b'rev':
            rev = rest.decode(errors='replace').strip() or 'unknown'
        elif kind == b'dirty':
            dirty = rest.strip() not in (b'', b'0')
        elif kind in (b't', b'u'):
            ftype, size, name = rest.split(b'\t', 2)
            name = os.fsdecode(name)
            if ftype.decode() not in _FILE_KINDS or name in files or _excluded(name, patterns):
                continue
            if _excluded(name, SECRET_PATTERNS):
                secrets.append(name)
                continue
            files[name] = (kind == b'u', int(size))
    names = sorted(files)
    size = sum(s for _, s in files.values())
    res = {'files': len(names), 'megabytes': round(size / 1e6, 1), 'remote_dir': norm, 'host': host,
           'local_dir': src}
    if secrets:
        res['skipped_secrets'] = sorted(set(secrets))[:20]
    if size > MAX_PUSH_MB * 1e6:
        return {**res, 'error': f'{size / 1e6:.0f} MB is over the {MAX_PUSH_MB} MB limit: download big files on the '
                                'cluster (in a job) instead of pushing them, or exclude them'}
    revision_sh = partial_sh = ''
    if mode == 'tracked':
        revision = rev + ('-dirty' if dirty or any(u for u, _ in files.values()) else '')
        res['revision'] = revision
        revision_sh = f'printf \'%s\\n\' {shlex.quote(revision)} > "$s/REVISION" || exit 6'
        partial_sh = f'printf \'%s\\n\' {shlex.quote(revision + "-partial")} > "$r/REVISION";'
    producer = _host_argv(host, prelude + source + _PUSH_PRODUCER)
    consumer = remote.ssh_base() + [_cluster_guard(norm, must_exist=False) +
                                    _fill(_PUSH_CONSUMER, REVISION=revision_sh, PARTIAL=partial_sh)]
    feed = b''.join(os.fsencode(n) + b'\0' for n in names)
    r, attempts = _transfer(producer, consumer, timeout, feed=feed, limit=MAX_PUSH_MB * 1e6)
    res['attempts'] = attempts
    if r.over:
        return {**res, 'error': f'the upload grew past {MAX_PUSH_MB} MB and was stopped; {norm} is unchanged'}
    if not r.ok:
        return {**res, **_transfer_failure(r, attempts, timeout)}
    if r.cons_out.strip():
        res['remote_path'] = r.cons_out.strip().splitlines()[-1]
    return res


# ---------------------------------------------------------------- fetch

_DEST = r'''d=$(resolve @PATH@) || exit 4
allowed "$d" || exit 4
[ ! -e "$d" ] || [ -d "$d" ] || { say "$d is not a directory"; exit 5; }
f=$d/@NAME@
allowed "$f" overlap || exit 4
allowed "$(realpath -m -- "$f")" overlap || exit 4
if [ -d "$f" ] && [ ! -L "$f" ]; then
  while IFS= read -r -d '' l; do
    [ -d "$l" ] || continue
    allowed "$(realpath -m -- "$l")" overlap || { say "$l links to a directory outside the allowed area; remove the link or fetch elsewhere"; exit 4; }
  done < <(find "$f" -type l -print0)
fi
'''

_FETCH_CHECK = r'''printf '%s\n' "$d"
'''

_FETCH_CONSUMER = r'''mkdir -p -- "$d" || exit 5
tar xzf - -C "$d" --no-same-owner --exclude='.*' --no-wildcards -- @NAME@ || exit 6
'''

_FETCH_SIZE = r'''printf '%s\n' "$r"
s=$(du -sb -- "$r" | cut -f1)
printf '%s\n' "${s:-0}"
find "$r" -mindepth 1 -name '.*' -prune -print 2>/dev/null | wc -l
'''

_FETCH_PRODUCER = r'''cd -- "$(dirname -- "$r")" || exit 5
exec tar czf - --transform=@XFORM@ -- "$(basename -- "$r")"
'''


def fetch(remote_path: str, local_dir: str, max_mb: float = 500, host: str | None = None,
          timeout: float = 900) -> dict:
    """Copy a remote file or directory into local_dir on `host` (created if needed), as local_dir/<its name>.
    A symlink is followed: its target's contents arrive under the requested name. Hidden files and directories
    inside it are skipped (they could replace .git, .venv or shell hooks on the host)."""
    host = default_host() if host is None else host
    if (err := _host_error(host)):
        return {'error': err, 'host': host}
    if (err := _local_path_error(local_dir)):
        return {'error': err, 'host': host}
    norm, err = normalize_remote(remote_path)
    if err:
        return {'error': err, 'host': host}
    if not _below_root(norm):
        return {'error': f'remote_path {remote_path!r} must name a file or directory below $DATA, $SCRATCH or ~/',
                'host': host}
    name = norm.rsplit('/', 1)[-1]
    if name.startswith('.'):
        return {'error': f'{name} is hidden: hidden files and directories are not fetched', 'host': host}
    dest = _fill(_DEST, PATH=shlex.quote(local_dir), NAME=shlex.quote(name))
    prelude = _host_prelude(host)
    code, out, err, _ = _run(_host_argv(host, prelude + dest + _FETCH_CHECK), min(timeout, 300))
    if code != 0:
        return _host_failure(host, code, err, 'checking the destination')
    dst = os.fsdecode(out).strip().splitlines()[-1]
    r = remote.run(_cluster_guard(norm, must_exist=True) + _FETCH_SIZE, timeout=min(timeout, 300), retry=True)
    if r.code in (3, 4):
        return {'error': '; '.join(_said(r.err)) or r.err.strip(), 'host': host}
    if not r.ok:
        return {'error': 'size check failed', **r.summary(2000), 'host': host}
    try:
        lines = r.out.strip().splitlines()
        resolved, size, hidden = lines[-3], int(lines[-2]), int(lines[-1])
    except (IndexError, ValueError):
        return {'error': 'unexpected output from the size check', **r.summary(2000), 'host': host}
    if size > max_mb * 1e6:
        return {'error': f'{size / 1e6:.0f} MB is over max_mb={max_mb:g}; fetch a subdirectory or raise max_mb',
                'host': host}
    xform = 's,^[^/]*,' + re.sub(r'([\\&,])', r'\\\1', name) + ',S'
    producer = remote.ssh_base() + [_cluster_guard(norm, must_exist=True) +
                                    _fill(_FETCH_PRODUCER, XFORM=shlex.quote(xform))]
    consumer = _host_argv(host, prelude + dest + _fill(_FETCH_CONSUMER, NAME=shlex.quote(name)))
    tr, attempts = _transfer(producer, consumer, timeout, limit=max_mb * 1e6 * 1.02 + 1e6)
    res = {'local_path': f'{dst.rstrip("/")}/{name}', 'megabytes': round(size / 1e6, 2), 'host': host,
           'attempts': attempts, 'remote_path': resolved}
    if hidden:
        res['skipped_hidden'] = hidden
    if tr.over:
        return {**res, 'error': f'the download grew past max_mb={max_mb:g} and was stopped; '
                                f'{res["local_path"]} may be incomplete'}
    if not tr.ok:
        return {**res, **_transfer_failure(tr, attempts, timeout),
                'warning': f'{res["local_path"]} may be incomplete'}
    return res


def quote_list(items: list[str]) -> str:
    return ' '.join(shlex.quote(i) for i in items)
