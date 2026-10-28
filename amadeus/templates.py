"""Tested job scripts as templates: this repository's jobs/ and the project's job directories.

The extra directories are [project] template_dirs in the config (AMADEUS_TEMPLATE_DIRS, os.pathsep-separated,
replaces them); directories that do not exist are skipped. Each script is described by its leading comments and its
#SBATCH options, and its source names the commit, with -dirty when the file differs from it (or is not committed).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import config
from . import slurm as S

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 60_000
# helper scripts a job sources or runs: jobs/lib/x.sh or lib/x.sh, at the start of a word or after $VAR/ or $(...)/,
# not deeper inside another path ($DATA/opt/tool/lib/x.sh is not a helper)
HELPER_RE = re.compile(r'(?:^|[\s"\'=(]|\$\{?\w+\}?/|\)/)((?:jobs/)?lib/[\w.-]+\.sh)\b', re.M)
RESOURCE_KEYS = ('partition', 'qos', 'nodes', 'gres', 'ntasks', 'cpus-per-task', 'mem', 'time', 'array', 'exclusive')


def dirs() -> list[Path]:
    out, seen = [], set()
    for d in [ROOT / 'jobs'] + config.template_dirs():
        if d.is_dir() and (r := d.resolve()) not in seen:
            seen.add(r)
            out.append(d)
    return out


def _repo(d: Path) -> str:
    return d.parent.name if d.name == 'jobs' else d.name


def _git(d: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(['git', '-C', str(d), *args], capture_output=True, text=True, timeout=10)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _state(d: Path) -> tuple[str, set[str]]:
    """The directory's commit and the file names in it that differ from it or are not committed."""
    rev = (_git(d, 'rev-parse', '--short', 'HEAD') or '').strip()
    changed = set((_git(d, 'diff', '--name-only', '--relative', 'HEAD', '--', '.') or '').split())
    changed |= set((_git(d, 'ls-files', '--others', '--exclude-standard', '--', '.') or '').split())
    return rev, changed


def _source(d: Path, f: Path, rev: str, changed: set[str]) -> str:
    if not rev:
        return _repo(d)
    return f'{_repo(d)}@{rev}' + ('-dirty' if f.name in changed else '')


def summary(text: str, limit: int = 400) -> str:
    """The leading comment lines (before the first command), without #SBATCH, #ASC and the shebang."""
    out = []
    for line in text.splitlines()[1:]:
        s = line.strip()
        if not s:
            continue
        if not s.startswith('#'):
            break
        if s.startswith(('#SBATCH', '#ASC', '#!')) or s.startswith('# shellcheck'):
            continue
        out.append(s.lstrip('#').strip())
    joined = ' '.join(x for x in out if x)
    return joined if len(joined) <= limit else joined[:limit - 1].rstrip() + '…'


def _entry(d: Path, f: Path, rev: str, changed: set[str]) -> dict:
    text = f.read_text(errors='replace')
    opts, _ = S.script_options(text)
    lines = text.splitlines()
    res = {k: opts[k] if opts[k] else True for k in RESOURCE_KEYS if k in opts}
    if len(lines) > 1 and lines[1].strip() == '#ASC --vanilla':
        res['vanilla'] = True
    return {'name': f'{_repo(d)}/{f.stem}', 'summary': summary(text), 'resources': res,
            'source': _source(d, f, rev, changed)}


def index() -> list[dict]:
    out = []
    for d in dirs():
        rev, changed = _state(d)
        for f in sorted(d.glob('*.slurm')):
            out.append(_entry(d, f, rev, changed))
    return out


def find(name: str) -> tuple[Path, Path] | None:
    """A template by its listed name (repo/stem; the first directory wins) or by its file name if that is unique."""
    name = name.strip().removesuffix('.slurm')
    by_stem = []
    for d in dirs():
        for f in sorted(d.glob('*.slurm')):
            if f'{_repo(d)}/{f.stem}' == name:
                return d, f
            if f.stem == name:
                by_stem.append((d, f))
    return by_stem[0] if len(by_stem) == 1 else None


def get(name: str) -> dict:
    hit = find(name)
    if hit is None:
        names = [e['name'] for e in index()]
        stem = name.strip().removesuffix('.slurm').split('/')[-1]
        close = [n for n in names if stem and stem in n]
        return {'error': f'no single template {name!r}' + (f'; did you mean {close[:5]}?' if close else '')
                + ' (job_templates() lists them)'}
    d, f = hit
    text = f.read_text(errors='replace')
    rev, changed = _state(d)
    out = {'name': f'{_repo(d)}/{f.stem}', 'path': f'{_repo(d)}/{d.name}/{f.name}', 'text': text[:MAX_TEXT],
           'source': _source(d, f, rev, changed)}
    helpers = {}
    for rel in dict.fromkeys(HELPER_RE.findall(text)):
        # jobs/lib/x.sh is relative to the repository (scripts run from its root), lib/x.sh to the script's directory
        p = d.parent / rel if rel.startswith('jobs/') else d / rel
        if p.is_file() and len(helpers) < 4:
            helpers[f'{_repo(d)}/{p.relative_to(d.parent)}'] = p.read_text(errors='replace')[:MAX_TEXT]
    if helpers:
        out['helpers'] = helpers
    return out
