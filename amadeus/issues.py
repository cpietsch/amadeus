"""Issue channel: problems with the server or the cluster interface, one Markdown file each in issues/.

Agents file issues with the report_issue tool; unexpected exceptions inside tools are filed automatically, once per
fingerprint (repeats raise the occurrence count of the open issue). A separate Claude session works through the open
issues (protocol in issues/README.md). `AMADEUS_ISSUES` sets the directory; `off` disables the channel.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ('open', 'in-progress', 'needs-human', 'fixed', 'wontfix', 'duplicate')
KINDS = ('bug', 'wrong-result', 'missing-feature', 'docs', 'cluster', 'other')
SEVERITIES = ('high', 'medium', 'low')
# never write secrets into an issue: bearer tokens, device links, private keys
REDACT = [(re.compile(r'(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}'), r'\1<redacted>'),
          (re.compile(r'https://auth\.asc\.ac\.at/device\?code=\d+'), '<device link redacted>'),
          (re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.S), '<key redacted>'),
          (re.compile(r'(?i)(token|password|secret)(["\']?\s*[:=]\s*["\']?)[^\s"\',}]{6,}'), r'\1\2<redacted>')]


def directory() -> Path | None:
    v = os.environ.get('AMADEUS_ISSUES', str(ROOT / 'issues'))
    return None if v.lower() in ('off', 'none', '') else Path(v)


def redact(text: str) -> str:
    for pat, rep in REDACT:
        text = pat.sub(rep, text)
    return text


def revision() -> str:
    p = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    rev = p.stdout.strip() or 'unknown'
    d = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no', '--', 'amadeus'], cwd=ROOT,
                       capture_output=True, text=True)
    return rev + ('-dirty' if d.stdout.strip() else '')


@contextmanager
def _locked(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / '.lock', 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:48] or 'issue'


def parse(path: Path) -> dict:
    """Front matter of an issue file plus its title."""
    text = path.read_text()
    meta: dict = {'path': str(path), 'file': path.name}
    m = re.match(r'^---\n(.*?)\n---\n', text, re.S)
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(':')
            if k.strip():
                meta[k.strip()] = v.strip()
    t = re.search(r'^# (.+)$', text, re.M)
    meta['title'] = t.group(1).strip() if t else path.stem
    return meta


def _front(meta: dict) -> str:
    return '---\n' + ''.join(f'{k}: {v}\n' for k, v in meta.items()) + '---\n'


def _write(d: Path, meta: dict, title: str, sections: list[tuple[str, str]]) -> Path:
    stamp = time.strftime('%Y%m%d-%H%M%S')
    meta = {'id': f'{stamp}-{secrets.token_hex(2)}', **meta}
    path = d / f'{meta["id"]}-{_slug(title)}.md'
    body = [_front(meta), f'# {redact(title)}\n']
    for head, text in sections:
        if text and text.strip():
            body.append(f'\n## {head}\n\n{redact(text.strip())}\n')
    body.append('\n## Resolution\n\n(for the fixer: what was wrong, the fix, the commit, the test that covers it)\n')
    path.write_text(''.join(body))
    return path


def report(title: str, what_happened: str, expected: str = '', reproduce: str = '', kind: str = 'bug',
           severity: str = 'medium', tool: str = '', project: str = '', client: str = '', context: str = '') -> dict:
    d = directory()
    if d is None:
        return {'filed': False, 'note': 'the issue channel is off (AMADEUS_ISSUES=off)'}
    kind = kind if kind in KINDS else 'other'
    severity = severity if severity in SEVERITIES else 'medium'
    meta = {'status': 'open', 'kind': kind, 'severity': severity, 'source': 'agent', 'tool': tool or '-',
            'project': project or '-', 'client': client or '-', 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'server_revision': revision()}
    with _locked(d):
        path = _write(d, meta, title, [('What happened', what_happened), ('Expected', expected),
                                       ('How to reproduce', reproduce), ('Context', context)])
    return {'filed': True, 'id': parse(path)['id'], 'path': str(path)}


def auto_report(tool: str, args: dict, exc: BaseException, client: str = '') -> str | None:
    """File an unexpected exception once per fingerprint; later occurrences raise the count. Returns the issue id."""
    d = directory()
    if d is None:
        return None
    tb = traceback.extract_tb(exc.__traceback__)
    where = next((f for f in reversed(tb) if '/amadeus/' in f.filename), tb[-1] if tb else None)
    loc = f'{Path(where.filename).name}:{where.lineno}' if where else '?'
    fp = hashlib.sha1(f'{tool}|{type(exc).__name__}|{loc}'.encode()).hexdigest()[:12]
    now = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    try:
        with _locked(d):
            for p in sorted(d.glob('*.md')):
                meta = parse(p)
                if meta.get('fingerprint') == fp and meta.get('status') in ('open', 'in-progress', 'needs-human'):
                    text = p.read_text()
                    n = int(meta.get('occurrences', '1')) + 1
                    text = re.sub(r'^occurrences: \d+$', f'occurrences: {n}', text, count=1, flags=re.M)
                    text = re.sub(r'^last_seen: .*$', f'last_seen: {now}', text, count=1, flags=re.M)
                    p.write_text(text)
                    return meta['id']
            meta = {'status': 'open', 'kind': 'bug', 'severity': 'high', 'source': 'automatic', 'tool': tool,
                    'project': '-', 'client': client or '-', 'created': now, 'last_seen': now, 'occurrences': 1,
                    'fingerprint': fp, 'server_revision': revision()}
            title = f'{tool}: {type(exc).__name__} at {loc}: {str(exc)[:120]}'
            path = _write(d, meta, title, [
                ('What happened', f'The tool `{tool}` raised an unexpected exception: `{type(exc).__name__}: {exc}`'),
                ('Arguments', '```json\n' + json.dumps(args, indent=1, default=str)[:4000] + '\n```'),
                ('Traceback', '```\n' + ''.join(traceback.format_exception(exc))[-6000:] + '```')])
            return parse(path)['id']
    except OSError:
        return None


def listing(status: str = 'open') -> list[dict]:
    d = directory()
    if d is None or not d.exists():
        return []
    out = []
    for p in sorted(d.glob('*.md')) + sorted((d / 'closed').glob('*.md')):
        if p.name == 'README.md':
            continue
        meta = parse(p)
        if status == 'all' or meta.get('status') == status:
            out.append({k: meta.get(k) for k in ('id', 'status', 'kind', 'severity', 'source', 'tool', 'project',
                                                 'created', 'occurrences', 'fixed_in', 'title', 'file')
                        if meta.get(k) is not None})
    return out
