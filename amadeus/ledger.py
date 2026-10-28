"""Local record of the jobs submitted or cancelled through the server, one JSON object per line.

Default ~/.local/state/amadeus/jobs.jsonl ($XDG_STATE_HOME/amadeus if set); AMADEUS_LEDGER replaces the path.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def path() -> Path:
    if (p := os.environ.get('AMADEUS_LEDGER')):
        return Path(p).expanduser()
    state = Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local' / 'state')
    return state / 'amadeus' / 'jobs.jsonl'


def append(entry: dict) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **entry}
    with p.open('a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def entries() -> list[dict]:
    p = path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def submitted() -> dict[str, dict]:
    """job id -> its submission entry."""
    return {e['job_id']: e for e in entries() if e.get('event') == 'submit' and e.get('job_id')}
