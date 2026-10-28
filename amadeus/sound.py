"""The banner and the two sounds of Amadeus.

The sounds are synthesised once into ~/.cache/amadeus/*.wav and played in the background with whatever player the
machine has (afplay on a Mac, pw-play/paplay/aplay on Linux, PowerShell from WSL2). No player or no audio device
(a container, a remote VM): silence, never an error. AMADEUS_SOUND=off switches them off.
"""
from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

BANNER = r"""
    _                        _
   / \   _ __ ___   __ _  __| | ___ _   _ ___
  / _ \ | '_ ` _ \ / _` |/ _` |/ _ \ | | / __|
 / ___ \| | | | | | (_| | (_| |  __/ |_| \__ \
/_/   \_\_| |_| |_|\__,_|\__,_|\___|\__,_|___/
        ♪  MUSICA · ASC · Wien  ♫
"""

RATE = 22050
ERROR_GAP_S = 10            # at most one error sound per this many seconds
_last_error = 0.0
_lock = threading.Lock()


def _note(name: str) -> float:
    """'G4' -> Hz (equal temperament, A4 = 440)."""
    steps = {'C': -9, 'D': -7, 'E': -5, 'F': -4, 'G': -2, 'A': 0, 'B': 2}[name[0]]
    steps += name.count('#') - name.count('b')
    return 440.0 * 2 ** ((steps + 12 * (int(name[-1]) - 4)) / 12)


# Eine kleine Nachtmusik, K. 525, the opening bars (in seconds; None = rest)
JINGLE = [('G4', .30), (None, .08), ('D4', .14), ('G4', .30), (None, .08), ('D4', .14),
          ('G4', .14), ('D4', .14), ('G4', .14), ('B4', .14), ('D5', .55)]
# a falling D minor line, as in the Lacrimosa: something went wrong
ERROR = [('A4', .16), ('F4', .16), ('D4', .16), ('C#4', .45)]


def _samples(score: list, volume: float = .35) -> bytes:
    out = bytearray()
    for name, dur in score:
        n = int(RATE * dur)
        if name is None:
            out += b'\0\0' * n
            continue
        f = _note(name)
        for i in range(n):
            t = i / RATE
            env = min(1.0, t / .01) * math.exp(-3.0 * t / dur)          # soft attack, decaying like a plucked string
            s = math.sin(2 * math.pi * f * t) + .3 * math.sin(4 * math.pi * f * t) + .1 * math.sin(6 * math.pi * f * t)
            out += struct.pack('<h', int(32767 * volume * env * s / 1.4))
    return bytes(out)


def wav(kind: str) -> Path:
    d = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'amadeus'
    p = d / f'{kind}.wav'
    if not p.exists():
        d.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f'.{os.getpid()}.tmp')
        with wave.open(str(tmp), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(_samples(JINGLE if kind == 'jingle' else ERROR))
        tmp.replace(p)
    return p


def _wsl() -> bool:
    try:
        return 'microsoft' in Path('/proc/version').read_text().lower()
    except OSError:
        return False


def _player(path: Path) -> list[str] | None:
    if sys.platform == 'darwin' and shutil.which('afplay'):
        return ['afplay', str(path)]
    if _wsl() and shutil.which('powershell.exe') and shutil.which('wslpath'):
        win = subprocess.run(['wslpath', '-w', str(path)], capture_output=True, text=True, timeout=5).stdout.strip()
        if win:
            return ['powershell.exe', '-NoProfile', '-Command',
                    f"(New-Object Media.SoundPlayer '{win}').PlaySync()"]
    for p in ('pw-play', 'paplay', 'aplay'):
        if shutil.which(p):
            return [p, '-q', str(path)] if p == 'aplay' else [p, str(path)]
    return None


def enabled() -> bool:
    return os.environ.get('AMADEUS_SOUND', 'on').strip().lower() not in ('0', 'off', 'no', 'false')


def play(kind: str) -> None:
    """Play 'jingle' or 'error' in the background; never blocks, never raises, never touches stdout."""
    global _last_error
    if not enabled():
        return
    if kind == 'error':
        with _lock:
            if time.monotonic() - _last_error < ERROR_GAP_S:
                return
            _last_error = time.monotonic()

    def run():
        try:
            if (cmd := _player(wav(kind))):
                subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


def failed(res) -> bool:
    """Whether a tool result reports a failure worth the error sound: an `error`, or a job that FAILED."""
    if not isinstance(res, dict):
        return False
    if res.get('error'):
        return True
    return any(str(e).startswith('FAILED ') for e in res.get('events') or [])


def greet() -> None:
    """Banner on stderr (stdout is the MCP channel) and the jingle."""
    try:
        print(BANNER, file=sys.stderr, flush=True)
    except Exception:
        pass
    play('jingle')
