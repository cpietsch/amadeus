import atexit
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("AMADEUS_SOUND", "off")   # no jingles from the test suite

# Independent of the user's own setup: the test project's config, a throwaway ledger, no overrides from the shell.
for _k in ("AMADEUS_ACCOUNT", "AMADEUS_PROJECT", "AMADEUS_FILESYSTEM", "AMADEUS_BUDGET", "AMADEUS_WORKDIR",
           "AMADEUS_TEMPLATE_DIRS", "AMADEUS_USER", "AMADEUS_LOCAL_ROOT", "AMADEUS_HOSTS", "AMADEUS_DEFAULT_HOST"):
    os.environ.pop(_k, None)
TEST_CONFIG = Path(__file__).resolve().parent / "config.toml"
os.environ["AMADEUS_CONFIG"] = str(TEST_CONFIG)
_state = tempfile.mkdtemp(prefix="amadeus-tests-")
atexit.register(shutil.rmtree, _state, True)
os.environ["AMADEUS_LEDGER"] = os.path.join(_state, "jobs.jsonl")
os.environ["XDG_STATE_HOME"] = _state           # Amadeus's ssh config (network.py) goes here, not to ~/.local/state
os.environ["MUSICA_VPN"] = "system"             # no network probe; tests that need wireproxy set it
