# Installing Amadeus, the MCP server for MUSICA

Written for a coding agent (Claude Code, Codex, …) that sets this up for its human. The human says something like
*"Install Amadeus following INSTALL.md in github.com/cpietsch/amadeus"*, and the agent does the
rest, asking the human only where noted. Keep it simple: use the platform's normal way to install things.

**What only the human can do** (ask for it when you get there, don't guess):
- an ASC account with MFA (https://auth.asc.ac.at), and their ASC username;
- their ASC project's details: name, Slurm account, filesystem, budget, working directory on the cluster (§6);
- for the ASC VPN: membership in the group `asc-wg-pub-access` and their WireGuard config from
  https://wg-pub.asc.ac.at (it contains a private key: never print it, copy it into place with mode 600);
- approving the login link in a browser, within about 60 s of it being shown;
- `sudo`, if a step needs it (ask first).

Run every step on the machine where the agent and its MCP client run (on Windows: inside WSL2, see below).
The steps: find the network path (§1), install the tools (§2), set up wireproxy if needed (§3), bootstrap `step`
(§4), register the server (§5), write the project config (§6), log in and check (§7).

## 1. Find the setup

MUSICA's login nodes only accept connections from partner-university networks or the ASC WireGuard VPN, and
only ASC-signed SSH certificates (made by the `step` CLI, valid about 16 h). How the network part works depends
on the machine:

| Machine | Network path | Notes |
|---|---|---|
| **Mac**, WireGuard app already set up with the ASC config | the system VPN (`MUSICA_VPN=system`) | Nothing to install for the network; the tunnel must be connected when the agent works |
| Linux desktop or laptop with WireGuard (`wg-quick`, NetworkManager) or on a university network | the system VPN | Same |
| **Dev machine without root or a TUN device**: a container, a remote dev VM, a cloud box | **wireproxy**, WireGuard in userspace, as a local SOCKS5 proxy | The server and `scripts/musica_auth.sh` start it themselves |
| **Windows** | use **WSL2** (Ubuntu) and follow the Linux steps inside it | Native Windows is not supported (bash scripts, Unix file locking): install and run Claude Code or Codex **inside the WSL shell** too, clone into the WSL filesystem (under `~`, not `/mnt/c`). If the WireGuard app on Windows is connected, WSL2's traffic usually goes through it (system VPN); otherwise use wireproxy inside WSL. The browser approval can happen in Windows. Not tested yet: report what worked |

Quick test whether the network path already works (system VPN up, or a university network):

```bash
nc -z -w 5 musica.vie.asc.ac.at 22 && echo reachable
```

- **Reachable:** skip wireproxy (§3). The server makes the same test and takes the system's path when it answers
  (`[cluster] vpn` in the config forces `system` or `wireproxy`). Once the system's path has worked on a machine,
  the server never starts wireproxy there on its own, so a VPN reconnect or an app not yet connected does not
  bring up a second tunnel with the same key.
- **Not reachable, and the human has a WireGuard app on this machine:** ask them to connect it and test again.
- **Otherwise** (a remote dev machine or a container: a WireGuard app only helps the machine it runs on):
  wireproxy, §3.

**Never run the same WireGuard config in two places at once**: with the same key both tunnels stall. For a second
machine the human downloads a second config from wg-pub.asc.ac.at, or disconnects the other one.

## 2. Install the tools

| Tool | Mac | Linux / WSL2 |
|---|---|---|
| `uv` (runs the server) | `brew install uv` or https://docs.astral.sh/uv/ | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `step` CLI (certificates) | `brew install step` | release binary from https://github.com/smallstep/cli/releases (`step_linux_<ver>_amd64.tar.gz`, or `arm64`) into `~/.local/bin`: survives container rebuilds, no root |
| `ssh`, `ssh-agent`, `ssh-keygen`, `pgrep` | built in | usually there (`openssh-client`, `procps`) |
| GNU `tar`, `find`, `stat`, `realpath` (for `push_files`/`fetch_files`) | `brew install coreutils findutils gnu-tar` (the server finds them itself) | built in |
| `nc` (only with wireproxy) | built in | the OpenBSD variant: `sudo apt install netcat-openbsd` (ask first). In a container without root, ask the human to add `netcat-openbsd` and `procps` to the image; a rebuild can remove packages installed by hand |
| `wireproxy` (only with wireproxy) | release binary into `~/.local/bin` | release binary from https://github.com/windtf/wireproxy/releases into `~/.local/bin` |
| `git` | built in / Xcode tools | usually there |

Make sure `~/.local/bin` (on a Mac also `/opt/homebrew/bin`) is on `PATH` in the shell's rc file, then start the
agent from a new shell: the MCP server inherits the agent's `PATH`. For a desktop (GUI) agent app, register the
server with the full path to `uv`.

## 3. Network (wireproxy only)

Ask the human for the WireGuard config and save it as `~/.config/wireguard/asc.conf` (mode 600). Then create
`~/.config/wireguard/wireproxy.conf`:

```ini
WGConfig = <absolute $HOME>/.config/wireguard/asc.conf

[Socks5]
BindAddress = 127.0.0.1:1080
```

`WGConfig` needs the absolute path (`/home/<user>/…` on Linux, `/Users/<user>/…` on a Mac). `access_status` and
logins start wireproxy when it is down (`scripts/vpn.sh up|down|status` by hand).

## 4. step

```bash
step ca bootstrap --context asc --ca-url https://auth.asc.ac.at:9000 \
  --fingerprint 44b048473242281db1da57124c2b843741d6c92a8fb5d0482dec032e957f2919
```

Nothing goes into `~/.ssh`: Amadeus writes its own SSH config, `~/.local/state/amadeus/ssh_config`, on every start
from `[cluster] user` (§6) and the network path (with wireproxy it adds the `nc` proxy line), with the ASC host CA
from `amadeus/asc_known_hosts` and its own ssh-agent socket next to it (the socket lets it renew the certificate
early). `scripts/mx` and `scripts/musica_auth.sh` use the same file. To reach the cluster from a terminal yourself:
`ssh -F ~/.local/state/amadeus/ssh_config musica`.

Installs from before, with a `musica` entry in `~/.ssh/config`, keep working: when that entry exists, Amadeus uses
it (plain `ssh musica`) and writes nothing. Remove the entry to switch to Amadeus's own file.

## 5. The server

```bash
git clone https://github.com/cpietsch/amadeus /path/to/amadeus   # any directory; its absolute path below
uv sync --project /path/to/amadeus        # Python and dependencies now, so the first start is quick
```

Register it for all projects:

- **Claude Code:** `claude mcp add --scope user amadeus -- uv run --quiet --project /path/to/amadeus amadeus`
- **Codex:** `codex mcp add amadeus -- uv run --quiet --project /path/to/amadeus amadeus`, or in
  `~/.codex/config.toml`:
  ```toml
  [mcp_servers.amadeus]
  command = "uv"
  args = ["run", "--quiet", "--project", "/path/to/amadeus", "amadeus"]
  ```
- **Crush** (charmbracelet/crush, tested with v0.98.1): in `~/.config/crush/crush.json` (or a project's
  `crush.json`):
  ```json
  {
    "$schema": "https://charm.land/crush.json",
    "mcp": {
      "amadeus": {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "--quiet", "--project", "/path/to/amadeus", "amadeus"],
        "timeout": 60
      }
    }
  }
  ```
  - `timeout` covers the server start only: Crush's default of 10 s is too short for a cold `uv run`. Tool calls
    have no time limit in Crush's code, so `wait_for_job` can block (measured up to 400 s).
  - Crush passes the server's instructions to the model, names the tools `mcp_amadeus_<tool>` and reads the
    runbook (`amadeus://runbook`) through its `read_mcp_resource` tool.
  - `crush run` (headless) approves every tool call without asking, Crush's own `bash` included. For unattended
    runs drop the job-changing tools (`"disabled_tools": ["submit_job", "cancel_job", "update_job",
    "push_files"]` in the `amadeus` entry) and Crush's shell (`"options": {"disabled_tools": ["bash"]}`), or use
    the interactive TUI, which asks before each call.
  - Keep Crush's data directory (`.crush/`, its session database) out of directories you push with
    `push_files(mode="all")`, or start it with `-D <dir>`.
  - Crush needs an LLM provider of its own (an API key, `crush login`, or a local model); Amadeus needs none.

File transfers (`push_files`, `fetch_files`) work on paths inside the home directory, outside hidden directories
(`local_root` in the config, §6). On start the server plays a short jingle (Eine kleine Nachtmusik) and on errors
a short sound, where the machine has a player (`afplay`, `pw-play`, `paplay`, `aplay`, or PowerShell from WSL2);
`AMADEUS_SOUND=off` in the server's environment switches them off.

## 6. The project config

Amadeus serves one ASC project at a time. Ask the human for these (the projects portal,
https://projects.asc.ac.at, shows them; don't guess any of them):

| Ask for | Example | Config key |
|---|---|---|
| Their ASC username (the login at auth.asc.ac.at) | `user1234` | `cluster.user` |
| ASC project name | `my-gpu-project` | `project.name` |
| Slurm account: `p` and the project number | `p123456` | `project.account` |
| Filesystem: `fs` and the same number | `fs123456` | `project.filesystem` |
| Total budget in tokens, and the project's end date | `1000`, `2027-06-30` | `project.budget_tokens`, `project.ends` |
| Working directory on the cluster (code and job scripts; usually a symlink into `$DATA`) | `~/my-project` | `project.workdir` |
| Optional: a Markdown file with the project's own notes (budgets approved, what is installed, results), and directories of the project's job scripts | `/path/to/my-project/PROJECT.md`, `["/path/to/my-project/jobs"]` | `project.notes`, `project.template_dirs` |
| Optional: other approval limits than the defaults (4 nodes, 20 tokens, heads-up at 16 nodes) | | `[limits]` |

The account is `p<number>`, not the filesystem name `fs<number>`: sbatch rejects that (some templates have it
wrong). Then write the config from the example in the clone:

```bash
mkdir -p ~/.config/amadeus
cp /path/to/amadeus/config.example.toml ~/.config/amadeus/config.toml        # then fill in the values
```

Good to know:
- `AMADEUS_CONFIG` points the server at another file. `AMADEUS_*` variables in the server's environment win over
  the file; the list is at the top of `config.example.toml` (an empty value counts as unset).
- Without an account, `check_job` and `submit_job` refuse and say so; the read-only tools work (`budget` then
  shows only your own jobs).
- The tools read the file again when it changes, but the server's instructions (project line, approval limits)
  are built at start: restart the agent after a change.
- The job ledger is kept in `~/.local/state/amadeus/jobs.jsonl` (`AMADEUS_LEDGER` moves it).

If the working directory does not exist on the cluster yet, create it after the first login (§7), e.g.
`mkdir -p $DATA/my-project && ln -s $DATA/my-project ~/my-project` through `run_command`.

## 7. First login and the check

Easiest after restarting the agent (so it loads the server): call the `start_login` tool. Without the server:

```bash
/path/to/amadeus/scripts/musica_auth.sh               # prints the link and returns; the login waits in the background
```

Either gives an `https://auth.asc.ac.at/device?code=…` link: **show it to the human at once** (first line of your
reply, don't wait for anything else first); they approve it in a browser within about 60 s. Then confirm with
`scripts/musica_auth.sh --check`. If it expired, start again when the human is ready. (`--wait` only returns after
the approval, too late to pass the link on from an agent's shell.)

Then call the `access_status` tool. Done when:
- `certificate.valid` is true;
- `ssh` starts with `ok:` (the login node's name);
- there is no `missing_tools` entry (it names anything the access path cannot find, with how to install it);
- `budget` shows the project name and budget from the config.

Optionally clone the ASC docs into the clone's `docs-vsc/` and `docs-extra/` (AGENTS.md §11): the runbook points
into them. Finally read `AGENTS.md` (the runbook: rules, budget, when to ask the human). The server gives agents
the same rules as `amadeus://runbook`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `access_status` lists `missing_tools` | Install what it names (§2); after a container rebuild check `nc` and `step` first |
| `ssh` fails with `Connection timed out during banner exchange` or `port 65535` | No network path: system VPN not connected, or wireproxy config wrong (`scripts/vpn.sh status`) |
| `Permission denied (publickey,keyboard-interactive)` | No valid certificate: §7 |
| `command not found: nc` | §2 (wireproxy setups only) |
| `check_job`/`submit_job` say no account is configured | §6 |
| The tunnel works on one machine but not another | Each machine needs its own install and its own daily login (serving a second machine: AGENTS.md §10). Two machines with the same WireGuard config stall each other |
| WSL2: ssh hangs during the key exchange behind the Windows VPN | Lower the MTU (`sudo ip link set eth0 mtu 1380`) or use `networkingMode=mirrored` |
| Mac: `push_files`/`fetch_files` fail with `realpath: illegal option` or tar/find errors | `brew install coreutils findutils gnu-tar` |

Problems with the server itself: the `report_issue` tool (it writes to `issues/` in your clone: push it or tell the
maintainer), or an issue at the repository.
