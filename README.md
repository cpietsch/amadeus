# Amadeus

**An MCP server that lets coding agents (Claude Code, Codex, …) work on MUSICA, the supercomputer of ASC
(Austrian Scientific Computing) in Vienna, without hand-written `ssh` and `sbatch`.**

The agent gets tools: sign in, run light commands, draft and check job scripts, submit, wait, read logs, move
files. The server enforces the runbook as it goes: the right account and QoS, time limits, the project's budget,
and the human's approval for big or expensive jobs. It works for any ASC project; your project's details come
from a small config file.

> Amadeus is an independent project, not an ASC service. Platform facts were verified on MUSICA Vienna in
> October 2026; when the cluster disagrees with the docs, trust the cluster and report it.

## Why

An agent with plain `ssh` to a supercomputer makes the same mistakes over and over: jobs on the wrong account, a
16-node run nobody approved, a `--time` that cannot be raised once the job runs, `sbatch` repeated after a dropped
connection, a build on the shared login node, logs grepped for "Error" by hand. Amadeus turns the lessons from
real agent sessions on MUSICA into tools that get these right by default, plus a runbook ([AGENTS.md](AGENTS.md))
the agent reads.

## Install

You need an ASC account and project on MUSICA, a network path to it (the ASC WireGuard VPN or a partner-university
network), macOS, Linux or Windows through WSL2, and an MCP client. Then tell your agent, on the machine that will
reach MUSICA:

```text
Install the Amadeus MCP server for me following INSTALL.md in https://github.com/cpietsch/amadeus.
Ask me for my ASC username, project details and WireGuard config when you need them.
```

[INSTALL.md](INSTALL.md) is written for the agent. It will ask you for your ASC username, your project's Slurm
account, filesystem, budget and working directory (the projects portal shows them), your WireGuard config if
the machine has no VPN yet, and to approve a sign-in link in your browser (within about 60 s). Each machine an
agent works on gets its own install.

By hand, after cloning (`/path/to/amadeus` is wherever you put it):

```bash
claude mcp add --scope user amadeus -- uv run --quiet --project /path/to/amadeus amadeus   # Claude Code
codex mcp add amadeus -- uv run --quiet --project /path/to/amadeus amadeus                 # Codex
```

Crush and other stdio clients: [INSTALL.md §5](INSTALL.md#5-the-server).

## Agent quickstart

Connected agents get these rules from the server itself. Paste the block into your project's `CLAUDE.md` or
`AGENTS.md` so they are in view from the start, or hand it to an agent as its brief:

```markdown
## MUSICA via Amadeus

Use the `amadeus` MCP tools for all work on MUSICA, not raw ssh or sbatch.

1. Start with `access_status`. Certificate missing or under 60 min: call `start_login` and put the link on the
   first line of your reply; the human has about 60 s to approve it.
2. The login node is for light work (`run_command`: ls, du, git, small checks). Builds, downloads and compute
   go into jobs.
3. Jobs: `draft_job` or `job_templates` → `check_job(script_text=…)` → save the script in this repo →
   `push_files` → `submit_job(script=…, purpose=…)`. Climb the test ladder: dev QoS (10 min), then 1, 2 and 4
   nodes, before 16. Set `--time` from a measured estimate × 1.3: a running job's limit cannot be raised.
4. Ask the human first and pass their words as `approval`: 4 or more nodes at once, more than 20 tokens, B200,
   long or priority QoS, deleting data, cancelling jobs you did not submit. A 16-node job needs its own
   heads-up. Never invent an approval. (Defaults; the config's `[limits]` can set others.)
5. Wait with `wait_for_job`, not sleep loops. Logs: `job_log`; a failed job: `job_details` (diagnosis).
   Results come back with `fetch_files`.
6. After a dropped connection, never resubmit blindly: `list_jobs` first.
7. A tool misbehaves or one is missing: `report_issue`.

Rules and patterns: `amadeus://runbook`; platform facts: `amadeus://notes`; this project: `amadeus://project`.
```

## Tools

| Area | Tools |
|---|---|
| Access | `access_status` (tunnel, certificate, SSH test; restarts a stalled tunnel), `start_login` (the ASC device login and early renewal; certificates last about 16 h) |
| Login node | `run_command`: light commands with a timeout. Slurm-changing commands, ssh to compute nodes and config edits are refused; deletions need approval |
| Writing jobs | `draft_job` (scripts in the form MUSICA accepts: CPU or GPU, part of a node or whole nodes, multi-node with launch retry, arrays, vLLM servers with a client), `job_templates` (tested scripts), `check_job` (runbook checks, `sbatch --test-only`, missing models, containers and envs) |
| Running jobs | `submit_job`, `update_job`, `cancel_job`, `list_jobs` |
| Watching | `wait_for_job` (start, end, failure, log milestones with deadlines, quiet-log alarm, expected outputs), `job_log` (cleaned logs, progress bars), `job_details` (diagnosis: infrastructure, script or program), `node_exec` (`nvidia-smi`, `ps`, `py-spy` on a running job's node) |
| Files | `push_files`, `fetch_files` (tar over SSH, staged, with the code's git revision; never deletes remote files) |
| Budget | `budget` (tokens spent against the project's budget), `cluster_load` (idle, planned and busy nodes), `ledger_entries` (the local record of every submission) |
| Issues | `report_issue`, `list_issues` |

Resources: `amadeus://runbook` ([AGENTS.md](AGENTS.md)), `amadeus://notes` ([NOTES.md](NOTES.md)),
`amadeus://project` (your project's own notes, if the config names a file). Every tool's details:
[AGENTS.md §4a](AGENTS.md#4a-the-mcp-server-amadeus).

## What the server enforces

- **Every job:** the project's account, a partition with a matching QoS, an explicit `--time` (at most 72 h; dev
  QoS 10 min and 2 nodes), CPUs per node within what the partition gives. Known script pitfalls are refused or
  flagged ([AGENTS.md §8](AGENTS.md#8-troubleshooting)).
- **The human's approval**, quoted in their words, for 4 or more nodes at once or more than 20 tokens (running
  and pending jobs count together), the B200 partition, long or priority QoS. 16 or more nodes need an approval
  that names the node count. The thresholds are configurable.
- **Slurm's own reading:** jobs are submitted on hold, checked again as Slurm parsed them, then released (or
  cancelled). After a dropped connection the server looks for the job instead of running `sbatch` again.
- **Destructive actions** (deleting or moving data, cancelling jobs the agent did not submit, killing processes on
  a node) need approval.
- **A ledger:** every submission is recorded with its purpose, estimate and approval.

These guards catch mistakes; they are not a security boundary. The agent works with your ASC account.

## Configuration

One file, `~/.config/amadeus/config.toml`, copied from [config.example.toml](config.example.toml):

```toml
[cluster]
user = "user1234"              # your ASC username

[project]
name = "my-project-2026"
account = "p123456"            # Slurm account: p<number>, not the filesystem fs<number>
filesystem = "fs123456"
budget_tokens = 1000
ends = "2026-12-31"
workdir = "~/my-project"       # code and job scripts on the cluster
```

Optional: `notes` and `template_dirs` (your project's notes and job scripts), `[limits]` (approval thresholds),
`[files]` (where `push_files`/`fetch_files` may read and write, other machines). `AMADEUS_*` environment
variables override the file ([INSTALL.md §6](INSTALL.md#6-the-project-config)). Without an account the read-only
tools work and submissions are refused. Restart the agent after a change.

## Without an agent

The shell helpers behind the server work on their own:

```bash
scripts/musica_auth.sh                  # tunnel + daily certificate; approve the printed link
scripts/mx 'squeue -u $USER'            # a command on the login node (timeout, batch mode)
scripts/mx 'cd <workdir> && sbatch jobs/test_gpu.slurm'
```

None of the server's checks apply there. Copying code by hand: [AGENTS.md §4](AGENTS.md#4-running-commands-and-moving-files).

## Docs

| File | What |
|---|---|
| [INSTALL.md](INSTALL.md) | Setup on a new machine (Mac, Linux, containers and dev VMs, WSL2), written for the agent |
| [AGENTS.md](AGENTS.md) | The runbook: rules, when to ask the human, job patterns, costs, troubleshooting |
| [NOTES.md](NOTES.md) | MUSICA reference: projects, hardware, Slurm, storage, software |
| [config.example.toml](config.example.toml) | The project config with placeholders |
| [issues/README.md](issues/README.md) | The issue channel and how fixes are made |

The runbook points into ASC's own docs; clone them next to the server ([AGENTS.md §11](AGENTS.md#11-where-to-look-things-up)).

## Development

```bash
uv run python -m unittest discover -s tests -t .     # against a fake cluster; nothing reaches MUSICA
```

```
amadeus/     the server: server.py (tools), guard.py (runbook checks), slurm.py (options, token estimates),
             drafts.py, templates.py (draft_job, job_templates), auth.py, network.py, remote.py, files.py,
             logs.py (cleaning, diagnosis), ledger.py, issues.py, config.py
jobs/        tested generic job scripts: smoke tests, GPU node probe, container pull, vLLM on N nodes
scripts/     shell helpers: musica_auth.sh, mx, vpn.sh
tests/       tests with a fake SSH shim and stub Slurm commands
issues/      problems filed by agents (report_issue) or automatically; one Markdown file each
```

Issues and pull requests are welcome. Keep it general: no values of a particular project (account, paths,
budgets, results) in code, docs or `jobs/`; examples use placeholders (`p123456`, `fs123456`). A lesson learned in
a project goes into AGENTS.md as the general rule. `report_issue` writes into `issues/` of the local clone: open
an issue or pull request here to send it upstream. [CLAUDE.md](CLAUDE.md) sets up a maintainer session.

On start the server shows a banner and plays a short jingle, on errors a short sound; `AMADEUS_SOUND=off` turns
them off.

## Credentials

Never commit or paste credentials: the WireGuard config, private keys, certificates, Hugging Face tokens. They
live in `~/.config/wireguard/`, `~/.ssh/` and `~/.step/`. The sign-in link goes only to the human. Issues are
redacted when they are written.

## History

Amadeus grew out of a hackathon workspace: from 2026-10-07 to 2026-10-16 the repository was team Socratech's
workspace for the European AI Hackathon on MUSICA, where a sweep of the ASC docs became the runbook and NOTES.md,
shell helpers became tools, and the benchmark and labelling jobs became the tested patterns behind `draft_job`.
On 2026-10-14 it was turned into an MCP server, on 2026-10-15 a review of past agent sessions added guard rules
and tools, and on 2026-10-16 it was named Amadeus and made general for any ASC project.

## License

MIT, see [LICENSE](LICENSE).
