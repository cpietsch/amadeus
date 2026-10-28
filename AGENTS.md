# Operating MUSICA with coding agents

Runbook for **AI coding agents** (Claude Code and similar) and the people supervising them. It covers how an
agent reaches the ASC MUSICA supercomputer, what it may do without asking, and the patterns that work. It
belongs to **Amadeus**, the MCP server in this repository (github.com/cpietsch/amadeus,
[README.md](README.md)), which serves it to every agent as `amadeus://runbook`; the server's tools build in the
rules below. It holds for any ASC project on MUSICA.

- Facts about **your project** (name, Slurm account, filesystem, budget, working directory, approval limits) come
  from the server's config, `~/.config/amadeus/config.toml` ([INSTALL.md](INSTALL.md) §6). Below they appear as
  `<account>` (`p<number>`), `<filesystem>` (`fs<number>`) and `<workdir>`. The project's own notes (budgets
  the human approved, installed models and envs, results) are served as `amadeus://project` when the config names
  them (`notes`).
- Platform facts (hardware, queues, storage, software) are in [NOTES.md](NOTES.md) (`amadeus://notes`).
- The official ASC docs can be cloned into `docs-vsc/`, and ASC's example repos into `docs-extra/`
  (see [§11](#11-where-to-look-things-up)). **Check them before guessing or debugging platform behaviour.**
- **Agents use Amadeus** (registered as `amadeus`, package `amadeus/`, §4a); the scripts in `scripts/` remain as
  fallback and for humans.
- Facts verified on MUSICA Vienna between 2026-10-07 and 2026-10-16, in the jobs of one ASC project.

---

## 0. Rules for agents

1. **Use Amadeus's tools** (the `amadeus` MCP server) when the session has them (§4a): `access_status`,
   `run_command`, `check_job`/`submit_job`, `wait_for_job`, `push_files`/`fetch_files` and the rest.
   They build in these rules (approval limits, account and QoS checks, the job ledger). Without the
   server, run commands with `scripts/mx '<cmd>'`, not raw `ssh`: it adds a timeout and batch mode.
   All work stays non-interactive: no `ssh -t`, editors, `salloc` shells or prompts.
2. **The login node is for light work only**: editing, copying, `sbatch`, `squeue`,
   small checks. ASC limits it to 4 cores and about 20 to 24 GB per user, and kills
   abusive processes. Anything compute-, memory- or I/O-heavy, including container
   builds and large downloads, **goes through `sbatch`**.
3. **Every job sets `--account=<account>`, a partition, a matching QoS and a `--time` from a measured
   estimate plus margin**: a running job's limit cannot be raised (§5 Patterns). Test with the `dev_*`
   QoS (10 min) first. Never run production work on dev QoS.
4. **Watch the budget** (`budget_tokens` in the config; the `budget` tool). A full H100 node costs 4 tokens/h,
   one GPU 1 token/h, a CPU node 0.96/h. Get human approval before anything over about **20 tokens** (for
   example more than 5 H100 node-hours), before any job with **4 or more nodes**, and before using the private
   B200 partition (see §9). **A 16-node job needs its own heads-up even inside an approved budget** (node count,
   maximum tokens, expected wall time), and only after a 4-node run of the same script has completed. These are
   the defaults; the config's `[limits]` can set others, and `submit_job` enforces what it says.
5. **Code goes in the project's working directory on the cluster** (`workdir` in the config, for example
   `~/<project>` as a symlink into `$DATA`). **Big files go in `$DATA` or `$SCRATCH`**: models, containers,
   datasets, caches, venvs. `$HOME` is 50 GB. Nothing on MUSICA is backed up.
6. **Don't edit `~/.bashrc` or `~/.bashrc.d/*` on the cluster.** ASC warns this breaks
   jobs. Put environment setup in job scripts.
7. **Never print, copy or commit secrets**: `~/.config/wireguard/asc.conf`, private keys,
   step certificates. Pass the sign-in link (`auth.asc.ac.at/device?code=…`) **only to
   the human**.
8. **When SSH fails, diagnose before retrying.** See [§8](#8-troubleshooting). Never blindly
   re-run `sbatch` after a dropped connection: check `squeue` first, because the job may
   already have been submitted.
9. **Don't use `pkill -f`/`pgrep -f` with a pattern that also appears in your own
   command line.** It kills the agent's own shell (exit 144). Anchor the pattern
   (`'^ssh -o …'`) or use PIDs.
10. **Wait in the background, not the foreground.** Use `wait_for_job`, or without the server a background task
    that polls `sacct -j <id> -X -n -o State` and gets you notified, instead of chaining `sleep`s.
11. **Jobs running at the same time count together** for the approval limits in rule 4: five
    1-node jobs at once are 5 nodes. Stagger them or ask first.
12. **The agent's shell may be zsh.** There `for x in $LIST` does not split words (nor does a variable
    holding a command), so loops silently run once; an unmatched glob aborts the command (`no matches
    found`: quote `--include='*.py'` and URLs with `?`); `echo ===` fails (quote `==`). Use arrays or
    `bash -c`. A background task's exit 0 does not prove a transfer worked: check counts or sizes. MCP
    tool arguments skip the local shell: `run_command` gets the command exactly as bash on the login node
    should read it, so add no extra escaping.
13. **Estimate before you start:** items × cost per item, or steps × s/step from a measured probe (§5
    Costs). Long CPU work (more than about 15 min on the agent's machine) is usually cheaper and faster on
    MUSICA (a CPU array; `dev_zen4_0768` for small jobs), within the approval limits (§5 Where work runs);
    the project's notes may say whether to send it without asking. Take node facts from NOTES.md (CPU node:
    190 usable cores; GPU node: 176), not from memory.

---

## 1. Project facts

Every ASC project has its own name, Slurm account, filesystem, budget and end date. Amadeus reads them from the
config (`[project]` in `~/.config/amadeus/config.toml`, INSTALL.md §6); `budget` and the server's instructions
show them. What holds for every project:

| Item | Value |
|---|---|
| Slurm account | **`p<number>`** (`project.account`). The filesystem name `fs<number>` is *rejected* as an account, even where a template uses it |
| Filesystem | `fs<number>` (`project.filesystem`): `$DATA=/data/<filesystem>/<user>`, `$SCRATCH=/scratch/<filesystem>/<user>` |
| QoS | `zen4_0768_h100x4` (72 h), `dev_zen4_0768_h100x4` (10 min, ≤2 nodes per user at once, across jobs), `zen4_0768`, `dev_zen4_0768` (same limits), `zen5_2304_b200x8` (guests, when idle). Other QoS (`long_*`, `fast_*`, `idle_*`) exist but are not assigned by default |
| Prices | in tokens (1 token ≈ €1, price sheet v1.4): H100 node 4.00/h, CPU node 0.96/h, partial nodes pro rata (1 GPU = 1/h), high priority double. B200 has no published price. When the budget runs out, jobs only run in the idle QoS (practically unusable) |
| Lifecycle | after the project's end date there is a grace period for submitting jobs, then a login deadline, then the data is deleted: the dates are on the projects portal (`docs-vsc/docs/musica/project_lifecycle.md`) |
| Storage | `$DATA` 10 TB, `$SCRATCH` 5 TB per project by default, `$HOME` 50 GB |
| Default site | Vienna, `musica.vie.asc.ac.at` (login01 = .21, login02 = .22) |

---

## 2. How access works

```
 agent machine                                           ASC Vienna
 ─────────────                                           ──────────
 ssh musica ── ProxyCommand nc -X 5 ──► wireproxy ══ WireGuard/UDP ══► wg-pub.asc.ac.at ──► login01/02
     │                      127.0.0.1:1080   (userspace)                                       :22
     └── IdentityAgent (own agent) ───────── step-ssh certificate (valid ~16 h, signed by ASC CA)
```

With a system VPN (WireGuard app, university network) the `ProxyCommand` and wireproxy drop out (INSTALL.md §1):
Amadeus takes the system's path when the login node answers directly, else wireproxy if configured.

| Piece | Where | Notes |
|---|---|---|
| Network | `~/.local/bin/wireproxy` + `~/.config/wireguard/{asc.conf,wireproxy.conf}`, or a system VPN | MUSICA only accepts partner-university IPs or ASC WireGuard. A container without `NET_ADMIN`/TUN can't run `wg-quick`; wireproxy runs WireGuard in userspace and exposes a SOCKS5 proxy. The tunnel routes Vienna and Linz only (`193.170.79.0/25`), **not Innsbruck** |
| Identity | `step` CLI (context `asc`) → cert in Amadeus's own ssh-agent (`~/.local/state/amadeus/agent.sock`) | The login nodes **only accept ASC-signed certificates.** Plain keys are rejected, whether registered at auth.asc.ac.at or in `authorized_keys` (tested on port 22; port 27 is closed) |
| SSH config | Amadeus's own, `~/.local/state/amadeus/ssh_config` (written from the config; `ssh -F` it by hand), hosts `musica`, `musica-lnz`, `musica-inn`; an own `musica` entry in `~/.ssh/config` wins | `ControlMaster no` on purpose (§8). `ServerAliveInterval 15`, `CountMax 4`, `ConnectTimeout 20` |
| Host trust | `@cert-authority *.asc.ac.at …` in `amadeus/asc_known_hosts` (the only known_hosts that config reads) | From the ASC docs; avoids host-key prompts |
| Compute-node hops | none that work | `ssh <node>` from the login node is **refused** even with your job on that node (`Permission denied (publickey,gssapi-keyex,gssapi-with-mic)`, 2026-10-08 and 2026-10-13), probably because the private half of the key in the cluster's `authorized_keys` is not on the cluster. Use `node_exec` or `srun --jobid=<id> --overlap` (§5 Monitoring) |

---

## 3. Start-of-session checklist

```bash
scripts/musica_auth.sh          # tunnel up? agent up? certificate valid? (--check: expiry only, no login)
scripts/mx 'hostname; date'     # smoke test
scripts/mx 'squeue -u $USER'    # what's running/pending
```

With Amadeus: `access_status` (tunnel, certificate, SSH test), then `list_jobs`.

**Signing in** (certificate missing, expired or under 60 min left; about once a day). The device link
expires after about 60 s (measured 2026-10-08: two codes expired unapproved; step logs `device
authorization grant expired`). A relayed link is dead unless the human is at a browser right then; when
they were, relayed links were approved within a minute. So:

1. When a message **from the human** arrives (not a cron, `/loop` or stop-hook prompt: those don't show
   that anyone is there) and the certificate is expired or under 60 min, call `start_login` (without the
   server: `scripts/musica_auth.sh --wait` as a background task) and put the link on the first line of
   the reply.
2. If it expires unapproved, don't loop or generate new codes while nobody is there: start a new one only
   when the human writes again. Each attempt is a fresh `step ssh login`.
3. Optional for the human: `! scripts/musica_auth.sh --wait` in Claude Code shows the link in their own
   terminal (a `!` line pasted as chat text does not run).

> Do **not** use the device flow built into plain `ssh` (QR code plus "press Enter").
> It needs Enter within about 60 s of approval, which an agent relaying through chat
> can't reliably manage.

**Plan renewals ahead.** The certificate lasts about 16 h, and the human may be asleep when it runs out.
Running jobs are not affected (only access is), but nothing can be submitted or fetched until it's
renewed (an overnight run easily outlasts it). The server adds
`certificate` to tool results when under 60 min are left (it says to start a login only when a message from the
human is being answered, as in step 1), and `certificate_warning` (`submit_job`,
`wait_for_job`, `list_jobs`) when your jobs end after the expiry: then, before the human goes offline,
offer an early renewal. `start_login` renews while the
certificate is still valid (by hand: `scripts/musica_auth.sh --renew` in the background, pass the link at
once; it logs in through a second agent and swaps the new certificate in after approval, giving a fresh
16 h). `scripts/musica_auth.sh --check` prints the expiry
without a login and exits 1 below 60 min. No reminder cron is needed; if the human wants one, run
`CronList` first (duplicate crons have run for hours) and delete it when the human says stop.

---

## 4. Running commands and moving files

```bash
scripts/mx 'squeue -u $USER'                       # quote the remote command once
MX_TIMEOUT=300 scripts/mx 'du -sh $DATA/*'        # longer timeout (default 120 s)
MX_LMOD=1 scripts/mx 'module -t avail CUDA'       # with Lmod initialised (see §6)
```

- Exit codes: the remote command's own code, `124` = timed out, `255` = SSH failed.
- `$VARS` inside single quotes expand **on the cluster** (`$USER`, `$DATA`, `$SCRATCH`).
- Long output: pipe through `tail` or `grep` on the remote side to save context.

**Files.** With Amadeus: `push_files` and `fetch_files` (§4a). The local project is the source of truth for
code: don't edit code directly on the cluster (the next push overwrites it); results flow back with
`fetch_files`. By hand, keep the cluster copy in `$DATA/<repo>` without git, push the tracked files with tar
over `mx`, and write the source revision into a `REVISION` file first, so every job can log what code it ran:

```bash
git rev-parse --short HEAD > REVISION     # add -dirty by hand if needed
{ git ls-files -z; printf 'REVISION\0'; } | tar czf - --null -T - | MX_TIMEOUT=300 scripts/mx 'mkdir -p $DATA/<repo> && cd $DATA/<repo> && tar xzf -'
MX_TIMEOUT=300 scripts/mx 'cd $DATA/<repo> && tar czf - results/<dir>' | tar xzf -     # fetch
scp -F ~/.local/state/amadeus/ssh_config musica:<workdir>/logs/x.out logs/              # single files (-F: §2)
```

Only tracked files travel this way: a new file caught by `.gitignore` (a pattern like `worlds/` also
matches `pkg/worlds/`) silently stays behind. SSH sometimes times out during the banner
exchange, so scripted transfers should retry (3 attempts, 20 s apart worked).

Checkpoints fetched from MUSICA carry absolute cluster paths: `base_model_name_or_path` in
`adapter_config.json` (and project configs) points at `/data/<filesystem>/<user>/models/…`. Rewrite it with
`sed` to the local Hugging Face snapshot (check that it is the same revision).

---

## 4a. The MCP server (`amadeus/`)

Agents reach MUSICA through **Amadeus**, the MCP server in this repository (registered as `amadeus`, package
and console script `amadeus`, stdio only). It runs on the machine that has the
network path and the certificate and wraps the same SSH path as `scripts/mx`. Each person runs their own, with
their own config (setup: [INSTALL.md](INSTALL.md), §10).

| Tool | What |
|---|---|
| `access_status` | tunnel, certificate expiry, SSH test. Call first. Starts wireproxy if it is down. With wireproxy, an SSH failure before login (banner exchange, port 65535 timeout; the command did not start: Amadeus has the remote shell print a start mark first, so an exit 255 of a command that ran never counts) while the certificate is valid restarts the tunnel (by PID, at most once a minute across calls and server processes) in every tool that runs commands on the login node; commands safe to repeat then run once more at once, and the result's `recovery` says so (§8). Not yet in the copying of `push_files`/`fetch_files`: a stall there is retried after 20 s without a restart |
| `start_login` | starts a login or an early renewal and returns the device link: show it to the human at once (about 60 s); one login at a time. Only when answering a message from the human, never from a cron, `/loop` or stop-hook turn (§3) |
| `run_command` | a light command on the login node (timeout, optional Lmod). Refused: interactive commands, Slurm-changing commands (`sbatch`, `scancel`, `scontrol update/hold/release`, `srun`, `salloc`: use the tools; `sacctmgr` except its read-only `show`/`list` forms: account administration is the human's), ssh to compute nodes (use `node_exec`) and changes to SSH, shell or VPN config. Deleting or moving files needs `approval`. Not repeated after an SSH failure: when `recovery` says the tunnel was restarted after a stall, nothing ran remotely, so run it again. A guard against mistakes, not a security boundary |
| `list_jobs` | my running and pending jobs (squeue) with their recorded purpose, and what they hold together (nodes, tokens they can still spend) |
| `job_details` | one job: state, cost, pending reason and cleaned log tails (all array tasks and requeues). An ended job that failed (or completed with errors in its log) gets a `diagnosis`: category (infrastructure = resubmit, script, program, benign), known signatures with meaning and action, the first real exceptions; harmless noise is not counted. A Slurm launch notice (`Prolog hung`, socket timeouts) decides the category only when the job failed fast with nothing more specific in the log; a cancelled job is benign |
| `job_log` | log tail, cleaned (ANSI, binary bytes, `\r` progress updates collapsed; `raw=True` as written), `grep`, `file=` for per-run logs (globs, `%j %x %A %a`). `progress=True`: the latest state of each progress bar, with kind loading / warmup / work (`Loading weights 723/723` is not training). `hide_benign=True`: drops known-harmless lines and tracebacks, counted in `benign_hidden`. A missing log in a job's first minutes is the node prolog |
| `wait_for_job` | waits for one job or several (`job_ids`; `until` = end, any_end, any_fail, start) up to 1 h per call and returns at the first event: STARTED, ENDED, FAILED (with diagnosis), MILESTONE / LATE (`milestones=[{pattern, within_s, file}]`), QUIET (`quiet_alarm_s`: logs not grown; 300 for big vLLM runs), OUTPUTS_MISSING (`expect`: output paths). Pass `next_cursor` back as `log_cursor` to get only new lines. State lives in Slurm: after a restart just call it again (alarms this server already returned come back as info and do not end the call). At most 8 running jobs or tasks are watched per call; others are listed in `unwatched`. Milestones are grepped on the cleaned log (as `job_log` shows it). No sleep/sacct watcher loops |
| `node_exec` | a short command on the node of one of your RUNNING jobs (`srun --jobid --overlap`, since compute-node ssh is refused): `nvidia-smi`, `top -b -n1`, `ps`, `free -g`, `py-spy dump`; `node` picks one of the job's nodes; 5–120 s; no background processes or `srun`; signals (`kill`, `pkill`, `fuser -k`, also behind `timeout`, `nice`, `env`, `xargs`, `bash -lc`) and deletions need `approval`. The check goes by command word: killing processes by any other means (e.g. `os.kill` in `python -c`) needs the human's approval too |
| `cluster_load`, `budget` | nodes per partition (allocated, idle, `planned`, other, total, and `states`: the count per raw Slurm state; `idle` is free now, `planned` nodes are held by the backfill scheduler for jobs next in line and are not free, although plain `sinfo` `%A`/`%F` counts them as idle; reserved, maintenance and not-responding nodes are other), pending jobs, my fair-share and priority factors; tokens per user since a date, against the project's budget from the config (without an account: your own jobs only) |
| `draft_job` | a job script in the form MUSICA accepts, from the patterns below (§5, §6, §8): part of a CPU node, whole CPU nodes, 1–3 GPUs, whole GPU nodes, several nodes (one `srun` step over all nodes, one task per node, under `set -e`; the whole step is launched again if it fails within 150 s, so make the work safe to restart), vLLM servers on every node with a client (1, 2 or 4 servers per node; `vllm_preset` for tested serve settings of some models (`qwen38-fp8`, `glm53-fp8`), reading the model from `$DATA/models/<repo name>` and the image from `$DATA/containers/` (download them first) unless `vllm_model`/`vllm_image` say otherwise; a server not loaded after `load_cap_s`, default 420, is dropped), arrays (`array_args`: one argument string per task as `$ARGS`, own `$OUT`, throttled under the approval limit), `runs`: up to 8 commands in parallel on one whole node (run r on GPU r % 4, own caches and log; `run_minutes` checked against `--time`), a venv, a container or modules. Every draft checks at job start that its env, container and model exist (exit 2 otherwise); GPU drafts set `HF_HUB_OFFLINE=1`, vLLM servers `OMP_NUM_THREADS=1`. The caller supplies the commands; the result includes the static check (cost, approvals). Nothing goes to the cluster |
| `job_templates` | tested job scripts: this repository's generic `jobs/` (§5) plus the directories in the config's `template_dirs` (`AMADEUS_TEMPLATE_DIRS` overrides), typically the `jobs/` of the projects that use the cluster; a summary each and `-dirty` on files that differ from the commit; with a name, the script and the helper scripts it sources |
| `check_job`, `submit_job` | runbook checks plus `sbatch --test-only` (`check_job(script_text=…)` checks a draft without pushing it); submission with the checks enforced. `script` is a path relative to `cwd` on the cluster: script text passed there is refused before any SSH call (check it with `script_text`, then save it in the project, `push_files`, and submit its path). Also: `missing_paths` (models, containers, envs, inputs the script reads, tested on the cluster; `submit_job` refuses them unless the job has a `--dependency`, or `allow_missing=True` for a job that fetches or creates them itself), `asc` (what ASC's comfort mode rewrote, e.g. `--cpus-per-task 8 -> 1`), a cold vLLM compile cache on the dev QoS, and `expected_minutes` (a `--time` far above it is flagged). Both refuse to run without an account in the config (read-only tools work) |
| `update_job` | new time limit or node count, hold or release; re-checked against the approval limits. Raising a running job's time limit is refused before `scontrol` (MUSICA denies it; raising works only while pending), with the options (checkpoint and resubmit, cancel and resubmit). Lowering is passed on to Slurm; for a running job not yet observed on MUSICA |
| `cancel_job` | jobs submitted through the server directly; others only with the human's approval |
| `push_files`, `fetch_files` | tar over SSH with retries: a code directory (git-tracked files plus a `REVISION` file, or all files) to `$DATA`/`$SCRATCH`/`~`, staged so a failed push changes nothing; results back. Never deletes remote files. Local paths must be inside `files.local_root` (default the home directory; on other hosts their home directory), outside hidden directories (`.ssh`, `.config`, `.git`, …). `host` names the machine the local paths are on: one of `files.hosts`, reached over SSH (§10), default `files.default_host`; `''` is the machine the server runs on. `push_files` warns (does not refuse) when your pending or running jobs work in or around the target: pending jobs run the pushed code, running ones may read changed files; push long chains to a per-revision directory. It also lists source files caught by `.gitignore` |
| `ledger_entries` | the local record of submissions and cancellations (`~/.local/state/amadeus/jobs.jsonl`; `AMADEUS_LEDGER` moves it) |
| `report_issue`, `list_issues` | file a problem with the server or the cluster interface (bug, wrong result, missing tool), and see the status of filed issues |

Resources: `amadeus://runbook` (this file), `amadeus://notes` (NOTES.md), `amadeus://project` (the project's
notes, when the config names a `notes` file).

**What `submit_job` enforces** (refused, with the reason, otherwise):
- the account from the config (added if missing), a partition and a matching QoS, an explicit `--time` (≤ 72 h),
  dev QoS ≤ 10 min and ≤ 2 nodes, CPUs per node within what the partition gives Slurm (176 physical
  cores on GPU nodes, 190 on CPU nodes, with `--threads-per-core=1`);
- script pitfalls from §8: no shebang, `set -u` before the Lmod init, a `bash -n` syntax error (extglob on),
  `$VARS` or `~` in `--output`/`--error`/`--chdir`, `--export` setting `DATA`/`SCRATCH`/`HOME`/`USER`/
  `TMPDIR`/`SLURM_*`, `--mem*` (or `--nodes` with 1–3 GPUs) on the GPU partition in comfort mode,
  `--gpu-memory-utilization` > 0.975 on H100, an array index ≥ 100000;
- warnings (not refused): `RC=$?` under `set -e`, `#ASC --vanilla` not on line 2, `-n N` in CPU comfort
  mode (the batch script gets 1 CPU), task/CPU options on whole GPU nodes, several vLLM servers without
  `OMP_NUM_THREADS`, vLLM on all interfaces without `--api-key`, a bare `wait` with a server in the
  background, `curl` loops without `-m`, caches under `$HOME`, `apptainer pull` without a local
  `APPTAINER_TMPDIR`, ≥ 4 nodes without an `srun` retry, `declare -A` without `=()` under `set -u`,
  `hf download --include A B`, a `--time` far above the stated expected runtime;
- known bad GPU nodes (NOTES.md, Hardware) excluded unless the job sets `--exclude` or `--nodelist`; log
  directories created;
- **human approval** (`approval`, in the human's words) for ≥ 4 nodes (counting the user's running and
  pending jobs, and array tasks at once), more than 20 tokens at the time limit, the B200 partition,
  and long or priority QoS; a job of **16 or more nodes** needs its own heads-up and an approval that
  names the node count (`16 nodes`, `16-node`, `-N16`): a budget approval does not cover it, nor do budget
  wordings such as `16 node-hours`, `16 nodes x 1h`, or a condition like `heads-up before 16 nodes`.
  The thresholds are the config's `[limits]` (`approval_nodes`, `approval_tokens`, `headsup_nodes`; defaults
  as given);
- the job is submitted **on hold**, read back with `scontrol show job`, checked again as Slurm reads it
  (nodes, array size, time, partition, QoS), then released; if Slurm's reading breaks a rule or needs an
  approval that was not given, the job is cancelled. Option spellings the static check misses cannot
  get around the gates;
- the user's running and pending jobs count together: nodes at once and the tokens they can still spend
  at their time limits (both against the node and token limits); submissions are serialised;
- a `purpose` line for every job, kept in the ledger with the estimate and the approval;
- after a dropped connection during `sbatch` it looks for a new job with the same name for 60 s and adopts
  it (then checks and releases it) instead of retrying; with several candidates it stops and lists them.

**Issue channel.** Agents that hit a problem with the server call `report_issue` (what happened, what they
expected, how to reproduce, their project); unexpected exceptions inside tools are filed automatically, once
per fingerprint, with arguments and traceback (expected failures such as SSH down are not). Each issue is a
Markdown file in `issues/`, secrets redacted. A maintainer Claude session in this repository works through
them (`/loop 30m work through the open issues in issues/ following issues/README.md`): claim, reproduce
with a test against the fake cluster, fix, close with the commit (sessions pick the fix up when they next start
the server). The protocol,
the file format and what goes to the human (`needs-human`) are in [issues/README.md](issues/README.md).
`AMADEUS_ISSUES` moves the directory; `off` switches the channel off.

**Connecting a client.**
- On the machine with the network path: clone `github.com/cpietsch/amadeus` (any directory, here `/path/to/amadeus`)
  and register it at user scope:
  `claude mcp add --scope user amadeus -- uv run --quiet --project /path/to/amadeus amadeus`.
  The repo's `.mcp.json` does the same for sessions started in this directory.
- Other clients over stdio work too: Codex, and Crush (tested with v0.98.1; config and its headless-mode caveats in
  INSTALL.md §5). Some clients prefix the tool names (Crush: `mcp_amadeus_access_status`); the rules here use the
  bare names.
- Every machine an agent works on gets its own install (INSTALL.md); there is no shared server.
- Tests: `uv run python -m unittest discover -s tests -t .` (a fake cluster: `tests/fake_ssh.sh`,
  stub Slurm commands in `tests/fakebin/`).

---

## 5. Jobs

### Workflow

With Amadeus: `draft_job` (or a script from `job_templates`) → `check_job(script_text=…)` → save it in the
project → `push_files` → `submit_job` (with `purpose`, and `approval` when needed) → `wait_for_job` →
`fetch_files`. By hand (fallback; the server's checks do not apply): push the code (§4), then

```bash
scripts/mx 'cd <workdir> && sbatch jobs/<script>.slurm'    # prints "Submitted batch job N"
scripts/mx 'sacct -j N -X -n -o State,Elapsed'           # poll from a BACKGROUND task; seff N at the end
```

and fetch logs and results back (§4).

### Job script skeleton

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=<account>
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=dev_zen4_0768_h100x4      # dev first; then zen4_0768_h100x4
#SBATCH --gres=gpu:4                    # comfort mode, see below
#SBATCH --nodes=1
#SBATCH --threads-per-core=1            # ASC recommendation on GPU nodes
#SBATCH --time=00:10:00                 # estimate x 1.3 (§5 Costs); cannot be raised once running; billing is for time used
#SBATCH --exclude=n3010-019,n3010-027,n3012-034   # known bad nodes (NOTES.md, Hardware)

source /cvmfs/software.eessi.io/versions/${EESSI_VERSION:-2025.06}/init/lmod/sh >/dev/null 2>&1
module use /cvmfs/software.asc.ac.at/init/modules
set -euo pipefail          # only AFTER the Lmod init (it reads unset variables)
export HF_HOME=$DATA/hf_cache APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
export APPTAINER_TMPDIR=/local/$USER/apptainer_tmp   # GPU nodes; unpacking on Weka took 30 min
```

Tested generic templates are in `jobs/` (also through `job_templates`): `test_gpu.slurm` (1 GPU smoke test),
`full_node_4gpu.slurm`, `probe_gpu_node.slurm` (internet, IB interfaces, `/local`, Apptainer, GPUs),
`pull_container.slurm`, `mcp_smoke.slurm` (the server's job path in seconds), `vllm_multinode.slurm` (vLLM on
N whole nodes: one replica per node behind vllm-router, or one DP × EP instance over all nodes; the model and
its flags are variables), helper `lib/vllm_node.sh` (one server per node). Model-specific scripts belong in the
project's own repository; list its `jobs/` in `template_dirs` so `job_templates` finds them.

### Patterns that work

- **Where work runs:** if the human has a local GPU workstation, use it for unit tests, builds and GPU smoke
  tests that fit there (e.g. a small vLLM model at a low `--gpu-memory-utilization`); it costs nothing and
  saves container setups on the cluster. MUSICA for long CPU work (rule 13) and all real GPU work, once the
  local checks pass. When moving a workload, rerun one baseline on the cluster and compare it paired with the
  workstation's result.
- **Test ladder before scale:** local tests (GPU parts on a workstation first), then a dry run of the
  whole job script with stand-ins for `srun`/`apptainer`/`vllm` (a syntax check misses a `set -u` abort),
  `sbatch --test-only` at every node count you plan to use, a dev-QoS smoke on 1–2 nodes,
  one real node, then 2 and 4 before 16. Also: before a GPU sweep over data subsets, check the subsets'
  composition with a quick CPU script ("first N records" subsets can hold only one kind of item and void
  the runs); a multi-node smoke must send more requests than nodes × per-server concurrency and check the
  per-server counts (a 2-node smoke once sent all requests to server 0).
- **Whole nodes, not single GPUs:** 1-GPU jobs waited hours for a `SharedNode` (estimates up to 9 h),
  full-node jobs started at once. Pack the work instead: four runs per node with
  `CUDA_VISIBLE_DEVICES=$i`, each with its own cache directory so parallel compiles don't collide, or one
  run with DDP (`torchrun --standalone --nproc_per_node=4`; about 2× the single-GPU rate for a 4B LoRA).
- **CPU work:** the CPU partition is often full (one user once held 40 of 69 nodes), `dev_zen4_0768`
  starts in minutes. In a pinch, the CPUs of a GPU node do it too (4 tokens/h instead of 0.96, but they
  start). On a whole CPU node, 188 worker processes with `OMP_NUM_THREADS=1` worked. On GPU nodes only 176
  physical cores are available to Slurm, so ask for `--gres=gpu:4 --cpus-per-task=176` (with
  `--threads-per-core=1`) and run 174 workers; 190 is rejected there (§8). With low fair-share, CPU jobs
  can wait a day (17–19 h on `Priority` with no start estimate while GPU nodes were idle); moving them to
  GPU nodes needs the human's approval of the higher rate.
- **Multi-node launch with retry** (one `srun` over all nodes): launches on 8–16 nodes fail now and then
  with `Prolog hung on node …` and `Socket timed out on send/recv operation`. Retry the step in the batch
  script, but only if it failed fast (within 150 s), wait 30 s, and time only the successful attempt. Under
  `set -e`, capture the exit code in the same statement, `srun … && RC=0 || RC=$?`; a bare `srun …; RC=$?`
  ends the script at the first failure and never retries. A failed launch is billed for its minutes. A retry
  can wait about 100 s for the prolog, so report the compute time (slowest task) next to the job time.
- **Size multi-node jobs to the fixed costs:** each process pays a fixed cost before useful work: launch
  0.5–2.5 min (most on 16 nodes), model load (tens of seconds), warm-up on the first batches, start-up of a
  simulation shard. With a few minutes of work per node, those dominate (efficiency per node fell to
  33–69% on 8–16 nodes). Give each node at least several minutes of work, or run independent single-node
  jobs for throughput. A fit of fixed cost + cost per item on 1–8 nodes predicts the 16-node run well
  (within 4% once). Tokenise or preprocess once (rank 0 or a separate CPU job, into `$SCRATCH`) and let the
  other ranks wait: done in every DDP rank it can cost a quarter of an hour per run.
- **Many vLLM replicas** (one per node, HTTP only): launch the per-node `srun` steps 0.5 s apart (16 at
  once timed out on two nodes: `Socket timed out on send/recv operation`) and relaunch a failed step once;
  serve on `0.0.0.0` (with `--api-key`) and reach servers by hostname (Ethernet); with several servers per
  node export `OMP_NUM_THREADS=1` for each; health-check all servers in parallel (`curl -m 5`) and drop one
  that is unreachable 120 s after `Application startup complete` or not up after 900 s; feed them from
  **one shared queue** in the client, not a static split, and send requests a failed server held to another
  one. With that, weak scaling was 100% per node at 1, 2, 4 and 16 nodes. Servers listen on the cluster
  network, so use a random per-job `--api-key` (never printed).
- **Respect the time limit:** read the job's end from `scontrol show job $SLURM_JOB_ID` (`EndTime=`) and
  stop clients 60 s before it, writing results as they arrive.
- **Chained jobs** (`--dependency=afterok`): make the first job exit non-zero when its output is incomplete
  (e.g. under 99.9% of records answered), or the next one runs on partial data.
- **Time limits are final once a job runs.** Raising a running job's limit is denied (`Access/permission
  denied for job N`). Only a pending job's limit can be raised (`update_job`); lowering a running job's
  limit is allowed by the tool but not yet observed on MUSICA. Set `--time` from a measured
  estimate (§5 Costs). A running job that won't fit: stop at a checkpoint and resubmit the rest.

### ASC "comfort mode" (default)

ASC's job submit filter rewrites your request; sbatch prints `applying job settings for >> <mode> <<` and
one `setting --opt=val` line per change (with an `error:` prefix when the request is then rejected).
Compare them with what you asked for:

| You ask for | You get |
|---|---|
| `--gres=gpu:N` (N = 1 to 3) | shared node, N tasks × 22 CPUs, N × 25 % of memory. No `--nodes` and no `--mem` (both rejected, §8) |
| `-N X --gres=gpu:4` | X exclusive nodes, 4 tasks × 22 CPUs each, all memory |
| `-N X` (CPU partition) | X exclusive nodes, 190 tasks; with `-c` > 1 rejected (`-N1 -n1 -c8`: "Requested node configuration is not available"). Part of a CPU node with several cores: `#ASC --vanilla` and `-N1 -n1 -c <cores> --threads-per-core=1 --mem=<GB>G` |
| `-n X --mem YG` (CPU) | shared, X single-core tasks (`-c` > 1 is set back to 1). The batch script itself gets **one CPU**: a program started without `srun` runs on one core. One program with X cores: `#ASC --vanilla` and `-n1 -c X --threads-per-core=1` (`draft_job(kind="cpu")`) |

Add `#ASC --vanilla` as the second line for plain Slurm (you then specify
`--ntasks-per-node`, `--cpus-per-task`, etc. yourself). GPU *n* sits on NUMA domain 3, 2, 6
or 4. `srun` pins tasks to the right NUMA domain by itself. Use `srun --gpu-bind=none` if a
program wants to see all GPUs.

### Monitoring

| Need | Command |
|---|---|
| My jobs | `squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"` |
| Estimated start (unreliable) | `squeue -j N --start` |
| Why is it pending / priority | `squeue -j N -o "%r"`, `sprio -j N` |
| Finished job | `sacct -j N -o JobID,State,ExitCode,Elapsed,NodeList`, `seff N` |
| Partition load | `sinfo -p zen4_0768_h100x4 -h -o "%T %D"` (per state; `%A` alloc/idle counts planned nodes as idle) |
| Will it be accepted? | `sbatch --test-only …` (validates account/QoS, no submission) |
| Cancel | `scancel N` |
| Look at a running node | `node_exec`; by hand from the login node `timeout 60 srun --jobid=N --overlap -N1 -n1 --cpus-per-task=1 [-w <node>] <cmd>` (`nvidia-smi`, `ps`, `top -b -n1`, `py-spy dump` from an env in `$DATA`; ptrace is allowed). `ssh <node>` is refused (§2) |

Tokens used so far, as GPU-hours (= tokens at normal priority; CPU jobs and high priority not
counted; the projects portal is authoritative):

```bash
sacct -u $USER -S <project start> -X -n -P -o ElapsedRaw,AllocTRES | awk -F'|' '{g=0; if (match($2,/gres\/gpu=[0-9]+/)) g=substr($2,RSTART+9,RLENGTH-9); t+=g*$1/3600} END {printf "%.1f\n", t}'
```

Whole project, all members, in tokens, CPU nodes included (the `budget` tool does the same). The billed node
share is Slurm's `gres/np_<partition>` in `AllocTRES` (100 = one node, 25 = a quarter); B200 has no published
price, so it shows as node-hours:

```bash
sacct -a -A <account> -S <project start> -X -n -P -o User,Partition,ElapsedRaw,AllocTRES | awk -F'|' '{s=0; if (match($4,/gres\/np_[a-z0-9_]+=[0-9]+/)) {x=substr($4,RSTART,RLENGTH); sub(/.*=/,"",x); s=x/100} h=s*$3/3600; if ($2=="zen4_0768_h100x4") c[$1]+=4*h; else if ($2=="zen4_0768") c[$1]+=0.96*h; else b[$1]+=h} END {for (u in c) printf "%s %.1f tokens\n", u, c[u]; for (u in b) printf "%s %.2f other node-h\n", u, b[u]}'
```

For per-job bookkeeping, query `sacct -X -D --array`: `-D` keeps earlier attempts of requeued
jobs and `--array` gives one line per array task (key rows by `DBIndex`).

`lastjobs` and `interactivejobs` from the general ASC docs **don't exist on MUSICA**.

Pending reasons: `Priority`/`Resources` = just waiting. Idle nodes may be held for larger
jobs, so a short `--time` helps backfill. `QOS*Limit` = a QoS limit was hit: `QOSMaxNodePerUserLimit`
(dev QoS, 2 nodes per user across jobs), `QOSGrpGRES` (a QoS-wide GPU cap shared with other users, e.g.
20 GPUs in the dev GPU QoS, 72 for B200 guests; waits until GPUs free up). Never starts, cancel and fix:
`QOSMaxWallDurationPerJobLimit`, `DependencyNeverSatisfied`.
`Invalid account or account/partition combination` = wrong `--account`. Out of memory shows
up as `NODE_FAIL`, a job stuck in CG, or `Exceeded step memory limit`.

### Costs

| Job | Tokens |
|---|---|
| 1 GPU, 10 min test | ~0.17 |
| 1 full H100 node, 1 h | 4 |
| 2 nodes, 1 h | 8 |
| 16 nodes, 1 h | 64 |
| CPU node, 1 h | 0.96 |
| 4 whole H100 nodes, 16 min (a vLLM scaling run incl. startup) | ~4.2 |
| a large model download on a whole GPU node (476 GiB, 6.5 min) | ~0.4 |

Measured costs of a project's own jobs belong in its notes (`amadeus://project`); the ledger keeps every
estimate.

**Estimating run time** (rule 13). Measure in a dev smoke of 30–50 steps (5–10 steps under-read training
speed 2–10× because of warm-up), then `--time` ≈ steps × s/step × 1.3 + load time. Fixed cost + cost per item
predicts well; take both from a probe, not from memory. Small models gain little on an H100 (a 9M-parameter
model ran 1.35× an RTX 4080 per step): use a workstation for those.

High priority costs double. Urgent priority bumps or walltime extensions are requests to
support@asc.ac.at, made by a human.

---

## 6. Software

- **Lmod only initialises itself for interactive shells.** For `mx` use `MX_LMOD=1`; in jobs
  use the two lines from the skeleton, and put them **before `set -u`**: the EESSI init reads
  unset variables, so under `set -u` the script exits silently. For a clean slate:
  `ml --force purge && ml load EESSI/2025.06 ASC/2025.06` (EESSI is "sticky").
- On the login node (2025.06): CUDA 12.6 to 13.0, cuDNN 9, NCCL 2.27.7, OpenMPI 5.0.x,
  UCX-CUDA 1.19, NVHPC 25.9, Python 3.12/3.13, Miniforge3. **GPU-only modules are invisible
  on the login node** (it has no GPUs). List them inside a GPU job. There is **no PyTorch
  or vLLM module**, and ASC installs no Python packages centrally.
- **Containers are the preferred route for ML stacks.** Apptainer 1.5.2 is installed on login and
  compute nodes (no module needed; ASC's page says 1.3.5). Anonymous NGC pulls work (no NGC key needed despite what the docs say). Latest as
  of 2026-10: `nvcr.io/nvidia/vllm:26.09-py3`, `nvcr.io/nvidia/pytorch:26.09-py3`. The node
  driver 595 / CUDA 13.2 supports them.
  ```bash
  sbatch jobs/pull_container.slurm nvcr.io/nvidia/vllm:26.09-py3   # → $DATA/containers/vllm_26.09-py3.sif
  apptainer exec --nv -B $DATA,$SCRATCH $DATA/containers/vllm_26.09-py3.sif python -c 'import torch; print(torch.cuda.device_count())'
  ```
  Apptainer mounts `$HOME` automatically. Bind `$DATA`/`$SCRATCH` with `-B`. Before guessing env var or
  flag names, grep the container's own code on the login node (`apptainer exec $SIF grep -rn … --include='*.py'`;
  that is how `DG_CACHE_DIR` turned up, and that vLLM sets `OMP_NUM_THREADS` itself).
- **Downloads:** compute nodes reach HF, PyPI, docker.io and nvcr.io at 0.8–2.1 GB/s with the `hf` CLI
  (in an env under `$DATA`; `HF_XET_HIGH_PERFORMANCE=1`, `--max-workers 16`; 173 GiB in 223 s, 476 GiB in
  363 s). The login node has internet too but is shared: only small downloads there (tokenizers,
  configs; a 4B model of 8.8 GB took 24 s). Whole large models and container pulls go in a job
  (`jobs/pull_container.slurm`; `dev_zen4_0768` if under 10 min; when the CPU queue is days out, a whole GPU
  node at 4× the rate: 476 GiB took 6.5 min, about 0.4 tokens). The job must exit non-zero unless the file
  count matches the HF listing (`curl -s -m 20 'https://huggingface.co/api/models/<repo>?blobs=true'`, quoted
  for zsh), so `afterok` chains never start on partial weights.
- **venv/conda** if needed: create them under `$DATA` (`module load Miniforge3`;
  `conda env create --prefix $DATA/envs/<name>`). Not in `$HOME`.
- **Extending the module stack:** `EESSI-extend` with `EESSI_USER_INSTALL=$DATA/eessi`. See
  `docs-vsc/docs/musica/software/compilation.md`.
- **Training environments without a container:** the system Python has no `Python.h`, and
  Triton compiles a C helper at run time, so a venv on it fails inside the job. Use uv
  with a standalone Python, built on the login node (`uv` is not on PATH there: put the release binary into
  `$DATA/envs/tools/bin`, for example):
  ```bash
  U=$DATA/envs/tools/bin/uv; export UV_PYTHON_INSTALL_DIR=$DATA/opt/uv-python UV_CACHE_DIR=$SCRATCH/uv-cache
  $U python install 3.12.14 && $U venv --python 3.12.14 $DATA/envs/<name> && $U pip install --python $DATA/envs/<name>/bin/python -r requirements.lock
  ```
  The login node has internet access for this; its system `python3` has no pip. A project env from a uv lock
  (about 2 min on the login node): install the requirements of `uv export --frozen --no-hashes
  --no-emit-project` without torch, then torch from `--index-url https://download.pytorch.org/whl/cu130`, then
  the project with `--no-deps -e .`.
- **Prebuilt CUDA 13 binaries outside containers** (e.g. llama.cpp; `libcudart.so.13 => not found`):
  install the CUDA 13 runtime and cuBLAS pip wheels into a directory under `$DATA` (`pip install --target`)
  and set `LD_LIBRARY_PATH` to its `nvidia/cu13/lib`.
- **Caches in jobs:** point every compile cache at `$SCRATCH`, one root per server configuration (or per
  GPU for parallel runs): `XDG_CACHE_HOME VLLM_CACHE_ROOT TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR
  DG_CACHE_DIR FLASHINFER_WORKSPACE_BASE`, plus `HF_HUB_OFFLINE=1` so nothing downloads mid-job. A warm cache
  saves minutes per server start. Stage weights to `/local` with parallel copies (create the directories
  first, then `find … -type f | xargs -P 16 -I{} cp --parents {} /local/$USER/…`: 173 GiB in about 20 s per
  node; parallel `cp --parents` without the directories races with `File exists` and leaves the model
  unstaged).
- `VLLM_NO_USAGE_STATS=1` in vLLM jobs: otherwise every log has a harmless `OSError: Could not find a suitable
  TLS CA certificate bundle` traceback from the usage-stats thread.

### Multi-node patterns (for 2, 4 and 16-node runs)

- MPI or container launch, from ASC's examples:
  `srun --ntasks-per-node=4 --cpu-bind=none --mpi=pmix singularity exec --nv --writable-tmpfs -B <mounts> <sif> <exe>`
- Ray (what vLLM uses across nodes): `docs-extra/example-job-scripts/98-frameworks-ray/02-ray-python-multi-node-gpu.sh`.
  Start the head on the batch host, then workers with `srun --relative=1 … ray start --address=head:6379 --block &`,
  then run the driver script. **Adapt it:** that example is for VSC-5 A100 nodes and reads
  the head IP from `ib0`, but on MUSICA the login node's InfiniBand interface is `ibp6s0`,
  and the compute nodes have 4 IB adapters. Check interface names inside a job first (`jobs/probe_gpu_node.slurm`).
- Cheapest way to test multi-node: 2 nodes on `dev_zen4_0768_h100x4` (10 min).
- **Proven pattern (vLLM, no Ray):** `#ASC --vanilla`, `--ntasks-per-node=1 --cpus-per-task=352
  --gres=gpu:4 --exclusive --mem=0`, then one `srun -N1 -n1 -w <node> --cpus-per-task=352 --gres=gpu:4
  --mem=0 …` per node from the batch script (`jobs/vllm_multinode.slurm`). Use the `ibs8` IP for
  cross-node traffic (`VLLM_HOST_IP`, `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`) and
  `NCCL_IB_HCA='^=mlx5_0'` (mlx5_0 is a 25 Gb/s RoCE port; mlx5_2..5 are the NDR200 adapters). Replicas need
  no IB at all: serve HTTP by hostname (§5 Patterns; an ibs8 IP was once unreachable from other nodes).
- **DDP across nodes:** one `srun` task per node running `torchrun --rdzv_backend=c10d
  --rdzv_endpoint=$HEAD:29500`, `NCCL_IB_HCA='^=mlx5_0'`, `OMP_NUM_THREADS=8` per process (4 per node on 176
  cores), one log file per node (`…-n$SLURM_PROCID.log`) instead of interleaved output, and the tokenised
  dataset built once by rank 0 on `$SCRATCH`/`$DATA` (not `/local`, which is per node) while the other ranks
  wait.

---

## 7. Storage rules

| Path | Put here | Don't |
|---|---|---|
| `<workdir>` (e.g. `~/<project>` → `$DATA/<project>`) | code, job scripts, small logs | — |
| `$DATA` (GPFS, 10 TB) | model weights, `.sif` images, venvs, results | treat as backed up (it isn't) |
| `$SCRATCH` (Weka flash, 5 TB) | caches, temp, hot data | keep anything you need later |
| `/local` (7 TB NVMe per GPU node) | stage model weights for a job | expect it after the job |
| `$HOME` (50 GB) | dotfiles | models, caches, conda envs |

Quota checks (verified; the forms in the ASC docs fail):
`mmlsquota --block-size auto -j $USER home_musica`, `mmlsquota --block-size auto -j <filesystem> data_ess`, `df -h $SCRATCH`.
The projects portal (projects.asc.ac.at) also shows usage and token consumption. It's only
reachable from allowed networks, and humans use it.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection timed out during banner exchange`, `Connection to UNKNOWN port 65535 timed out` | Tunnel stalled (wireproxy logs `Retrying handshake` about once a minute; most are harmless, some stall connections for 10 to 60 s, and a stalled wireproxy often does not recover by waiting) or a login-node hiccup. With an expired certificate the same timeout shows | With wireproxy, Amadeus restarts the tunnel itself (stop by PID, `scripts/vpn.sh up`; at most once a minute, only while the certificate is valid, only when the command did not start) and runs commands that are safe to repeat once more; `recovery` in the result says so. The copying in `push_files`/`fetch_files` does not restart it yet (it retries after 20 s): if a transfer keeps failing this way, call `access_status` (its SSH test restarts a stalled tunnel), then push or fetch again. `run_command` is not repeated: run it again. `submit_job` never repeats `sbatch`; it looks for the job itself (§4a). If it persists: `access_status`, `scripts/vpn.sh status`; if Linz answers but Vienna doesn't, it's ASC's side. Check https://status.asc.ac.at. With a system VPN, check the VPN app or network |
| `client_loop: send disconnect: Broken pipe` mid-command | same | Retry read-only commands. For `sbatch`, check `squeue` first |
| `Permission denied (publickey,keyboard-interactive)` | Certificate expired or missing in the agent | `start_login` (or `scripts/musica_auth.sh`) → show the link to the human at once (§3) |
| `Permission denied (publickey,gssapi-keyex,gssapi-with-mic)` on `ssh <node>` from the login node | Compute-node ssh does not work (§2) | `node_exec` / `srun --jobid=N --overlap` (§5 Monitoring) |
| Commands hang forever | (Historical) a stuck ControlMaster socket | `ControlMaster` is now `no`. If re-enabled, remove `~/.ssh/cm-*` and kill the `[mux]` process |
| `An error occurred during authentication` after the QR/device prompt | The 60 s Enter window of plain-ssh device auth expired | Use step-ssh (§3) |
| `module: command not found` | Lmod not initialised (non-interactive shell) | `MX_LMOD=1` / skeleton lines |
| `Invalid account or account/partition combination` | Wrong account (often the filesystem name `fs<number>`) | `--account=p<number>` (the config's `account`) |
| `check_job`/`submit_job` refuse: no account configured | No `[project] account` in the config | Ask the human for it and write `~/.config/amadeus/config.toml` (INSTALL.md §6) |
| `Access/permission denied for job N` on `scontrol update … TimeLimit=` | Users can't raise a running job's time limit (pending works) | Stop at a checkpoint and resubmit the rest; next time estimate first (§5 Costs) |
| Logs land in `<workdir>/$DATA/…` or a literal `~` directory | sbatch expands neither `$VARS` nor `~` in `#SBATCH` lines | Paths relative to the submit directory (`logs/%x-%j.out`) or absolute (`/data/<filesystem>/<user>/…`) |
| `Memory: A specific memory setting was requested but gpu partitions only support full node/half node memory via gpu count`, then `Invalid generic resource (gres) specification` | `--mem` on the GPU partition in comfort mode | Drop `--mem` (memory follows the GPU count), or `#ASC --vanilla` with `--mem=0` for whole nodes |
| `GPU: Cannot specify number of GPUs when requesting full nodes with "--nodes"`, then `Job size specification needs to be provided` | `--nodes` together with `--gres=gpu:1`–`3` in comfort mode | 1–3 GPUs: `--gres=gpu:N` without `--nodes`; whole nodes: `-N n --gres=gpu:4` |
| A multi-process program on the CPU partition runs on one core (`nproc` 1) | Comfort mode: `-n N` gives N single-core tasks; the batch script gets one of them | `#ASC --vanilla` with `-n1 -c N --threads-per-core=1`, or start the tasks with `srun` (§5 comfort mode) |
| 1–3 GPU job pending far in the future even though nodes are idle | Partial-node jobs only run on nodes with feature `SharedNode` (about 40 of the H100 nodes, often full). Whole-node jobs can use any node | `sinfo -p zen4_0768_h100x4 -o "%f %T %D"`. Request the whole node (`--gres=gpu:4`) and compare with `sbatch --test-only` |
| Job pending with an estimate far in the future even though nodes are idle | The "idle" nodes are `planned`: held free by the backfill scheduler for higher-priority jobs next in line (plain `sinfo` `%A`/`%F` counts them as idle) | `cluster_load` (`planned` per partition) or `sinfo -o "%T %D"`. Shorter `--time` (only a job that ends before their planned start can use them). Use dev QoS for tests. Check `sprio` |
| Jobs pending for many hours on `Priority`, `squeue --start` shows `N/A` | Low fair-share: in `sprio` the FAIRSHARE part (about 1,800) can be tiny next to the QOS part (100,000), so older jobs of other users go first (CPU queue, 178 waiting, 2026-10-13) | Check `cluster_load` for idle (not `planned`) nodes elsewhere; the CPUs of GPU nodes start at once at 4× the rate (ask the human) |
| `sbatch --test-only` predicts a start hours away, the real job starts at once | The test-only estimate is not a reliable start forecast (2026-10-14: predicted 11:45 and 13:43, started within a minute) | Use `--test-only` to validate a job, not to predict its start |
| `allocation failure: Requested node configuration is not available` for part of a CPU node (`-N1 -c8`) | Comfort mode reads `--nodes` on the CPU partition as whole nodes of 190 single-core tasks (without `--nodes` it sets `--cpus-per-task=1` instead, so the job gets one core) | `#ASC --vanilla` on line 2 with `-N1 -n1 -c <cores> --threads-per-core=1 --mem=…` (`draft_job(kind="cpu")` writes it) |
| `allocation failure: Requested node configuration is not available` on the GPU partition, for a CPU-heavy job | `--threads-per-core=1 --cpus-per-task=190` asks for more physical cores than GPU nodes give Slurm (176 of 192; `scontrol show node <n>` shows `CoreSpecCount=16`, `CPUEfctv=352`) | `--cpus-per-task=176` with `--threads-per-core=1` (or up to 352 without) |
| A retry loop around `srun` never retries; the job ends at the first launch failure | `set -e` exits on the failing `srun` before the loop checks `$?` | `srun … && RC=0 \|\| RC=$?` (§5 Patterns) |
| A job you just submitted seems gone (empty `squeue -j`) | `squeue` can miss a job for a moment right after `sbatch` | Poll `sacct -j <id> -X -n -o State` instead (always has a state) |
| Router/server returns 503 or `error sending request` at high concurrency | Jobs inherit the login node's soft open-file limit (1024) | `ulimit -n 65536` in the job script (hard limit 524288) |
| `SLURM_NODEID` is 0 on every node | It's relative to the step; in 1-node `srun` steps it is always 0 | Derive the index from `scontrol show hostnames "$SLURM_JOB_NODELIST"` |
| `hf download --include A B` fetches only B | The new `hf` CLI reads B as a filename and then ignores `--include` | One `--include` per pattern |
| Batch script ends silently right after the Lmod init | `set -u` was active while the EESSI init ran (unbound `EESSI_NO_MODULE_PURGE_ON_INIT`) | Init Lmod first, then `set -u` |
| GPU modules / `nvidia-smi` missing | You're on the login node | Run it inside a GPU job |
| Exit 144 from the Bash tool | `pkill -f` matched the agent's own shell | See rule 9 |
| Some per-node `srun` steps fail with `Socket timed out on send/recv operation` | Many simultaneous step launches (16 at once) | Launch steps 0.5 s apart and relaunch a failed one once (§5 Patterns) |
| A single multi-node `srun` fails with `Prolog hung on node …` and socket timeouts | Slurm prolog slow on some nodes at 8–16 nodes | Retry the step after 30 s (§5 Patterns); the retry may wait about 100 s for the prolog |
| One vLLM replica stops answering mid-run, `shm_broadcast` `TimeoutError` in its log | Engine worker hang (seen once) | Health checks during the run; move its requests to the other servers |
| `NODE_FAIL` within a minute of the start, nothing in the log | Node problem on ASC's side | Resubmit; billed only for the minute |
| Triton or `torch.compile` fails in a job: `Python.h: No such file or directory` | venv on the system Python, which has no headers | uv with a standalone Python (§6) |
| Batch script aborts right after start with an `unbound variable` | Under `set -u`, bash treats a declared but empty associative array as unset | Initialise it (`declare -A X=()`) or test with `${X[k]:-}`; dry-run job scripts before submitting |
| A loop over a space-separated list runs once; `no matches found: --include=*.md`; `(eval):1: == not found` | The agent's shell is zsh (rule 12) | Use an array or `bash -c`; quote globs, URLs with `?` and `==` |
| Same code builds different data on MUSICA and a workstation | Different system libraries (EL10 has cairo 1.18.2, Ubuntu 1.18.0; NOTES.md Software) | Build once, copy it, check the checksum |

---

## 9. When to stop and ask the human

- **Sign-in:** every certificate renewal (the link must be approved by the human).
- **Spending:** any job or series over about 20 tokens. Any ≥ 4-node job; a 16-node job gets its own
  heads-up even inside an approved budget (rule 4; thresholds from the config's `[limits]`). Anything on
  `zen5_2304_b200x8` (private ISTA nodes; available to guests only when idle). Long QoS
  (`long_*`) or priority requests. A budget the human approved for one experiment covers that experiment
  only; the project's notes list the approved budgets.
- **Outward-facing actions:** support tickets or email to support@asc.ac.at, Matrix posts,
  anything sent to the project's mentor or other people.
- **Destructive actions:** deleting from `$DATA`/`$SCRATCH`, `scancel` of jobs the agent
  didn't start, overwriting results.
- **Access configuration:** changing `~/.ssh/config`, the WireGuard config, step contexts,
  or the cluster's `authorized_keys` (including any attempt to make compute-node ssh work: `ssh -A`, a
  login-node key).
- **Project configuration:** the account, budget and limits in `~/.config/amadeus/config.toml` come from the
  human; never guess or change them.
- **Licensing:** gated models (e.g. Llama) need a Hugging Face token from the human. Never
  invent or reuse someone else's.

Support requests (written by a human) should include: system (MUSICA Vienna), username,
job IDs, the path to the job script and logs (left unmodified), the exact error text, and
what was expected. One ticket per issue. Office hours are 09:00 to 17:00 Vienna time.

---

## 10. Setting up a new machine or agent (teammates)

**[INSTALL.md](INSTALL.md)** has the setup, written for an agent to carry out (Mac with the WireGuard app, Linux,
dev containers and VMs with wireproxy, Windows through WSL2), starting from a clone of
`github.com/cpietsch/amadeus`, and ending with the project config. Every person
uses **their own ASC account**: the server runs commands as the user in its config (`[cluster] user`), so a teammate
installs the server on their own machine.

- **Other machines of the same person:** each gets its own install and its own daily login. To move files
  between the cluster and another machine without installing there, list that machine's SSH alias in the
  config's `files.hosts` (or `AMADEUS_HOSTS`) and pass it as `host` to `push_files`/`fetch_files`.
- **Reuse shared assets:** a member's `$DATA` (models, SIFs, envs) is readable by the project group (the
  account name, `p<number>`, mode `drwxr-x---`): point `MODEL`/`SIF` at `/data/<filesystem>/<user>/…` instead of
  downloading again; the project's notes list what is there. HF's `.cache/huggingface/` metadata under
  local-dir downloads is owner-only, so skip `.cache` when copying or staging a model (`find <path> ! -perm
  -g+r`). Compute-node ssh does not work as set up; inspect running nodes with `node_exec` (§5 Monitoring).

---

## 11. Where to look things up

Clone the ASC docs once next to the server (both directories are git-ignored):

```bash
git clone https://gitlab.tuwien.ac.at/vsc-public/documentation docs-vsc
git clone https://gitlab.tuwien.ac.at/vsc-public/job-scripts/example-job-scripts.git docs-extra/example-job-scripts
git clone https://gitlab.tuwien.ac.at/vsc-public/training/vsc-intro.git docs-extra/vsc-intro
```

| Topic | File |
|---|---|
| Login: device flow, step, compute nodes | `docs-vsc/docs/musica/login/{index,device_auth,step_ssh,compute_node}.md` |
| WireGuard | `docs-vsc/docs/login/how_to_login.md` (section "Wireguard") |
| Partitions, QoS | `docs-vsc/docs/musica/musica_queues.md` |
| Comfort vs vanilla Slurm | `docs-vsc/docs/musica/slurm_modes.md` |
| Example job scripts (incl. multi-node container) | `docs-vsc/docs/musica/example_job_scripts.md` |
| NUMA/GPU/NIC topology, cores reserved for Weka | `docs-vsc/docs/musica/node_config.md` |
| Storage and quotas | `docs-vsc/docs/musica/storage.md` |
| Modules, EESSI, setup | `docs-vsc/docs/musica/software/{usage,module_environment,examples}.md` |
| Containers | `docs-vsc/docs/musica/software/containers.md` |
| Python venv / conda | `docs-vsc/docs/musica/software/python/` |
| Building software (EESSI-extend, buildenv) | `docs-vsc/docs/musica/software/compilation.md` |
| Known issues | `docs-vsc/docs/musica/software/issues.md` |
| Tokens, project dates, extensions | `docs-vsc/docs/musica/{projects,project_lifecycle}.md`, `docs-vsc/docs/files/ASC_Price_Sheet.pdf` |
| Login node limits | `docs-vsc/docs/running_jobs/login_node.md` |
| Scheduling, backfill, priorities | `docs-vsc/docs/running_jobs/scheduling/` |
| Monitoring, reason codes, OOM | `docs-vsc/docs/running_jobs/{monitoring_jobs,oom}.md` |
| Support etiquette | `docs-vsc/docs/support/support.md` |
| Ray / PyTorch / TF example jobs (VSC-5, adapt them) | `docs-extra/example-job-scripts/9{6,7,8}-frameworks-*` |
| Job and NUMA introspection scripts | `docs-extra/example-job-scripts/util/` |

Pages under `docs-vsc/docs/running_jobs`, `storage`, `software` and `login` are mostly
written for VSC-4/5. Their *concepts* apply, but their partition names, paths, module names,
`lastjobs`/`interactivejobs`, SMS OTP and port 27 do **not**. Known errata on MUSICA itself: Apptainer
is 1.5.2 (the containers page says 1.3.5), and `ssh` to compute nodes (`login/compute_node.md`: needs `ssh -A`
or a key created on the cluster) is refused as set up here; `srun --overlap` works.

Refresh the clones: `git -C docs-vsc pull`, `git -C docs-extra/example-job-scripts pull`.
Live docs: https://docs.asc.ac.at. Status: https://status.asc.ac.at.
Community: Matrix `#asc-musica-users:tuwien.ac.at`.
