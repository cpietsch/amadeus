# Issues of Amadeus, the MCP server

Problems with Amadeus (`amadeus/`, registered as `amadeus`) or the way agents reach MUSICA through it. One
Markdown file per issue, `<id>-<slug>.md`, written by the server:

- **By agents** in other projects, through the `report_issue` tool (`source: agent`).
- **Automatically**, when a tool raises an unexpected exception (`source: automatic`): one issue per fingerprint
  (tool, exception type, code location). Repeats raise `occurrences` and `last_seen` on the open issue instead of
  creating new files. Expected operational failures (SSH down, a Slurm command failing) are not filed.

Secrets are redacted on writing (bearer tokens, device links, private keys). Open issues sit in this directory;
closed ones move to `closed/`.

## File format

```markdown
---
id: 20261014-171502-a1b2
status: open                 # open | in-progress | needs-human | fixed | wontfix | duplicate
kind: bug                    # bug | wrong-result | missing-feature | docs | cluster | other
severity: medium             # high (blocks work) | medium | low
source: agent                # agent | automatic
tool: submit_job
project: my-project
client: claude-code 2.1.289
created: 2026-10-14T17:15:02+0000
server_revision: abc1234     # code the server ran when it was filed
---
# Title

## What happened / Expected / How to reproduce / Context   (agent issues)
## What happened / Arguments / Traceback                   (automatic issues)
## Resolution                                              (filled in by the fixer)
```

The fixer adds `claimed: <time>` when it starts, and `fixed_in: <commit>` when it is done.

## Protocol for the maintainer session

A Claude session in this repository checks this directory regularly, for example with
`/loop 30m work through the open issues in issues/ following issues/README.md`. For each run:

1. **Pick** the open issues, highest severity first, then oldest. Skip those in `in-progress` with a `claimed` time
   under 2 hours old (another run has them).
2. **Claim:** set `status: in-progress` and add `claimed: <time>`.
3. **Triage:**
   - Not a server problem but a cluster or platform behaviour (`kind: cluster`): record it in AGENTS.md §8 or NOTES.md
     if it is new, then close it as `fixed` (documentation) or `wontfix`.
   - Needs the human (budget, approvals, access configuration, anything in AGENTS.md §9, a design decision): set
     `status: needs-human`, write the question under Resolution, and leave it.
   - Duplicate: `status: duplicate`, name the other id under Resolution.
4. **Reproduce** against the fake cluster first: add a failing test to `tests/` (see `tests/test_server.py`; the fake
   SSH shim and Slurm stubs are in `tests/fake_ssh.sh` and `tests/fakebin/`). Use the real cluster only for
   read-only checks (`access_status`, `list_jobs`, `job_details`), never submit jobs or move data to reproduce.
5. **Fix** in `amadeus/`, keep the runbook rules intact (AGENTS.md §0, §4a, §9), and run
   `uv run python -m unittest discover -s tests -t .` until everything passes.
6. **Deploy:** nothing to restart here; sessions pick the fix up when they next start the server.
7. **Close:** fill in Resolution (cause, fix, test), set `status: fixed` and `fixed_in: <commit>`, move the file to
   `closed/`, commit (`amadeus: <fix> (issue <id>)`) and push.
8. If a fix changes how agents should use a tool, update the tool's description and AGENTS.md §4a.

Never weaken a guard (approval limits, account and QoS checks, path checks) to make an issue go
away: if an agent finds a guard in its way, that is `needs-human`.
