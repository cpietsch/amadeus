# CLAUDE.md

This repository is Amadeus, the MCP server (`amadeus/`, AGENTS.md §4a) through which coding agents use the
ASC MUSICA supercomputer, for any ASC project, with the agent runbook built in. You are its maintainer session.

- Keep it general: no values of a particular project (account, filesystem, budget, user names, paths in someone's
  `$DATA`, results) in code, docs or `jobs/`. Project values come from the config (`amadeus/config.py`,
  `config.example.toml`, INSTALL.md §6); examples use placeholders (`p123456`, `fs123456`). Project-specific job
  scripts and notes live in the projects' own repositories (`template_dirs`, `notes`).
- What goes where: platform facts in NOTES.md, rules and patterns for agents in AGENTS.md (keep the section
  numbers: tools, guards and other docs refer to them), setup in INSTALL.md, tested generic job scripts in
  `jobs/`. A lesson learned in a project goes in as the general rule, without the project's numbers.
- When a tool's behaviour changes, update its description, AGENTS.md §4a and, if the rules change, the server's
  instructions (`INSTRUCTIONS` in `amadeus/server.py`).
- Run the tests after every change (`uv run python -m unittest discover -s tests -t .`; a fake cluster, nothing
  reaches MUSICA).
- Work through `issues/` following [issues/README.md](issues/README.md).
- Operate the cluster through Amadeus itself, following the runbook below: its rules (§0) and its "when to ask
  the human" list (§9). Reproduce problems against the fake cluster; on the real one only read-only checks
  unless the human asks for more.

@AGENTS.md

Platform reference: @NOTES.md
