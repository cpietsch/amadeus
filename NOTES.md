# MUSICA quick reference

Platform reference behind Amadeus, the MCP server in this repo (see [README.md](README.md)), served to agents as
`amadeus://notes`. It holds for any ASC project; facts about your own project (account, budget, what is
installed in your `$DATA`, results) belong in the project's notes, which the server serves as `amadeus://project`
(`notes` in `~/.config/amadeus/config.toml`).

Condensed from the ASC docs (local clone in `docs-vsc/`, source:
https://gitlab.tuwien.ac.at/vsc-public/documentation; clone commands in AGENTS.md §11). Most relevant pages:
`docs/musica/login/`, `musica_queues.md`, `slurm_modes.md`, `example_job_scripts.md`,
`node_config.md`, `storage.md`, `project_lifecycle.md`, `software/` (incl. `python/`).
ASC example repos (`example-job-scripts`, `vsc-intro`) go in `docs-extra/`.
**How agents operate the cluster: [AGENTS.md](AGENTS.md).** Facts verified on MUSICA Vienna, 2026-10-07 to
2026-10-16.

## Projects and accounts

- Each ASC project has a name, a number `<n>`, a Slurm account **`p<n>`** and a filesystem **`fs<n>`**. The
  account is `p<n>`: `fs<n>`, as some templates (and mentors' examples) have it, is rejected by sbatch. Users
  have the project account as their primary group.
- **Budget in tokens.** 1 token ≈ €1. Price sheet v1.4 (`docs-vsc/docs/files/ASC_Price_Sheet.pdf`):
  H100 node 4.00 tokens/h, CPU node 0.96/h, partial nodes billed pro rata (1 GPU = 1/h),
  high priority costs double. A 16-node run costs about 64 tokens/hour. B200 nodes have no published price.
  When the budget runs out, jobs only run in the idle QoS (low priority, 24 h limit).
- **Lifecycle:** after the project's end date a grace period for submitting jobs, then a login deadline, then
  the data is deleted (`project_lifecycle.md`; the dates are on projects.asc.ac.at).
- QoS a project typically has: `zen4_0768` (default), `zen4_0768_h100x4`, `dev_zen4_0768_h100x4`,
  `dev_zen4_0768`, `zen5_2304_b200x8` (guest access).
- Default site: Vienna (`musica.vie.asc.ac.at`).
- `$DATA=/data/<filesystem>/<user>` (10 TB per project), `$SCRATCH=/scratch/<filesystem>/<user>` (5 TB),
  `$HOME` 50 GB.
- Default job time is 24 h, max 72 h.

## Connecting

`scripts/musica_auth.sh` (add `--wait` to block until done; INSTALL.md has the setup) checks three things:

1. **Network path.** A system VPN (WireGuard app, `wg-quick`) or a partner-university network; where there is
   no NET_ADMIN or `/dev/net/tun` (containers, many dev VMs), **wireproxy** (`~/.local/bin/wireproxy`) runs
   WireGuard in userspace (config `~/.config/wireguard/asc.conf`) and provides a SOCKS5 proxy on
   `127.0.0.1:1080`. The tunnel routes only Vienna and Linz (193.170.79.0/25), not Innsbruck. Control it
   with `scripts/vpn.sh up|down|status`.
2. **ssh-agent**: the `IdentityAgent` of Amadeus's own ssh config (`~/.local/state/amadeus/agent.sock`), or of an
   own `musica` entry in `~/.ssh/config`.
3. **step-ssh certificate**, valid about 16 h. When it has expired, the script starts
   `step ssh login <user> --context asc --console` and prints an
   `https://auth.asc.ac.at/device?code=…` link. Approve it in any browser (with the
   second factor) **within about 60 s**: the device grant expires after that (measured
   2026-10-08), so run `scripts/musica_auth.sh --wait` when you're ready to click.

What doesn't work: plain SSH keys (whether registered at auth.asc.ac.at or in
`~/.ssh/authorized_keys`) are rejected on the login nodes, which only accept
CA-signed certificates. The browser sign-in inside `ssh` (no `step`) needs Enter
within about 60 s. A shared SSH connection (ControlMaster) got stuck after tunnel
drops, so it's switched off. Each `ssh musica` opens a fresh connection.
`ssh <node>` from the login node is **refused** even with your job on that node, also with a public key in the
cluster's `~/.ssh/authorized_keys` (`Permission denied (publickey,gssapi-keyex,gssapi-with-mic)`, 2026-10-08 and
2026-10-13; probably because the private key is not on the cluster). ASC's routes (`ssh -A`, a key created on
the cluster) would be access changes for the human.
What works: `timeout 60 srun --jobid=<id> --overlap -N1 -n1 --cpus-per-task=1 [-w <node>] <cmd>` from the
login node (Amadeus's `node_exec`), including `py-spy` (ptrace allowed).

## Access (general, from the docs)

- SSH only from partner-university networks or the ASC WireGuard service.
- Login uses SSO via https://auth.asc.ac.at plus TOTP/WebAuthn: either the device flow
  on plain `ssh`, or `step ssh` (bootstrap:
  `step ca bootstrap --context asc --ca-url https://auth.asc.ac.at:9000 --fingerprint 44b048473242281db1da57124c2b843741d6c92a8fb5d0482dec032e957f2919`).
- Host CA line (Amadeus ships it as `amadeus/asc_known_hosts`; for a known_hosts file of your own):
  ```
  @cert-authority *.asc.ac.at ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBADeLjSj4J0MFrdxdn9M/mcZi1fqKnQB6ZHjysUlontzFXI9PVoxPjYqMDDO0hM6IM6NCnkiiFqveEvSoXDnGWI=
  ```
- No JupyterHub or NoMachine yet.

## Hardware

| Partition          | Nodes (Vienna) | Per node                                      |
|--------------------|----------------|-----------------------------------------------|
| `zen4_0768`        | 69             | 2x EPYC 9654 (192 cores), 768 GB              |
| `zen4_0768_h100x4` | 109            | same + 4x H100 SXM5 94 GB (NVLink NV6), 4x NDR200, 7.68 TB NVMe |
| `zen5_2304_b200x8` | 28             | 8x B200, 2304 GB. Privately owned (ISTA); available to others when idle. Slurm sees 352 CPUs (CPUTot 384, CPUEfctv 352; 176 physical cores with `--threads-per-core=1`), like the H100 nodes |

GPU nodes have 8 NUMA domains of 22 usable physical cores each (WEKA reserves the rest),
352 virtual or 176 physical cores for Slurm. GPU0 to GPU3 sit on NUMA 3, 2, 6 and 4,
each with its own NIC. `srun` pins tasks to the GPU's NUMA domain automatically. CPU nodes have 190 usable
cores (`CoreSpecCount` 2, 384 threads with SMT). H100 nodes: driver 595.71.05, CUDA 13.2.

Network inside GPU nodes: four NDR200 InfiniBand adapters (`mlx5_2..5`) plus a 25 Gb/s RoCE port (`mlx5_0`;
exclude it from NCCL with `NCCL_IB_HCA='^=mlx5_0'`); the IB interfaces (`ibs8`, `ibs9`, …) share one subnet,
so cross-node traffic to an `ibs8` IP depends on per-node routing. Ethernet (hostnames) is the reliable path for
HTTP between nodes.

**Known bad GPU nodes** (2026-10-08; `submit_job` excludes them unless a job sets `--exclude` or `--nodelist`):
`n3010-027` (its `ibs8` IP unreachable from other nodes), `n3010-019` (Apptainer fails with `unknown userid`),
`n3012-034` (about 25% slower than the others in a vLLM run).

## Slurm

| QoS                    | Limit                     |
|------------------------|---------------------------|
| `zen4_0768_h100x4`     | 72 h                      |
| `dev_zen4_0768_h100x4` | 10 min, max 2 nodes per user at once across all jobs (a second dev job waits with `QOSMaxNodePerUserLimit`). Use for quick tests |
| `zen4_0768` / `dev_zen4_0768` | CPU equivalents   |

- Rocky Linux 10.2 (kernel 6.12) on all nodes, Slurm 25.11.8 (`seff`, `sprio`, `sshare`, `sstat` available),
  `MaxArraySize` 100000. QoS-wide GPU caps (`GrpTRES`): `dev_zen4_0768_h100x4` 20 GPUs, B200 guests 72 GPUs;
  jobs waiting on them show `QOSGrpGRES`.
- **Time limits:** users can raise a job's `TimeLimit` only while it is pending. For a running job it is denied
  (`Access/permission denied for job N`).
- **Comfort mode** (default): `--gres=gpu:N` (N=1-3) gives a shared node with N tasks,
  22 CPUs/task and N×25% of node memory (`--nodes` or `--mem` with it is rejected). `-N X --gres=gpu:4`
  gives X exclusive nodes with 4 tasks each. On the CPU partition `-n N` gives N single-core tasks and the
  batch script one CPU. Add `#ASC --vanilla` for plain Slurm behaviour.
- Partial-node GPU jobs only run on nodes with the feature `SharedNode` (about 40 of the H100 nodes, often full);
  whole-node jobs start sooner.
- The docs recommend `#SBATCH --threads-per-core=1` on GPU nodes (physical cores only).
- Multi-node container pattern (from the docs):
  `srun --ntasks-per-node=4 --cpu-bind=none --mpi=pmix singularity exec --nv ...`
- Queue: `squeue -u $USER`. A full node can take more than an hour to start, or start at once; the
  `sbatch --test-only` estimate is not a reliable forecast. Fair-share priority plus backfill:
  tight `--time` limits start sooner. Billing covers only the time actually used.
- Jobs inherit the login node's soft open-file limit (1024; hard 524288): servers with many connections need
  `ulimit -n 65536` in the job script.
- Multi-node launches at 8–16 nodes sometimes fail with `Prolog hung on node …` and socket timeouts, and a
  node can end a job in `NODE_FAIL` within a minute: retry or resubmit (AGENTS.md §5 Patterns).
- Useful: `seff <id>`, `sprio -j <id>`, `squeue -j <id> --start`, `sbatch --test-only`.
  **`lastjobs` and `interactivejobs` (general ASC docs) don't exist on MUSICA.**
- Other QoS exist (`long_*` 10 d, `fast_*`, `idle_*`) but aren't assigned by default.
- InfiniBand interface on the login node: `ibp6s0` (not `ib0`). GPU nodes have 4 IB adapters.
- Do compute-heavy work in jobs only. The login node is for editing, copying and
  light preparation.

## Storage

| Var        | FS             | Quota (default site) | Use                     |
|------------|----------------|----------------------|-------------------------|
| `$HOME`    | GPFS           | 50 GB                | code, configs           |
| `$DATA`    | GPFS (tiered)  | 10 TB project        | models, datasets, results, envs |
| `$SCRATCH` | Weka all-flash | 5 TB project         | fast temp data, no backup |
| `/local`   | node NVMe      | ~7 TB on GPU nodes   | job-only (e.g. stage model weights) |

**No backup on MUSICA at all.** Point `HF_HOME`, `APPTAINER_CACHEDIR` and similar caches at
`$DATA` or `$SCRATCH`, never `$HOME`. Keep a project's working copy in `$DATA/<project>`, with
`~/<project>` as a symlink if you like (ASC: don't run jobs from `$HOME`); Amadeus's `workdir` points at it.

Measured: staging 173 GiB of weights from `$DATA` to `/local` took 17–23 s per node, also on 16 nodes at once
(about 120 GB/s from `$DATA` in total). Unpacking a container image on Weka (`$SCRATCH`) took 30 min; set
`APPTAINER_TMPDIR` to `/local`.

Quota (verified): `mmlsquota --block-size auto -j $USER home_musica`,
`mmlsquota --block-size auto -j <filesystem> data_ess`, `df -h $SCRATCH`.

Files in a member's `$DATA` are readable by the project group (`drwxr-x---`), so teammates can share models,
images and envs. HF's `.cache/huggingface/` metadata under local-dir downloads is owner-only.

## Software

- **Lmod setup only happens for interactive shells** (`~/.bashrc.d/asc-samrc` checks `$PS1`).
  In batch jobs and non-interactive `ssh` commands, run:
  ```
  source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/sh
  module use /cvmfs/software.asc.ac.at/init/modules
  ```
- Visible on the login node (2025.06): CUDA 12.6 to 13.0, cuDNN 9.x, NCCL 2.27.7,
  OpenMPI 5.0.x, UCX-CUDA 1.19, NVHPC 25.9, Python 3.12/3.13, Miniforge3 25.3.
  GPU-only builds are **invisible on the login nodes** (they have no GPUs);
  list them inside a GPU job.
- There is no PyTorch, vLLM or TensorFlow module. Clean slate: `ml --force purge && ml load EESSI/2025.06 ASC/2025.06`.
  EESSI is a sticky module: use `module --force purge` to remove it.
- **NGC containers work with anonymous pulls** (no NGC key needed). Latest: `nvcr.io/nvidia/{vllm,pytorch}:26.09-py3`.
  Pull them with `jobs/pull_container.slurm` (CPU job) to `$DATA/containers/`.
- Extending the module stack: `EESSI-extend` with `EESSI_USER_INSTALL=$DATA/eessi` (see `software/compilation.md`).
- **There are no central Python packages** (ASC policy): use venv or conda
  (`module load Miniforge3`), placed on `$DATA` because of `$HOME` quota and file counts.
  The system Python has no `Python.h`, which Triton needs at run time: build venvs for GPU code
  on uv's standalone Python (AGENTS.md §6).
- Apptainer 1.5.2 (`apptainer` or `singularity`) on login and compute nodes (ASC's page says 1.3.5). For
  example, `apptainer pull docker://vllm/vllm-openai:<tag>` (in a job: `jobs/pull_container.slurm`), then
  run with `--nv`.
- **Login nodes** (login01/02): Rocky 10.2, 168 cores, about 300 users, per user 4 cores and 20–24 GB,
  `/local` 11 GB. System `python3` 3.12.13 without the pip module or `Python.h` (don't count on numpy);
  no `uv` on PATH (put a binary into `$DATA`); no GPUs or `nvidia-smi`. Internet works, but one HF
  sample came at about 52 MB/s. Helpers that need packages: `$DATA/envs/<env>/bin/python` or
  `apptainer exec -B $DATA,$SCRATCH $SIF python3 …`.
- **Compute nodes** reach HF, PyPI, docker.io and nvcr.io; HF downloads ran at 0.8–2.1 GB/s
  (AGENTS.md §6 Downloads).
- **System libraries differ from a typical workstation:** EL10 has cairo 1.18.2, pixman 0.43.4, freetype 2.13.2
  (an Ubuntu workstation had cairo 1.18.0). The same rendering code built a corpus with a different hash (up to
  about 15% of test icons differed). For bit-identical data, build once and copy with a checksum check.

## Support

Matrix room `#asc-musica-users:tuwien.ac.at` (general: `#asc:tuwien.ac.at`), or email support@asc.ac.at /
https://support.asc.ac.at. Office hours 09:00 to 17:00 Vienna time. One ticket per issue, with username,
job IDs, script and log paths, and the exact error text. Status page: https://status.asc.ac.at.
Don't customise `~/.bashrc` on the cluster (ASC: it can make jobs fail).
