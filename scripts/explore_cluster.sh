#!/bin/bash
# Run on a MUSICA login node to get an overview of the platform.
# Usage: bash scripts/explore_cluster.sh | tee logs/explore-$(date +%F).txt

section() { echo; echo "===== $* ====="; }

section "Identity"
whoami; id; hostname; date -Is

section "OS / kernel"
cat /etc/os-release 2>/dev/null | head -4
uname -r

section "Storage (home / scratch / project)"
echo "HOME=$HOME"
env | grep -iE '^(DATA|SCRATCH|WORK|PROJECT|TMPDIR)' || true
df -h "$HOME" 2>/dev/null
ls -ld /scratch* /data* /gpfs* /lustre* 2>/dev/null || true
command -v mmlsquota >/dev/null && mmlsquota 2>/dev/null || true
command -v lfs >/dev/null && lfs quota -u "$USER" "$HOME" 2>/dev/null || true

section "Slurm partitions"
sinfo -s 2>/dev/null
sinfo -o "%20P %5a %10l %6D %8c %10m %30G %N" 2>/dev/null

section "My accounts / QOS"
sacctmgr show user "$(id -u)" withassoc format=user,defaultaccount,account,qos%60s,defaultqos%20s 2>/dev/null
sacctmgr -nP show qos format=name,maxwall,maxtrespu,grptres 2>/dev/null | head -30

section "Queue (mine / overall)"
squeue -u "$USER" 2>/dev/null
squeue -h 2>/dev/null | wc -l | xargs echo "total jobs in queue:"

section "Modules"
module --version 2>&1 | head -2
module avail 2>&1 | head -80

section "Containers / Python tooling"
for c in apptainer singularity enroot podman conda mamba micromamba uv python3; do
    printf "%-12s %s\n" "$c" "$(command -v $c || echo '-')"
done
