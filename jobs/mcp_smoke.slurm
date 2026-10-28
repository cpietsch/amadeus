#!/bin/bash
#SBATCH --job-name=mcp-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=zen4_0768
#SBATCH --qos=dev_zen4_0768
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
# Smallest end-to-end check of the MCP server's job path (hold, check, release, wait, log): one core, seconds.
echo "host $(hostname) at $(date -Is), revision $(cat REVISION 2>/dev/null)"
sleep 5
echo done
