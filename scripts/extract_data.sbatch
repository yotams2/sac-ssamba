#!/bin/bash
#SBATCH --job-name=extract_sac_data
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out

# Submit from the sac-ssamba root: sbatch scripts/extract_data.sbatch
# Extract data.tar (top-level "data/") into /scratch200/yotams2/sac-ssamba/data.
# Overwrites files from the earlier partial extraction, so a truncated last file gets rewritten too.
set -euo pipefail
cd /scratch200/yotams2/sac-ssamba
echo "start: $(date)"
tar -xf data.tar --checkpoint=100000 --checkpoint-action=echo="%T"
echo "done: $(date)"
ls data data/*/simu
