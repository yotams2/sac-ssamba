#!/bin/bash
#SBATCH --job-name=prune_now
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --constraint=ib_dj
#SBATCH --exclude=compute-0-351
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Manual prune with a shorter age window (11-25: quota full; finished runs' resume states < 24 h old).
# Usage: sbatch scripts/backup/prune_now.sbatch <min age h> [--dry-run]
python3 ~/sac-ssamba/scripts/backup/prune.py --min-age-h "$1" ${2:-}; echo "$(date) exit $?"
