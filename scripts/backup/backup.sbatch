#!/bin/bash
#SBATCH --job-name=backup_scratch
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --constraint=ib_dj
#SBATCH --exclude=compute-0-351
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/a/home/cc/students/enginer/yotams2/exp-records/backup_logs/%x_%j.out

# Backs up /scratch200/yotams2 experiment outputs to /moshe-data and lightweight records to ~/exp-records
# (policy: backup.py docstring; overrides: keep.txt, skip.txt next to this file).
#   sbatch scripts/backup/backup.sbatch            one backup, then the next one is queued for tomorrow (daily chain)
#   sbatch scripts/backup/backup.sbatch --dry-run  report only, no chain
# scrontab is disabled on this cluster, hence the self-resubmitting chain. Stop it: touch ~/exp-records/STOP_CHAIN
# (and scancel the pending job). Only one pending backup job is ever queued.
set -uo pipefail
HERE=/a/home/cc/students/enginer/yotams2/sac-ssamba/scripts/backup
REC=$HOME/exp-records
mkdir -p "$REC/backup_logs"
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $*"
python3 "$HERE/backup.py" "$@"; rc=$?
echo "$(date) backup.py exit $rc"

if [[ " $* " != *" --dry-run "* ]] && [ -n "${SLURM_JOB_ID:-}" ] && [ ! -e "$REC/STOP_CHAIN" ]; then
    pending=$(squeue -u "$USER" -h -n backup_scratch -t PENDING -o %i | grep -v "^${SLURM_JOB_ID}$" | head -1)
    if [ -z "$pending" ]; then
        sbatch --begin=now+1day "$HERE/backup.sbatch" | sed 's/^/next: /'
    else
        echo "next: already queued ($pending)"
    fi
fi
exit $rc
