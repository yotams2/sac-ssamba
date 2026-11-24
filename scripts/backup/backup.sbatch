#!/bin/bash
#SBATCH --job-name=backup_scratch
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --constraint=ib_dj
#SBATCH --exclude=compute-0-351
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/a/home/cc/students/enginer/yotams2/exp-records/backup_logs/%x_%j.out

# Backs up /scratch200/yotams2 experiment outputs to /moshe-data and lightweight records to ~/exp-records
# (policy: backup.py docstring; overrides: keep.txt, skip.txt next to this file), then prunes finished runs (prune.py).
#   sbatch scripts/backup/backup.sbatch            one backup, then the next one is queued 12 h later (chain; daily until 11-23)
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
# Then delete regenerable weights of finished runs (policy: prune.py docstring), only after a clean backup
if [ $rc -eq 0 ] && [[ " $* " != *" --dry-run "* ]]; then
    python3 "$HERE/prune.py"; echo "$(date) prune.py exit $?"
fi

# Scratch usage after the prune (added 11-23, after the quota filled up mid-day): log it; above 90% put a warning on
# top of ~/STATUS.md (replaced each run, removed when back under).
q=$(timeout 300 quota -s -f /scratch200 2>/dev/null | tail -1 | awk '{print $1, $3}' | tr -d 'G*')
if [ -n "$q" ]; then
    used=${q% *}; lim=${q#* }
    # per-job temp dirs (TMPDIR of jobs sourcing SAC-SAR-SSL scripts/job_tmp.sh, 11-24): size and count, 4th / 5th columns
    T=/scratch200/$USER/tmp
    tmp_gb=$( [ -d "$T" ] && timeout 300 du -s -BG "$T" 2>/dev/null | cut -f1 | tr -d G || echo 0)
    tmp_n=$( [ -d "$T" ] && ls "$T" | wc -l || echo 0)
    printf "%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$used" "$lim" "${tmp_gb:-?}" "$tmp_n" >> "$REC/quota_log.tsv"
    echo "$(date) scratch ${used}/${lim} GB (per-job tmp ${tmp_gb:-?} GB in $tmp_n dirs)"
    S=$HOME/STATUS.md
    sed -i '/^\*\*WARNING: scratch quota/d' "$S"
    if [ "$used" -ge $(( lim * 9 / 10 )) ] 2>/dev/null; then
        sed -i "1a **WARNING: scratch quota ${used}/${lim} GB at $(date '+%F %H:%M')** (per-job tmp ${tmp_gb:-?} GB in $tmp_n dirs under $T; backup job; prune ran; see ~/exp-records/quota_log.tsv)" "$S"
    fi
fi

if [[ " $* " != *" --dry-run "* ]] && [ -n "${SLURM_JOB_ID:-}" ] && [ ! -e "$REC/STOP_CHAIN" ]; then
    pending=$(squeue -u "$USER" -h -n backup_scratch -t PENDING -o %i | grep -v "^${SLURM_JOB_ID}$" | head -1)
    if [ -z "$pending" ]; then
        sbatch --begin=now+12hours "$HERE/backup.sbatch" | sed 's/^/next: /'
    else
        echo "next: already queued ($pending)"
    fi
fi
exit $rc
