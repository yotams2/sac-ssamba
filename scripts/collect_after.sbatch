#!/bin/bash
#SBATCH --job-name=collect_after
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

# Close the record of a job that ended without being able to report it. Slurm's SIGTERM (preemption, time limit,
# scancel) reaches only the batch shell, which can't run a trap while its foreground command runs, so it gets SIGKILLed
# 30 s later with the record still at `running`. results_hook.sh submits this at every job start. It does nothing if the
# job reported on its own, or was requeued. By hand, for a job started before it had the hook, from the sac-ssamba root:
#   sbatch --dependency=afterany:<job id> scripts/collect_after.sbatch <task> <exp dir> <job id> <log> [commit] [s3prl commit] [probe file]
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
TASK=${1:?usage: collect_after.sbatch <task> <exp dir> <job id> <log> [commit] [s3prl commit] [probe file]}
EXP_DIR=${2:?}
JOB=${3:?}
LOG=${4:-}
COMMIT=${5:-}
S3PRL=${6:-}
PROBE=${7:-}

ROOT=${SAC_RESULTS_ROOT:-$HOME/sac-ssamba-results}
SUMMARY="${ROOT}/results/${TASK}/$(basename "$EXP_DIR")__${JOB}/summary.json"
if [ -f "$SUMMARY" ] && ! grep -q '"status": "running"' "$SUMMARY"; then
    echo "job ${JOB} already reported"
    exit 0
fi
state=$(sacct -j "$JOB" -X -n -o State%20 | head -1 | awk '{print $1}')
case "$state" in
    COMPLETED) status=done ;;
    CANCELLED|TIMEOUT|PREEMPTED|NODE_FAIL|DEADLINE) status=interrupted ;;
    FAILED|OUT_OF_MEMORY|BOOT_FAIL) status=failed ;;
    *) echo "job ${JOB} is ${state}, not finished"; exit 0 ;;   # requeued: its next start submits a new one of these
esac
echo "job ${JOB}: ${state} -> ${status}"
python3 "${REPO_ROOT}/scripts/collect_result.py" --task "$TASK" --exp_dir "$EXP_DIR" --job_id "$JOB" --status "$status" \
    ${LOG:+--log "$LOG"} ${COMMIT:+--git_commit "$COMMIT"} ${S3PRL:+--s3prl_commit "$S3PRL"} ${PROBE:+--probe_file "$PROBE"}
bash "${REPO_ROOT}/scripts/sync_results.sh" "results: ${TASK} $(basename "$EXP_DIR") job ${JOB} ${status}"
