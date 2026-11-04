#!/bin/bash
#SBATCH --job-name=collect_after
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

# Report a job that was already running when its launch script got the results hook (Slurm keeps its own copy of the
# script, so a change doesn't reach it). Submit from the sac-ssamba root:
#   sbatch --dependency=afterany:<job id> scripts/collect_after.sbatch <task> <exp dir> <job id> <its log>
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
TASK=${1:?usage: collect_after.sbatch <task> <exp dir> <job id> <log>}
EXP_DIR=${2:?}
JOB=${3:?}
LOG=${4:?}
state=$(sacct -j "$JOB" -X -n -o State%20 | head -1 | awk '{print $1}')
case "$state" in
    COMPLETED) status=done ;;
    CANCELLED|TIMEOUT|PREEMPTED) status=interrupted ;;
    *) status=failed ;;
esac
echo "job ${JOB}: ${state} -> ${status}"
python3 "${REPO_ROOT}/scripts/collect_result.py" --task "$TASK" --exp_dir "$EXP_DIR" --job_id "$JOB" --status "$status" --log "$LOG"
bash "${REPO_ROOT}/scripts/sync_results.sh" "results: ${TASK} $(basename "$EXP_DIR") job ${JOB} ${status}"
