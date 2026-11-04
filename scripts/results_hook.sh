# Report a run to the `results` branch. Source this from a launch script, after the experiment dir is known:
#
#   source <repo>/scripts/results_hook.sh && results_hook <task> <exp_dir> [probe_file]
#
# It records the run as `running` right away, then done / failed / interrupted when the script exits. The start record
# also catches what no trap can: a node failure, or a SIGKILL 30 s after a preemption (Slurm's KillWait).
# Does nothing without the worktree (RunAI), and never changes the job's exit code.

_RH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_rh_git_id() {
    git --no-optional-locks -C "$1" describe --always --dirty --abbrev=40 --match=__none__ 2>/dev/null || true
}

# _rh_record <status> <seconds allowed> [exit code]
_rh_record() {
    local py
    py=$(command -v python3 || echo /bin/python3)
    timeout "$2" "$py" "${_RH_DIR}/collect_result.py" --task "$_RH_TASK" --exp_dir "$_RH_EXP" --job_id "$_RH_JOB" \
        --status "$1" --git_commit "$_RH_COMMIT" --s3prl_commit "$_RH_S3PRL" \
        ${_RH_LOG:+--log "$_RH_LOG"} ${_RH_PROBE:+--probe_file "$_RH_PROBE"} ${3:+--exit_code "$3"} || true
    RESULTS_LOCK_WAIT=$2 timeout "$2" bash "${_RH_DIR}/sync_results.sh" \
        "results: ${_RH_TASK} $(basename "$_RH_EXP") job ${_RH_JOB} ${1}" || true
}

_rh_exit() {
    local rc=$?
    { set +x; set +e; } 2>/dev/null
    trap '' TERM
    if [ -n "${_RH_SIGNALLED:-}" ] || [ "$rc" = 143 ]; then
        # About 30 s before the SIGKILL. A sync killed halfway is cleaned up by the next one.
        _rh_record interrupted 12 143
    elif [ "$rc" = 0 ]; then
        _rh_record done 200 0
    else
        _rh_record failed 200 "$rc"
    fi
}

results_hook() {
    [ -e "${SAC_RESULTS_ROOT:-$HOME/sac-ssamba-results}/.git" ] || return 0
    { local xtrace=$-; set +x; } 2>/dev/null
    _RH_TASK=$1
    _RH_EXP=$(realpath -m "$2")
    _RH_PROBE=${3:+$(realpath -m "$3")}
    _RH_JOB=${SLURM_JOB_ID:-local-$(date +%Y%m%d%H%M%S)}
    # At job start, not submit: the job runs the working tree as it is now, however long it sat in the queue
    _RH_COMMIT=$(_rh_git_id "${_RH_DIR}/..")
    _RH_S3PRL=""
    [ -n "${S3PRL_ROOT:-}" ] && _RH_S3PRL=$(_rh_git_id "$S3PRL_ROOT")
    _RH_LOG=""
    [ -n "${SLURM_JOB_ID:-}" ] && _RH_LOG=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *StdOut=//p' | head -1)
    _rh_record running 60
    # Without a TERM trap bash still runs the EXIT trap on SIGTERM, but with $?=0: a preempted job would count as done
    trap '_RH_SIGNALLED=1; exit 143' TERM
    trap _rh_exit EXIT
    case "$xtrace" in *x*) set -x ;; esac
    return 0
}
