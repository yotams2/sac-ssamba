#!/bin/bash
#SBATCH --job-name=mono_ds_gate
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Runs when a pretraining job ends, whatever its state: afterany also fires on a time limit or failure. Submits the 3
# fine-tuning runs only if pretraining reached its end, and with the settings of the amba-* baseline runs.
# Usage: sbatch --dependency=afterany:<pretraining job> scripts/mono_downstream_gate.sbatch <experiment dir> \
#            [resubmits left] [pretraining --export list, e.g. RECON_LAMBDA=10.0,CLASSIF_LAMBDA=1.0,EXP_NAME_SUFFIX=-x]
# With resubmits left > 0, a pretraining that stopped early (2-day limit) is submitted again with the same exports,
# which continues from models/latest.pth, and this gate chains itself after it with one resubmit fewer. It only
# resubmits when latest.pth moved since the last gate, so a run that crashes before saving is not retried forever.
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
source "${REPO_ROOT}/scripts/mono_ds_paths.sh"
EXP=$1 LEFT=${2:-0} PRE_EXPORTS=${3:-}
cd "${REPO_ROOT}"
for v in $(env | grep -o '^SLURM[A-Z_]*'); do unset "$v"; done
if [ ! -f "${EXP}/models/audio_model.final.pth" ]; then
    LATEST=${EXP}/models/latest.pth STAMP=${EXP}/models/.gate_latest_mtime
    if [ "${LEFT}" -gt 0 ] && [ -n "${PRE_EXPORTS}" ] && [ -f "${LATEST}" ] \
        && [ "$(stat -c %Y "${LATEST}")" != "$(cat "${STAMP}" 2>/dev/null)" ]; then
        stat -c %Y "${LATEST}" > "${STAMP}"
        PRE=$(sbatch --parsable --exclude='compute-0-[18,53,58,300,376,419,420]' --export=ALL,${PRE_EXPORTS} scripts/mono_sac.sbatch)
        GATE=$(sbatch --parsable --dependency=afterany:${PRE} scripts/mono_downstream_gate.sbatch "${EXP}" $((LEFT - 1)) "${PRE_EXPORTS}")
        echo "RESUBMITTED: pretraining not finished; pretraining ${PRE}, gate ${GATE} ($((LEFT - 1)) resubmits left)"
        exit 0
    fi
    echo "NOT SUBMITTED: pretraining did not finish, no audio_model.final.pth in ${EXP} (resubmits left ${LEFT})"; exit 1
fi
[ -f "${SAC_EXP_ROOT}/_smoke/mono_ds/.passed" ] || { echo "NOT SUBMITTED: smoke test has not passed"; exit 1; }
sbatch src/finetune/speechcommands_v2/run_sc_amba.sh "${EXP}"
sbatch src/finetune/iemocap/run_er.sh "${EXP}" 1e-4 ssamba_local false   # the amba-* runs named _1e-5 ran at 1e-4
sbatch src/finetune/voxceleb1/run_sid.sh "${EXP}" 1e-4 ssamba_local false
