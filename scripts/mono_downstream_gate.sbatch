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
# Usage: sbatch --dependency=afterany:<pretraining job> scripts/mono_downstream_gate.sbatch <experiment dir>
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
source "${REPO_ROOT}/scripts/mono_ds_paths.sh"
EXP=$1
[ -f "${EXP}/models/audio_model.final.pth" ] || { echo "NOT SUBMITTED: pretraining did not finish, no audio_model.final.pth in ${EXP}"; exit 1; }
[ -f "${SAC_EXP_ROOT}/_smoke/mono_ds/.passed" ] || { echo "NOT SUBMITTED: smoke test has not passed"; exit 1; }
cd "${REPO_ROOT}"
for v in $(env | grep -o '^SLURM[A-Z_]*'); do unset "$v"; done
sbatch src/finetune/speechcommands_v2/run_sc_amba.sh "${EXP}"
sbatch src/finetune/iemocap/run_er.sh "${EXP}" 1e-4 ssamba_local false   # the amba-* runs named _1e-5 ran at 1e-4
sbatch src/finetune/voxceleb1/run_sid.sh "${EXP}" 1e-4 ssamba_local false
