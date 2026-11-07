#!/bin/bash
#SBATCH --job-name=mono_ds_smoke
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,58,300,376,419,420]
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Short end-to-end run of the 3 mono downstream tasks on a copy of a mono SAC checkpoint. Writes only under
# $SAC_EXP_ROOT/_smoke/mono_ds and reports nothing to the results branch. scripts/mono_downstream_gate.sbatch needs its .passed.
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
source "${REPO_ROOT}/scripts/mono_ds_paths.sh"
export SAC_EXP_ROOT=${SAC_EXP_ROOT}/_smoke/mono_ds NO_RESULTS_HOOK=1
EXP=${SAC_EXP_ROOT}/sac-base-smoke
export SAC_SSAMBA_SRC=${REPO_ROOT}/src
rm -f "${SAC_EXP_ROOT}/.passed"
S3PRL_ROOT=$(dirname "${REPO_ROOT}")/sac-s3prl
PYTHONPATH=${S3PRL_ROOT}:${VIM_ROOT}:${VIM_ROOT}/vim:${VIM_ROOT}/mamba-1p1p1 "${PYTHON}" scripts/check_mono_upstream_load.py "${EXP}/models/best_audio_model.pth"
SHORT=",,config.runner.total_steps=40,,config.runner.eval_step=20,,config.runner.log_step=20"
EPOCHS=1 bash src/finetune/speechcommands_v2/run_sc_amba.sh "${EXP}"
EXTRA_OVERRIDES=${SHORT} bash src/finetune/iemocap/run_er.sh "${EXP}" 1e-4 ssamba_local false
EXTRA_OVERRIDES=${SHORT} bash src/finetune/voxceleb1/run_sid.sh "${EXP}" 1e-4 ssamba_local false
ls "${SAC_EXP_ROOT}"/speechcommands_v2/*/eval_result.csv
ls "${SAC_EXP_ROOT}"/iemocap/*/unfreeze_cross-valid-on-fold1/log.log "${SAC_EXP_ROOT}"/voxceleb1/*/log.log
touch "${SAC_EXP_ROOT}/.passed"
echo "SMOKE PASSED: $(date)"
