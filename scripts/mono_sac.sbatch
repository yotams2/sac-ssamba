#!/bin/bash
#SBATCH --job-name=sac_mono
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18     # RTX PRO 6000 Blackwell (sm_120): torch 2.2.2+cu121 has no kernels for it
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Mono SAC-SSAMBA pretraining with the fixed F0 and HNR, recipe of exp6_true_bs64_ckpt: mpg + 0.02 SAC, no mpc,
# batch 64, lr 4e-4, tau 0.12, offline median sigma, the 15 universal features.
# Needs $SAC_DATA_ROOT/LibriSpeech and $SAC_DATA_ROOT/librispeech_{train,eval}.json.
# Submit from the sac-ssamba root: sbatch scripts/mono_sac.sbatch
# After a preemption or time limit, submit it again: it continues from models/latest.pth.
# Any setting in src/sac/run_sac.sh can be overridden through its variable, e.g. sbatch --export=ALL,CLASSIF_LAMBDA=1.0,EXP_NAME_SUFFIX=-x ...
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
for split in train eval; do
    if [ ! -f "${SAC_DATA_ROOT}/librispeech_${split}.tau.json" ] && [ -z "${TR_DATA:-}" ]; then
        ${PYTHON} "${REPO_ROOT}/scripts/localize_librispeech_manifest.py" "${SAC_DATA_ROOT}/librispeech_${split}.json" \
            "${SAC_DATA_ROOT}/librispeech_${split}.tau.json" "${SAC_DATA_ROOT}/LibriSpeech"
    fi
done
export TR_DATA=${TR_DATA:-${SAC_DATA_ROOT}/librispeech_train.tau.json}
export TE_DATA=${TE_DATA:-${SAC_DATA_ROOT}/librispeech_eval.tau.json}
export LABEL_CSV=${REPO_ROOT}/src/finetune/audioset/data/class_labels_indices.csv
export RECON_LAMBDA=${RECON_LAMBDA:-1.0} CLASSIF_LAMBDA=${CLASSIF_LAMBDA:-0.0} SAC_LAMBDA=${SAC_LAMBDA:-0.02}
export SAC_TEMPERATURE=${SAC_TEMPERATURE:-0.12} LOCAL_SIGMA_MODE=${LOCAL_SIGMA_MODE:-offline_global_median}
export EXP_NAME_SUFFIX=${EXP_NAME_SUFFIX:--exp6_pitchfix}
export EXP_DESCRIPTION=${EXP_DESCRIPTION:-Exp6 recipe rerun on TAU with the fixed F0 and HNR}
export USE_WANDB=${USE_WANDB:-false} LATEST_EVERY=${LATEST_EVERY:-200} NUM_WORKERS=${NUM_WORKERS:-8}
export DIAGNOSTIC_STEPS=${DIAGNOSTIC_STEPS:-0}  # the diagnostics need packages the shared venv leaves out
cd "${REPO_ROOT}/src/sac"
bash run_sac.sh
