#!/bin/bash

#SBATCH --job-name=amba_sid
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,58,300,376,419,420]  # 18: Blackwell, no kernels in this torch; the rest are the slow cards
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out  # relative to the submit dir: submit from the sac-ssamba root

set -x
# source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
# conda activate superv
SCRIPT_PATH=${BASH_SOURCE[0]}
# under sbatch BASH_SOURCE is slurm's spooled copy, so take the submitted script path instead
if [ "$(basename "$SCRIPT_PATH")" = "slurm_script" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "${SCRIPT_DIR}/../../../env.sh"
S3PRL_ROOT=${S3PRL_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)/sac-s3prl}
export SAC_SSAMBA_SRC=${SAC_SSAMBA_SRC:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
export TORCH_HOME="${SCRIPT_DIR}/../../pretrained_models"

# Default parameters
# Available models: ssast_patch400_base, ssamba_sac_feat_universal_mode_sqrt_dim, ssamba_sac_feat_universal_mode_offline_global_median, etc.
model_input=${1:-ssast_patch400_base}  # default to 'ssast_patch400_base'
lr=${2:-1e-4}
if [[ "$(basename "$model_input")" == *"binaural"* ]]; then  # mono SAC runs are called sac-*, and the repo path holds "sac" too
    default_upstream="binaural_sac_local"
else
    default_upstream="ssamba_local"
fi
upstream_name=${3:-$default_upstream}
frozen=${4:-${FROZEN:-false}}

# check if using binaural sac to override dataset
if [[ "$upstream_name" == *"binaural"* ]] || [[ "$upstream_name" == *"sac"* ]]; then
    file_path_override=",,config.downstream_expert.datarc.file_path=${SAC_DATA_ROOT}/VoxCeleb1"
else
    file_path_override=",,config.downstream_expert.datarc.file_path=${VOXCELEB1_ROOT:-${SAC_DATA_ROOT}/mono/voxceleb1}"
fi
file_path_override="${file_path_override},,config.downstream_expert.datarc.meta_data=${S3PRL_ROOT}/s3prl/downstream/voxceleb1/veri_test_class.txt"

# Check if model_input is a directory path
if [ -d "$model_input" ]; then
    upstream_args="-u $upstream_name -k ${model_input}/models/best_audio_model.pth"
    expname_prefix=$(basename "$model_input")
elif [ -f "$model_input" ]; then
    upstream_args="-u $upstream_name -k ${model_input}"
    expname_prefix=$(basename $(dirname "$model_input"))
else
    upstream_args="-u $model_input"
    expname_prefix=$model_input
fi

# Configure frozen vs fine-tuned (trainable upstream)
# In s3prl, -f (--upstream_trainable) enables fine-tuning. Omitting -f freezes the upstream model.
if [ "$frozen" = "true" ] || [ "$frozen" = "frozen" ] || [ "$frozen" = "1" ]; then
    trainable_flag=""
    expname=sid_${expname_prefix}_frozen_${lr}
    echo "Running in FROZEN upstream mode (upstream weights frozen)"
else
    trainable_flag="-f"
    expname=sid_${expname_prefix}_${lr}
    echo "Running in FINE-TUNING mode (upstream weights trainable)"
fi

expdir=${SAC_EXP_ROOT:+${SAC_EXP_ROOT}/voxceleb1}  # set by env.sh; unset: ./exp
expdir=${expdir:-./exp}/$expname
mkdir -p $expdir

# Prevent PyTorch/NumPy threading crashes in DataLoader forks
export OMP_NUM_THREADS=1

# Report this run to the results branch, including failures (scripts/results_hook.sh)
[ -z "${NO_RESULTS_HOOK:-}" ] && source "${SCRIPT_DIR}/../../../scripts/results_hook.sh" && results_hook voxceleb1 "$expdir"
cd ${S3PRL_ROOT}/s3prl
PYTHONPATH=${S3PRL_ROOT}:${VIM_ROOT}:${VIM_ROOT}/vim:${VIM_ROOT}/mamba-1p1p1 ${PYTHON} run_downstream.py --expdir $expdir -m train $upstream_args -d voxceleb1 -c ${S3PRL_ROOT}/s3prl/downstream/voxceleb1/config.yaml -s hidden_states -o "config.optimizer.lr=$lr,,config.runner.total_steps=50000$file_path_override${EXTRA_OVERRIDES:-}" $trainable_flag -a

