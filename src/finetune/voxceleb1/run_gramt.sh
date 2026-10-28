#!/bin/bash

#SBATCH --job-name=gramt_sid
#SBATCH --partition=gpu-general-pool
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=logs/%x_%j.out  # relative to the submit dir: submit from the sac-ssamba root

# GRAM-T baseline on binaural VoxCeleb1 SID (s3prl), for comparison with run_binaural_sac_frozen.sh
# Usage: run_gramt.sh [upstream=gramt_binaural|gramt_binaural_time|gramt_mono] [lr=1e-4] [frozen=true]
set -x
SCRIPT_PATH=${BASH_SOURCE[0]}
# under sbatch BASH_SOURCE is slurm's spooled copy, so take the submitted script path instead
if [ "$(basename "$SCRIPT_PATH")" = "slurm_script" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "${SCRIPT_DIR}/../../../env.sh"
S3PRL_ROOT=${S3PRL_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)/sac-s3prl}
export SAC_SSAMBA_SRC=${SAC_SSAMBA_SRC:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
PYTHON_BIN=${PYTHON}

upstream_name=${1:-gramt_binaural}
lr=${2:-1e-4}
frozen=${3:-${FROZEN:-true}}

# In s3prl, -f (--upstream_trainable) enables fine-tuning. Omitting -f freezes the upstream model.
if [ "$frozen" = "true" ] || [ "$frozen" = "frozen" ] || [ "$frozen" = "1" ]; then
    trainable_flag=""
    expname=sid_${upstream_name}_frozen_${lr}
else
    trainable_flag="-f"
    expname=sid_${upstream_name}_${lr}
fi

expdir=${SAC_EXP_ROOT}/voxceleb1/$expname
mkdir -p $expdir

# Prevent PyTorch/NumPy threading crashes in DataLoader forks
export OMP_NUM_THREADS=1

cd ${S3PRL_ROOT}/s3prl
PYTHONPATH=${S3PRL_ROOT} $PYTHON_BIN run_downstream.py --expdir $expdir -m train -u $upstream_name \
    -d voxceleb1 -c ${S3PRL_ROOT}/s3prl/downstream/voxceleb1/config.yaml -s hidden_states \
    -o "config.optimizer.lr=$lr,,config.runner.total_steps=50000,,config.downstream_expert.datarc.file_path=${SAC_DATA_ROOT}/VoxCeleb1,,config.downstream_expert.datarc.meta_data=${S3PRL_ROOT}/s3prl/downstream/voxceleb1/veri_test_class.txt" \
    $trainable_flag -a
