#!/bin/bash

#SBATCH --job-name=gramt_er
#SBATCH --partition=power-general-shared-pool
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

# GRAM-T baseline on IEMOCAP emotion recognition (s3prl), for comparison with run_binaural_sac_frozen.sh
# Usage: run_gramt.sh [upstream=gramt_binaural|gramt_binaural_time|gramt_mono] [lr=1e-4] [frozen=true]
# Utterances/folds/labels come from s3prl's meta_data; audio comes from the binaural simulation
# (iemocap_spatial_*.json under SAC_DATA_ROOT), as in run_er.sh for the binaural SAC model.
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
IEMOCAP_ROOT=${IEMOCAP_ROOT:-/storage/data/IEMOCAP_audio_only/IEMOCAP_full_release}  # unused when spatial_data_root is set
PYTHON_BIN=${PYTHON}

upstream_name=${1:-gramt_binaural}
lr=${2:-1e-4}
frozen=${3:-${FROZEN:-true}}

# In s3prl, -f (--upstream_trainable) enables fine-tuning. Omitting -f freezes the upstream model.
if [ "$frozen" = "true" ] || [ "$frozen" = "frozen" ] || [ "$frozen" = "1" ]; then
    trainable_flag=""
    mode_name="freeze"
    expname=emotion_${upstream_name}_frozen_${lr}
else
    trainable_flag="-f"
    mode_name="unfreeze"
    expname=emotion_${upstream_name}_${lr}
fi

expdir=${SCRIPT_DIR}/exp/$expname
mkdir -p $expdir

# Prevent PyTorch/NumPy threading crashes in DataLoader forks
export OMP_NUM_THREADS=1

cd ${S3PRL_ROOT}/s3prl
#for test_fold in fold1 fold2 fold3 fold4 fold5;
for test_fold in fold1;
do
  echo "running cross-validation on $test_fold"
  mkdir -p $expdir/${mode_name}_cross-valid-on-${test_fold}
  PYTHONPATH=${S3PRL_ROOT} $PYTHON_BIN run_downstream.py --expdir $expdir/${mode_name}_cross-valid-on-${test_fold} -m train -u $upstream_name \
      -d emotion -c ${S3PRL_ROOT}/s3prl/downstream/emotion/config.yaml -s hidden_states \
      -o "config.optimizer.lr=$lr,,config.downstream_expert.datarc.test_fold='$test_fold',,config.downstream_expert.datarc.root=${IEMOCAP_ROOT},,config.downstream_expert.datarc.meta_data=${S3PRL_ROOT}/s3prl/downstream/emotion/meta_data,,config.downstream_expert.datarc.spatial_data_root=${SAC_DATA_ROOT}" \
      $trainable_flag -a
done
