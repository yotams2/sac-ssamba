#!/bin/bash

#SBATCH --job-name=amba_er
#SBATCH --gres=gpu:a40:1         # Request an A40 GPU
#SBATCH --cpus-per-task=12        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

set -x
# source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
# conda activate superv
export TORCH_HOME=../../pretrained_models

# Default parameters
# Available models: ssast_patch400_base, ssamba_sac_feat_universal_mode_sqrt_dim, ssamba_sac_feat_universal_mode_offline_global_median, etc.
model_input=${1:-ssast_patch400_base}  # default to 'ssast_patch400_base'
lr=${2:-1e-5}

# Check if model_input is a directory path
if [ -d "$model_input" ]; then
    upstream_args="-u ssamba_local -k ${model_input}/models/best_audio_model.pth"
    expname_prefix=$(basename "$model_input")
elif [ -f "$model_input" ]; then
    upstream_args="-u ssamba_local -k ${model_input}"
    expname_prefix=$(basename $(dirname "$model_input"))
else
    upstream_args="-u $model_input"
    expname_prefix=$model_input
fi

expname=emotion_${expname_prefix}_${lr}
expdir=./exp/$expname
mkdir -p $expdir

# Prevent PyTorch/NumPy threading crashes in DataLoader forks
export OMP_NUM_THREADS=1

# Use 6.0 second window for IEMOCAP
export SSAMBA_WINDOW_SECS=6.0

#for test_fold in fold1 fold2 fold3 fold4 fold5;
for test_fold in fold1;
do
  echo "running cross-validation on $test_fold"
  mkdir -p $expdir/unfreeze_cross-valid-on-${test_fold}; mkdir -p ./log/emotion/unfreeze_cross-valid-on-${test_fold}
  PYTHONPATH=/storage/yotam/s3prl python3 /storage/yotam/s3prl/s3prl/run_downstream.py --expdir $expdir/unfreeze_cross-valid-on-${test_fold} -m train $upstream_args -d emotion -c /storage/yotam/s3prl/s3prl/downstream/emotion/config.yaml -s hidden_states -o "config.downstream_expert.datarc.test_fold='$test_fold'" -f -a
done
