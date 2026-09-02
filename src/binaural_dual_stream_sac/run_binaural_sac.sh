#!/bin/bash
# ==============================================================================
# Binaural Dual-Stream SSAMBA + Dual-Branch SAC Pretraining Script
# ==============================================================================

# Change directory to the script's directory so exp/ is saved here
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

# CUDA / GPU environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,4,5"}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Datasets (absolute paths)
data_train="/scratch/yotam/ssamba/data/binaural_train.json"
data_val="/scratch/yotam/ssamba/data/binaural_val.json"

# Training Hyperparameters
batch_size=32
lr=1e-4
epochs=10
epoch_iter=4000
time_mask_ratio=0.5

# Loss Lambdas
sac_spec_lambda=1.0
sac_spat_lambda=1.0
recon_lambda=1.0

# Sigma calculation mode (default: offline_global_median)
local_sigma_mode="offline_global_median"

# WandB Experiment Tracking
use_wandb=true
wandb_project="SSAMBA-Binaural"
exp_name_suffix="-binaural_ccsr_dual_sac"
exp_description="Binaural SSAMBA CCSR pretraining with dual SAC branches (Spectral CA branch + Spatial Projection head), offline_global_median sigma."

exp_name="binaural-sac-bs${batch_size}-lr${lr}-mode_${local_sigma_mode}${exp_name_suffix}"
exp_dir="./exp/${exp_name}"

mkdir -p ${exp_dir}/models

# Save experiment description
echo "${exp_description}" > ${exp_dir}/description.log

echo "Starting Binaural SSAMBA Pretraining..."
echo "  Script Directory:     ${SCRIPT_DIR}"
echo "  Experiment Directory: ${exp_dir}"
echo "  Local Sigma Mode:     ${local_sigma_mode}"

python3 run_binaural_amba.py \
  --data_train ${data_train} \
  --data_val ${data_val} \
  --exp_dir ${exp_dir} \
  --exp_name ${exp_name} \
  --exp_description "${exp_description}" \
  --wandb_project ${wandb_project} \
  $( [ "${use_wandb}" = "true" ] && echo "--use_wandb" ) \
  --batch_size ${batch_size} \
  --lr ${lr} \
  --epochs ${epochs} \
  --epoch_iter ${epoch_iter} \
  --sac_spec_lambda ${sac_spec_lambda} \
  --sac_spat_lambda ${sac_spat_lambda} \
  --recon_lambda ${recon_lambda} \
  --local_sigma_mode ${local_sigma_mode} \
  --time_mask_ratio ${time_mask_ratio}
