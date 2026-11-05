#!/bin/bash
#SBATCH --job-name=sac_pretrain
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18     # RTX PRO 6000 Blackwell (sm_120): torch 2.2.2+cu121 has no kernels for it
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out    # relative to the submit dir: submit from the sac-ssamba root
# ==============================================================================
# Binaural Dual-Stream SSAMBA + Dual-Branch SAC Pretraining Script (Exp A + B)
# Exp A: projection_head_spat uses BatchNorm1d
# Exp B: standardized c_spatial ([-1, 1]) + standardized offline median heuristic
# ==============================================================================

# Change directory to the script's directory so exp/ is saved here
SCRIPT_PATH=${BASH_SOURCE[0]}
# under sbatch BASH_SOURCE is slurm's spooled copy, so take the submitted script path instead
if [ "$(basename "$SCRIPT_PATH")" = "slurm_script" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
fi
SCRIPT_DIR="$( cd "$( dirname "$SCRIPT_PATH" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/../../env.sh"
cd "${SCRIPT_DIR}"

# CUDA / GPU environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Datasets (absolute paths)
data_train="${DATA_TRAIN:-${SAC_DATA_ROOT}/binaural_train.json}"
data_val="${DATA_VAL:-${SAC_DATA_ROOT}/binaural_val.json}"

# Training Hyperparameters
batch_size=32
lr=1e-4
epochs=${EPOCHS:-10}
epoch_iter=4000
time_mask_ratio=0.5
mask_block=${MASK_BLOCK:-16}  # frames; 16 = one patch width, so masked tokens are fully hidden
latest_every=200              # rolling models/latest.pth; the run resumes from it after preemption (segments can die within 30-60 min)

# Frozen SC2/SID/IEMOCAP probes on the spectral encoder (pretrain_probe.py), at step 0 and every probe_every steps; 0 disables.
# ~6.6k clips loaded once into RAM; results in ${exp_dir}/probe_metrics.jsonl
probe_every=${PROBE_EVERY:-${epoch_iter}}

# Loss Lambdas
sac_spec_lambda=1.0
sac_spat_lambda=1.0
recon_lambda=1.0

# SAC Temperatures
spec_sac_temperature=0.3
spat_sac_temperature=0.15

# Sigma calculation mode (default: offline_global_median)
local_sigma_mode="offline_global_median"

# WandB Experiment Tracking
use_wandb=${USE_WANDB:-false}  # not installed in the shared venv
wandb_project="SSAMBA-Binaural"
exp_name_suffix="${EXP_NAME_SUFFIX:--binaural_ccsr_dual_sac_bn_standardized_spatial}"
exp_description="${EXP_DESCRIPTION:-Binaural SSAMBA CCSR pretraining with dual SAC branches (Spectral CA + Spatial Projection Head with BatchNorm1d and Standardized Spatial SAC), offline_global_median sigma.}"

exp_name="binaural-sac-bs${batch_size}-lr${lr}-mode_${local_sigma_mode}${exp_name_suffix}"
exp_dir="${SAC_EXP_ROOT:+${SAC_EXP_ROOT}/binaural_sac_pretrain}"  # set by env.sh; unset: ./exp
exp_dir="${exp_dir:-./exp}/${exp_name}"

mkdir -p ${exp_dir}/models

# Save experiment description
echo "${exp_description}" > ${exp_dir}/description.log

# Report this run to the results branch, including failures (scripts/results_hook.sh)
source "${SCRIPT_DIR}/../../scripts/results_hook.sh" && results_hook binaural_sac_pretrain "${exp_dir}"
echo "Starting Binaural SSAMBA Pretraining (BatchNorm1d + Standardized Spatial SAC)..."
echo "  Script Directory:     ${SCRIPT_DIR}"
echo "  Experiment Directory: ${exp_dir}"
echo "  Local Sigma Mode:     ${local_sigma_mode}"

${PYTHON} run_binaural_amba.py \
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
  --spec_sac_temperature ${spec_sac_temperature} \
  --spat_sac_temperature ${spat_sac_temperature} \
  --local_sigma_mode ${local_sigma_mode} \
  --time_mask_ratio ${time_mask_ratio} \
  --mask_block ${mask_block} \
  --latest_every ${latest_every} \
  --resume "${exp_dir}/models/latest.pth" \
  --probe_every ${probe_every} \
  ${EXTRA_ARGS:-} \
  --use_checkpointing
