#!/bin/bash
# =============================================================
# SSAMBA + SAC Loss Pretraining Launch Script
# 
# This script launches pretraining of the SSAMBA model with the
# Soft Acoustic Contrastive (SAC) loss integrated.
#
# The SAC loss adds geometry-aware regularization using single-channel
# acoustic features (F0, HNR, Spectral Centroid, Spectral Flux, ZCR).
#
# Usage:
#   bash run_sac.sh
# =============================================================

set -x
export TORCH_HOME=../../pretrained_models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Dataset Configuration ----
dataset=librispeech
tr_data=/storage/yotam/ssamba/librispeech_train.json
te_data=/storage/yotam/ssamba/librispeech_eval.json
label_csv=/storage/yotam/ssamba/src/finetune/audioset/data/class_labels_indices.csv
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
num_mel_bins=128
sample_rate=16000

# ---- Model Configuration ----
model_size=base
fshape=16
tshape=16
mask_patch=300

# Vision Mamba parameters
patch_size=16
embed_dim=768
depth=24
rms_norm='false'
residual_in_fp32='false'
fused_add_norm='false'
if_rope='false'
if_rope_residual='false'
bimamba_type="v2"
drop_path_rate=0.1
stride=16
channels=1
num_classes=1000
drop_rate=0.
norm_epsilon=1e-5
if_bidirectional='true'
final_pool_type='none'
if_abs_pos_embed='true'
if_bimamba='false'
if_cls_token='true'
if_devide_out='true'
use_double_cls_token='false'
use_middle_cls_token='false'

# ---- Training Configuration ----
batch_size=64
lr=4e-4
lr_patience=2
n_epochs=10
epoch_iter=4000

# ---- Multi-Loss Configuration ----
recon_lambda=1.0      # Weight for reconstruction loss (MPG): 1.0 (matching EXP2 dual loss baseline)
classif_lambda=0.0    # Weight for classification loss (MPC): set to 0.0 for EXP8 dual loss
sac_lambda=0.02       # Weight for SAC contrastive loss: 0.02 (matching EXP2 dual loss baseline)
sac_temperature=0.8502  # Temperature τ for cosine similarity (winning value from Optuna v5_1 Trial #49)
sac_sigma=1.0        # Gaussian kernel bandwidth σ (ignored by optuna_optimal)

# === PREDEFINED FEATURE LISTS ===
# Uncomment the list that matches your downstream task.

# 1. Speaker Identification (VoxCeleb)
# Captures biometric vocal tract resonances and timbre. Omits channel/noise-sensitive metrics (flux, zcr).
# sac_features="mfcc,formants,f0_mean,f0_var,hnr"
# feature_alias="sid"

# 2. Emotion Recognition (IEMOCAP)
# Focuses on expressive prosody, speaking cadence, and vocal effort.
# sac_features="f0_mean,f0_var,rhythm,hnr,flux"
# feature_alias="emo"

# 3. Audio Event Detection (AudioSet / ESC-50)
# Captures broad spectral envelopes and noise profiles typical of non-speech sounds.
# sac_features="centroid,flux,zcr_mean,hnr"
# feature_alias="events"

# 4. Speech Commands (Keyword Spotting)
# Focuses on phonetic envelope, vowel formants, and fricative/consonant noise.
# sac_features="mfcc,formants,zcr_mean,centroid"
# feature_alias="sc"

# 5. Universal Representation (All Groups)
# Uses the factorized cross-attention to learn all acoustic properties simultaneously.
sac_features="f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm"
feature_alias="universal"

proj_dim=128         # Projection head output dimension
local_sigma_mode="optuna_optimal" # dynamic_batch_median, offline_global_median, chi2_median, sqrt_dim, static_entropy_optimal, optuna_optimal
use_cross_attention="true" # Set to "false" to use the legacy SAC logic (no cross-attention)
num_queries_per_group=1   # Number of queries per acoustic feature group

# To resume from a checkpoint, set this to the path of the .pth file (e.g., ./exp/.../models/audio_model.15.pth)
resume_checkpoint="" 

# ---- Experiment Description ----
# Free-text description of the run to easily identify it later. This is saved to description.log in the exp_dir.
exp_description="Experiment 8: Optuna-Calibrated Factorized SAC (Dual Loss: recon_lambda=1.0, classif_lambda=0.0, sac_lambda=0.02, tau=0.8502, mode=optuna_optimal v5_1, BS64 LR 4e-4)"
exp_name_suffix="-exp8_optuna_v5_1"

# ---- Experiment Directory ----
exp_dir=./exp/sac-${model_size}-f${fshape}-t${tshape}-b${batch_size}-lr${lr}-lam${sac_lambda}-sig${sac_sigma}-feat_${feature_alias}-mode_${local_sigma_mode}-${dataset}${exp_name_suffix}
mkdir -p ${exp_dir}/models

# Save description
echo "${exp_description}" > "${exp_dir}/description.log"

# ---- Launch ----
CUDA_CACHE_DISABLE=1 PYTHONUNBUFFERED=1 python -u -W ignore run_pretrain_sac.py \
    --sac-loss \
    --use_wandb \
    --dataset ${dataset} \
    --data-train ${tr_data} \
    --data-val ${te_data} \
    --label-csv ${label_csv} \
    --exp-dir ${exp_dir} \
    --lr ${lr} \
    --lr_patience ${lr_patience} \
    --n-epochs ${n_epochs} \
    --batch-size ${batch_size} \
    --num-workers 16 \
    --n-print-steps 100 \
    --epoch_iter ${epoch_iter} \
    --task pretrain_joint \
    --mask_patch ${mask_patch} \
    --dataset_mean ${dataset_mean} \
    --dataset_std ${dataset_std} \
    --target_length ${target_length} \
    --num_mel_bins ${num_mel_bins} \
    --sample_rate ${sample_rate} \
    --model_size ${model_size} \
    --fshape ${fshape} \
    --tshape ${tshape} \
    --recon-lambda ${recon_lambda} \
    --classif-lambda ${classif_lambda} \
    --sac-lambda ${sac_lambda} \
    --sac-temperature ${sac_temperature} \
    --sac-sigma ${sac_sigma} \
    --sac_features ${sac_features} \
    --local_sigma_mode ${local_sigma_mode} \
    --use_cross_attention ${use_cross_attention} \
    --use_checkpointing true \
    --num_queries_per_group ${num_queries_per_group} \
    --proj-dim ${proj_dim} \
    --patch_size ${patch_size} \
    --embed_dim ${embed_dim} \
    --depth ${depth} \
    --rms_norm ${rms_norm} \
    --residual_in_fp32 ${residual_in_fp32} \
    --fused_add_norm ${fused_add_norm} \
    --if_rope ${if_rope} \
    --if_rope_residual ${if_rope_residual} \
    --bimamba_type ${bimamba_type} \
    --drop_path_rate ${drop_path_rate} \
    --stride ${stride} \
    --channels ${channels} \
    --num_classes ${num_classes} \
    --drop_rate ${drop_rate} \
    --norm_epsilon ${norm_epsilon} \
    --if_bidirectional ${if_bidirectional} \
    --final_pool_type ${final_pool_type} \
    --if_abs_pos_embed ${if_abs_pos_embed} \
    --if_bimamba ${if_bimamba} \
    --if_cls_token ${if_cls_token} \
    --if_devide_out ${if_devide_out} \
    --use_double_cls_token ${use_double_cls_token} \
    --use_middle_cls_token ${use_middle_cls_token} \
    --resume "${resume_checkpoint}"
