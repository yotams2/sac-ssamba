#!/usr/bin/env bash
# ==============================================================================
# Serial Spatialization & Downstream Manifest Generation Pipeline Script
# ==============================================================================
# Usage:
#   chmod +x run_all_remaining.sh
#   ./run_all_remaining.sh
#
# Log output will be saved to generation_pipeline.log
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="${SCRIPT_DIR}/generation_pipeline.log"
PYTHON_BIN="python3"
GPU_ID="[0]"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "======================================================================"
log "Starting Spatial Audio Generation Pipeline for Remaining Datasets"
log "Log File: ${LOG_FILE}"
log "======================================================================"

run_command() {
    local step_name="$1"
    shift
    log "----------------------------------------------------------------------"
    log "STARTING STEP: ${step_name}"
    log "Command: $*"
    log "----------------------------------------------------------------------"
    
    if "$@" >> "$LOG_FILE" 2>&1; then
        log "COMPLETED STEP: ${step_name} successfully."
    else
        local exit_code=$?
        log "ERROR: STEP '${step_name}' failed with exit code ${exit_code}!"
        exit ${exit_code}
    fi
}

export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# 1. Speech Commands v2 (preval & pretest)
# ------------------------------------------------------------------------------
run_command "Speech Commands v2 (preval - 9,981 items)" \
    $PYTHON_BIN gen_simu_speechcommands.py --stage preval --data_num 9981 \
    --sc_root /storage/yotam/ssamba/src/finetune/speechcommands_v2/data/speech_commands_v0.02 \
    --label_csv /storage/yotam/ssamba/src/finetune/speechcommands_v2/data/speechcommands_class_labels_indices.csv \
    --save_to ../data/SpeechCommands/simu --gpus "$GPU_ID"

run_command "Speech Commands v2 (pretest - 11,005 items)" \
    $PYTHON_BIN gen_simu_speechcommands.py --stage pretest --data_num 11005 \
    --sc_root /storage/yotam/ssamba/src/finetune/speechcommands_v2/data/speech_commands_v0.02 \
    --label_csv /storage/yotam/ssamba/src/finetune/speechcommands_v2/data/speechcommands_class_labels_indices.csv \
    --save_to ../data/SpeechCommands/simu --gpus "$GPU_ID"

# ------------------------------------------------------------------------------
# 2. VoxCeleb1 (pretrain, preval, pretest)
# ------------------------------------------------------------------------------
run_command "VoxCeleb1 (pretrain - 140,000 items)" \
    $PYTHON_BIN gen_simu_voxceleb1.py --stage pretrain --data_num 140000 \
    --vox1_root /storage/data/voxceleb1 \
    --meta_file /storage/yotam/s3prl/s3prl/downstream/voxceleb1/veri_test_class.txt \
    --save_to ../data/VoxCeleb1/simu --gpus "$GPU_ID"

run_command "VoxCeleb1 (preval - 4,000 items)" \
    $PYTHON_BIN gen_simu_voxceleb1.py --stage preval --data_num 4000 \
    --vox1_root /storage/data/voxceleb1 \
    --meta_file /storage/yotam/s3prl/s3prl/downstream/voxceleb1/veri_test_class.txt \
    --save_to ../data/VoxCeleb1/simu --gpus "$GPU_ID"

run_command "VoxCeleb1 (pretest - 4,874 items)" \
    $PYTHON_BIN gen_simu_voxceleb1.py --stage pretest --data_num 4874 \
    --vox1_root /storage/data/voxceleb1 \
    --meta_file /storage/yotam/s3prl/s3prl/downstream/voxceleb1/veri_test_class.txt \
    --save_to ../data/VoxCeleb1/simu --gpus "$GPU_ID"

# ------------------------------------------------------------------------------
# 3. IEMOCAP (pretrain, preval, pretest)
# ------------------------------------------------------------------------------
run_command "IEMOCAP (pretrain - 3,556 items)" \
    $PYTHON_BIN gen_simu_iemocap.py --stage pretrain --data_num 3556 \
    --iemocap_root /storage/data/IEMOCAP_audio_only/IEMOCAP_full_release \
    --meta_dir /storage/yotam/s3prl/s3prl/downstream/emotion/meta_data \
    --save_to ../data/IEMOCAP/simu --gpus "$GPU_ID"

run_command "IEMOCAP (preval - 890 items)" \
    $PYTHON_BIN gen_simu_iemocap.py --stage preval --data_num 890 \
    --iemocap_root /storage/data/IEMOCAP_audio_only/IEMOCAP_full_release \
    --meta_dir /storage/yotam/s3prl/s3prl/downstream/emotion/meta_data \
    --save_to ../data/IEMOCAP/simu --gpus "$GPU_ID"

run_command "IEMOCAP (pretest - 1,085 items)" \
    $PYTHON_BIN gen_simu_iemocap.py --stage pretest --data_num 1085 \
    --iemocap_root /storage/data/IEMOCAP_audio_only/IEMOCAP_full_release \
    --meta_dir /storage/yotam/s3prl/s3prl/downstream/emotion/meta_data \
    --save_to ../data/IEMOCAP/simu --gpus "$GPU_ID"

# ------------------------------------------------------------------------------
# 4. Generate Downstream JSON Manifests
# ------------------------------------------------------------------------------
run_command "Build Speech Commands Manifests" \
    $PYTHON_BIN generate_downstream_manifests.py --dataset speechcommands \
    --simu_dir /storage/yotam/ssamba/data/SpeechCommands/simu \
    --out_dir /storage/yotam/ssamba/data

run_command "Build VoxCeleb1 Manifests" \
    $PYTHON_BIN generate_downstream_manifests.py --dataset voxceleb1 \
    --simu_dir /storage/yotam/ssamba/data/VoxCeleb1/simu \
    --out_dir /storage/yotam/ssamba/data

run_command "Build IEMOCAP Manifests" \
    $PYTHON_BIN generate_downstream_manifests.py --dataset iemocap \
    --simu_dir /storage/yotam/ssamba/data/IEMOCAP/simu \
    --out_dir /storage/yotam/ssamba/data

log "======================================================================"
log "ALL STAGES COMPLETED SUCCESSFULLY!"
log "======================================================================"
