#!/usr/bin/env bash
# ==============================================================================
# Full Pipeline: LibriSpeech Spatial Audio Generation + SAC Pretraining
# ==============================================================================
# Usage:
#   chmod +x run_pretrain_pipeline.sh
#   ./run_pretrain_pipeline.sh
#
# Output log will be saved to pretrain_pipeline.log
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="${SCRIPT_DIR}/pretrain_pipeline.log"
PYTHON_BIN="python3"
GPU_ID="[0]"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "======================================================================"
log "Starting LibriSpeech Spatial Audio Generation + SAC Pretraining"
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
# 1. LibriSpeech Spatial Audio Generation (pretrain, preval, pretest)
# ------------------------------------------------------------------------------
run_command "LibriSpeech Spatial Generation (pretrain - 50,000 items)" \
    $PYTHON_BIN gen_simu.py --mode sig --stage pretrain --data_num 50000 \
    --src_dir /storage/data/LibriSpeech \
    --save_to ../data/MicSig/simu --gpus "$GPU_ID"

run_command "LibriSpeech Spatial Generation (preval - 4,000 items)" \
    $PYTHON_BIN gen_simu.py --mode sig --stage preval --data_num 4000 \
    --src_dir /storage/data/LibriSpeech \
    --save_to ../data/MicSig/simu --gpus "$GPU_ID"

run_command "LibriSpeech Spatial Generation (pretest - 4,000 items)" \
    $PYTHON_BIN gen_simu.py --mode sig --stage pretest --data_num 4000 \
    --src_dir /storage/data/LibriSpeech \
    --save_to ../data/MicSig/simu --gpus "$GPU_ID"

# ------------------------------------------------------------------------------
# 2. Generate binaural_train.json and binaural_val.json Manifests
# ------------------------------------------------------------------------------
log "----------------------------------------------------------------------"
log "STARTING STEP: Generate binaural_train.json and binaural_val.json"
log "----------------------------------------------------------------------"

$PYTHON_BIN -c "
import os, glob, json
from pathlib import Path

out_dir = Path('/storage/yotam/ssamba/data')
out_dir.mkdir(parents=True, exist_ok=True)

simu_dir = Path('/storage/yotam/ssamba/data/MicSig/simu')

for split, json_name in [('pretrain', 'binaural_train.json'), ('preval', 'binaural_val.json')]:
    split_dir = simu_dir / split
    wav_files = sorted(glob.glob(str(split_dir / '*.wav')))
    items = [{'wav': w} for w in wav_files]
    out_json = out_dir / json_name
    with open(out_json, 'w') as f:
        json.dump({'data': items}, f, indent=2)
    print(f'Generated {out_json} with {len(items)} items.')
" >> "$LOG_FILE" 2>&1

log "COMPLETED STEP: Generated binaural manifests successfully."

# ------------------------------------------------------------------------------
# 3. CD to src/binaural_dual_stream_sac and run run_binaural_sac.sh
# ------------------------------------------------------------------------------
SAC_DIR="/storage/yotam/ssamba/src/binaural_dual_stream_sac"
log "----------------------------------------------------------------------"
log "Changing directory to ${SAC_DIR} and executing run_binaural_sac.sh"
log "----------------------------------------------------------------------"

cd "$SAC_DIR"
run_command "Binaural SAC Pretraining (run_binaural_sac.sh)" ./run_binaural_sac.sh

log "======================================================================"
log "ALL PRETRAINING PIPELINE STAGES COMPLETED SUCCESSFULLY!"
log "======================================================================"
