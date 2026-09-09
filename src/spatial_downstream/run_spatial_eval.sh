#!/bin/bash
# Script to run spatial downstream finetuning and evaluation for SSAMBA models
# Usage:
#   bash src/spatial_downstream/run_spatial_eval.sh <model_checkpoint> [trainmode] [ds_embed] [gpu_id]

set -e

MODEL_PATH=${1:-""}
TRAINMODE=${2:-"finetune"}    # choices: finetune, lineareval, scratchLOW
DS_EMBED=${3:-"spat"}         # choices: spat, spec, spec_spat
GPU_ID=${4:-"0"}

if [ -z "$MODEL_PATH" ] && [ "$TRAINMODE" != "scratchLOW" ]; then
    echo "Usage: bash src/spatial_downstream/run_spatial_eval.sh <path_to_model_ckpt.pth> [finetune|lineareval] [spat|spec|spec_spat] [gpu_id]"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SSAMBA_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

export PYTHONPATH="${SSAMBA_ROOT}/src:${PYTHONPATH}"

TASKS=("TDOA" "DRR" "T60" "C50" "ABS")

echo "======================================================="
echo "Starting SSAMBA Spatial Downstream Evaluation"
echo "Model Checkpoint: ${MODEL_PATH}"
echo "Training Mode:    ${TRAINMODE}"
echo "Target Stream:    ${DS_EMBED}"
echo "GPU ID:           ${GPU_ID}"
echo "======================================================="

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "-------------------------------------------------------"
    echo "Running downstream task: ${TASK}"
    echo "-------------------------------------------------------"

    python3 "${SCRIPT_DIR}/run_downstream.py" \
        --ds-train \
        --ds-trainmode "${TRAINMODE}" \
        --simu-exp \
        --ds-nsimroom 8 \
        --ds-task "${TASK}" \
        --ds-embed "${DS_EMBED}" \
        --pretrained_mdl_path "${MODEL_PATH}" \
        --gpu-id "${GPU_ID}"
done

echo ""
echo "======================================================="
echo "All spatial downstream tasks completed successfully."
echo "======================================================="
