#!/bin/bash
# =============================================================
# Launch Script for Optuna Hyperparameter Optimization of Factorized SAC Loss
# =============================================================

set -e

STUDY_NAME="ssamba_sac_hyperparameter_tuning"
BASE_OUT_DIR="/storage/yotam/ssamba/src/metrics/optuna_hyperparameters_search"
OUT_DIR="${BASE_OUT_DIR}/${STUDY_NAME}"
mkdir -p ${OUT_DIR}

# Defaults to python in PATH or environment
PYTHON_BIN=${PYTHON_BIN:-python}

echo "============================================================="
echo "  Launching SSAMBA Factorized SAC Optuna Search"
echo "  Study Name: ${STUDY_NAME}"
echo "  Output Directory: ${OUT_DIR}"
echo "  Python Binary: ${PYTHON_BIN}"
echo "============================================================="

CUDA_CACHE_DISABLE=1 PYTHONUNBUFFERED=1 ${PYTHON_BIN} -u /storage/yotam/ssamba/src/sac/optuna_sac_search.py \
    --data-train /storage/yotam/ssamba/librispeech_train.json \
    --data-val /storage/yotam/ssamba/librispeech_eval.json \
    --label-csv /storage/yotam/ssamba/src/finetune/audioset/data/class_labels_indices.csv \
    --sac-features "f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm" \
    --batch-size 64 \
    --lr 4e-4 \
    --probe-steps 100 \
    --eval-last-steps 20 \
    --n-trials 60 \
    --use-cross-attention true \
    --mode shared_sigma \
    --tau-min 0.05 \
    --tau-max 1.00 \
    --sigma-scale-min 0.10 \
    --sigma-scale-max 0.80 \
    --min-rank 2.5 \
    --max-uniformity -0.45 \
    --study-name "${STUDY_NAME}" \
    --out-dir "${OUT_DIR}" \
    "$@"
