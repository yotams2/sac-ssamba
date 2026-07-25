#!/bin/bash
# =============================================================
# Launch Script for Optuna Hyperparameter Optimization of Factorized SAC Loss
# =============================================================

set -e

OUT_DIR="/storage/yotam/ssamba/src/metrics/optuna_hyperparameters_search"
mkdir -p ${OUT_DIR}

# Defaults to python in PATH or environment
PYTHON_BIN=${PYTHON_BIN:-python}

echo "============================================================="
echo "  Launching SSAMBA Factorized SAC Optuna Search"
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
    --probe-steps 25 \
    --eval-last-steps 5 \
    --n-trials 30 \
    --mode shared_sigma \
    --tau-min 0.05 \
    --tau-max 1.00 \
    --sigma-scale-min 0.2 \
    --sigma-scale-max 5.0 \
    --min-rank 3.2 \
    --max-uniformity -1.5 \
    --out-dir ${OUT_DIR} \
    "$@"
