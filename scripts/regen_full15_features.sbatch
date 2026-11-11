#!/bin/bash
#SBATCH --job-name=regen_full15
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Writes binaural_{train,val}.full15.json next to the pitchfix manifests, with all 15 spectral features in the
# 5 groups (the mono `universal` list). Spatial features and wav order are copied, and the inputs are not touched.
# Use them with DATA_TRAIN, DATA_VAL and `--spectral_features <the list below>`.
# Submit from the sac-ssamba root. Safe to rerun, also after a preemption.
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
FEATURES="f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm"
cd "${REPO_ROOT}/src"
for split in val train; do
    ${PYTHON} binaural_dual_stream_sac/precompute_acoustic_features.py --keep_spatial --num_workers "${SLURM_CPUS_PER_TASK}" \
        --features_list "${FEATURES}" \
        --input_json "${SAC_DATA_ROOT}/binaural_${split}.pitchfix.json" --output_json "${SAC_DATA_ROOT}/binaural_${split}.full15.json" 2>&1 | tr '\r' '\n' | tail -6
done
${PYTHON} "${REPO_ROOT}/scripts/check_full15_manifest.py"
