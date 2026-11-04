#!/bin/bash
#SBATCH --job-name=regen_binaural
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Writes binaural_{train,val}.pitchfix.json next to the originals, with F0 and HNR from the current
# src/sac/acoustic_features.py. Spatial features and wav order are copied, and the originals are not touched.
# Submit from the sac-ssamba root. Safe to rerun, also after a preemption.
#
# To use them, either pass DATA_TRAIN and DATA_VAL to run_binaural_sac.sh, or, once no job reads the originals:
#   cd $SAC_DATA_ROOT
#   for s in train val; do mv binaural_$s.json binaural_$s.json.v1; mv binaural_$s.pitchfix.json binaural_$s.json; done
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
cd "${REPO_ROOT}/src"
for split in val train; do
    ${PYTHON} binaural_dual_stream_sac/precompute_acoustic_features.py --keep_spatial --num_workers "${SLURM_CPUS_PER_TASK}" \
        --input_json "${SAC_DATA_ROOT}/binaural_${split}.json" --output_json "${SAC_DATA_ROOT}/binaural_${split}.pitchfix.json" 2>&1 | tr '\r' '\n' | tail -6
done
${PYTHON} "${REPO_ROOT}/scripts/check_regen_manifest.py"
