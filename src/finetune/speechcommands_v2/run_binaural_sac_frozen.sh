#!/bin/bash

#SBATCH --job-name=binaural_sac_sc_frozen
#SBATCH --gres=gpu:l40:1         # Request an L40 GPU
#SBATCH --cpus-per-task=8        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

# Runner script for frozen upstream evaluation (linear probing) of the SAC binaural model on Speech Commands v2
MODEL_DIR="/storage/yotam/ssamba/src/binaural_dual_stream_sac/exp/binaural-sac-bs32-lr1e-4-mode_offline_global_median-binaural_ccsr_dual_sac"
LR=${1:-2.5e-4}
FROZEN="true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_sc_amba.sh" "$MODEL_DIR" "$LR" "$FROZEN"
