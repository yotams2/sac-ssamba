#!/bin/bash

#SBATCH --job-name=binaural_sac_er_frozen
#SBATCH --gres=gpu:a40:1         # Request an A40 GPU
#SBATCH --cpus-per-task=12        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

# Runner script for frozen upstream evaluation (linear probing) of the SAC binaural model on IEMOCAP ER
MODEL_DIR="/storage/yotam/ssamba/src/binaural_dual_stream_sac/exp/binaural-sac-bs32-lr1e-4-mode_offline_global_median-binaural_ccsr_dual_sac"
LR=${1:-1e-5}
UPSTREAM="binaural_sac_local"
FROZEN="true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_er.sh" "$MODEL_DIR" "$LR" "$UPSTREAM" "$FROZEN"
