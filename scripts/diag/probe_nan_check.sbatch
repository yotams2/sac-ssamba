#!/bin/bash
#SBATCH --job-name=probe_nan
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# Usage from sac-ssamba root: sbatch scripts/diag/probe_nan_check.sbatch <ckpt> ...   (first ckpt = fbank stats + 'random' arch)
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd "${REPO_ROOT}/src"
${PYTHON} -u "${REPO_ROOT}/scripts/diag/probe_nan_check.py" "$@"
echo "$(date) done"
