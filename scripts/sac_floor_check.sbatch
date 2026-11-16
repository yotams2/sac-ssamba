#!/bin/bash
#SBATCH --job-name=sac_floor
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18     # RTX PRO 6000 Blackwell (sm_120): torch 2.2.2+cu121 has no kernels for it
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# SAC loss vs its floor H(P) per feature group, for finished mono pretrainings (src/sac/sac_floor_check.py).
# Submit from the sac-ssamba root:  sbatch scripts/sac_floor_check.sbatch <exp dir> [<exp dir> ...]
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
cd "${REPO_ROOT}/src/sac"
${PYTHON} -u -W ignore sac_floor_check.py --n_batches 40 ${FLOOR_ARGS:-} --exp_dir "$@"
