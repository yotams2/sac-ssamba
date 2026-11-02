#!/bin/bash
#SBATCH --job-name=sac_probe_diag
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18     # RTX PRO 6000 Blackwell (sm_120): torch 2.2.2+cu121 has no kernels for it
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Frozen binaural-SAC probe sanity check on SC2: pretrained vs random encoder, all-frame vs real-frame pooling.
# Submit from the sac-ssamba root:  sbatch scripts/probe_diagnostic.sbatch <ckpt.pth> [extra args]
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
CKPT=${1:?usage: sbatch scripts/probe_diagnostic.sbatch <path/to/best_audio_model.pth>}
shift
cd "${REPO_ROOT}/src"
${PYTHON} binaural_dual_stream_sac/probe_diagnostic.py --ckpt "${CKPT}" --workdir "${TMPDIR:-/tmp}" "$@"
