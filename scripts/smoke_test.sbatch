#!/bin/bash
#SBATCH --job-name=gramt_smoke
#SBATCH --partition=gpu-general-pool
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Check of the shared venv: GRAM-T wrapper vs the official HF representation (gramt_wrapper/smoke_test.py).
# Submit from the sac-ssamba root after setup_env.sbatch:  sbatch scripts/smoke_test.sbatch [model_id]
# The test also runs on CPU (it does not use the mamba kernels); when the GPU queue is long:
#   sbatch -p power-general-shared-pool --gres=none scripts/smoke_test.sbatch
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"

cd "${REPO_ROOT}/src"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi -L
    ${PYTHON} -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"
else
    echo "no GPU allocated: running on CPU"
fi
${PYTHON} -m gramt_wrapper.smoke_test "${1:-labhamlet/gramt-binaural-frame}"
