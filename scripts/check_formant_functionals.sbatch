#!/bin/bash
#SBATCH --job-name=check_formant_fn
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out

# Submit from the sac-ssamba root: sbatch scripts/check_formant_functionals.sbatch [old git commit]
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
${PYTHON} "${REPO_ROOT}/scripts/check_formant_functionals.py" "$@"
