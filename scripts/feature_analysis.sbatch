#!/bin/bash
#SBATCH --job-name=feat_analysis
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Submit from the sac-ssamba root: sbatch scripts/feature_analysis.sbatch [old git commit]
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
${PYTHON} "${REPO_ROOT}/scripts/feature_analysis.py" "$@"
