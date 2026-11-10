#!/bin/bash
#SBATCH --job-name=test_pretrain_resume
#SBATCH --partition=power-general-shared-pool
#SBATCH --account=public-blaufer_v2
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
# CPU test of the step-exact resume in src/sac/run_pretrain_sac.py. See scripts/test_pretrain_resume.py.
# Usage (from the sac-ssamba root): sbatch scripts/test_pretrain_resume.sbatch
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
"$PYTHON" -I -W ignore "${REPO_ROOT}/scripts/test_pretrain_resume.py"
