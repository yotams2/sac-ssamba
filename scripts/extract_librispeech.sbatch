#!/bin/bash
#SBATCH --job-name=extract_libri
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Extract LibriSpeech.tar into $SAC_DATA_ROOT, then delete the tar, only if tar succeeded and every listed file is on disk.
# Safe to resubmit: nothing is removed until that check passes.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/env.sh"
TAR=/scratch200/yotams2/data/LibriSpeech.tar
LIST=$(mktemp)
echo "start: $(date)"; quota -s 2>/dev/null | tail -3 || true
cd "${SAC_DATA_ROOT}"
tar -xvf "${TAR}" > "${LIST}"
LISTED=$(grep -vc '/$' "${LIST}")
FOUND=$(find LibriSpeech -type f | wc -l)
echo "files in tar: ${LISTED}, on disk: ${FOUND}"; du -sh LibriSpeech
[ "${LISTED}" -gt 0 ] && [ "${LISTED}" -eq "${FOUND}" ] || { echo "MISMATCH: keeping ${TAR}"; exit 1; }
rm -f -- "${TAR}"
echo "deleted ${TAR}; done: $(date)"
