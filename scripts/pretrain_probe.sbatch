#!/bin/bash
#SBATCH --job-name=sac_pretrain_probe
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18     # RTX PRO 6000 Blackwell (sm_120): torch 2.2.2+cu121 has no kernels for it
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Frozen SC2 / VoxCeleb1-SID / IEMOCAP-ER probes (src/binaural_dual_stream_sac/pretrain_probe.py) on saved checkpoints.
# Submit from the sac-ssamba root. Model specs: checkpoint paths or quoted globs, 'random', 'gramt[:<hf id>]'.
#   sbatch scripts/pretrain_probe.sbatch <out.jsonl> random gramt '<exp>/models/audio_model.*.pth'
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
OUT=${1:?usage: sbatch scripts/pretrain_probe.sbatch <out.jsonl> <model spec>...}
shift
[ $# -gt 0 ] || { echo "no model specs given"; exit 1; }
cd "${REPO_ROOT}/src"
# Report this run to the results branch, including failures (scripts/results_hook.sh)
source "${REPO_ROOT}/scripts/results_hook.sh" && results_hook probe "$(dirname "${OUT}")" "${OUT}"
${PYTHON} binaural_dual_stream_sac/pretrain_probe.py --out "${OUT}" --models "$@"
