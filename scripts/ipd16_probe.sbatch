#!/bin/bash
#SBATCH --job-name=ipd16_probe
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18     # RTX PRO 6000 Blackwell (sm_120): torch 2.2.2+cu121 has no kernels for it
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Sprint RQ2: frozen spatial (ipd16) SSAMBA encoders on the binaural SC2 / SID / IEMOCAP probes
# (pretrain_probe.py --audio binaural --input_repr ipd16; 4.112-s windows, fp32). 'gramt' = GRAM-T anchor on the same clips.
#
# Usage, from the sac-ssamba root:
#   sbatch scripts/ipd16_probe.sbatch <out.jsonl> "<extra probe args>" <encoder_*.pth | random | gramt> ...
#   (the first ipd16 checkpoint supplies the channel statistics for 'random')
set -euo pipefail
OUT=$1; EXTRA=($2); shift 2
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
export HF_HUB_OFFLINE=1
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git $(git -C "$REPO_ROOT" rev-parse --short HEAD)$(git -C "$REPO_ROOT" diff --quiet || echo -dirty)"
cd "${REPO_ROOT}/src"
${PYTHON} -u binaural_dual_stream_sac/pretrain_probe.py --audio binaural --input_repr ipd16 --out "$OUT" "${EXTRA[@]}" --models "$@"
echo "$(date) done"
