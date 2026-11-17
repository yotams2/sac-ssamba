#!/bin/bash
#SBATCH --job-name=mono_probe
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18     # RTX PRO 6000 Blackwell (sm_120): torch 2.2.2+cu121 has no kernels for it
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Frozen-embedding probes (src/binaural_dual_stream_sac/pretrain_probe.py --audio mono) on saved mono SSAMBA
# checkpoints: SC2 kNN / class-mean / linear accuracy, VoxCeleb1 SID class-mean / linear / EER, IEMOCAP UA.
# Proxies to compare checkpoints, not downstream accuracies. Each checkpoint's architecture and fbank stats come
# from its run's args.pkl; 'random' = the first checkpoint's architecture, untrained.
# The same probe runs inside new pretrainings (run_sac.sh PROBE_EVERY, default 2000 steps -> probe_metrics.jsonl).
#
# Usage, from the sac-ssamba root:
#   sbatch scripts/mono_probe.sbatch <out.jsonl> <checkpoint or glob | random> ...
set -euo pipefail
OUT=$1; shift
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git $(git -C "$REPO_ROOT" rev-parse --short HEAD)$(git -C "$REPO_ROOT" diff --quiet || echo -dirty)"
cd "${REPO_ROOT}/src"
${PYTHON} -u binaural_dual_stream_sac/pretrain_probe.py --audio mono --out "$OUT" --models "$@"
echo "$(date) done"
