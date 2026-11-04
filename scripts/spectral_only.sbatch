#!/bin/bash
#SBATCH --job-name=sac_spectral_only
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18     # RTX PRO 6000 Blackwell (sm_120): torch 2.2.2+cu121 has no kernels for it
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# One spectral encoder without CCSR, on the binaural pretraining set: does CCSR hold it back, and does raw STFT
# input cost quality compared with log-mel? The two runs differ only in the input. Why, and how to read the
# results: docs/spectral_only_experiment.md
# Submit from the sac-ssamba root:
#   sbatch --job-name=sac_spec_stft   scripts/spectral_only.sbatch stft
#   sbatch --job-name=sac_spec_logmel scripts/spectral_only.sbatch logmel
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
INPUT=${1:?usage: sbatch scripts/spectral_only.sbatch <stft|logmel>}

# Shared: SSAMBA masked patches with mpg + mpc, spectral SAC at the mono recipe's weight
COMMON="--spectral_only --input_repr ${INPUT} --classif_lambda 1.0 --sac_spec_lambda 0.02"
case "${INPUT}" in
    # 600 of 1024 tokens; weight 1 on an MSE that starts at ~2.9
    stft)   EXTRA="--mask_patch 600 --recon_lambda 1.0" ;;
    # 300 of 512 tokens as in mono SSAMBA; weight 10 on an MSE that starts at ~0.25.
    # Stats from 2000 clips of binaural_train.json (the mono constants come from clean LibriSpeech)
    logmel) EXTRA="--mask_patch 300 --recon_lambda 10.0 --fbank_mean -3.98963 --fbank_std 3.39167" ;;
    *) echo "unknown input: ${INPUT}"; exit 1 ;;
esac

export EPOCHS=6 PROBE_EVERY=1000
export EXP_NAME_SUFFIX="-spectral_only_${INPUT}"
export EXP_DESCRIPTION="Spectral-only (no CCSR), input ${INPUT}: SSAMBA mpg + mpc + spectral SAC. See docs/spectral_only_experiment.md."
export EXTRA_ARGS="${COMMON} ${EXTRA}"
bash "${REPO_ROOT}/src/binaural_dual_stream_sac/run_binaural_sac.sh"
