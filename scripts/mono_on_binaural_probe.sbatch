#!/bin/bash
#SBATCH --job-name=mono_on_bin
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18     # RTX PRO 6000 Blackwell (sm_120): torch 2.2.2+cu121 has no kernels for it
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# How much of the mono -> binaural drop is the harder (reverberant binaural) downstream audio? Frozen probes
# (pretrain_probe.py) of the mono `amba` seeds on the mono sets vs on the binaural sets collapsed to (L+R)/2 and to L,
# plus the binaural log-mel spectral-only model (23348511) on the same binaural subsets. SID / IEMOCAP use the same
# utterances in every mode; SC2 draws from the mono lists in mono mode. One jsonl per mode in exp/_probes/.
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git $(git -C "$REPO_ROOT" rev-parse --short HEAD)$(git -C "$REPO_ROOT" diff --quiet || echo -dirty)"
E=${SAC_EXP_ROOT}/sac_pretrain
AMBA=()
for s in 1 2 3; do AMBA+=("$E/sac-base-f16-t16-b64-lr4e-4-lam0-sig1.0-feat_none-mode_offline_global_median-librispeech-amba_seed$s/models/audio_model.final.pth"); done
BIN=${SAC_EXP_ROOT}/binaural_sac_pretrain/binaural-sac-bs32-lr1e-4-mode_offline_global_median-spectral_only_logmel/models/audio_model.final.pth
OUT=${SAC_EXP_ROOT}/_probes/mono_on_binaural_$(date +%F)
cd "${REPO_ROOT}/src"
P="${PYTHON} -u binaural_dual_stream_sac/pretrain_probe.py"
echo "===== mono model, mono audio";        $P --audio mono          --out ${OUT}_mono.jsonl --models "${AMBA[@]}"
echo "===== mono model, binaural (L+R)/2";  $P --audio binaural_mid  --out ${OUT}_bin_mid.jsonl --models "${AMBA[@]}" random
echo "===== mono model, binaural L";        $P --audio binaural_left --out ${OUT}_bin_left.jsonl --models "${AMBA[0]}"
echo "===== binaural log-mel model, binaural audio"
$P --audio binaural --input_repr logmel --fbank_mean -3.98963 --fbank_std 3.39167 --out ${OUT}_binmodel.jsonl --models "$BIN"
echo "$(date) done"
