#!/bin/bash
#SBATCH --job-name=sac_env_setup
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

# Builds the single venv shared by SAC-SSAMBA, the GRAM-T baseline and sac-s3prl ($SAC_VENV),
# clones Vim ($VIM_ROOT) and downloads the default GRAM-T weights ($HF_HOME); paths come from env.sh.
# Runs as a job because the Jupyter node (1 GB RAM) is OOM-killed by `import torch`.
# Submit from the sac-ssamba root:  sbatch scripts/setup_env.sbatch
# Re-running is safe: pip skips installed packages, the clone and the download are skipped if present.
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
[ -f "${REPO_ROOT}/env.sh" ] || { echo "submit from the sac-ssamba root (no env.sh in ${REPO_ROOT})"; exit 1; }
source "${REPO_ROOT}/env.sh"
S3PRL_ROOT=${S3PRL_ROOT:-$(dirname "${REPO_ROOT}")/sac-s3prl}

# keep pip's build/unpack files off the small /tmp and home
export TMPDIR=${SAC_SCRATCH}/tmp/pip_${SLURM_JOB_ID:-$$}
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}"' EXIT
echo "start: $(date)  venv=${SAC_VENV}  vim=${VIM_ROOT}  hf=${HF_HOME}"

# 1. venv
[ -x "${SAC_VENV}/bin/python" ] || /usr/bin/python3.11 -m venv "${SAC_VENV}"
PIP="${PYTHON} -m pip install --no-cache-dir"
${PIP} --upgrade pip wheel

# 2. torch (CUDA 12.1 build), pinned to requirements.txt
${PIP} torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 3. prebuilt mamba / causal-conv1d CUDA kernels for torch 2.2 + CUDA 12 + py3.11 (no nvcc needed).
#    The Python side of mamba_ssm comes from Vim's mamba-1p1p1 fork (bimamba v2), put first on sys.path
#    by models/both_models.py; only the compiled selective_scan_cuda / causal_conv1d_cuda come from here.
${PIP} --no-deps \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
    "https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# 4. everything else in one resolve, so numpy stays < 2 (torch 2.2 is built against numpy 1.x).
#    sac-ssamba: einops timm scipy scikit-learn matplotlib soundfile; GRAM-T: transformers safetensors;
#    sac-s3prl runner (used from its checkout via PYTHONPATH, not pip-installed): the rest.
#    Left out to save space: wandb (optional in the code), librosa, optuna/umap/seaborn (analysis only).
${PIP} "numpy<2" einops==0.8.0 timm==1.0.3 scipy scikit-learn matplotlib soundfile \
    transformers==4.46.3 safetensors huggingface_hub \
    pyyaml tqdm omegaconf tensorboardX filelock packaging pandas gdown "protobuf>=4.21.1"

# 5. Vision Mamba
[ -d "${VIM_ROOT}/vim" ] || git clone --depth 1 https://github.com/hustvl/Vim.git "${VIM_ROOT}"

# 6. GRAM-T weights + remote code (other variants are fetched on first use)
${PYTHON} - <<'EOF'
from huggingface_hub import snapshot_download
print("GRAM-T snapshot:", snapshot_download("labhamlet/gramt-binaural-frame"))
EOF

echo "--- sizes"
du -sh "${SAC_VENV}" "${VIM_ROOT}" "${HF_HOME}" 2>/dev/null || true

# 7. import checks (CPU only; the GPU check is scripts/smoke_test.sbatch)
echo "--- import checks"
${PYTHON} -c "import torch, torchaudio, numpy, transformers, einops, timm, causal_conv1d, selective_scan_cuda; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| numpy', numpy.__version__, '| transformers', transformers.__version__)"
(cd "${REPO_ROOT}/src" && ${PYTHON} -c "import models, dataloader, gramt_wrapper; print('sac-ssamba imports OK (Vim + mamba fork)')")
(cd "${S3PRL_ROOT}" && PYTHONPATH="${S3PRL_ROOT}" ${PYTHON} -c "import s3prl.hub as hub; assert hasattr(hub, 'gramt_binaural'); print('s3prl hub OK')")
echo "done: $(date)"
