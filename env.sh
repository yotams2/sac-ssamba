# Shared cluster environment for sac-ssamba (binaural SAC + GRAM-T baseline) and sac-s3prl.
# Usage: source env.sh   (the run_*.sh scripts source it themselves)
# Everything heavy lives on /scratch200: the home quota is ~2 GB.
SAC_SCRATCH=${SAC_SCRATCH:-/scratch200/yotams2}

export SAC_VENV=${SAC_VENV:-${SAC_SCRATCH}/venvs/sac}                  # one venv for SAC, GRAM-T and s3prl
export VIM_ROOT=${VIM_ROOT:-${SAC_SCRATCH}/code/Vim}                   # hustvl/Vim checkout, read by models/both_models.py
export HF_HOME=${HF_HOME:-${SAC_SCRATCH}/hf}                           # GRAM-T weights (labhamlet/gramt-*)
export SAC_DATA_ROOT=${SAC_DATA_ROOT:-${SAC_SCRATCH}/sac-ssamba/data}  # replaces /storage/yotam/ssamba/data in the jsons
export PIP_NO_CACHE_DIR=1
export PYTHON=${SAC_PYTHON:-${SAC_VENV}/bin/python}
