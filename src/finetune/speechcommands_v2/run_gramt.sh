#!/bin/bash

#SBATCH --job-name=gramt_sc
#SBATCH --partition=gpu-general-pool
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=logs/%x_%j.out  # relative to the submit dir: submit from the sac-ssamba root

# GRAM-T baseline on binaural Speech Commands v2 (run_amba.py), for comparison with run_binaural_sac_frozen.sh
# Usage: run_gramt.sh [model_id=labhamlet/gramt-binaural-frame] [lr=2.5e-4] [frozen=true]
set -x
SCRIPT_PATH=${BASH_SOURCE[0]}
# under sbatch BASH_SOURCE is slurm's spooled copy, so take the submitted script path instead
if [ "$(basename "$SCRIPT_PATH")" = "slurm_script" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "${SCRIPT_DIR}/../../../env.sh"
PYTHON_BIN=${PYTHON}

model_id=${1:-labhamlet/gramt-binaural-frame}
lr=${2:-2.5e-4}
frozen=${3:-${FROZEN:-true}}

if [ "$frozen" = "true" ] || [ "$frozen" = "frozen" ] || [ "$frozen" = "1" ]; then
    frozen_flag="--freeze_base"
    exp_task_name="ft_avgtok-frozen"
else
    frozen_flag=""
    exp_task_name="ft_avgtok"
fi

tr_data=${TR_DATA:-${SAC_DATA_ROOT}/speechcommands_spatial_pretrain.json}
val_data=${VAL_DATA:-${SAC_DATA_ROOT}/speechcommands_spatial_preval.json}
eval_data=${EVAL_DATA:-${SAC_DATA_ROOT}/speechcommands_spatial_pretest.json}

batch_size=128
epoch=30
mixup=0.6

exp_dir=${SCRIPT_DIR}/exp/gramt-speechcommands-b${batch_size}-lr${lr}-${exp_task_name}-$(basename ${model_id})

cd ${SCRIPT_DIR}
CUDA_CACHE_DISABLE=1 $PYTHON_BIN -W ignore ../../run_amba.py --use_wandb --dataset speechcommands \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/speechcommands_class_labels_indices.csv --n_class 35 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--mixup ${mixup} --bal none --warmup True --task ft_avgtok \
--model_size gramt --gramt_model_id ${model_id} --clip_secs 1.0 --adaptschedule False \
--head_lr 1 --noise False \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss BCE --metrics acc \
$frozen_flag
