#!/bin/bash
#SBATCH --job-name=esc
#SBATCH --gres=gpu:l40:1         # Request an L40 GPU
#SBATCH --cpus-per-task=8        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="${SCRIPT_DIR}/../../pretrained_models"
mkdir -p "${SCRIPT_DIR}/exp"

# prep speechcommands dataset and download the pretrained model
if [ -e "${SCRIPT_DIR}/data/datafiles" ]
then
    echo "speechcommands already downloaded and processed."
else
    (cd "${SCRIPT_DIR}" && python prep_sc.py)
fi

# source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
# conda activate new_env_name

input_arg=${1:-amba_base_400}
if [ "$2" = "true" ] || [ "$2" = "frozen" ]; then
    lr=2.5e-4
    frozen="true"
else
    lr=${2:-2.5e-4}
    frozen=${3:-${FROZEN:-false}}
fi

# Configure frozen vs fine-tuned (linear probing)
if [ "$frozen" = "true" ] || [ "$frozen" = "frozen" ] || [ "$frozen" = "1" ]; then
    frozen_flag="--freeze_base"
    exp_task_name="${task}-frozen"
    echo "Running in FROZEN upstream mode (upstream weights frozen / linear probing)"
else
    frozen_flag=""
    exp_task_name="${task}"
    echo "Running in FINE-TUNING mode (upstream weights trainable)"
fi

if [ -d "$input_arg" ]; then
    pretrain_path="${input_arg}/models/best_audio_model.pth"
    pretrain_model=$(basename "$input_arg")
elif [ -f "$input_arg" ]; then
    pretrain_path="$input_arg"
    if [[ "$input_arg" == *"/models/"* ]]; then
        pretrain_model=$(basename $(dirname $(dirname "$input_arg")))
    else
        pretrain_model=$(basename $(dirname "$input_arg"))
    fi
else
    pretrain_model="$input_arg"
    pretrain_path="./amba/${pretrain_model}.pth"
fi

pretrain_exp="amba"

dataset=speechcommands
dataset_mean=-6.845978
dataset_std=5.5654526
target_length=128
noise=True
tr_data=./data/datafiles/speechcommand_train_data.json
val_data=./data/datafiles/speechcommand_valid_data.json
eval_data=./data/datafiles/speechcommand_eval_data.json

bal=none
freqm=48
timem=48
mixup=0.6
epoch=30
batch_size=128
fshape=16
tshape=16
fstride=10
tstride=10

depth=24
rms_norm='false'
residual_in_fp32='false'
fused_add_norm='false'
if_rope='false'
if_rope_residual='false'
bimamba_type="v2"
drop_path_rate=0.1
stride=10
channels=1
num_classes=1000
drop_rate=0.
norm_epsilon=1e-5
if_bidirectional='true'
final_pool_type='none'
if_abs_pos_embed='true'
if_bimamba='false'
if_cls_token='true'
if_devide_out='true'
use_double_cls_token='false'
use_middle_cls_token='false'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/exp"

task=ft_avgtok
model_size=base
pool_real_time_only=${POOL_REAL_TIME_ONLY:-false}
if [[ $pretrain_model == *"tiny"* ]]; then
  model_size="tiny"
  embed_dim=192
elif [[ $pretrain_model == *"small"* ]]; then
  model_size="small"
  embed_dim=384
elif [[ $pretrain_model == *"binaural"* ]] || [[ $pretrain_model == *"sac"* ]]; then
  model_size="binaural_sac"
  embed_dim=768
  tr_data=${SAC_DATA_ROOT:-/storage/yotam/ssamba/data}/speechcommands_spatial_pretrain.json
  val_data=${SAC_DATA_ROOT:-/storage/yotam/ssamba/data}/speechcommands_spatial_preval.json
  eval_data=${SAC_DATA_ROOT:-/storage/yotam/ssamba/data}/speechcommands_spatial_pretest.json
  noise=False
  freqm=0
  timem=0
  mixup=0
  dataset_mean=0
  dataset_std=1
  target_length=1024
else
  model_size="base"
  embed_dim=768
fi

head_lr=1

base_exp_dir="${SCRIPT_DIR}/exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${exp_task_name}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}"

exp_dir="${SCRIPT_DIR}/exp/test01-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${exp_task_name}-${model_size}-$pretrain_exp-${pretrain_model}-${head_lr}x-noise${noise}"

PYTHON_BIN=${PYTHON:-/storage/yotam/venvs/ssamba_ide/bin/python}
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

CUDA_CACHE_DISABLE=1 $PYTHON_BIN -W ignore "${SCRIPT_DIR}/../../run_amba.py" --use_wandb --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv "${SCRIPT_DIR}/data/speechcommands_class_labels_indices.csv" --n_class 35 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --embed_dim ${embed_dim} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False --pool_real_time_only ${pool_real_time_only} \
--pretrained_mdl_path ${pretrain_path} --if_cls_token 'true' --final_pool_type 'mean' --if_abs_pos_embed 'true' --if_devide_out 'true' --use_middle_cls_token 'true' \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss BCE --metrics acc --embed_dim ${embed_dim} --depth ${depth} \
--rms_norm ${rms_norm} --residual_in_fp32 ${residual_in_fp32} \
--fused_add_norm ${fused_add_norm} --if_rope ${if_rope} --if_rope_residual ${if_rope_residual} \
--bimamba_type ${bimamba_type} --use_middle_cls_token ${use_middle_cls_token} --drop_path_rate ${drop_path_rate} --stride ${stride} --channels ${channels} --num_classes ${num_classes} --drop_rate ${drop_rate} --norm_epsilon ${norm_epsilon} --if_bidirectional ${if_bidirectional} --final_pool_type ${final_pool_type} --if_abs_pos_embed ${if_abs_pos_embed} --if_bimamba ${if_bimamba} --if_cls_token ${if_cls_token} --if_devide_out ${if_devide_out} --use_double_cls_token ${use_double_cls_token} $frozen_flag 
