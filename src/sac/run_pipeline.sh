#!/bin/bash
set -e

# Parse arguments
PRETRAIN_EXP_PATH=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --pretrain_exp_path)
      PRETRAIN_EXP_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$PRETRAIN_EXP_PATH" ]; then
    echo "========================================"
    echo "Starting Pretraining"
    echo "========================================"
    bash run_sac.sh
    
    # Pretraining is done, let's find the most recent experiment directory in ./exp
    # since run_sac.sh generates the name dynamically based on config
    LATEST_EXP=$(ls -td ./exp/*/ | head -1)
    PRETRAIN_EXP_PATH=${LATEST_EXP%/}
    PRETRAIN_EXP_PATH=$(realpath "$PRETRAIN_EXP_PATH")
    echo "Pretraining finished. Using newly created exp dir: $PRETRAIN_EXP_PATH"
else
    echo "========================================"
    echo "Skipping Pretraining, using provided path"
    echo "========================================"
    PRETRAIN_EXP_PATH=$(realpath "$PRETRAIN_EXP_PATH")
    echo "Using pretrain path: $PRETRAIN_EXP_PATH"
fi

echo ""
echo "========================================"
echo "Starting Downstream Fine-tuning Pipeline"
echo "========================================"

# SID (VoxCeleb1)
echo ""
echo "--> Running Speaker ID (VoxCeleb1)..."
cd ../finetune/voxceleb1
bash run_sid.sh "$PRETRAIN_EXP_PATH" 1e-4
cd ../../sac

# ER (IEMOCAP)
echo ""
echo "--> Running Emotion Recognition (IEMOCAP)..."
cd ../finetune/iemocap
bash run_er.sh "$PRETRAIN_EXP_PATH" 1e-5
cd ../../sac

# SC2 (Speech Commands v2)
echo ""
echo "--> Running Speech Commands v2..."
cd ../finetune/speechcommands_v2
bash run_sc_amba.sh "$PRETRAIN_EXP_PATH"
cd ../../sac

echo ""
echo "========================================"
echo "Pipeline finished successfully!"
echo "========================================"
