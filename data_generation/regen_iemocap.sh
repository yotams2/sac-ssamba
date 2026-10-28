#!/usr/bin/env bash
# Regenerates binaural IEMOCAP so that every utterance of s3prl's emotion meta_data is simulated exactly once,
# at its full length, then rebuilds iemocap_spatial_{pretrain,preval,pretest}.json and verifies them.
# Replaces the first release, which sampled sources with replacement (only 3,519 of 5,531 utterances, some
# up to 6 times) and cut/padded every utterance to 6 s.
#
# Needs: a GPU, gpuRIR + jsonargparse + soundfile + scipy in $PYTHON, the original IEMOCAP_full_release.
# Defaults are the RunAI paths (where the first release was generated); override any of them, e.g.
#   IEMOCAP_ROOT=... META_DIR=... DATA_ROOT=... GPUS="[0]" bash regen_iemocap.sh
#
# Output: $DATA_ROOT/IEMOCAP/simu_v2/{pretrain,preval,pretest}/ and $DATA_ROOT/iemocap_spatial_*.json
# (the old jsons are kept as iemocap_spatial_*.json.v1). The old simu/ folder is left untouched.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IEMOCAP_ROOT=${IEMOCAP_ROOT:-/storage/data/IEMOCAP_audio_only/IEMOCAP_full_release}
META_DIR=${META_DIR:-/storage/yotam/s3prl/s3prl/downstream/emotion/meta_data}
DATA_ROOT=${DATA_ROOT:-/storage/yotam/ssamba/data}
GPUS=${GPUS:-[0]}
PYTHON=${PYTHON:-python3}
SIMU_DIR=${DATA_ROOT}/IEMOCAP/simu_v2
STAGING=${SIMU_DIR}/manifests

for d in "$IEMOCAP_ROOT/Session1" "$META_DIR/Session1" "$DATA_ROOT"; do
    [ -d "$d" ] || { echo "missing $d"; exit 1; }
done
$PYTHON -c "import gpuRIR, jsonargparse, soundfile, scipy" || { echo "$PYTHON lacks gpuRIR/jsonargparse/soundfile/scipy"; exit 1; }
if [ -d "$SIMU_DIR/pretrain" ] && [ -z "${RESUME:-}" ]; then
    echo "$SIMU_DIR already exists: delete it, or set RESUME=1 to regenerate into it"; exit 1
fi

cd "$SCRIPT_DIR"
echo "start: $(date)"
# 1. simulate each split; data_num and the container length T default to "all utterances" / "longest utterance"
for stage in pretrain preval pretest; do
    echo "=== $stage"
    $PYTHON gen_simu_iemocap.py --stage $stage --iemocap_root "$IEMOCAP_ROOT" --meta_dir "$META_DIR" \
        --test_fold fold1 --save_to "$SIMU_DIR" --gpus "$GPUS"
done

# 2. manifests into a staging folder, refused if any utterance is duplicated
$PYTHON generate_downstream_manifests.py --dataset iemocap --simu_dir "$SIMU_DIR" --out_dir "$STAGING"

# 3. coverage / labels / lengths against meta_data and the original audio
$PYTHON verify_binaural_iemocap.py --data_root "$STAGING" --meta_dir "$META_DIR" --iemocap_root "$IEMOCAP_ROOT"

# 4. only now replace the jsons the evaluations read
for split in pretrain preval pretest; do
    f="$DATA_ROOT/iemocap_spatial_${split}.json"
    [ -f "$f" ] && [ ! -f "$f.v1" ] && mv "$f" "$f.v1"
    cp "$STAGING/iemocap_spatial_${split}.json" "$f"
done
du -sh "$SIMU_DIR"
echo "done: $(date)"
