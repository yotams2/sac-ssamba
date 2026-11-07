#!/bin/bash
#SBATCH --job-name=extract_mono_ds
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --constraint=ib_dj
#SBATCH --exclude=compute-0-351
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Extract the mono downstream sets from /moshe-data into $SAC_DATA_ROOT/mono and build the Speech Commands manifests.
# Reads the archives only, deletes nothing. A dataset already marked done is skipped, so this is safe to resubmit.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/env.sh"
SRC=/moshe-data/blaufer-data/yotams2/data
OUT=${SAC_DATA_ROOT}/mono
mkdir -p "${OUT}"
echo "start: $(date)"; quota -s 2>/dev/null | tail -3 || true

extract() {  # name archive target_dir tar_flags
    local name=$1 archive=$2 dest=$3 flags=$4 list
    [ -f "${OUT}/.done_${name}" ] && { echo "${name}: already done"; return; }
    list=$(mktemp); mkdir -p "${dest}"
    tar "${flags}" "${SRC}/${archive}" -C "${dest}" > "${list}"
    echo "${name}: $(grep -vc '/$' "${list}") files in the archive, $(date)"
    mv "${list}" "${OUT}/.list_${name}"
    touch "${OUT}/.done_${name}"
}
extract speechcommands speech_commands_v0.02.tar.gz "${OUT}/speech_commands_v0.02" -xzvf
extract iemocap IEMOCAP.tar "${OUT}" -xvf
extract voxceleb1 voxceleb1.tar.gz "${OUT}" -xzvf

"${PYTHON}" "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/prep_mono_speechcommands.py" "${OUT}/speech_commands_v0.02" \
    "${SLURM_SUBMIT_DIR:-$(pwd)}/src/finetune/speechcommands_v2/data/speechcommands_class_labels_indices.csv" "${OUT}/speechcommands_datafiles"
ls -la "${OUT}"
find "${OUT}" -maxdepth 3 -type d \( -name 'Session1' -o -name 'wav' \)
quota -s 2>/dev/null | tail -3 || true
echo "done: $(date)"
