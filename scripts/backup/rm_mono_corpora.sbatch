#!/bin/bash
#SBATCH --job-name=rm_mono_corpora
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# 11-25 (Yotam's call, conditions checked): delete the raw mono corpora on scratch. (a) Backed up: raw archives in
# /moshe-data/blaufer-data/yotams2/data/ (IEMOCAP.tar, LibriSpeech.tar, speech_commands_v0.02.tar.gz, voxceleb1.tar.gz);
# the scratch copies were extracted from them by scripts/extract_mono_downstream.sbatch / extract_librispeech.sbatch,
# which re-stage them (the .done_* markers are removed so the extract script does not skip). (b) No reader: the V1/V3
# trainers use cached teacher targets / pre-rendered tars, the speech probes read the binaural */simu* sets (kept), and
# no queued job reads them. Kept: mono/speechcommands_datafiles (manifests), mono/.list_* (archive listings).
S=/scratch200/yotams2/sac-ssamba/data
for d in "$S/LibriSpeech" "$S/mono/voxceleb1" "$S/mono/data" "$S/mono/speech_commands_v0.02"; do
  [ -d "$d" ] || { echo "missing $d"; continue; }
  kb=$(du -sk "$d" | cut -f1)
  rm -rf -- "$d" && printf "%s\t%s\t%.1f\traw mono corpus, archive on moshe data/; Yotam 11-25 (backed up + no reader)\t%s\n" \
    "$(date '+%F %T')" "$SLURM_JOB_ID" "$(echo "$kb/1024" | bc -l)" "$d" >> ~/exp-records/prune_log.tsv
  echo "$(date) deleted $d ($((kb/1048576)) GB)"
done
rm -f "$S/mono/.done_iemocap" "$S/mono/.done_speechcommands" "$S/mono/.done_voxceleb1"
quota -s -f /scratch200; echo "$(date) done"
