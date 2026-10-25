#!/usr/bin/env bash
# ==============================================================================
# Transfer Pretraining & Downstream Datasets to TAU via SSH Multiplexing
# Prompts for password only ONCE at the start.
# ==============================================================================
set -euo pipefail

SOCK="/tmp/ssh_tau_$(id -u).sock"
REMOTE="yotams2@slurmlogin.tau.ac.il"
TARGET="/moshe-data/blaufer-data/yotams2/data"

cleanup() {
    if [ -e "$SOCK" ]; then
        ssh -S "$SOCK" -O exit "$REMOTE" 2>/dev/null || true
        rm -f "$SOCK"
    fi
}
trap cleanup EXIT INT TERM

echo "--------------------------------------------------"
echo "1. Establishing SSH Master Connection..."
echo "   (Enter your TAU password once below)"
echo "--------------------------------------------------"
rm -f "$SOCK"
ssh -M -S "$SOCK" -fnN "$REMOTE"

echo "SSH master connection established successfully."
echo "--------------------------------------------------"

# Ensure destination directory exists on TAU
ssh -S "$SOCK" "$REMOTE" "mkdir -p $TARGET"

# 1. LibriSpeech (uncompressed tar, files are already .flac)
echo "=== [1/3] Transferring LibriSpeech.tar (~36 GB)... ==="
tar -C /storage/data -cf - LibriSpeech | ssh -S "$SOCK" "$REMOTE" "cat > $TARGET/LibriSpeech.tar"
echo "-> LibriSpeech.tar transfer complete."

# 2. VoxCeleb1 (compressed on the fly with gzip)
echo "=== [2/3] Transferring voxceleb1.tar.gz (~30 GB compressed)... ==="
tar -C /storage/data -czf - voxceleb1 | ssh -S "$SOCK" "$REMOTE" "cat > $TARGET/voxceleb1.tar.gz"
echo "-> voxceleb1.tar.gz transfer complete."

# 3. Speech Commands v2 (pre-compressed .tar.gz)
echo "=== [3/3] Transferring speech_commands_v0.02.tar.gz (2.3 GB)... ==="
cat /storage/yotam/ssamba/src/finetune/speechcommands_v2/data/speech_commands_v0.02.tar.gz | \
    ssh -S "$SOCK" "$REMOTE" "cat > $TARGET/speech_commands_v0.02.tar.gz"
echo "-> speech_commands_v0.02.tar.gz transfer complete."

echo "--------------------------------------------------"
echo "ALL TRANSFERS COMPLETED SUCCESSFULLY!"
echo "Files are located at: ${REMOTE}:${TARGET}/"
echo "--------------------------------------------------"
