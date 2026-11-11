#!/bin/bash
#SBATCH --job-name=test_sc_resume
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,58,300,376,419,420]
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# Checks that SC2 fine-tuning killed mid-epoch and resumed from models/resume.pth matches an uninterrupted run.
# Three runs on a 1/5 subset of SC2 for 6 epochs (lr decay starts at epoch 5, warm-up lasts the whole run):
#   A, B: uninterrupted, same seed (B - A = the GPU's own run-to-run noise)
#   C:    kill -9 during epoch 5, then the same command again (resumes after epoch 4)
# Pass: C - A no larger than B - A, per epoch and on the test set.
# Usage (from the sac-ssamba root): sbatch scripts/test_sc_resume.sbatch [pretraining exp dir]
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "${REPO_ROOT}/env.sh"
PRE=${1:-${SAC_EXP_ROOT}/sac_pretrain/sac-base-f16-t16-b64-lr4e-4-lam0.02-sig1.0-feat_universal-mode_offline_global_median-librispeech-baseline_objective_pitchfix}
T=${SAC_EXP_ROOT}/_test_sc_resume
rm -rf "$T"; mkdir -p "$T/data"
SRC=${SAC_DATA_ROOT}/mono/speechcommands_datafiles
${PYTHON} - "$SRC" "$T/data" <<'EOF'
import json, sys
src, dst = sys.argv[1:]
for split in ['train', 'valid', 'eval']:
    d = json.load(open(f'{src}/speechcommand_{split}_data.json'))
    d['data'] = d['data'][::5]
    json.dump(d, open(f'{dst}/speechcommand_{split}_data.json', 'w'))
    print(split, len(d['data']))
EOF
export SC_DATAFILES=$T/data EPOCHS=6 NO_RESULTS_HOOK=1
SC=${REPO_ROOT}/src/finetune/speechcommands_v2/run_sc_amba.sh
run() { SAC_EXP_ROOT=$T/$1 bash "$SC" "$PRE" > "$T/$1.log" 2>&1; }
expdir() { ls -d "$T/$1"/speechcommands_v2/test01-*; }

run A; echo "A done $(date)"
run B; echo "B done $(date)"

# C: start, wait until epoch 4's resume.pth is written and epoch 5 is under way, kill the whole process group
SAC_EXP_ROOT=$T/C setsid bash "$SC" "$PRE" > "$T/C.log" 2>&1 &
PID=$!
until grep -q "current #epochs=5" "$T/C.log" 2>/dev/null; do sleep 5; done
sleep 45
grep -q "epoch 5 training time" "$T/C.log" && { echo "FAIL: epoch 5 finished before the kill"; exit 1; }
kill -9 -- -"$PID"; wait "$PID" || true
echo "C killed during epoch 5 $(date)"
SAC_EXP_ROOT=$T/C bash "$SC" "$PRE" > "$T/C_resumed.log" 2>&1
grep "resumed from" "$T/C_resumed.log" || { echo "FAIL: C did not resume"; exit 1; }
echo "C done $(date)"

${PYTHON} - "$(expdir A)" "$(expdir B)" "$(expdir C)" <<'EOF'
import sys, numpy as np, torch
A, B, C = sys.argv[1:]
r = {k: np.loadtxt(d + '/result.csv', delimiter=',') for k, d in zip('ABC', (A, B, C))}
e = {k: np.loadtxt(d + '/eval_result.csv') for k, d in zip('ABC', (A, B, C))}
np.set_printoptions(precision=6, suppress=True, linewidth=150)
print('per-epoch dev acc / train loss / lr')
for k in 'ABC':
    print(k, r[k][:, 0], r[k][:, 5], r[k][:, 9])
print('test [val_acc, val_auc, test_acc, test_auc]')
for k in 'ABC':
    print(k, e[k])
def wdiff(x, y):
    a = torch.load(x + '/models/best_audio_model.pth', map_location='cpu')
    b = torch.load(y + '/models/best_audio_model.pth', map_location='cpu')
    return max((a[n].float() - b[n].float()).abs().max().item() for n in a)
nBA, nCA = np.abs(r['B'] - r['A']).max(0), np.abs(r['C'] - r['A']).max(0)
print('max |B-A| (noise floor): dev acc %.6f, train loss %.6f, lr %.3g, test acc %.6f, weights %.3g'
      % (nBA[0], nBA[5], nBA[9], abs(e['B'][2] - e['A'][2]), wdiff(A, B)))
print('max |C-A| (resume):      dev acc %.6f, train loss %.6f, lr %.3g, test acc %.6f, weights %.3g'
      % (nCA[0], nCA[5], nCA[9], abs(e['C'][2] - e['A'][2]), wdiff(A, C)))
EOF
