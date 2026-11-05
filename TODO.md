# TODO

Working list for the binaural SAC-SSAMBA evaluation on this cluster (TAU power cluster).
Cluster paths and the shared venv are set in [env.sh](env.sh); heavy files go on `/scratch200/yotams2`
(the home quota is ~2 GB). Run every step as a Slurm job, because the Jupyter node has 1 GB of RAM and
`import torch` gets OOM-killed there.

## Now: GRAM-T baseline on the 3 downstream tasks

Submit every command from the `sac-ssamba` root. Logs go to `logs/<job-name>_<jobid>.out`. Slurm does not create
the folder, so it is kept in git as `logs/.gitkeep`. Once a run is done, copy its numbers (with the job id) into the results below.
Checkpoints and experiment outputs go to `$SAC_EXP_ROOT/<task>/<experiment>` (`/scratch200/yotams2/sac-ssamba/exp`, set in `env.sh`).

- [x] **Environment** builds `/scratch200/yotams2/venvs/sac`, clones Vim, downloads `labhamlet/gramt-binaural-frame`.
  `sbatch scripts/setup_env.sbatch`. It is done when the log `logs/sac_env_setup_<jobid>.out` ends with `done:`.
  The empty `/scratch200/yotams2/venvs/gramt` from an earlier attempt can then be deleted.
- [x] **Smoke test** (passed on CPU, job 23244826): `sbatch scripts/smoke_test.sbatch`. The log `logs/gramt_smoke_<jobid>.out` must end with `OK`.
  GPU jobs must use `gpu-general-pool`; `power-general-shared-pool` rejects GPU requests. The GPU queue can be days long,
  and the test also runs on CPU: `sbatch -p power-general-shared-pool --gres=none scripts/smoke_test.sbatch`.
- [x] **Speech Commands v2** (frozen probe, run_amba.py), job 23244879, done 2026-10-28 (30 epochs, ~3.5 min/epoch):
  `sbatch src/finetune/speechcommands_v2/run_gramt.sh labhamlet/gramt-binaural-frame 2.5e-4 true`
  Output moved to `/scratch200/yotams2/sac-ssamba/exp/speechcommands_v2/gramt-speechcommands-b128-lr2.5e-4-ft_avgtok-frozen-gramt-binaural-frame`.
- [ ] **IEMOCAP ER** (frozen, s3prl, fold1 only). Unblocked: regenerated binaural IEMOCAP (v2) is in place on TAU (2026-11-02):
  `sbatch src/finetune/iemocap/run_gramt.sh gramt_binaural 1e-4 true`
- [x] **VoxCeleb1 SID** (frozen, s3prl), job 23305339, done 2026-11-02 (50k steps; 23305273 failed on the Blackwell node, see below) (output in `$SAC_EXP_ROOT/voxceleb1/sid_gramt_binaural_frozen_1e-4`):
  `sbatch src/finetune/voxceleb1/run_gramt.sh gramt_binaural 1e-4 true`
- [ ] Collect the results in a table next to the SAC numbers.

## Next: the effect of SAC (main question)

The thesis question is what adding SAC changes, compared with the same model pretrained without SAC
(same data, same compute). GRAM-T is a large-scale reference point, not the target: it saw about 31M samples
(325k steps x 96, AudioSet) vs about 1.3M for binaural SAC (40k steps x 32, 50k simulated LibriSpeech clips).

- [ ] **Debug the frozen probe** (frozen SAC: SC2 13.5%, VoxCeleb1 14.5%; GRAM-T: 89.6%, 76.9%).
  - [x] `downstream_experts.py` used to skip a missing checkpoint path and load with `strict=False`, so a wrong path
    silently gave a random encoder. It now raises on a missing path or missing encoder weights, and prints what it loaded.
  - [x] Pooling over padding is ruled out (checked on RunAI 2026-11-02): both SC2 SAC runs (frozen and fine-tuned)
    had `pool_real_time_only=True`, and the s3prl path (VoxCeleb1, IEMOCAP) trims hidden states to each utterance's
    length (`upstream/ssamba/expert.py`, `Featurizer.tolist` uses `wav.shape[-1]`) and pools with a length mask.
  - [ ] Run `sbatch scripts/probe_diagnostic.sbatch <ckpt>` (needs the SAC checkpoint on TAU, or run the same script on
    RunAI). It tests the remaining suspects: pretrained vs random encoder; the 1 s clip zero-padded to 10.24 s (~90%
    zeros, as in the runs) vs tiled to fill the window; and input level (raw STFT, no log/normalisation) of SC2 vs the
    pretraining set. If pretrained ~= random, look at the pretraining objective (raw Re/Im reconstruction is
    phase-dominated) rather than the probe.
- [ ] **Frozen probes during pretraining** (`src/binaural_dual_stream_sac/pretrain_probe.py`): SC2 k-NN / class-mean /
  linear, VoxCeleb1 SID class-mean / linear / EER, IEMOCAP fold1 k-NN / class-mean / linear UA (v2 manifests in
  `IEMOCAP/simu_v2/manifests`) on fixed ~6.6k-clip subsets, spectral encoder pooled as downstream.
  On by default in `run_binaural_sac.sh` (step 0 + every `epoch_iter`, `PROBE_EVERY=0` disables) -> `<exp>/probe_metrics.jsonl`.
  Calibrated with job 23317158 (2026-11-02; `<RunAI run exp>/probe_metrics_offline.jsonl`). Linear probe, SC2 acc / SID acc / IEMOCAP UA:
  random 3.7 / 1.9 / 39.0, GRAM-T 71.2 / 62.4 / 62.9, SAC run at 4k / 8k / 12k / final steps
  11.4 / 16.1 / 48.4, 12.6 / 16.4 / 47.3, 11.6 / 15.6 / 47.8, 11.1 / 16.0 / 47.3.
  The ordering matches downstream and the SAC values are close to the real frozen SC2 13.5 / SID 14.5, so the proxy works.
  **The SAC encoder plateaus before step 4k**: later checkpoints change nothing (within ~1 pt noise). Look at the
  objective, not the training length. ~56 s per probe (vs ~10 h per 4000 steps), 167 s one-off data loading.
  Rerun: `sbatch scripts/pretrain_probe.sbatch <out.jsonl> random gramt '<exp>/models/audio_model.*.pth'`.
- [x] **Fix the reconstruction objective**, found with the shortcut test (job 23318585, RunAI final checkpoint; temporary script, not kept).
  - `_reconstruct` and the MPC loss weight patches with `adaptive_avg_pool1d(1 - W, 1024)`: 1024 frames -> 1024 patches is an identity, so
    patch p (frequency-major order) gets the mask of frame p. Correlation with the patch's real masked fraction: 0.000. In the code since 71028e7,
    so every run so far has it. The loss is half on visible frames.
  - W masks single frames, patches are 16 frames wide: patches are ~50% visible on average (almost no token is fully hidden), and 32 ms frames at a 10 ms hop overlap.
  - MSE / MSE of predicting zeros, on masked | visible frames: model 0.113 | 0.019; no right ear in the spectral input 0.402 | 0.100;
    spatial input zeroed 0.340 | 0.288; raw copy of the right ear 0.557 | 0.554. Zeroed inputs are out of distribution, so read these as rough.
  - Fixed: the recon and MPC losses use per-element weights on masked frames, W masks whole 16-frame blocks (`--mask_block`, 1 = old
    single-frame mask). All checkpoints and downstream numbers from before this fix are pre-fix: don't compare their recon losses.
- [ ] **New baseline with the fixed objective and fixed F0/HNR**: job 23380446, submitted 2026-11-05, queued (6 epochs, probes every 1000; ~15 s/step, so ~39 h plus preemptions):
  `sbatch --export=ALL,EPOCHS=6,PROBE_EVERY=1000,EXP_NAME_SUFFIX=-lossfix_blockmask_pitchfix_baseline,DATA_TRAIN=$SAC_DATA_ROOT/binaural_train.pitchfix.json,DATA_VAL=$SAC_DATA_ROOT/binaural_val.pitchfix.json src/binaural_dual_stream_sac/run_binaural_sac.sh`
  Output in `$SAC_EXP_ROOT/binaural_sac_pretrain/...-lossfix_blockmask_pitchfix_baseline` (`probe_metrics.jsonl`). After a time-limit kill, resubmit the same command: it resumes from `models/latest.pth`.
  **If this run keeps failing before it finishes, stop resubmitting and debug the cause.**
  Previous attempt, job 23347188 (old manifests), failed 2026-11-05 on resume: 0c3f0dc changed the size of `cpredlayer`, so its `latest.pth` no longer loads. Only 2000 steps: SC2 / SID / UA linear 12.0 / 16.5 / 45.7, EER 31.0.
  Old pre-fix run for comparison: SC2 linear ~12, SID linear ~16, EER ~29, IEMOCAP linear UA ~47, flat from step 4k.
- [ ] **Spectral-only runs** (one encoder, no CCSR; raw STFT vs log-mel, nothing else differs): reasoning, reading guide and limits in
  [docs/spectral_only_experiment.md](docs/spectral_only_experiment.md). `sbatch --job-name=sac_spec_stft scripts/spectral_only.sbatch stft`, same with `logmel`. Both used the old manifests.
  - `logmel`, job 23348511: still running 2026-11-05, in the last epoch (ends ~12.4k steps; each resume redoes that epoch's data, so compare by step, not epoch). Isolated NaN gradients, skipped by AMP.
  - `stft`, job 23348510: failed with NaN loss at step 8021. Resubmitted as job 23380447 (queued), resuming from step 8000, to finish the run and test the NaN. **If it NaNs again or keeps failing, debug the masked forward pass.**
  - NaN facts: every weight and buffer in the step 8000 checkpoint is finite and normal in size. Recon and MPC went NaN, first in validation at step 8000 after 3000 clean steps in that job segment, while Spec SAC stayed finite. Cause unknown: numerics in the masked pass, or a transient GPU fault. The `spatial_encoder contains NaN` line is only a label, since shared weights are named that way.
  - Linear probes, step 0 -> 2000 -> 8000 -> latest. SC2: `stft` 4.7 / 5.7 / 8.1, `logmel` 5.2 / 12.6 / 16.8 / 19.8 (step 12000), dual 5.0 / 12.0. SID: `stft` 2.1 / 8.9 / 16.7, `logmel` 38.8 / 29.5 / 33.5 / 33.1, dual 2.7 / 16.5. SID EER: `stft` 40.2 / 30.3 / 30.6, `logmel` 26.6 / 20.9 / 23.5 / 23.7, dual 40.3 / 31.0. IEMOCAP UA: `stft` 38.9 / 39.0 / 45.5, `logmel` 50.6 / 52.0 / 54.0 / 53.8, dual 38.9 / 45.7.
  - Reading so far: raw STFT costs far more than CCSR. `logmel` gains ~15 on SC2 vs 3.4 for `stft`. Removing CCSR did not help STFT: dual at step 2000 beats `stft` at step 8000 on SC2 (SAC weight differs, 1.0 vs 0.02). CCSR on log-mel is untested. Beyond SC2, `logmel` barely improves on its random encoder (best EER 20.2 at step 3000, SID linear falls from 39), and it stays far below GRAM-T.
  - Next, once the baseline has its curve: build the spatial probe, then design the log-mel spectral stream.
- [ ] **Ablations against that baseline**, ~4k+ steps each, scored with the probes: no SAC; same mask on both ears in the spectral stream;
  compressed input and log-magnitude target. Revisit after the spectral-only runs, which may replace them. Only if a run clearly beats the baseline: the 3 real downstream runs. Profile the step time first.
  Nothing measures the spatial encoder yet.
- [ ] **SAC vs no-SAC on all 3 tasks**, with 3 seeds each, so gains of 1-2 points can be told from noise
  (VoxCeleb1 fine-tuned: SAC 41.46% vs CCSR-only 40.25%, one run each).
- [ ] **Fine-tune GRAM-T** too, for a fine-tuned vs fine-tuned comparison. The runners already support it:
  pass `false` as the 3rd argument (`run_gramt.sh <model> <lr> false`); SC2 drops `--freeze_base`, s3prl adds `-f`.
  Training the whole encoder needs more GPU memory and possibly a lower lr than the frozen runs.
- [ ] Optional, if there is compute: pretrain SAC longer or on more data and plot downstream accuracy vs samples seen.

## Results

| Model | SC2 acc | IEMOCAP acc (fold1) | VoxCeleb1 acc | Job ids / notes |
|---|---|---|---|---|
| GRAM-T binaural-frame (frozen) | 89.59 (val 90.83) | | 76.88 (dev 76.97, best dev at step 45k) | SC2: job 23244879, lr 2.5e-4, AUC 0.9941; VoxCeleb1: job 23305339, lr 1e-4 |
| Binaural SAC (frozen) | | | | |

All IEMOCAP numbers from before the regeneration (including the RunAI "binaural IEMOCAP" runs) were on **mono**
IEMOCAP: the binaural setting did not exist yet, so the SAC model got mono audio copied to all 4 STFT channels.

## Open issues / to check

- [ ] **Binaural spectral SAC runs on 3 feature groups by accident.** The design says 15 features in 5 groups, but `run_binaural_sac.sh` never passes
  `--spectral_features`, so since 71028e7 it has used the default `f0,hnr,centroid,flux,zcr` (Prosody, Voice_Quality, Scene). Nothing records a reason.
- [ ] **F0 and HNR are fixed in the code, not yet in the manifests.** `src/sac/acoustic_features.py` now has a proper pitch tracker (checked by
  `scripts/check_pitch_features.sbatch`, job 23354831). `binaural_{train,val}.json` still hold the old values, which barely track real F0 (rank correlation -0.11 on train, 0.19 on val), so binaural runs,
  including those in progress, use the old features. Corrected copies are ready as `binaural_{train,val}.pitchfix.json` (job 23355202, 0 failures, only F0 and HNR differ):
  baseline job 23380446 uses them through `DATA_TRAIN` / `DATA_VAL`. Don't do the `mv` swap from `scripts/regen_binaural_features.sbatch` until jobs 23348511 and 23380447 are done, since they resume on the old manifests. A pitchfix spectral-only run needs a new `EXP_NAME_SUFFIX`, or it resumes the old run. 230 of 50,000 training clips have F0 exactly 0, probably a no-voiced-frames fallback (unchecked). Mono runs pick up the fix at once: `OPTUNA_OPTIMAL_SIGMAS` and `OPTIMAL_SIGMAS` were fitted to the old features.
- [ ] **Frozen SAC on SC2: 13.5% or 5.75%?** This file and the frozen-probe debugging above quote 13.5%, but the only frozen SC2 run on this
  cluster (`--frozen-binaural_sac-...binaural_ccsr_dual_sac`, copied from RunAI) has test 5.75% and best validation 5.50% in `eval_result.csv`
  and `result.csv`. Find out which run gave 13.5%, then correct whichever is wrong. VoxCeleb1 frozen agrees (14.5% quoted, 14.81% at best dev).
- [ ] **No seed in `run_amba.py` or `run_binaural_amba.py`**, so the SC2 and pretraining summaries have `seed: null`. Add a `--seed` before the
  3-seed SAC vs no-SAC runs. s3prl always uses 1337 unless told otherwise, so pass a different one per run there.
- [ ] **Binaural downstream sets were sampled with replacement.** In `data_generation/utils_simu_rir_sig.py`,
  `generate_microphone_signal` picks `src_idx = np.random.randint(0, len(srcdataset))` instead of `src_idx = idx`.
  So each simulated split covers only ~63% of its source utterances, and the rest of its clips are repeats in other rooms
  (IEMOCAP: 3,519 unique of 5,531; VoxCeleb1 and SC2: ~63% per split). No train/test overlap.
  - IEMOCAP + `spatial_data_root` (`run_er.sh` for binaural/SAC upstreams, `iemocap/run_gramt.sh`) **crashes**:
    `emotion/dataset.py` asserts that every s3prl fold utterance has a simulation (fold1: 398 test / 1,614 train missing).
  - VoxCeleb1 and SC2 run, but on the reduced, duplicated splits: fine for comparing SAC with GRAM-T, not the standard test sets.
  - The first IEMOCAP release also cut or zero-padded every utterance to 6 s before simulating (and computed the SNR over
    the padding), unlike the full-length mono IEMOCAP.
  - [x] **IEMOCAP: regenerated (v2), copied to TAU 2026-11-02** (`IEMOCAP/simu_v2/`, jsons in `$SAC_DATA_ROOT`, old ones as `*.json.v1`; 5,531 utterances, no duplicates, fold1 fully covered; length check vs the original IEMOCAP not run here). `gen_simu_iemocap.py` now simulates every utterance once
    (`sequential_src`), at full length (container `T` = longest utterance, padding cut off before the noise is mixed in).
    `regen_iemocap.sh` runs the 3 splits, builds the manifests, checks them with `verify_binaural_iemocap.py` (every fold
    utterance exactly once, right label, original length), and only then replaces `iemocap_spatial_*.json` (old ones kept
    as `*.json.v1`). It needs gpuRIR and the original IEMOCAP, so run it on RunAI:
    `cd sac-ssamba/data_generation && bash regen_iemocap.sh`.
    Then copy `<data>/IEMOCAP/simu_v2/` and the 3 `iemocap_spatial_*.json` to `/scratch200/yotams2/sac-ssamba/data/` here and
    re-run the check: `$PYTHON data_generation/verify_binaural_iemocap.py --data_root $SAC_DATA_ROOT --meta_dir ../sac-s3prl/s3prl/downstream/emotion/meta_data --iemocap_root <IEMOCAP_full_release>`
    (the label/coverage part needs no audio; the length part needs the original IEMOCAP).
  - [ ] VoxCeleb1 and SC2: their generators still sample with replacement. Pass `sequential_src=True` there too and
    regenerate if the standard test sets are needed.
  - The s3prl IEMOCAP loader refuses the old release (missing or duplicated utterances), so it cannot be used by mistake.
- [x] **The venv cannot run on compute-0-18** (RTX PRO 6000 Blackwell, sm_120): torch 2.2.2+cu121 has no kernels for it
  (`CUDA error: no kernel image is available`). The GPU scripts now have `#SBATCH --exclude=compute-0-18`. Upgrading torch
  would also mean new mamba / causal-conv1d wheels, so keep the exclude unless that node is needed.
- [ ] **wandb is not installed** in the shared venv (to save space). `--use_wandb` is then silently skipped.
  If you want it, run `$PYTHON -m pip install --no-cache-dir wandb` and `wandb login` once.
- [ ] **IEMOCAP runs only fold1** (`run_gramt.sh`, `run_er.sh`). Run all 5 folds for the final numbers.
- [ ] **Reading data from /scratch200 may be slow** in the SC2 job (about 85k small wavs per epoch, 30 epochs). If the epochs
  are I/O-bound, copy the SC2 `simu/` folders to node-local disk at job start, or cut `--num-workers`.
- [ ] **Bring the SAC side to this cluster** for the comparison. The SAC checkpoint
  (`run_binaural_sac_frozen.sh`: `/storage/yotam/ssamba/src/binaural_dual_stream_sac/exp/...`) and the
  SAC scripts still point at `/storage/yotam/...` on the old machine:
  - [ ] copy the checkpoint to `/scratch200/yotams2/` and update the `MODEL_DIR` paths
  - [ ] port `run_sc_amba.sh`, `run_er.sh`, `run_sid.sh` and the `run_binaural_sac_frozen.sh` wrappers to `env.sh`
        (sbatch-safe `SCRIPT_DIR`, `--partition`, `$PYTHON`). The binaural SC2 json paths already use `SAC_DATA_ROOT`.
- [ ] **Optional GRAM-T variants**: `gramt_binaural_time`, `gramt_mono` (weights download on first use, ~0.4 GB each).

## Done

- [x] Results sync (2026-11-04): every launch script reports to the `results` branch, see "Results sync" in [CLAUDE.md](CLAUDE.md). Backfilled:
  the finished TAU runs and 74 RunAI runs. Jobs 23347188, 23348510, 23348511 were started before the hook; `collect_after.sbatch` jobs report them when they end.

- [x] Frozen GRAM-T dropout: not an issue. `encoder_dropout` and `encoder_attention_dropout` are 0.0 in the HF config, so
  `train()` putting the frozen encoder in train mode changes nothing.

- [x] Binaural SC2 eval fix (the SAC encoder now gets the 4-channel STFT used in pretraining, not repeated
  mono fbanks): merged to `main` (ecdca0f).
- [x] GRAM-T wrapper (`src/gramt_wrapper`), SC2 path in `run_amba.py`, s3prl upstream `gramt_*` in sac-s3prl.
- [x] `/scratch200/yotams2/sac-ssamba/data.tar` deleted after extracting `data/` (2026-10-28).
- [x] `SAC_DATA_ROOT` remapping of the json wav paths (`src/data_paths.py`), used by both the GRAM-T and the binaural SAC SC2 datasets.
