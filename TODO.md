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

## Thesis scope review (2026-11-06)

Estimate only, assumes near full-time work; course status and deadlines not checked. Must-have: 3+ pretraining seeds for CCSR / CCSR+SAC / SAC only (the paper says "different seeds", all runs are seed 1); mono SAC vs no-SAC with the same objective, 3 seeds, 3 tasks, IEMOCAP 5 folds (needs `--seed`); write the thesis. Binaural: time-box, build the spatial probe, report what is known. Future work, not thesis: Sinkhorn, GRAM-T retrain, SARL, curriculum, global vs time-varying split. Don't quote the Notion mono gains (61.4 vs 59.2, 60.5 vs 56.6): best-of-N, confounded objective, old F0/HNR.

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
- [ ] **New baseline with the fixed objective and fixed F0/HNR**: job 23380446, done 2026-11-07 12:35 on compute-0-419, results pushed, ~12 s/step (6 epochs, probes every 1000). Val total 7.75 (step 4000), 7.69 (8000). NaN only at steps 22 and 3538, one step each. Linear probes at 0 / 2000 / 9000: SC2 4.2 / 11.2 / 11.0, SID 2.4 / 16.5 / 23.4, EER 40.2 / 28.5 / 28.6, UA 38.5 / 46.6 / 46.1, so flat since ~2-3k except SID:
  `sbatch --export=ALL,EPOCHS=6,PROBE_EVERY=1000,EXP_NAME_SUFFIX=-lossfix_blockmask_pitchfix_baseline,DATA_TRAIN=$SAC_DATA_ROOT/binaural_train.pitchfix.json,DATA_VAL=$SAC_DATA_ROOT/binaural_val.pitchfix.json src/binaural_dual_stream_sac/run_binaural_sac.sh`
  Output in `$SAC_EXP_ROOT/binaural_sac_pretrain/...-lossfix_blockmask_pitchfix_baseline` (`probe_metrics.jsonl`). After a time-limit kill, resubmit the same command: it resumes from `models/latest.pth`.
  **If this run keeps failing before it finishes, stop resubmitting and debug the cause.**
  Previous attempt, job 23347188 (old manifests), failed 2026-11-05 on resume: 0c3f0dc changed the size of `cpredlayer`, so its `latest.pth` no longer loads. Only 2000 steps: SC2 / SID / UA linear 12.0 / 16.5 / 45.7, EER 31.0.
  Old pre-fix run for comparison: SC2 linear ~12, SID linear ~16, EER ~29, IEMOCAP linear UA ~47, flat from step 4k.
- [ ] **Spectral-only runs** (one encoder, no CCSR; raw STFT vs log-mel, nothing else differs): reasoning, reading guide and limits in
  [docs/spectral_only_experiment.md](docs/spectral_only_experiment.md). `sbatch --job-name=sac_spec_stft scripts/spectral_only.sbatch stft`, same with `logmel`. Both used the old manifests.
  - `logmel`, job 23348511: done 2026-11-05 19:53 (`audio_model.final.pth`, results pushed). Isolated NaN gradients, skipped by AMP. Each resume redid that epoch's data, so compare by step, not epoch.
  - `stft`, job 23348510: failed with NaN loss at step 8021. Resubmitted as job 23380447, done 2026-11-06 17:09 (results pushed, final val 7.64, step 12000 probe SC2 10.0 / SID 19.7 / EER 29.3 / UA 49.1, still rising at the end). Probes at 9000 / 10000 / 11000: SC2 9.05 / 9.33 / 9.05, SID 19.25 / 17.75 / 19.50, EER 30.09 / 30.58 / 29.25, IEMOCAP UA 46.42 / 45.60 / 47.31. Step 12000 validation finite: total 7.6389, Recon 1.5269, MPC 6.0472 (4.86%), best so far; linear SC2 10.0, SID 19.75. Only 2 NaN steps in the job, so the run is usable.
  - NaN facts: every weight and buffer in the step 8000 checkpoint is finite and normal in size. Recon and MPC went NaN, first in validation at step 8000 after 3000 clean steps in that job segment, while Spec SAC stayed finite. Cause unknown: numerics in the masked pass, or a transient GPU fault. This is not the old spatial-collapse NaN: its fixes (BatchNorm1d in `projection_head_spat`, fp32 cosine similarity in `_compute_single_vector_sac`) are still in the code, but `spectral_only` never runs that branch (`sac_spat_lambda=0`). First lead: `AMBAModel.mpg` / `mpc` have their own forward loop, separate from `_encode_with_mamba`, which stayed finite. The `spatial_encoder contains NaN` line is only a label, since shared weights are named that way.
  - NaN update 2026-11-06 (job 23380447): loss NaN at steps 8001 (Recon finite, so MPC only) and 8003 (Recon too), then none in ~3700 steps. It starts within a few steps of the step 8000 weights in both jobs, on different nodes and GPU models (L40S on compute-0-102, A6000 on compute-0-420), so a transient GPU fault is very unlikely. It looks tied to the weights near step 8000 plus particular batches: the old job hit 14 NaN steps and reached 5 in a row, this one only 2 before updates moved the weights on. The `nan` on every later log line is display only: `AverageMeter` keeps the NaN until `.reset()` at the next validation, which runs every `epoch_iter`=4000 steps, not every probe. Those meter values also go to wandb `train/*`. Fix to make: skip non-finite values in the meters. Baseline job 23380446 had 1 NaN gradient step and no NaN loss so far.
  - Linear probes, step 0 -> 2000 -> 8000 -> latest. SC2: `stft` 4.7 / 5.7 / 8.1, `logmel` 5.2 / 12.6 / 16.8 / 19.8 (step 12000), dual 5.0 / 12.0. SID: `stft` 2.1 / 8.9 / 16.7, `logmel` 38.8 / 29.5 / 33.5 / 33.1, dual 2.7 / 16.5. SID EER: `stft` 40.2 / 30.3 / 30.6, `logmel` 26.6 / 20.9 / 23.5 / 23.7, dual 40.3 / 31.0. IEMOCAP UA: `stft` 38.9 / 39.0 / 45.5, `logmel` 50.6 / 52.0 / 54.0 / 53.8, dual 38.9 / 45.7.
  - Reading so far: raw STFT costs far more than CCSR. `logmel` gains ~15 on SC2 vs 3.4 for `stft`. Removing CCSR did not help STFT: dual at step 2000 beats `stft` at step 8000 on SC2 (SAC weight differs, 1.0 vs 0.02). CCSR on log-mel is untested. Beyond SC2, `logmel` barely improves on its random encoder (best EER 20.2 at step 3000, SID linear falls from 39), and it stays far below GRAM-T.
  - **Decision 2026-11-06: raw STFT dropped as the spectral encoder's input** (`logmel` MPC val accuracy reached 28.8% at step 12000, `stft` training MPC stays ~5%). Job 23380447 is left running: its step 12000 validation still tests the NaN, and `logmel` uses the same `mpg` / `mpc` code. Not settled by this: whether CCSR helps or hurts log-mel. `logmel` uses `AMBAModel.mpg` / `mpc` + spectral SAC and ignores W, so there is no log-mel + CCSR arm. The STFT evidence does not support CCSR hurting the spectral encoder (dual 11-12 vs `stft` 5.7 on SC2 at step 2000), but SAC weight (1.0 vs 0.02) and recon weight differ there too. A CCSR log-mel arm or a throwaway content encoder needs design: CCSR's target is the complex STFT, and the spatial encoder still needs STFT phase.
  - Next, once the baseline has its curve: build the spatial probe, then design the log-mel spectral stream.
- [ ] **Proposal, undecided 2026-11-06: rerun with fixed F0/HNR before any pivot.**
  - `logmel` + pitchfix manifests, as a pair: `sac_spec_lambda` 0.02 and 0. Job 23348511 used old manifests, where 2 of the 3 active SAC groups (Prosody, Voice_Quality) were broken, and has no no-SAC control. SAC is <2% of its total loss (0.02 x 3.2 of ~3.75). `spectral_only.sbatch` hard-codes `EXP_NAME_SUFFIX` and the lambda, so it needs a small change. Pass `DATA_TRAIN` / `DATA_VAL`, no `mv` swap needed.
  - 2026-11-06: mono path made runnable here: `scripts/mono_sac.sbatch` (exp6 recipe; tau 0.12 and no mpc come from exp6's config.json and git, f79ec65), `scripts/localize_librispeech_manifest.py` (rewrites paths, stops on a missing file, since the mono loader falls back to silence), `scripts/extract_librispeech.sbatch`, `run_sac.sh` settings through env vars, `latest.pth` + auto-resume in `run_pretrain_sac.py`. Diagnostics off, since seaborn and sklearn may be missing from the venv.
  - Smoke test passed, job 23410039 (GPU, 12 clips, 17 min): 2 passes, exit 0, resume restored step 6 / epoch 2 and optimizer state and continued to step 12. Job 23410019 on CPU cannot work: the Mamba kernel needs CUDA. Time per step is not measured: the smoke saved `latest.pth` (1.2 GB) every 2 steps. Check the first log lines of the real run, because SAC features are computed in the training loop.
  - Mono rerun submitted 2026-11-06: job 23410602, `--dependency=afterok:23410017`, user's choice of the baseline objective: `sbatch --export=ALL,RECON_LAMBDA=10.0,CLASSIF_LAMBDA=1.0,EXP_NAME_SUFFIX=-baseline_objective_pitchfix scripts/mono_sac.sbatch`. 132,553 train clips, 2,703 eval, ~2071 steps per epoch, 10 epochs. Output in `$SAC_EXP_ROOT/sac_pretrain/...-baseline_objective_pitchfix`. tau 0.12 (spectral-only runs use 0.3). Manifests copied to `$SAC_DATA_ROOT`; originals are git-ignored in the repo root.
  - GPU check 2026-11-07: jobs are GPU-bound (100% utilization, data workers near idle). gpu-general-pool has V100 (compute-0-300), A6000 (58, 419, 420), A5000 (53), RTX_6000 (376), RTX 6000 Ada (100), L40S x36 (102, 223, 291-293, 377), A100 (10, 196, 282-285), H100 (62), H200 (103). Same `stft` job: 4.7 s/step on L40S (compute-0-223, step 0 to 4000 in 5h14, probes included) vs 9.0 on A6000 (compute-0-420, 8000 to 12000 in 10h03); probe 55 vs 96 s. V100 probe ~99 s, same as A6000. A100 is unclear (logmel probe 50 s but 5.6 s/step on compute-0-282). H100 and H200 untested. RunAI GPU is an RTX 6000 Ada Generation, 48 GB, 300 W (nvidia-smi from the user 2026-11-07; `CLAUDE.md` says L40, which is wrong but the same generation); no RunAI step time recorded apart from the ambiguous "~10 h per 4000 steps" above. Owner-pool preemption hit these jobs on L40S, A100 and V100 on 2026-11-04/05; compute-0-419/420 ran uninterrupted, one sample. Next mono resubmit: `--exclude=compute-0-[18,53,58,300,376,419,420]`. Untested idea: drop `--use_checkpointing` (uses 20 GB of 32 GB now). `sacct -X` Elapsed shows only the last requeue segment; use `sacct -D`.
  - Mono status 2026-11-07 12:24: started 2026-11-06 18:32 on compute-0-300, paths OK (0 missing), MPC runs, step ~5300 of 20710, 12.1 s/step (~70 h total), no NaN, val 4.99 at step 4000, train MPC acc ~40%. Hits the 2-day limit 2026-11-08 18:32 at ~step 14.3k; resubmit the same command, ~21.5 h more. Closer: 23411657. Resume queued 2026-11-07: job 23416521, `--dependency=afterany:23410602`, `--exclude=compute-0-[18,53,58,300,376,419,420]`.
  - Focus decided 2026-11-07: mono and profiling first, binaural on hold. Same-recipe old numbers (b64, lr 4e-4, single seed, `amba` vs SAC lam 0.02 offline median): SC2 97.34 vs 97.46, SID 60.5 vs 61.29, IEMOCAP 57.24 vs 56.13. Not a clean test: different objective (SAC lacked mpc), old F0/HNR. The current mono run is the first with the same objective and fixed features. Rule agreed 2026-11-07: within ~1.5 pts on all 3 tasks = no sign of a gain, stop investing and run only the seeds the thesis needs; a clear gain on at least one task = 3 seeds to confirm. SAC-SAR-SSL: the user's Interspeech paper shows SAC helps there, pending the multi-seed check (read the paper, not just the summary tables, for its numbers). Seed std in the 2026-09-16 table comes from `gen_summary_table.py`, check what it measures (6126b7c) before trusting it; that table has no Original row; the 2025-12-29 table does (mean TDOA 0.483 vs 0.422, DRR 2.02 vs 1.81, T60 and ABS no better).
  - Profiling 2026-11-07: job 23416548 on branch `profiling` (worktree `~/sac-ssamba-profiling`, so the working copy that the queued resume reads stays untouched), `scripts/profile_pretrain.py`, output `$SAC_EXP_ROOT/_profile/<jobid>/results.jsonl`. Found by reading: binaural runs use `num_workers=0` (default, never set in `run_binaural_sac.sh`); mono runs 3 full encoder passes per step (mpg, mpc, SAC) under gradient checkpointing with no AMP; binaural checks every parameter gradient with `.item()` each step; formants use `torch.linalg.eigvals` on the GPU. Timings pending.
  - Profiler first results 2026-11-07 (mono on A100-80GB, 20 steps): 8.22 s/step = backward 6.11 + forward 2.00 + data ~0 + to_device/features 0.05 + optimizer 0.01. Inside forward: mamba layers 1.88 s over 3 passes, mask ids 0.04, SAC loss 0.003. Peak alloc 18.6 GB. `no_ckpt` OOM at 79 GB (fails on any card in this pool at b64). `formants_eigvals_ab` said GPU 5.26 s vs CPU 0.024 s, identical output, but that GPU figure looks like a cold first call: per-step formants cost 0.03 s in the breakdown, so no gain there.
  - Profiler final 2026-11-07 (23416548, COMPLETED 1:13 h, A100, 20 steps each, 0 nonfinite). Mono: base 8.22, amp 4.53 (13.9 GB), amp_no_ckpt 3.94 (63.8 GB), no_ckpt OOM. Logmel (b32, AMP): w0 5.79-7.28, w8 2.29, w8_nochecks 2.24, w8_no_ckpt 1.84 (32.5 GB). Dual (b32, AMP): w0 8.78-9.26, w8 5.72, w8_nochecks 5.60, w8_no_ckpt OOM. Data wait is 3-5 s/step at num_workers=0. Grad checks cost 0.05-0.12 s, leave them. RAM peak 19.5 GB of 32.
  - Fixes on local branch `profiling` (`12f979c`, not merged, not pushed): `--amp` in `run_pretrain_sac.py` (`AMP=true` in `run_sac.sh`, default off, scaler saved in latest.pth, logs skipped steps), `--num_workers ${NUM_WORKERS:-8}` in `run_binaural_sac.sh`. `31034e4` holds the same mono code that is now on main as `85eaf13` (byte-identical, checked), so a merge should not conflict there. `scripts/check_full15_manifest.py` and `regen_full15_features.sbatch` are still uncommitted on main.
  - AMP check: job 23417434 (running on compute-0-223 since 15:19), `sac_mono_ampval`, submitted from the worktree, 2 epochs, log `~/sac-ssamba-profiling/logs/sac_mono_ampval_23417434.out`, exp suffix `-baseline_objective_pitchfix_ampval`. Compare with 23410602: step 4000 val 4.99, train acc, no skipped steps. Unseeded, so match within noise only. Why it matters: every run with NaNs so far used AMP; `both_models.py:621` already forces fp32 for an fp16 overflow on raw STFT. Do not switch the running mono to AMP: it must stay comparable with `amba-*`. Downstream check on AMP weights still needed before using AMP for seeds. 8-worker binaural only tested 20+ steps; the segfault comment at `run_binaural_amba.py:364` is about the stats loader, kept at 0.
  - Mono downstream chain, set up 2026-11-07: extract 23417704 (`scripts/extract_mono_downstream.sbatch`, /moshe-data archives to `$SAC_DATA_ROOT/mono`, deletes nothing, does not use the scratch copies `/scratch200/yotams2/data/{speech_commands_v0.02.tar.gz,voxceleb1.tar.gz}`, 36 GB, delete after the smoke test if the user agrees; COMPLETED 17:07 in 1:08 h, 342508 new files: SC2 105841, IEMOCAP 83151, VoxCeleb1 153516, more than the 270k estimated, 1251 speakers, SC2 splits 84843/9981/11005; shared filesystem is at 96% of inodes) -> smoke 23417710 (`afterok`, `scripts/smoke_mono_downstream.sbatch`, on a copy of the step-4000 checkpoint in `_smoke/mono_ds`; `check_mono_upstream_load.py` verifies the encoder weights really load on both paths, since loading is strict=False) -> gate 23417711 (`afterany:23416521`, `scripts/mono_downstream_gate.sbatch`) submits SC2, IEMOCAP, VoxCeleb1 only if `audio_model.final.pth` exists and the smoke test passed. If the gate says NOT SUBMITTED, nothing reruns by itself: resume pretraining and submit a new gate.
  - Smoke 23417710 FAILED in 2.5 min at the load check: SC2 path fine (432 encoder tensors match), s3prl path built a 192-dim model. Cause: sac-s3prl `7edd93ec` (2026-09-09) kept `model_size="base_a"`; every mono s3prl run since then would crash at load, July baselines predate it. Fixed in sac-s3prl (`ea490c91`). Rerun 23417899 passed the load check on both paths (432 and 408 tensors) and failed at `run_amba.py` argument parsing: `run_sc_amba.sh` passed `--pool_real_time_only false`, which `ast.literal_eval` rejects; default now `False`. Third run 23417912 COMPLETED 2026-11-07 18:08 in 52 min on an A100, `.passed` written, step-4000 checkpoint: SC2 1 epoch test acc 44.6%, IEMOCAP 40 steps 45.8%, VoxCeleb1 40 steps 0.16% (near chance, only 40 steps; these are pipeline checks, not results). Measured A100 speed: SC2 ~16 min/epoch, IEMOCAP ~2.5 s/step, VoxCeleb1 ~2.4 s/step, evals 4-9 min, so ~9 h, ~22 h, ~35 h. Slower than RunAI (5, 13, 25 h) but under the 2-day limit; preempted s3prl jobs resume, SC2 restarts from scratch. Other mono-path changes since July checked, harmless. s3prl path uses rms_norm=True while pretraining uses False: same in the baselines, left alone for comparability.
  - IEMOCAP on scratch checked 2026-11-07, file by file: mono `$SAC_DATA_ROOT/mono/data/IEMOCAP_audio_only/IEMOCAP_full_release` has all 5531 utterances of the s3prl meta_data (none missing or empty); binaural `$SAC_DATA_ROOT/IEMOCAP/simu_v2` has all 5531 in `iemocap_spatial_{pretrain,preval,pretest}.json` (3556 + 890 + 1085, no duplicates, every meta_data utterance covered). `IEMOCAP/simu` is the old v1 set, not used by the current manifests. Content of the wavs not checked.
  - Recipe read from the old RunAI exp dirs, identical for `amba-base-f16-t16-b64-lr4e-4` and exp6: fine-tuning (`-f`), `best_audio_model.pth`, SC2 mono model_size base lr 2.5e-4 30 epochs; VoxCeleb1 lr 1e-4 50000 steps; IEMOCAP fold1 30000 steps **real lr 1e-4**. The IEMOCAP dirs are named `_1e-5` but only `test_fold` was overridden, so the config value 1e-4 applied; the script now does apply `lr`, so the gate passes 1e-4. Fixed: `*sac*` name matching sent mono SAC runs to the binaural path (repo path contains "sac" too); now only `*binaural*` in the basename. Old run times (RunAI): SC2 ~5 h, IEMOCAP ~13 h, VoxCeleb1 ~25 h. Baseline `amba` numbers stay from RunAI: different machine and venv, not rerun here.
  - **Old mono comparisons are confounded**: the no-SAC baseline `amba-*` is `mpc + 10 mpg`, while exp6-style SAC runs are `1.0 mpg + 0.02 SAC` with no mpc, so SAC vs no-SAC there also changes the objective. Check before using the table below.
  - Mono SAC-SSAMBA rerun, same recipe as exp6 (b64, lr 4e-4, lam 0.02, universal features, offline_global_median), then the 3 downstream tasks. All mono runs predate the fix (2084dd3). Needs clean LibriSpeech and manifests copied from `/moshe-data` to scratch (not on scratch now; RunAI checkpoints and pretrain time on TAU unchecked). `OPTUNA_OPTIMAL_SIGMAS` / `OPTIMAL_SIGMAS` are stale, so don't use the optuna mode.
  - Existing fine-tuned mono test results, no-SAC vs SAC runs (test at best dev, single seed): SC2 97.3-97.7 vs 95.2-97.7; SID 56.1-60.5 (59.2 for `ssamba_baseline`) vs 52.7-61.4; IEMOCAP 54.7-57.2 vs 52.5-60.6 (best is `feat_emo` lam 0.02 sig 2.0, the only hint of a gain, best of 9 on fold 1). No clear SAC effect so far.
  - [x] 2026-11-06: job 23409908 done in 8.5 min (`scripts/regen_full15_features.sbatch`, `scripts/check_full15_manifest.py`): `binaural_{train,val}.full15.json`, 0 failed clips, order, spatial and shared features unchanged, all finite, clipping under ~2%. Group median distances: Prosody 0.52, Vocal_Tract 0.70, Timbre 0.96, Voice_Quality 0.32, Scene 0.79. Exact-value pile-ups in `f0_mean` (~2% per value at 195-200 Hz) are the integer lag grid of the pitch tracker (16000/80, /81, /82), not a fallback. Open: train medians F1 / F2 / F3 are 414 / 1054 / 2038 Hz, F2 and F3 lower than typical speech; compare with clean LibriSpeech once on scratch, same code. centroid, zcr_mean and mfcc1 correlate at ~0.9-0.96, so Scene and Timbre overlap. The `logmel` pair is on hold; the user is copying the mono data and rerunning mono pretraining first.
  - Use all 15 features (5 groups) in the `logmel` pair, not the accidental 5: `f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm`, the mono `universal` list. Steps: `precompute_acoustic_features.py --keep_spatial --features_list ...` to a new manifest (`binaural_{train,val}.full15.json`, CPU job; the F0/HNR-only regen took 12 min on 16 CPUs, formants and MFCC add to that), then `--spectral_features` with the same list through `EXTRA_ARGS`. Per-group medians are recomputed at startup, so no sigma table is involved. Formants and MFCC have only run on clean mono LibriSpeech, never on the reverberant mean-of-ears signal, so check their distributions and failures after the regen (`check_regen_manifest.py` assumes the 5-feature layout, needs a variant). Dual baseline 23380446 stays on 5 features.
- [ ] **Proposal, undecided 2026-11-06: retrain GRAM-T on our pretraining set** (same 50k binaural clips, batch 32, 40k steps = same ~1.3M samples), scored with the same frozen probes. Tells data scale from architecture/objective: near `logmel` means the gap is mostly data, far above means our model or objective. Code in `~/GRAM-T` (`train.py`, `patching=frame`, `data.mask_patch=100`). Its README gives ~72 h for 200k steps on an A100, ~1.3 s/step vs our ~13 s/step on different GPUs. Needs: a data adapter (it reads AudioSet webdataset shards and adds RIRs and noise online, ours are simulated clips) and its own env (Python 3.9, torch 2.5.1; our venv has torch 2.2.2). Our clips appear to be 16 kHz, GRAM-T runs at 32 kHz (check before building). If it scores far above `logmel`, consider adding SAC to GRAM-T.
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
  - [x] port `run_sc_amba.sh`, `run_er.sh`, `run_sid.sh` to `env.sh` (2026-11-07, mono path, smoke test pending)
  - [ ] port the `run_binaural_sac_frozen.sh` wrappers to `env.sh`
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
