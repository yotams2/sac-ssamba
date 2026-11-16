# TODO

Current state only. History goes in [docs/log/](docs/log/) (`2026-11.md`; full pre-compaction TODO archived there 11-16 12:30).
**Current sprint: [SPRINT.md](SPRINT.md) (11-16 → 11-30): one model good at spectral + spatial tasks.** One-screen summary: `~/STATUS.md`.

## Sprint work (see SPRINT.md for the why)
- [ ] RQ2 single encoder: `pretrain_probe.py --audio binaural --input_repr ipd16` (`scripts/ipd16_probe.sbatch <out> "<extra>" <ckpts|random|gramt>`): SAC-SAR-SSL `ssamba_spatial` wrapper + `stft_reim`, clips cut into 4.112-s windows (tail < 0.5 s dropped), fp32, pooling as the other probes. Smoke 23663228 (tiny subsets) -> full 23663229 (afterok): ipd16_sac / ipd16_nosac final, random, gramt -> `exp/_probes/ipd16_probe_2026-11-17.jsonl`. Caveat: probe audio is the sac-ssamba binaural sets, not SAR-SSL's free-field 2-mic arrays the encoder saw.
- [ ] RQ1 GRAM-T on SAR-SSL tasks: `--upstream gramt` (SAC-SAR-SSL `code/gramt_spatial.py` over sac-ssamba `gramt_wrapper`: 16→32 kHz, 2-ch log-mel, "raw" tokens mean-pooled over real audio, 6144-d, LayerNorm+Linear head; uncommitted). Launcher `scripts/gramt_spatial_eval.sbatch <label> "<tasks>"` (R8, 20 ep, lr 1e-3/1e-4, finetune+lineareval; no scratchLOW). Smoke 23655896 passed 11-16 20:51 (TDOA 1 ep: finetune 3.03, linear 2.94). TDOA 23655897 done: finetune 2.65, lineareval 2.94 (~mean predictor; test MSE 13.2 at an untrained head). Not a bug: magnitude-only log-mel input + free-field 3-20 cm mic pair = no TDOA cue. Report GRAM-T TDOA as 'not applicable by design'; judge GRAM-T on T60/DRR/C50/ABS. Jobs: `gramt_bf` TDOA 23655897, T60 98, DRR 99, C50 900, ABS 901 (afterok). If the smoke fails, the 5 stay pending forever: cancel + resubmit.
- [ ] RQ2 frozen ipd16 encoders (SAC/no-SAC final) on binaural SC2/SID/IEMOCAP: ipd16 front end + length policy in `pretrain_probe.py`.
- [ ] RQ3 frozen concat (log-mel `amba` s1 on L + ipd16 SAC) on both task families; small cross-attention decoder reconstructing log-mel L/R + IPD from spectral / spatial / both.
- [ ] RQ4 after RQ2/3: SAIL-style two-tokenizer SSAMBA + shared masked reconstruction, 10-ep pilot on WSJ0 simulated set.
- [ ] SAR-SSL reference numbers for DRR/C50/ABS (results branch).

## Running / waiting
- **Spatial DRR/C50/ABS** 23652530/31/32 (pre_nosac_final, pre_sac_final finetune+frozen; ctrl_ipd16_n256). DRR finetune already: no SAC 1.81, SAC 1.58, random 2.17. Must-have.
- **`amba` seeds downstream** (no-SAC mono baseline, thesis must-have): s2 SID 23638875 preempted 11-16 17:10 at 30.9k/50k, requeued (resumes from 29k ckpt), s2 SC2 rerun 23640910, s3 IEMOCAP 23652540 (resumes from 18k). Collect jobs 23638928/23650768.
- **Mono probe** 23654005 FAILED 11-16 20:57 (stale NFS handle reading the running k=5 `latest.pth`; probe copies instead). k=5 audio_model.2 scored exactly chance on all probes (random init above chance): diag 23663212 (resubmit of 23663181, import path; `scripts/diag/probe_nan_check.*`) checks weights + fp16 vs fp32 embeddings; suspect fp16 autocast in `_sac_batch` (also used by the in-loop probe). Fix before rerun. Data load 102-712 s; use --time=3:00:00. (`scripts/mono_probe.sbatch` -> `exp/_probes/mono_probe_2026-11-16.jsonl`). In-loop probe (PROBE_EVERY=2000) untested until a new pretraining starts: check its first `---- Step 0 probe` line.
- **Mono SAC perplexity** k=5 23647242 running (epoch 4/10, ~1 day left) + gate 23647243. k=10 23652538 + gate 23652539 **held 11-16 for the sprint** (`scontrol release` to resume).
- **Mono SAC seeds 2/3** (median rule) 23640933/35 held, gates 23640934/36: cancel at the k=5 read if perplexity wins.
- Backup + prune daily (23653353).

## Results
Spatial ipd16 SSAMBA, test MAE, R8, 20 ep, single seed (finetune / frozen):
| Encoder | TDOA | T60 |
|---|---|---|
| no SAC final | 0.26 / 2.43 | 0.050 / 0.051 |
| SAC final | **0.23 / 1.31** | 0.049 / 0.047 |
| random init 256 frames | 1.04 / - | 0.133 / - |
| SAR-SSL ref | 0.29-0.48 / 1.1-2.1 | 0.107 / - |
Original raw Re/Im input: TDOA 2.39 finetune, 2.92 frozen, 2.86 random (mean predictor 3.07).

Mono, test at best dev, single seed (SC2 / IEMOCAP fold1 / SID):
| Run | SC2 | IEMOCAP | SID |
|---|---|---|---|
| SAC old features | 97.58 | 60.18 | 71.40 |
| SAC egemaps18 | 97.50 | 56.7 | 67.5 |
| `amba` s1 | 97.45 | 58.71 | 67.9 |
| `amba` s2 | pending | 55.9 | running |
| `amba` s3 | 97.61 | pending | 67.7 |
| `amba` RunAI (old baseline) | 97.34 | 57.24 | 60.50 |
| GRAM-T binaural-frame frozen | 89.59 | - | 76.88 |

## Decisions in force
- **Sprint focus 11-16 → 11-30** (supervisor meeting): see SPRINT.md. No new mono SAC runs or spatial seeds this sprint.
- **Spatial input = ipd16** (11-12); raw STFT dropped as spectral input (11-06), log-mel instead.
- **Mono baseline = our 3 `amba` seeds** (11-15), not RunAI. **SAC features = egemaps18** (11-15; [docs/feature_selection.md](docs/feature_selection.md)). SAC sigma: perplexity rule under test (`--sac-perplexity k`).
- **1.5-pt rule** (11-07): SAC within ~1.5 pts of no-SAC on all 3 tasks = no gain; a clear gain on any task = 3 seeds.
- **Thesis must-haves** (11-06): 3+ seeds for the final arms, mono SAC vs no-SAC on 3 tasks with IEMOCAP 5 folds. Don't quote the old Notion mono gains.
- No AMP for mono (NaNs). Keep the old-feature downstream runs.

## Open tasks (not this sprint unless noted)
- [ ] Commit: `src/traintest.py` resume fix and other uncommitted items (SAC-SAR-SSL: figures scripts, opt.py `--ds-lr/--ds-nepoch`, pretrain code).
- [ ] Spatial seeds 2-3 of both ipd16 arms (after the sprint).
- [ ] IEMOCAP all 5 folds for final numbers (`run_er.sh`, `run_gramt.sh` run fold1 only).
- [ ] Formal tau/sigma choice for SAC (SNE frame: perplexity calibration; tau learnable).
- [ ] ER/SID resume is not exact (s3prl `-a`): your call whether to fix.
- [ ] GRAM-T frozen IEMOCAP (`run_gramt.sh gramt_binaural 1e-4 true`): cheap, fills the "both families" table.
- [ ] Binaural VoxCeleb1/SC2 sets sampled with replacement (~63% unique): regenerate if standard test sets are needed.
- [ ] `AverageMeter` keeps NaN until next validation. Frozen SAC SC2 13.5% vs 5.75% discrepancy.
- Superseded by the sprint: `logmel` pair, CCSR on log-mel design, `run_binaural_sac.sh` feature-list bug (only matters if the old binaural recipe returns).

## Facts and caveats
- **Cluster:** preempted jobs requeue; s3prl resumes (`-a`), SC2 resumes via `resume.pth`. Exclude `compute-0-[18,53,58,300,376,419,420]`. L40S ~1.8x A6000. Use `sacct -D`. Priority = fairshare only. Replace launch scripts instead of editing in place while a job runs them.
- **Storage:** scratch 285/400 GB; never unpack `MicSig/simu_pretrain.tar` (read through `simu_pretrain.tar.index.json`). Shared FS near 100% inodes. Daily backup + `scripts/backup/prune.py`; pin in `keep.txt`.
- **Spatial pretraining:** `SAC-SAR-SSL/code/ssamba_spatial_pretrain.py`, `scripts/ssamba_spatial_pretrain.sbatch`; wrapper `code/ssamba_spatial.py` (`--ssamba-nframes` for random control); launcher `scripts/ssamba_spatial_downstream.sbatch <TASK> <ckpt> <label> [nsimroom] [mode]`. Encoders in `SAC-SAR-SSL/exp/ssamba_pretrain/ipd16_{nosac,sac}/encoder_final.pth`.
- **Mono pipeline:** SAC features from the first 11.24 s; normalisation stats stored in the checkpoint (new features need a new `EXP_NAME_SUFFIX`). Old IEMOCAP dirs named `_1e-5` ran at 1e-4.
- **GRAM-T:** venv `/scratch200/yotams2/venvs/gramt`; binaural-frame input = 2-ch log-mel at 32 kHz (n_fft 1024, hop 320, 128 mel), no phase.
- **Venv:** `/scratch200/yotams2/venvs/sac` (parselmouth 0.4.7; no wandb).
