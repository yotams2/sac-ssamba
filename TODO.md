# TODO

Current state only. History goes in [docs/log/](docs/log/) (`2026-11.md`; full pre-compaction TODO archived there 11-16 12:30).
**Current sprint: [SPRINT.md](SPRINT.md) (11-16 → 11-30): one model good at spectral + spatial tasks.** One-screen summary: `~/STATUS.md`.

## Sprint work (see SPRINT.md for the why)
- [x] RQ1 GRAM-T on SAR-SSL tasks: done 11-17 (results below). n/a on TDOA by design (magnitude-only input).
- [x] RQ2 ipd16 frozen probes on binaural SC2/SID/IEMOCAP: done 11-17 (23663229, results below). Caveat: probe audio is the sac-ssamba binaural sets, not SAR-SSL's free-field arrays.
- [ ] **Joint pilots A/B** (spec: Notion Agent Handoff 11-17; code `SAC-SAR-SSL/code/joint_pretrain.py`, `scripts/joint_pretrain.sbatch`). Smoke 23665154 failed (mamba_ssm import, fixed); rerun smoke 23673591 -> Pilot A spec 23673592 / spat 23673593 (afterok). A-spec epoch-2 gate: MPC acc vs chance 1.8%, MPG falling; else fallback 6/16 columns for all pilots. Pilot B after that. Eval to build: spectral probe input for the new log-mel (shared STFT, 4.1-s windows), fusion-on eval.
- [ ] **RQ3b after Pilot A, on A's encoders: reconstruction from frozen embeddings only** (your call 11-17; frozen-concat downstream dropped). Spectral encoder = **binaural 2-ch log-mel `spectral_only_logmel`** (23348511, `exp/binaural_sac_pretrain/...-spectral_only_logmel/models/audio_model.final.pth`; your call 11-17, it sees both ears and the reverberant binaural domain; caveat: its SAC term ran on the old broken manifests at lambda 0.02, so treat it as ~no-SAC). Optional control row: mono `amba` s1 on L. Spatial = frozen ipd16 SAC on SAR-SSL simulated binaural clips; small cross-attention decoder reconstructs log-mel L/R + IPD on one grid; per-component error from spectral / spatial / both.
- [ ] RQ4 after RQ2/3: SAIL-style two-tokenizer SSAMBA + shared masked reconstruction, 10-ep pilot on WSJ0 simulated set.
- [ ] SAR-SSL reference numbers for DRR/C50/ABS (results branch).

## Running / waiting
- **Account change (seen 11-17):** `gpu-general-pool` accepts only `public-users_v2`. Scripts with `--account=public-blaufer_v2` pend forever (PartitionConfig): fix each before reuse; `scontrol update job <id> Account=public-users_v2` for queued/held ones (k=10 23652538, seeds 23640933/35 on release).
- **Home quota:** downstream logs ~20 MB each in `logs/`; move finished ones to `/scratch200/yotams2/logs/sac-ssamba_home_logs/` regularly (quota hit 11-17 18:25).
- **k=5 downstream:** SC2 97.60 (23666191); IEMOCAP 23673595 / SID 23673596 resubmitted after the quota crash.
- **`amba` seeds downstream** (thesis must-have): s2 SID 23638875 pending (resumes from 29k ckpt), s2 SC2 rerun 23640910 running since 11-17 06:38. s3 IEMOCAP **23665001** (resubmitted 11-17 08:30, resumes from 18k; 23652540 failed on the old wrong `IEMOCAP_ROOT` default in `run_er.sh:40`, now fixed to `data/mono/data/IEMOCAP_audio_only/IEMOCAP_full_release`).
- **Mono probe fp16 bug** (diag 23663212): fp16 autocast makes trained-model embeddings 90-100% NaN -> exactly chance; fp32 is fine and above random for k=5 ep2/ep4, eg18, `amba` s1. Fixed 11-17: `_sac_batch` runs fp32 (covers the in-loop probe). Rerun 23665002 done: k=5 ep5 linear SC2/SID/EER/UA 33.7/36.4/3.0/58.0 vs eg18 final 38.2/30.5/3.3/58.4, `amba` s1 34.5/30.1/3.3/58.4, random 6.7/24.1/13.3/49.4 (`exp/_probes/mono_probe_2026-11-17.jsonl`).
- **Mono SAC perplexity** k=5 23647242 running (epoch 5/10 saved 11-17; 2-day limit hit ~22:00 11-17, then requeue) + gate 23647243. k=10 23652538 + gate 23652539 **held 11-16 for the sprint** (`scontrol release` to resume).
- **Mono SAC seeds 2/3** (median rule) 23640933/35 held, gates 23640934/36: cancel at the k=5 read if perplexity wins.
- Backup + prune daily (23653353).

## Results
Spatial SAR-SSL tasks, test MAE, R8, 20 ep, single seed (finetune / frozen):
| Encoder | TDOA | T60 | DRR | C50 | ABS |
|---|---|---|---|---|---|
| ipd16 no SAC final | 0.26 / 2.43 | 0.050 / 0.051 | 1.81 / 1.56 | 0.59 / 0.42 | 0.052 / 0.049 |
| ipd16 SAC final | **0.23 / 1.31** | 0.049 / 0.047 | 1.58 / 1.57 | 0.63 / 0.39 | 0.050 / 0.051 |
| random init 256 frames | 1.04 / - | 0.133 / - | 2.17 / - | 1.08 / - | 0.063 / - |
| GRAM-T `gramt_bf` | 2.65 / 2.94 | 0.095 / 0.070 | 1.49 / 1.40 | 0.41 / 0.40 | 0.063 / 0.067 |
| SAR-SSL ref | 0.29-0.48 / 1.1-2.1 | 0.107 / - | ? | ? | ? |
Original raw Re/Im input: TDOA 2.39 finetune, 2.92 frozen, 2.86 random (mean predictor 3.07).

Mono, test at best dev, single seed (SC2 / IEMOCAP fold1 / SID):
| Run | SC2 | IEMOCAP | SID |
|---|---|---|---|
| SAC old features | 97.58 | 60.18 | 71.40 |
| SAC egemaps18 | 97.50 | 56.7 | 67.5 |
| `amba` s1 | 97.45 | 58.71 | 67.9 |
| `amba` s2 | 97.50 | 55.9 | pending |
| `amba` s3 | 97.61 | 58.8 | 67.7 |
| SAC k=5 perplexity | 97.60 | running | running |
| `amba` RunAI (old baseline) | 97.34 | 57.24 | 60.50 |
| GRAM-T binaural-frame frozen | 89.59 | - | 76.88 |

RQ2 frozen probes on binaural speech sets (linear acc SC2 / SID / IEMOCAP UA; SID EER): ipd16 SAC 19.4 / 13.0 / 51.3 (31.8), ipd16 no SAC 19.1 / 15.8 / 49.5 (31.6), random 3.1 / 9.1 / 46.8 (32.3), GRAM-T 71.2 / 62.4 / 62.9 (13.9). Chance 2.9 / 0.5 / 25.

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
