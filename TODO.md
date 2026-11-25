# TODO

Current state only. History goes in [docs/log/](docs/log/) (`2026-11.md`; full pre-compaction TODO archived there 11-16 12:30).
**Current sprint: [SPRINT.md](SPRINT.md) (11-16 → 11-30): one model good at spectral + spatial tasks.** One-screen summary: `~/STATUS.md`.

## Sprint work (see SPRINT.md for the why; history in docs/log/2026-11.md)
- [ ] After the sprint: C3' (logL 1, ILD 0, cos/sin IPD 2). C3 dropped as specified (check 3); flag (dry tar `simu_paired_dry.tar` ready, md5-checked), C1 scene bottleneck, C2 + small head if C2's SC2/SID drop. SAC seeds after the sprint.
- [ ] SAR-SSL reference numbers for DRR/C50/ABS (results branch).
- [ ] Commit new SAC-SAR-SSL code (swap_pretrain.py C2, swap_valcheck.py, make_random_spec.py, room_share.py, build_registry.py, sync hook, eval scripts) when git is reachable.
- **Results registry:** `SAC-SAR-SSL-results/results/RUNS.md` (rebuilt on every sync); add a `results/labels.tsv` line per new run/label.

## Running / waiting
- [ ] **Post-sprint prep (decisions 11-23; no training until base C/C2 is picked):** data v2 done (paired v2 + unpaired 8.224 s, per-scene gpuRIR seed, RIRs saved; crops verified identical across renderings); V3 WPE shards done (simu_unpaired8_wpe/, 25 shards). Trainer has --cka-lambda (V2) + --lr-schedule cosine. V1 targets: v1_teacher_amba_s1.npy; **global per-dim z-score (training-set stats, MicSig/v1_teacher_amba_s1.stats.npz, done) in the loader before the cosine loss; not instance norm (keeps speaker info)**. **Released 11-23:** C 10 ep v2 23743825, swap-only 23745185, C 30 ep 23747481 (gates 23743826 / 23745186 / 23743828). To build: V2 / V3 / V1 launch configs (V3 loader: halves + WPE + long-T60 RIR pool + EQ; V1: RIR encoder on simu_paired_pretrain_v2.tar.rirs.npz). Space candidates: paired v1 tar 13.2 GB, simu_pretrain.tar (re-stageable).
- **Slurm account:** `public-users_v2` everywhere since 11-17 (gpu-general-pool rejects `public-blaufer_v2`); moshe (ib_dj) CPU jobs use `public-blaufer_v2` on power-general-shared-pool.
- [ ] **V1b SID flag (11-24):** teacher SID 40.3 (linear, binaural mid) < C 43.6; soft weights flat (ESS 61/127), same-speaker 4.4x chance per item, exact pair never > 0.8. Optional: GPU ridge dump of the teacher.
- [x] **11-25 quota incident fixed:** 259/400 GB (freed resume states 14.5, epoch snapshots 23.0, mono corpora 92 GB: backed up in moshe data/, no reader). Guard scripts/space_guard.sh (< 60 GB free refuses) in launch_v1/v2/csac; new runs: empty --snapshot-epochs (latest + best + final). prune.py --min-age-h. Open: C30 epoch10/20 snapshots (1.5 GB) keep or delete?
- [ ] **V1 status (11-25):** v1bA_l1 resumed from step 3000 (verified) 23773446, gate 23773447. Lost evals rerunning: 23773452/53 (A' l0.1 spat + spec leakage), 23773454-56 (V1a DRR + spec leakage, B l1 spec DRR); concat cells not rerun (appendix). Check 4a: swap_shufdiag skips lat_pred.* (logged); V1a done; B l1/l01, A' l01 already OK; A' l1 via gate.
- [ ] **V1 results (vs C v2 10 ep 2.38/0.067/1.40/19.4/43.6):** V1b B l1 1.37/0.065/1.43/20.5/42.9 (spec T60 leak 0.134 vs 0.064); B l0.1 1.73/0.069/1.34/21.3/46.6; A' l0.1 SC2 22.3/SID 46.1, ridge spat TDOA 1.43; V1a l1 TDOA 1.75, T60 0.164 (regression), SC2 26.0/SID 48.7, spat leak 7.1/1.8. RUNS.md decision 11-25: hybrid candidate V1a spectral + V1b spatial loss.
- [ ] **C+SAC lambda control (Yotam 11-25, most important for V1b):** scripts/launch_csac.sh ready (pilotc_csac_l1 / _l01; hand-crafted spatial cues, spatial encoder only, sigma/temp 0.613/0.15 default via SAC_SIGMA/SAC_TEMP). Waiting: Yotam's GPU priority vs V2/V3 + sigma/tau source.
- [ ] **V2 (CKA 0.1 / 1):** scripts/launch_v2.sh written + dry run OK (11-24); Yotam launches 25 Nov (not me). **V3:** Yotam launches 25 Nov. Loader built 11-24 (v3_data.py; `--v3` in _swap_pretrain_v1.py; h1 = first half, WPE(h1) * paired-v2 training RIR + EQ 3-5 peaks ±6 dB + tilt ±3 dB/oct; spatial = h2; target = h1). Sanity PASSED 11-24 (exp/v3/sanity.json). Launch: --v3 --tar simu_unpaired8_pretrain.tar --val-scenes 2500.
- `swap_pretrain.py` must stay backward compatible while C2 / seed 2 run (12-h pieces re-read it).

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
| `amba` s2 | 97.50 | 55.9 | 65.3 |
| `amba` s3 | 97.61 | 58.8 | 67.7 |
| SAC k=5 perplexity | 97.60 | 58.7 | 67.6 |
| `amba` RunAI (old baseline) | 97.34 | 57.24 | 60.50 |
| GRAM-T binaural-frame frozen | 89.59 | - | 76.88 |

RQ2 frozen probes on binaural speech sets (linear acc SC2 / SID / IEMOCAP UA; SID EER): ipd16 SAC 19.4 / 13.0 / 51.3 (31.8), ipd16 no SAC 19.1 / 15.8 / 49.5 (31.6), random 3.1 / 9.1 / 46.8 (32.3), GRAM-T 71.2 / 62.4 / 62.9 (13.9). Chance 2.9 / 0.5 / 25.

Mono `amba` on binaural audio (23673898, linear SC2 / SID / IEMOCAP UA, s1-3 mean): mono sets 33.6 / 28.7 / 57.6; binaural (L+R)/2 25.2 / 38.6 / 55.3; binaural L (s1) 24.4 / 38.9 / 52.4; binaural 2-ch log-mel model 19.7 / 32.4 / 53.2; random on (L+R)/2 4.8 / 35.0 / 48.1. SID sets differ (binaural random 35% vs mono ~24%), so compare SID only within a set. SC2 drops ~8 pts from the audio alone; mono `amba` beats the binaural model on its own binaural audio.

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

## Facts and caveats
- **Cluster:** preempted jobs requeue; s3prl resumes (`-a`), SC2 resumes via `resume.pth`. Exclude `compute-0-[18,53,58,300,376,419,420]`. L40S ~1.8x A6000. Use `sacct -D`. Priority = fairshare only. Replace launch scripts instead of editing in place while a job runs them.
- **Storage:** scratch ~41 GB free (11-23); never unpack `MicSig/simu_pretrain.tar` (read through `simu_pretrain.tar.index.json`). Shared FS near 100% inodes. Daily backup + `scripts/backup/prune.py`; pin in `keep.txt`.
- **Spatial pretraining:** `SAC-SAR-SSL/code/ssamba_spatial_pretrain.py`, `scripts/ssamba_spatial_pretrain.sbatch`; wrapper `code/ssamba_spatial.py` (`--ssamba-nframes` for random control); launcher `scripts/ssamba_spatial_downstream.sbatch <TASK> <ckpt> <label> [nsimroom] [mode]`. Encoders in `SAC-SAR-SSL/exp/ssamba_pretrain/ipd16_{nosac,sac}/encoder_final.pth`.
- **Mono pipeline:** SAC features from the first 11.24 s; normalisation stats stored in the checkpoint (new features need a new `EXP_NAME_SUFFIX`). Old IEMOCAP dirs named `_1e-5` ran at 1e-4.
- **GRAM-T:** venv `/scratch200/yotams2/venvs/gramt`; binaural-frame input = 2-ch log-mel at 32 kHz (n_fft 1024, hop 320, 128 mel), no phase.
- **Venv:** `/scratch200/yotams2/venvs/sac` (parselmouth 0.4.7; no wandb).
