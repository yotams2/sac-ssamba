# TODO

Current state only. **History** (job-by-job narrative, diagnostics, dead ends, superseded decisions) goes in
[docs/log/](docs/log/), one file per month (`2026-11.md`), as dated entries; the full pre-split TODO is archived there.
Topic write-ups live in [docs/](docs/) (e.g. [feature_selection.md](docs/feature_selection.md)) and are linked, not
repeated. The one-screen summary is `~/STATUS.md`. Rule: replace stale text here; move anything worth keeping
(job ids, numbers, why something was ruled out) to the log.

## Running / waiting

- **Mono pretraining, egemaps18 features: job 23502629.** Started 2026-11-09 16:53 on compute-0-102 (L40S), preempted
  4x; running again since 2026-11-10 23:12 on compute-0-223 (L40S). At 2026-11-11 07:55: ~step 11100/20710 (epoch 6,
  batch 745), ~6.7 s/step, no NaN/OOM; ~18 h left if not preempted. Resume fix confirmed in log (see below).
  Step 4000 val loss 5.0244. Nominal end 20710 steps (10 x 2071); ~30 h of GPU time left plus resume overhead.
  Command if it ever needs resubmitting:
  `sbatch --exclude='compute-0-[18,53,58,300,376,419,420]' --export=ALL,RECON_LAMBDA=10.0,CLASSIF_LAMBDA=1.0,EXP_NAME_SUFFIX=-baseline_objective_egemaps18 scripts/mono_sac.sbatch`
  - **Resume bug fixed 2026-11-10 (22430a6):** the run is now exactly n_epochs x len(loader) = 20710 steps; resumes
    continue mid-epoch with a stored per-epoch shuffle seed. Tested on CPU (`scripts/test_pretrain_resume.sbatch`,
    job 23574751, bit-identical to an uninterrupted run). Confirmed 2026-11-10 23:12: log shows "Training position:
    step 6400/20710 = epoch 4, batch 187/2071" and the "older format" note (the rest of epoch 4 gets a fresh permutation). The
    ~3300 steps already repeated stay in. The old-feature run got ~1770 extra steps (22480 total), so it is not exactly
    matched to `amba`.
  - Downstream gate **23504394** (afterany 23502629): if pretraining hit the 2-day limit it resubmits it with the same
    exports and chains itself (2 resubmits allowed); once `audio_model.final.pth` exists it submits SC2, IEMOCAP, SID.
- **Old-feature mono run, downstream** (pretraining 23410602 + 23416521, `-baseline_objective_pitchfix`):
  SC2 23474187 done (97.58), IEMOCAP 23474188 done (60.18, best dev at step 1000, the first eval), SID 23474189 running since
  2026-11-10 23:21 on compute-0-223, step ~41160/50000 at 07:55 (~4.5 h left). Interim: latest dev 72.39, last two
  test evals 70.55 / 70.37 (`amba` 60.50; check it is the same metric/split before trusting the gap). Compare with `amba` under the 1.5-pt rule once SID is in.

## Decisions in force

- **Focus (2026-11-07):** mono first; binaural on hold until mono reports.
- **1.5-pt rule (2026-11-07):** SAC within ~1.5 pts of no-SAC on all 3 tasks = no sign of a gain: stop investing, run
  only the thesis seeds. A clear gain on any task = 3 seeds to confirm.
- **Features (2026-11-09): eGeMAPS-style 18-value set** (`egemaps18`, default in `run_sac.sh`), chosen and justified
  in [docs/feature_selection.md](docs/feature_selection.md). Every earlier SAC run used features with level and padding
  confounds and near-random formants; don't present those features as the method.
- **Raw STFT dropped** as spectral-encoder input (2026-11-06); log-mel is the way forward.
- **No AMP for mono** (2026-11-07): NaNs from step 626, loss scale to 0. Mono stays fp32.
- **Keep the old-feature downstream runs** (user, 2026-11-09): they are the SAC-vs-no-SAC check with the old features.
- **Thesis scope (2026-11-06):** must-have = 3+ seeds for the binaural arms and mono SAC vs no-SAC (same objective,
  3 tasks, IEMOCAP 5 folds), then write. Binaural time-boxed. Future work: Sinkhorn, GRAM-T retrain, SARL, curriculum.
  Don't quote the old Notion mono gains (best-of-N, confounded objective, old F0/HNR).

## Open tasks

Storage
- [ ] Backup flow `scripts/backup/` (2026-11-11, uncommitted): daily self-requeued `backup.sbatch` (first 23583088 done 11-11: 8 weight files 6.9 GB + 303 records; next 23583637). Policy in `backup.py` docstring; pin/skip in `keep.txt`/`skip.txt`; manifest `~/exp-records/manifest.tsv`. Covers top-level `sac-ssamba/data/*.json` manifests (features computed here); audio dirs come from /moshe-data. Unverified: that IEMOCAP/simu_v2 (RunAI regen) is on /moshe-data.

Mono (critical path)
- [x] Resume + seed code committed 2026-11-11 (`1086cd6`): `run_amba.py --auto_resume True` (models/resume.pth per
  epoch, exact; SC2 test 23504361 passed), `--seed` in `run_amba.py`/`run_binaural_amba.py`, `SEED=n` in `run_sc_amba.sh`.
  Pretraining (`run_pretrain_sac.py`) still has no seed.
- [ ] ER/SID resume is not exact (s3prl `-a`): restarts from the last `states-*.ckpt` (every 1000 steps), loses up to
  1000 steps, begins a fresh epoch (new shuffle, RNG re-seeded to 1337), so a resumed run takes a different path than
  an uninterrupted one. Old-feature IEMOCAP/SID 23474188-9 already resumed this way. Your call whether to make it exact.
- [ ] IEMOCAP: all 5 folds for final numbers (`run_er.sh` and `run_gramt.sh` run fold1 only).
- [ ] Formal choice of tau and sigma (journal; user 2026-11-07). Proposed frame: per-group SAC = SNE objective
  H(P) + KL(P||Q); sigma by perplexity calibration (median rule as baseline); tau learnable per group. First cheap
  step: H(P) per group at batch 64 on CPU. Features are settled (feature_selection.md).

Binaural (on hold)
- [ ] `logmel` pair (`sac_spec_lambda` 0.02 vs 0): regenerate the binaural manifests with the egemaps18 list first
  (`scripts/regen_full15_features.sbatch`, uncommitted: change its FEATURES; old full15 manifests have the old features).
  `spectral_only.sbatch` hard-codes `EXP_NAME_SUFFIX` and the lambda.
- [ ] `run_binaural_sac.sh` never passes `--spectral_features`, so binaural runs used 3 groups (`f0,hnr,centroid,flux,zcr`)
  by accident; pass the full list.
- [ ] **Spatial encoder on SAR-SSL tasks (started 2026-11-11, in `~/SAC-SAR-SSL`, uncommitted):** SAR-SSL's own
  `code/run_downstream.py --upstream ssamba_spatial --ssamba-ckpt <pth>` (wrapper `code/ssamba_spatial.py`: spatial encoder
  only, pretraining front-end = peak 0.9, STFT 512/hop 160, clip tiled to 10 s; SAR-SSL head LayerNorm+Linear).
  Launcher `scripts/ssamba_spatial_downstream.sbatch <TASK> <ckpt> <label> [nsimroom=8] [mode]`, runs in the sac venv.
  Checkpoint: `binaural_sac_pretrain/...lossfix_blockmask_pitchfix_baseline/models/audio_model.final.pth` (old 3-group features).
  Data staged (simu_ds R1..R256+val+test in /scratch200/yotams2/SAC-SAR-SSL/data). Wrapper check 23582050 passed
  (front-end = wav2stft to 3e-7; finetune 0.55 s/batch8, 11.4 GB; frozen 0.17 s). Smoke 23582593 (TDOA R8 `--my-tmp`,
  exp_smoke; sbatch now `--workers 0`: forked workers deadlocked in 23582088). Next: 5 tasks x finetune at R8; compare with SAR-SSL rows on `results`.
  The older port `src/spatial_downstream/` (own driver, SAR-SSL STFT input, strict=False load) is superseded.
- [ ] CCSR on log-mel needs a design (CCSR's target is the complex STFT).
- [ ] Merge the binaural `num_workers=8` fix from local branch `profiling` (`12f979c`; not the `--amp` part).
- [ ] Frozen-probe debugging: `scripts/probe_diagnostic.sbatch <ckpt>` (pretrained vs random, padding vs tiling, input
  level). Frozen SAC SC2 is quoted as 13.5% but the only run here says 5.75%: find which is right.
- [ ] `AverageMeter` keeps NaN until the next validation: skip non-finite values.

GRAM-T and data
- [ ] GRAM-T frozen IEMOCAP: `sbatch src/finetune/iemocap/run_gramt.sh gramt_binaural 1e-4 true`.
- [ ] Fine-tune GRAM-T for a fine-tuned vs fine-tuned comparison (`run_gramt.sh <model> <lr> false`).
- [ ] Proposal, undecided: retrain GRAM-T on our 50k binaural clips (data scale vs architecture). Needs a data adapter
  and its own env (Python 3.9, torch 2.5.1); check 16 vs 32 kHz first.
- [ ] Binaural VoxCeleb1 and SC2 sets were sampled with replacement (~63% unique per split, no train/test overlap):
  regenerate with `sequential_src=True` (on RunAI) if standard test sets are needed. IEMOCAP is already fixed (v2).
- [ ] Port `run_binaural_sac_frozen.sh` wrappers to `env.sh`; copy the binaural SAC checkpoint from RunAI.
- [ ] Optional: GRAM-T variants `gramt_binaural_time`, `gramt_mono`; SAC with more data/steps.

## Results

Test accuracy at best dev, single seed. Mono rows: fine-tuned, LibriSpeech pretraining, b64, lr 4e-4.

| Model | SC2 | IEMOCAP fold1 | VoxCeleb1 | Notes |
|---|---|---|---|---|
| `amba` no SAC (RunAI) | 97.34 | 57.24 | 60.50 | objective mpc + 10 mpg |
| exp6 SAC (RunAI) | 97.46 | 56.13 | 61.29 | no mpc: confounded objective; old features |
| Mono SAC, old features, same objective as `amba` | 97.58 | 60.18 | pending | 23474187-9 |
| Mono SAC, egemaps18 | | | | 23502629 pretraining ~step 6400/20710, gate 23504394 |
| GRAM-T binaural-frame (frozen) | 89.59 | | 76.88 | 23244879, 23305339 |

All IEMOCAP numbers from before the binaural v2 regeneration were on mono IEMOCAP.

## Facts and caveats

- **Cluster:** GPU jobs on `gpu-general-pool` are preempted and requeued; s3prl runs resume (`-a`), SC2 doesn't.
  Exclude slow/broken cards: `compute-0-[18,53,58,300,376,419,420]` (18 = Blackwell, no kernels). Mono s/step:
  A100 8.2, V100 12.1; L40S ~1.8x A6000. `sacct -X` shows only the last requeue segment: use `sacct -D`.
  `scontrol top` is not allowed; use `nice` to reorder own jobs. Shared filesystem at ~96% of inodes.
- **Mono pipeline:** SAC features come from the first 11.24 s of each clip (zero-padded), deterministic. Feature
  normalisation stats and group medians are computed at the start of a fresh run and stored in the checkpoint: a run
  resumed across a feature change keeps the old stats, so a new feature set needs a new `EXP_NAME_SUFFIX`.
- **Mono baselines:** `amba-*` numbers stay from RunAI (not rerun here). Old IEMOCAP dirs named `_1e-5` really ran
  at 1e-4. s3prl path uses `rms_norm=True` vs False in pretraining, as in the baselines; left for comparability.
- **Binaural data:** `binaural_{train,val}.pitchfix.json` have fixed F0/HNR; the plain ones hold the old values.
  `full15` manifests have the old (buggy) formants. Binaural IEMOCAP v2 (`IEMOCAP/simu_v2`) covers all 5531 utterances.
- **Venv:** `/scratch200/yotams2/venvs/sac`; parselmouth 0.4.7 installed (Praat reference for feature checks);
  wandb not installed (`--use_wandb` is skipped silently).
- **Feature checks:** `scripts/check_formant_features.sbatch` (vs Praat), `scripts/check_formant_stats.sbatch`,
  `scripts/feature_analysis.sbatch` (full table for feature_selection.md, ~10 min CPU).
