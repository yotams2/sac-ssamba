# TODO

Current state only. **History** (job-by-job narrative, diagnostics, dead ends, superseded decisions) goes in
[docs/log/](docs/log/), one file per month (`2026-11.md`), as dated entries; the full pre-split TODO is archived there.
Topic write-ups live in [docs/](docs/) (e.g. [feature_selection.md](docs/feature_selection.md)) and are linked, not
repeated. The one-screen summary is `~/STATUS.md`. Rule: replace stale text here; move anything worth keeping
(job ids, numbers, why something was ruled out) to the log.

## Running / waiting

- **Mono pretraining, egemaps18 features: job 23502629** (pending). Same recipe as the old-feature run except features:
  `sbatch --exclude='compute-0-[18,53,58,300,376,419,420]' --export=ALL,RECON_LAMBDA=10.0,CLASSIF_LAMBDA=1.0,EXP_NAME_SUFFIX=-baseline_objective_egemaps18 scripts/mono_sac.sbatch`
  (from the repo root). ~2 days on an A100 + one resubmit of the same command after the 2-day limit (resumes from
  `models/latest.pth`). At start check: peak GPU memory (per-frame LPC, ~1 GB transient), s/step vs 8.2 on A100,
  group medians in the log.
  - [ ] Set up the downstream gate for it (copy `scripts/mono_downstream_gate.sbatch`, point it at the new exp dir).
- **Old-feature mono run, downstream** (pretraining 23410602 + 23416521, `-baseline_objective_pitchfix`, final val 4.22):
  - SC2 23474187: running; preempted once and **restarted from epoch 1** (no resume in `run_sc_amba.sh`).
  - IEMOCAP 23474188, SID 23474189: preempted, resume from checkpoints; `nice=100` so 23502629 starts first.
  - Compare with `amba` lr 4e-4 under the 1.5-pt rule (below). IEMOCAP so far: best dev 68.8 at step 1000 (first
    eval), test 60.2 vs 57.2; later devs ≤ 65.3.

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

Mono (critical path)
- [ ] Add resume to `run_sc_amba.sh` (a preemption restarts SC2 from epoch 1, ~3 h lost each time).
- [ ] `--seed` in `run_amba.py` and `run_binaural_amba.py` (summaries say `seed: null`); s3prl defaults to 1337, pass
  a different seed per run. Needed before any 3-seed runs.
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
- [ ] Spatial probe (nothing measures the spatial encoder); CCSR on log-mel needs a design (CCSR's target is the complex STFT).
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
| Mono SAC, old features, same objective as `amba` | running | 60.2 (partial) | pending | 23474187-9 |
| Mono SAC, egemaps18 | | | | 23502629 pending |
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
