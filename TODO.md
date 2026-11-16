# TODO

Current state only. **History** (job-by-job narrative, diagnostics, dead ends, superseded decisions) goes in
[docs/log/](docs/log/), one file per month (`2026-11.md`), as dated entries; the full pre-split TODO is archived there.
Topic write-ups live in [docs/](docs/) (e.g. [feature_selection.md](docs/feature_selection.md)) and are linked, not
repeated. The one-screen summary is `~/STATUS.md`. Rule: replace stale text here; move anything worth keeping
(job ids, numbers, why something was ruled out) to the log.

## Running / waiting

- [ ] **Mono probe (11-16):** first run 23654005 (`scripts/mono_probe.sbatch`, out `exp/_probes/mono_probe_2026-11-16.jsonl`) on k=5
  audio_model.2 / latest, eg18 median final + seed2/3 latest, `amba` s1 final, random. In-loop probe (PROBE_EVERY=2000) is
  live for pretrainings that start from now (k=10 23652538, held SAC seeds); untested in-loop until one starts: check the first
  `---- Step 0 probe` line in its log. Snapshots of finished runs are pruned, so the in-loop log is the only training curve.

- [x] **Scratch quota full (11-16 ~01:00), handled 09:40 (your OK).** Deleted 69 SAR-SSL `model<n>.tar` (24 GB, 374/400 after).
  SAR-SSL `run_downstream.py` now prunes per-epoch weights outside the ensemble window and all of them after ensembling;
  `run_pretrain_sac.py` fsyncs `latest.pth` and moves an unreadable one aside (uncommitted, both). Resubmitted: spatial
  23652530/31/32 (labels pre_nosac_final / pre_sac_final / ctrl_ipd16_n256, DRR C50 ABS; DRR finetune done: 1.81 / 1.58 / 2.17),
  perplexity k=10 23652538 + gate 23652539 (from step 0), `amba` s3 IEMOCAP 23652540 (from 18k). `amba` s3 SID final
  from the dead run: best dev 68.5 @35k, test 67.7. Then (your call): deleted `LibriSpeech/train-clean-100.tar.gz` and `IEMOCAP/simu` (v1, unused). prune.py now also
  deletes dev-best / SC2 best / SAR-SSL best+ensemble of finished runs, and spatial pretraining snapshots once
  `encoder_final.pth` (now backed up) is on moshe. Dry run 171 files / 67 GB (`~/exp-records/prune_dry_2026-11-16.out`); first
  real run with backup 23641358 (10:28). Check its log + prune_log.tsv. Open: size of `sac-ssamba/data`.

- **Mono SAC, egemaps18 features: pretraining 23502629 finished 2026-11-12 01:55** (all 20710 steps; final val loss
  4.2523; `audio_model.final.pth` + `best_audio_model.pth` saved). Slurm says FAILED and the results record says
  `failed` only because `run_sac.sh` was edited in place (the `--seed` line, 11-11 14:37) while the job was running, and
  bash hit a syntax error after Python exited. The weights are fine. Gate 23504394 then submitted downstream (all load
  `best_audio_model.pth`; job names are `amba_*`, the generic downstream names): SC2 **23591975 done: 97.50**
  (best dev ep 24; finished 13:40, but Slurm requeued it at preemption, so it is pending again; a rerun only re-evaluates.
  Cancel?), IEMOCAP **23591976** done 11-14 (best dev 67.3 @4k, test 56.7), SID **23591977 done 11-14 22:16** (best dev 69.6 @50k, test 67.5). Old features win 2 of 3 (60.18 / 71.40 vs 56.7 / 67.5; SC2 tie). **Seeds 2/3 with egemaps18 (your call 11-15)**: pretraining 23640933 / 23640935 (exports `RECON_LAMBDA=10.0,CLASSIF_LAMBDA=1.0,SAC_LAMBDA=0.02,SEED=n,FEATURE_ALIAS=egemaps18,EXP_NAME_SUFFIX=-sac_eg18_seedn`; dirs `sac_pretrain/...-lam0.02-...-feat_egemaps18-...-sac_eg18_seedn`; args differ from `amba` seeds only in sac_lambda), gates 23640934 / 23640936 (2 resubmits). Unseeded 23502629 = SAC seed 1. **Held 11-15 19:30** until the sigma runs report (`scontrol release`). The sigma-multiplier change is a no-op for them (sigma 1.0).
- **Old-feature mono run, downstream** (pretraining 23410602 + 23416521, `-baseline_objective_pitchfix`):
  SC2 23474187 done (97.58), IEMOCAP 23474188 done (60.18, best dev at step 1000, the first eval), SID 23474189 done 2026-11-11 13:58
  (test at best dev 71.40, best dev 72.39 at step 40000; final-step test 70.60). Passes the 1.5-pt rule vs `amba`
  (IEMOCAP +2.94, SID +10.90) -> 3 seeds.
- **`amba` seeds (2026-11-11, your call):** design 3x1 (3 pretraining seeds x 1 fine-tune each, fine-tune seed fixed).
  Pretraining 23586532/34/36 (seeds 1/2/3), gates 23586533/35/37 (2 resubmits). Exports
  `RECON_LAMBDA=10.0,CLASSIF_LAMBDA=1.0,SAC_LAMBDA=0,SEED=n,FEATURE_ALIAS=none,EXP_NAME_SUFFIX=-amba_seedn`; dirs
  `sac_pretrain/sac-base-...-lam0-...-feat_none-...-amba_seedn`. New optional `--seed` in `run_pretrain_sac.py` / `SEED` in
  `run_sac.sh` (unset = old unseeded path, so 23502629 is unaffected); **uncommitted**, don't edit these files until
  the 3 jobs have started. All 3 pretrainings done (s1 11-14 14:00, s3 18:54, s2 11-15 00:09). Downstream (test at best dev): s1 SC2 23634632 **97.45**, IEMOCAP 23634633 **58.71**, SID 23634634 **done 11-15 18:06: best dev 68.0 @40k, test 67.9** (above SAC egemaps18 67.5 and RunAI `amba` 60.50). s3 SC2 23636475 **97.61**, IEMOCAP 23636476 pending, SID 23636477 running (30k: best dev 67.4, test 67.9). s2 SC2 23638873 FAILED (truncated resume.pth, see below) -> rerun **23640910**; IEMOCAP 23638874 at 29k/30k (best dev 67.0 @4k, test 55.9), SID 23638875 pending. Seeds now done in `src/sac/` -> those files can be edited again. SAC seeds: wait for egemaps18 downstream, then 2 more SAC seeds with the better features.

- [x] **Scratch space (2026-11-11, approved):** one-time job 23587025 deleted the 56 listed scratch files (35 GB) and the 2
  abandoned latest.pth on moshe (3.5 GB, now in skip.txt). List `~/exp-records/prune_onetime_2026-11-11.tsv`. Quota 258/400.
- [ ] **Auto prune (uncommitted):** `scripts/backup/prune.py`, run by `backup.sbatch` after a clean backup. Deletes only
  weight files of finished runs (by each run's own end marker, >24 h old, keep.txt wins): pretrain snapshots + resume
  state once the final is on moshe; s3prl states-*.ckpt once states-<total_steps> exists; SC2 optimizer/resume once
  eval_result.csv exists; SAR-SSL model<n>.tar + latest once trained and tested; smoke weights >7 days; moshe resume
  states of finished pretrainings. Keeps final/best/dev-best/ensemble. Moshe dry run 23587027: 2 tiny files. First real run with
  the next backup (23583637, 11-12 10:20): check `~/exp-records/backup_logs/` and `prune_log.tsv`.

- [ ] Queue (2026-11-11 19:35): ~1400 jobs ahead of ours on fairshare. Sweep 23586231-33 limit cut to 6 h for backfill (resubmit if one hits 6 h). Your call: shorter limits for the `amba` pretrainings (more gate resubmits).
- [ ] Node exclusions relaxed 2026-11-11: sweep `ExcNodeList=compute-0-18`, amba seeds + `mono_downstream_gate.sbatch` resubmit
  line `compute-0-[18,53,376]` (uncommitted). Queued gates 23504394, 23586533/35/37 keep the old list: if one resubmits a
  pretraining, `scontrol update JobId=<new> ExcNodeList=compute-0-[18,53,376]`.

- [ ] **Supervisor meeting report**: Notion (private) https://app.notion.com/p/41578b64ef458168a724cf6b59af1f7f. Updated 11-16: §2 spatial (lr sweep, input probe, ipd16 pretraining, TDOA/T60/DRR table, input figure with patch rectangles + TDOA chart; scripts `~/SAC-SAR-SSL/code/figures/ssamba_{input,tdoa}_fig.py`, uncommitted) and mono status. Add DRR/C50/ABS + SAR-SSL refs when 23652530-32 finish.

## Decisions in force

- **Focus (2026-11-07):** mono first; binaural on hold until mono reports.
- [ ] **SAC weight (approved 2026-11-15; step 1 = job 23646477 (eg18 done: H(P) 4.00 vs ln63 4.14, loss 4.02, 86% of gap closed -> targets near uniform; lambda sweep on hold), rerun 23646555 done: eg18 learnable gap 0.02 at sigma x1 vs 0.41/1.24/2.05 at x0.5/0.3/0.2 (eff. neighbours 29/11/5); old-feature check unusable (today's extractor != features that encoder trained on). **Perplexity rule (your call 20:00):** multiplier runs cancelled unstarted. `local_sigma_mode=perplexity` + `--sac-perplexity` (sac_model.py `perplexity_targets`, run_pretrain_sac.py, run_sac.sh `SAC_PERPLEXITY`). Check 23647067 passed. **Running: eg18 k=10 23647240 (gate 23647241), k=5 23647242 (gate 23647243)**, lambda 0.02, SEED=1, dirs `...-sig1.0-feat_egemaps18-mode_perplexity-librispeech-sac_eg18_perp<k>`. `--sac-sigma` is also a multiplier in offline_global_median (1.0 = old rule). SAC seeds 2/3 (23640933/35) **held** pending this; uncommitted: sac_model.py, run_sac.sh, sac_floor_check.*, `scripts/sac_floor_check.sbatch` + `src/sac/sac_floor_check.py`, uncommitted):** at lambda 0.02 SAC is ~3% of total loss, flat after epoch 2 (eg18 4.04 -> 4.02; old features 3.96). Step 1: H(P) per group at batch 64 (CPU, the floor). Step 2: if well below 4.0, egemaps18 pretrain at lambda 0.1 and 0.5, 1 seed each, 3 tasks.
- **Mono baseline (2026-11-15):** compare against our own 3 `amba` seeds run here, not the RunAI `amba` numbers (our s1 SID 65.7 @30k vs RunAI 60.50).
- **Mono SAC features (2026-11-15):** egemaps18 for the seeds and the thesis, even though the old features scored higher on IEMOCAP/SID.
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

- [ ] Commit (git missing on node compute-0-373, 11-15): `src/traintest.py` resume fix (fsync before rename; unreadable resume.pth -> `.corrupt`, restart from epoch 1), plus the earlier uncommitted items listed below.

Storage
- [ ] Backup flow `scripts/backup/` (2026-11-11, `c73060e`): daily self-requeued `backup.sbatch` (first 23583088 done 11-11: 8 weight files 6.9 GB + 303 records; next 23583637). Policy in `backup.py` docstring; pin/skip in `keep.txt`/`skip.txt`; manifest `~/exp-records/manifest.tsv`. Covers top-level `sac-ssamba/data/*.json` manifests (features computed here); audio dirs come from /moshe-data. Unverified: that IEMOCAP/simu_v2 (RunAI regen) is on /moshe-data.

Mono (critical path)
- [x] Resume + seed code committed 2026-11-11 (`1086cd6`): `run_amba.py --auto_resume True` (models/resume.pth per
  epoch, exact; SC2 test 23504361 passed), `--seed` in `run_amba.py`/`run_binaural_amba.py`, `SEED=n` in `run_sc_amba.sh`.
  Pretraining seed added 2026-11-11 (`--seed`, uncommitted).
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
  (front-end = wav2stft to 3e-7; finetune 0.55 s/batch8, 11.4 GB; frozen 0.17 s). Committed SAC-SAR-SSL `4facbc5`. Smoke 23582593 TDOA R8 1 lr 20 ep: MAE 2.39
  (SAR-SSL finetune 0.29-0.48, scratchLOW 0.50, lineareval 1.1-2.1; mean predictor 3.07). Controls: 23583698 lineareval 2.92 (~mean predictor); 23583697 scratchLOW (random SSAMBA) 2.86 vs SAR-SSL scratchLOW 0.50: architecture/input is the bottleneck. TDOA lr sweep (new `--ds-lr`/`--ds-nepoch` in opt.py, uncommitted), label `ssamba_*_lrsweep`, test MAE at best-val lr: 23586231 finetune **2.93** (lr 1e-3 picked over 1e-5), 23586232 lineareval **2.89** (1e-3 over 1e-2); 23586233 scratchLOW **2.93** (done 11-11 23:25). All ~mean predictor; the earlier 2.39 finetune (default lr) remains the best. Decision: nothing well below ~2.4 -> 5-task grid skipped. **Input probe 23594658 done 11-12 17:42** (SAC-SAR-SSL `0958ee1`, `scripts/ssamba_input_probe.sbatch`, labels `inprobe_*`, results `exp/{TDOA,T60}/inprobe_*/downstream_test_res_scratchLOW.json`): random init, R8, 20 ep. Test MAE TDOA / T60: sarssl_ctrl 0.64 / 0.107, **ipd16 0.66 / 0.133**, ipdframe 0.65 / 0.185, reimframe_stem 0.99 / 0.175, reimframe 2.31 / 0.210, reim16 2.93 / 0.212, wave16_short 2.92 / 0.199. **Winner: ipd16.** 
- [ ] **Spatial SSAMBA pretraining (your call 11-12: SAR-SSL WSJ0 simulated data, Interspeech SAC features, TOP priority).**
  Single encoder (decided; dual only if it later fails binaural speech tasks). In SAC-SAR-SSL, uncommitted:
  `code/ssamba_spatial_pretrain.py` (ipd16 via the wrapper's own `stft_frontend`, per-channel z-norm stats saved in the
  checkpoint `meta`; loss 10*mpg + 1*mpc (mask 150 of 256) + sac_lambda*SAC; SAC = SAC-SAR-SSL's own
  `spatial_features_from_stft` + `SARSSL._calc_single_feat_sim_loss`, 5 features, tau 0.15, sigma 0.613 (the
  09151152 "production baseline"), z-norm from 50 batches; b128, AdamW 4e-4, ReduceLROnPlateau, 30 epochs = 11.7k
  steps; resume from latest.pth every 200 steps), `scripts/ssamba_spatial_pretrain.sbatch <label> [args]` (12-h
  pieces, each queues the next with afterany). Wrapper `ssamba_spatial.py` now loads these checkpoints (meta: pad to
  256 frames, apply stats); new `--ssamba-nframes` for a matching random-init control. Data: `stage_simu_pretrain.sbatch`
  (23613636) packed moshe `MicSig/simu/{pretrain,preval,pretest}` wav+info.npz (no spec_feats.pt) into ONE tar
  `/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_pretrain.tar` (15 GB, 116,002 files = source counts), read in place
  through a cached offset index (scratch has ~310k free inodes; never unpack it there). Smoke 23613762 (SAC, resume,
  no-SAC, wrapper load); smoke failed (plain python3, no venv; fixed to $PYTHON). 11-13: arms 23623085/86 crashed in setup (my code set requires_grad on all encoder params, incl. the integer size constants p_input_fdim/tdim; fixed: the full encoder trains, the constants are left alone); CPU checks pass; resubmitted as 23624668 `ipd16_nosac`, 23624669 `ipd16_sac --sac-lambda 0.02` (running since 11-13 13:51). **Epoch-10 reads done** (test MAE, finetune/frozen): TDOA no-SAC 0.33/2.51, SAC 0.32/1.43, random-init 256-frame ctrl 1.04; T60 no-SAC 0.073/0.068, SAC 0.064/0.065, ctrl 0.133. `ipd16_nosac` finished 30 ep 11-14 10:14; `ipd16_sac` ends ~11-14 17:00. **Final reads done 11-14 19:32** (test MAE finetune/frozen): TDOA no-SAC 0.26/2.43, SAC **0.23/1.31**; T60 no-SAC 0.050/0.051, SAC 0.049/0.047. Both beat SAR-SSL finetune 0.29-0.48; SAC's clear effect = frozen TDOA. **Your call 11-15:** read DRR, C50, ABS first, then seeds. Submitted 23640930 pre_nosac_final, 23640931 pre_sac_final (finetune+lineareval), 23640932 ctrl_ipd16_n256 (scratchLOW), `EVAL_TASKS="DRR C50 ABS"`. Need SAR-SSL references for these 3 tasks (results branch). Then seeds 2-3 of both arms. Tar index cached (`simu_pretrain.tar.index.json`; building it took ~30 min) (mono SSAMBA's SAC weight; your
  Interspeech runs used 1.0 relative to CCSR/no recon); the 6 other pending GPU jobs niced +1 (11-13; was +1000). A Python crash cancels the next chained piece. Eval: TDOA+T60 R8,
  finetune + lineareval, at epoch-2/5/10/20 snapshots and final, vs random-init ipd16 with `--ssamba-nframes 256`.
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
| Mono SAC, old features, same objective as `amba` | 97.58 | 60.18 | 71.40 | 23474187-9 |
| Mono SAC, egemaps18 | 97.50 | 56.7 | 67.5 | 23591975-77 |
| `amba` seed 1 (ours, same pipeline) | 97.45 | 58.71 | 67.9 | 23634632-34 |
| `amba` seed 3 (ours) | 97.61 | | | 23636475-77 |
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
