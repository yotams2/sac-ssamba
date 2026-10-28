# TODO

Working list for the binaural SAC-SSAMBA evaluation on this cluster (TAU power cluster).
Cluster paths and the shared venv are set in [env.sh](env.sh); heavy files go on `/scratch200/yotams2`
(the home quota is ~2 GB). Run every step as a Slurm job, because the Jupyter node has 1 GB of RAM and
`import torch` gets OOM-killed there.

## Now: GRAM-T baseline on the 3 downstream tasks

Submit every command from the `sac-ssamba` root. Logs go to `logs/<job-name>_<jobid>.out`. Slurm does not create
the folder, so it is kept in git as `logs/.gitkeep`. Once a run is done, copy its numbers (with the job id) into the results below.

- [x] **Environment** builds `/scratch200/yotams2/venvs/sac`, clones Vim, downloads `labhamlet/gramt-binaural-frame`.
  `sbatch scripts/setup_env.sbatch`. It is done when the log `logs/sac_env_setup_<jobid>.out` ends with `done:`.
  The empty `/scratch200/yotams2/venvs/gramt` from an earlier attempt can then be deleted.
- [x] **Smoke test** (passed on CPU, job 23244826): `sbatch scripts/smoke_test.sbatch`. The log `logs/gramt_smoke_<jobid>.out` must end with `OK`.
  GPU jobs must use `gpu-general-pool`; `power-general-shared-pool` rejects GPU requests. The GPU queue can be days long,
  and the test also runs on CPU: `sbatch -p power-general-shared-pool --gres=none scripts/smoke_test.sbatch`.
- [ ] **Speech Commands v2** (frozen probe, run_amba.py), running as job 23244879 (about 4 h):
  `sbatch src/finetune/speechcommands_v2/run_gramt.sh labhamlet/gramt-binaural-frame 2.5e-4 true`
- [ ] **IEMOCAP ER** (frozen, s3prl, fold1 only). Blocked: crashes until the IEMOCAP simulation is fixed (see below):
  `sbatch src/finetune/iemocap/run_gramt.sh gramt_binaural 1e-4 true`
- [ ] **VoxCeleb1 SID** (frozen, s3prl):
  `sbatch src/finetune/voxceleb1/run_gramt.sh gramt_binaural 1e-4 true`
- [ ] Collect the results in a table next to the SAC numbers.

## Results

| Model | SC2 acc | IEMOCAP acc (fold1) | VoxCeleb1 acc | Job ids / notes |
|---|---|---|---|---|
| GRAM-T binaural-frame (frozen) | | | | |
| Binaural SAC (frozen) | | | | |

## Open issues / to check

- [ ] **Binaural downstream sets were sampled with replacement.** In `data_generation/utils_simu_rir_sig.py`,
  `generate_microphone_signal` picks `src_idx = np.random.randint(0, len(srcdataset))` instead of `src_idx = idx`.
  So each simulated split covers only ~63% of its source utterances, and the rest of its clips are repeats in other rooms
  (IEMOCAP: 3,519 unique of 5,531; VoxCeleb1 and SC2: ~63% per split). No train/test overlap.
  - IEMOCAP + `spatial_data_root` (`run_er.sh` for binaural/SAC upstreams, `iemocap/run_gramt.sh`) **crashes**:
    `emotion/dataset.py` asserts that every s3prl fold utterance has a simulation (fold1: 398 test / 1,614 train missing).
  - VoxCeleb1 and SC2 run, but on the reduced, duplicated splits: fine for comparing SAC with GRAM-T, not the standard test sets.
  - Fix: use `src_idx = idx` for the downstream sets and re-simulate IEMOCAP, VoxCeleb1 and SC2. Until then,
    either keep IEMOCAP mono (drop the `spatial_data_root` override) or evaluate fold1 on the simulated splits as they are.
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
- [ ] **/scratch200 space**: 249 of 400 GB used. `/scratch200/yotams2/sac-ssamba/data.tar` (~130 GB) can be
  deleted once the extracted `data/` is confirmed complete.

## Done

- [x] Frozen GRAM-T dropout: not an issue. `encoder_dropout` and `encoder_attention_dropout` are 0.0 in the HF config, so
  `train()` putting the frozen encoder in train mode changes nothing.

- [x] Binaural SC2 eval fix (the SAC encoder now gets the 4-channel STFT used in pretraining, not repeated
  mono fbanks): merged to `main` (ecdca0f).
- [x] GRAM-T wrapper (`src/gramt_wrapper`), SC2 path in `run_amba.py`, s3prl upstream `gramt_*` in sac-s3prl.
- [x] `SAC_DATA_ROOT` remapping of the json wav paths (`src/data_paths.py`), used by both the GRAM-T and the binaural SAC SC2 datasets.
