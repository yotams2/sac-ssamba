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
- [ ] **Speech Commands v2** (frozen probe, run_amba.py), running as job 23244879 (about 4 h):
  `sbatch src/finetune/speechcommands_v2/run_gramt.sh labhamlet/gramt-binaural-frame 2.5e-4 true`
  This run started before the move to `SAC_EXP_ROOT` and writes to the home folder. When it is done, move it:
  `mkdir -p /scratch200/yotams2/sac-ssamba/exp/speechcommands_v2 && mv src/finetune/speechcommands_v2/exp/gramt-* /scratch200/yotams2/sac-ssamba/exp/speechcommands_v2/`
- [ ] **IEMOCAP ER** (frozen, s3prl, fold1 only). Blocked until binaural IEMOCAP is regenerated (see below):
  `sbatch src/finetune/iemocap/run_gramt.sh gramt_binaural 1e-4 true`
- [ ] **VoxCeleb1 SID** (frozen, s3prl):
  `sbatch src/finetune/voxceleb1/run_gramt.sh gramt_binaural 1e-4 true`
- [ ] Collect the results in a table next to the SAC numbers.

## Results

| Model | SC2 acc | IEMOCAP acc (fold1) | VoxCeleb1 acc | Job ids / notes |
|---|---|---|---|---|
| GRAM-T binaural-frame (frozen) | | | | |
| Binaural SAC (frozen) | | | | |

All IEMOCAP numbers from before the regeneration (including the RunAI "binaural IEMOCAP" runs) were on **mono**
IEMOCAP: the binaural setting did not exist yet, so the SAC model got mono audio copied to all 4 STFT channels.

## Open issues / to check

- [ ] **Binaural downstream sets were sampled with replacement.** In `data_generation/utils_simu_rir_sig.py`,
  `generate_microphone_signal` picks `src_idx = np.random.randint(0, len(srcdataset))` instead of `src_idx = idx`.
  So each simulated split covers only ~63% of its source utterances, and the rest of its clips are repeats in other rooms
  (IEMOCAP: 3,519 unique of 5,531; VoxCeleb1 and SC2: ~63% per split). No train/test overlap.
  - IEMOCAP + `spatial_data_root` (`run_er.sh` for binaural/SAC upstreams, `iemocap/run_gramt.sh`) **crashes**:
    `emotion/dataset.py` asserts that every s3prl fold utterance has a simulation (fold1: 398 test / 1,614 train missing).
  - VoxCeleb1 and SC2 run, but on the reduced, duplicated splits: fine for comparing SAC with GRAM-T, not the standard test sets.
  - The first IEMOCAP release also cut or zero-padded every utterance to 6 s before simulating (and computed the SNR over
    the padding), unlike the full-length mono IEMOCAP.
  - [ ] **IEMOCAP: fixed in the generator, needs regenerating.** `gen_simu_iemocap.py` now simulates every utterance once
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

- [x] Frozen GRAM-T dropout: not an issue. `encoder_dropout` and `encoder_attention_dropout` are 0.0 in the HF config, so
  `train()` putting the frozen encoder in train mode changes nothing.

- [x] Binaural SC2 eval fix (the SAC encoder now gets the 4-channel STFT used in pretraining, not repeated
  mono fbanks): merged to `main` (ecdca0f).
- [x] GRAM-T wrapper (`src/gramt_wrapper`), SC2 path in `run_amba.py`, s3prl upstream `gramt_*` in sac-s3prl.
- [x] `/scratch200/yotams2/sac-ssamba/data.tar` deleted after extracting `data/` (2026-10-28).
- [x] `SAC_DATA_ROOT` remapping of the json wav paths (`src/data_paths.py`), used by both the GRAM-T and the binaural SAC SC2 datasets.
