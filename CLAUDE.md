# sac-ssamba

SSAMBA (self-supervised audio Mamba, fork of SiavashShams/ssamba) extended with the **Soft Acoustic Contrastive (SAC)**
loss and a **binaural dual-stream SAC** model. The current work evaluates binaural SAC against a **GRAM-T** baseline on
three binaural (simulated) downstream tasks: Speech Commands v2, IEMOCAP ER, and VoxCeleb1 SID.

- Research background: [project_context.md](project_context.md), [ssamba_project_summary.md](ssamba_project_summary.md),
  [exp_session_1.md](exp_session_1.md), and the binaural design docs in [docs/](docs/).
- **Current status, open issues and results table: [TODO.md](TODO.md).** Read it first and keep it up to date
  (job ids, numbers, what's blocked).

## History: two machines

The project started on an **external RunAI server** (paths like `/storage/yotam/...`, `/storage/data/...`, L40 GPUs).
It is being moved to the **TAU power cluster** (Slurm). Some scripts, json manifests and checkpoint paths still point at
`/storage/...`. When you touch one, port it to `env.sh` (see "Porting a script" below) rather than hardcoding new paths.
Jobs that need gpuRIR or the original datasets (e.g. `data_generation/regen_iemocap.sh`) still have to run on RunAI.

## Storage on this cluster

| Location | What | Notes |
|---|---|---|
| `~/sac-ssamba` (home) | this git repo (code only) | home quota is ~2 GB: no data, checkpoints or venvs here |
| `/scratch200/yotams2/` | working copy of everything heavy: `sac-ssamba/data`, `sac-ssamba/exp`, `venvs/sac`, `hf`, `code/Vim` | reachable from all nodes, large, but **scratch: may be wiped, not long-term storage**. Also slow: avoid recursive scans/`du`/`tar t` |
| `/moshe-data/blaufer-data/yotams2/` | long-term copy of what came over from RunAI (e.g. `data/` with LibriSpeech, VoxCeleb1, SC2 archives; see [transfer_datasets.sh](transfer_datasets.sh)) | **not mounted on all nodes** (not on the Jupyter node). Don't point jobs at it; treat it as the backup |

`/scratch200/yotams2/sac-ssamba/src/` holds files copied from RunAI's `src/` tree (old experiment outputs, metrics), not code to edit.

Consequences:
- Anything that only exists on `/scratch200` (new results, checkpoints, regenerated data) should be copied to
  `/moshe-data` (from a node that mounts it) or kept in git (small results/metrics) if it matters.
- If `/scratch200` gets wiped, restore data from `/moshe-data` and rebuild the venv with `scripts/setup_env.sbatch`.

## Environment

- `source env.sh` sets everything: `SAC_VENV`, `PYTHON`, `HF_HOME`, `VIM_ROOT`, `SAC_DATA_ROOT`
  (`/scratch200/yotams2/sac-ssamba/data`), `SAC_EXP_ROOT` (`/scratch200/yotams2/sac-ssamba/exp`). The `run_*.sh`
  scripts source it themselves.
- One shared venv for SAC, GRAM-T and s3prl: `/scratch200/yotams2/venvs/sac` (torch 2.2.2+cu121, mamba kernels),
  built by `sbatch scripts/setup_env.sbatch`. Install with `--no-cache-dir`. wandb is not installed.
- `src/data_paths.py` remaps the old `/storage/yotam/ssamba/data` paths inside the json manifests to `SAC_DATA_ROOT`.

## Running jobs

- The interactive (Jupyter) node has 1 GB RAM and no GPU: `import torch` gets OOM-killed. Locally only do static
  checks (`python -m py_compile`, `bash -n`). Everything else goes through `sbatch`.
- **Submit from the repo root**: scripts write to `logs/%x_%j.out` (relative to the submit dir; `logs/` is gitignored
  except `.gitkeep`).
- Slurm account `public-blaufer_v2`. GPU jobs: `--partition=gpu-general-pool` only (queue can be days long).
  CPU jobs: `power-general-shared-pool` (rejects GPU requests).
- Always keep `#SBATCH --exclude=compute-0-18` on GPU jobs: it's a Blackwell (sm_120) node with no kernels in this torch.
- `gpu-general-pool` is **preemptible** (`PreemptMode=REQUEUE`): jobs get killed and requeued. Make runs resumable
  (s3prl: pass `-a` and keep `save_step` reasonable) and add `#SBATCH --open-mode=append`, otherwise the requeued job
  truncates its log (`JobFileAppend=0` here).
- Check jobs: `squeue -u $USER`, `sacct -u $USER -S <date> --format=JobID,JobName%20,State,Elapsed,NodeList`.

## Code map

- `src/run_amba.py`, `src/traintest*.py`, `src/models/`: SSAMBA pretrain/finetune (also the SC2 path for GRAM-T).
- `src/sac/`: single-channel SAC model (`SSAMBASACModel`, factorized SAC loss, acoustic features).
- `src/binaural_dual_stream_sac/`: binaural dual-stream SAC model, dataloader, spatial features. `run_binaural_sac.sh` is the pretraining
  entry point (`--spectral_only` for one encoder without CCSR), `pretrain_probe.py` the frozen downstream probes.
- `src/gramt_wrapper/`: GRAM-T encoder wrapper (+ `smoke_test.py`, run via `scripts/smoke_test.sbatch`).
- `src/finetune/<task>/`: per-task launch scripts. The ported ones are `run_gramt.sh` (SC2, IEMOCAP, VoxCeleb1);
  `run_binaural_sac_frozen.sh`, `run_er.sh`, `run_sid.sh`, `run_sc_amba.sh` still use RunAI paths.
- `data_generation/`: gpuRIR binaural simulation of the downstream sets and manifest generation (runs on RunAI).
- IEMOCAP and VoxCeleb1 run through **s3prl**: the sibling repo `../sac-s3prl` (github yotams2/sac-s3prl) has the
  `gramt_*` and SAC upstreams and the binaural loaders. Changes often span both repos.
- `../GRAM-T`: upstream GRAM-T code, reference only.

## Porting a script to this cluster

Follow `src/finetune/voxceleb1/run_gramt.sh`: `#SBATCH` header (partition, exclude, `--output=logs/%x_%j.out`),
sbatch-safe `SCRIPT_DIR` (resolve via `scontrol show job` when `BASH_SOURCE` is `slurm_script`), `source env.sh`,
use `$PYTHON`, `$SAC_DATA_ROOT`, `$SAC_EXP_ROOT` instead of absolute paths.

## Keeping the status current

`~/STATUS.md` (home dir, outside git) is the user's executive summary of where things stand. After **every** change or
action (job submitted or finished, result read, decision made, code committed), update both `~/STATUS.md` and `TODO.md`
before ending the turn. Keep `STATUS.md` to one screen: where we are, what we're waiting for, latest results, next
action. Replace stale text instead of appending, and put the detail in `TODO.md`.

## Results sync and the cloud Claude

I also work with Claude in the claude.ai app ("cloud Claude"). It cannot reach this cluster. It reads this repo from
GitHub (code on `main`, results on the `results` branch) and the Notion thesis archive. Anything not pushed is
invisible to it.

- **Results**: launch scripts source `scripts/results_hook.sh` and call `results_hook <task> <exp_dir>`. It records the
  run as `running` at start and `done` or `failed` at exit (EXIT trap), through `scripts/collect_result.py` and
  `scripts/sync_results.sh`. Traps can't catch preemption, time limits or scancel here: Slurm signals only the batch
  shell, which can't run a trap during a foreground command. So the hook also submits `scripts/collect_after.sbatch`
  with `--dependency=afterany`, which closes the record from Slurm's final state (`interrupted`). Each run is `results/<task>/<exp>__<jobid>/summary.json` plus small raw files, in the
  worktree `~/sac-ssamba-results` (branch `results`), and compute nodes push directly. Never commit results to `main`.
  Format and status meanings: `README.md` on that branch. Report `metrics.test_at_best_dev`.
- **New launch scripts**: add the hook, after the experiment dir is known and before the training command. Pipelines
  that only call other launch scripts don't need it.
- **Changing a launch script**: replace the file (`sed -i`, or write + rename), don't edit it in place, if a job may be
  running it from the repo. Bash reads a script as it goes, and `scripts/spectral_only.sbatch` runs
  `run_binaural_sac.sh` from the repo, not from Slurm's copy. A job already running doesn't get a new hook: report it
  with `sbatch --dependency=afterany:<job id> scripts/collect_after.sbatch <task> <exp dir> <job id> <log>`.
- **Start of every session**: run `scripts/sync_results.sh` (it pushes anything a job couldn't; `scrontab` is disabled
  here, so nothing else would), `git pull` on main, and read the Notion page "Agent Handoff" (under SAC Loss) for
  requests from the cloud Claude.
- **End of every session**: run `scripts/sync_results.sh`, update `TODO.md` (status, job ids, numbers in the results
  table), commit with a descriptive message, and **push `main`**. Unpushed work doesn't exist for the cloud Claude.
- **Notion** (only if the Notion tools are available here): selective updates, no new page per exchange or minor fix.
  Add or update a row in the Experiments Log only when an experiment is finished and has a verdict (check for an
  existing row first). Tick items in Global TODO when they're done. To ask the cloud Claude something, add a dated line
  to "Agent Handoff". Delete lines from that page once they're handled; it's an inbox, not an archive.

## Git

Remote `git@github.com:yotams2/sac-ssamba.git`. Push over SSH (HTTPS fails through the VS Code credential helper).
