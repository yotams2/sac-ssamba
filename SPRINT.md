# Sprint 2026-11-16 → 2026-11-30

**Focus: one SSAMBA model that is good at both spectral (speech) and spatial (SAR-SSL) tasks.**
Set after the 11-16 supervisor meeting. Anything that does not answer RQ1-RQ4 below waits until 12-01, unless it is a
thesis must-have that is already running (see "Pending jobs").

Starting point (end of last sprint, Notion report 11-16):
- Spectral: an SSAMBA encoder works on log-mel; raw complex STFT fails (MPC ~5%, SC2 probe 9.5).
- Spatial: an SSAMBA encoder works on ipd16 (log|L|, ILD, cos/sin IPD; TDOA 0.23 finetune, beats SAR-SSL); raw Re/Im fails.
- Each cue has to be close to linear in the input the patch embedding sees. So the old dual-encoder + CCSR recipe (both
  streams on raw STFT, reconstruct the complex STFT) can't be reused as is.

## Research questions

Each RQ ends in a number that changes what we build next. The order is the run order: cheap reads before new pretraining.

### RQ1. Does a strong general binaural model (GRAM-T) solve the spatial tasks? *(meeting bullet 1)*
- **Why:** this is the external reference for "good on both". GRAM-T binaural-frame takes 2-channel log-mel (L, R; 32 kHz,
  n_fft 1024, hop 320, 128 mel). It has no phase input. Hypothesis: it does well on T60/DRR/C50 (level and decay cues) and
  poorly on TDOA (an inter-ear phase cue). If that holds, it motivates an explicit spatial stream.
- **Do:** add a `gramt` upstream to SAC-SAR-SSL `run_downstream.py`, next to `ssamba_spatial.py`. Resample 16 → 32 kHz and
  use HF checkpoint `labhamlet/gramt-binaural-frame` (cached in /scratch200/yotams2/hf), in the `gramt` venv or the sac venv.
  Run 5 tasks (TDOA, DRR, T60, C50, ABS) at R8 for 20 epochs, finetune and frozen. Same protocol as the ipd16 rows.
- **Output:** a 5-task × {finetune, frozen} table with rows GRAM-T, ipd16 SSAMBA (SAC / no SAC), SAR-SSL, and random init.
- **Cost:** ~1 day of code plus ~10 jobs. Check first that the HF model code runs in the venv we pick.

### RQ2. Is a single encoder enough? *(bullet 2, "single encoder")*
- **Why:** ipd16 already includes a spectral channel (log|L|), and the encoder was pretrained on speech (WSJ0). It may
  already carry speech information. If it does, "one encoder" is a small step, not a new model.
- **Do:** run the existing frozen ipd16 encoders (SAC and no-SAC final) on binaural SC2 / SID / IEMOCAP. Use the frozen
  probes (`pretrain_probe.py`: kNN, linear, EER) first, then fine-tune on SID only if a probe looks reasonable. This needs
  an ipd16 front end in the probe/s3prl path, plus a length policy: ipd16 is 256 frames ≈ 4.1 s, SC2 is 1 s, VoxCeleb is
  longer.
- **Output:** spectral-task numbers for the spatial encoder, next to log-mel SSAMBA and GRAM-T.
- **Decision:** if ipd16 is within a few points of log-mel on the probes, prefer a single encoder: ipd16 or SAIL-style,
  see RQ4. If it is far below, we need two token streams.

### RQ3. Are the two existing encoders complementary, and is a shared target learnable from both? *(bullets 3 and 5)*
> **Update 2026-11-19:** RQ3 runs on Pilot A's encoders (RQ4), not the old ones (agreed 11-17: shared grid and data). Part 1 = A-concat, frozen, queued. Part 2 (RQ3b reconstruction) waits for the A/B results.

- **Why:** this is a cheap feasibility test for a joint objective, with no new pretraining. It reuses the frozen
  encoders: log-mel SSAMBA (`amba` s1, mono LibriSpeech) and ipd16 SSAMBA (SAC, WSJ0).
- **Do, on SAR-SSL simulated binaural clips** (both encoders can take them; log-mel encoder on the L channel):
  1. **Downstream on the concatenation.** For both task families, compare a frozen probe on [spectral; spatial] with each
     encoder alone. This is the "two specialists glued together" baseline that any joint model has to match.
  2. **Reconstruction from frozen embeddings** (the supervisor's test). Train only a small decoder that cross-attends to
     the token sequence and reconstructs the binaural target. The target is log-mel L/R plus IPD on the same grid, and
     error is reported per component. Train it from spectral only, spatial only, and both.
     - Complementary encoders: log-mel error is lowest with the spectral encoder, IPD error lowest with the spatial
       encoder, and "both" is best on everything.
     - If one encoder already gives near-best error on both components, a shared target adds little.
- **Note:** the token grids differ. Log-mel uses 16×16 patches over 128 mel × 1024 frames at 10 ms, so a token spans
  160 ms. ipd16 uses 256 bins × 256 frames at 16 ms, so a token spans 256 ms. A cross-attention decoder avoids aligning
  them, which is why it was chosen over frame-wise concatenation.
- **Caveat:** this measures the information in the embeddings, not whether training with the objective helps. It can
  rule a shared target out, but it can't prove it in.
- **Cost:** ~2 days of code. Decoder runs are short single-GPU jobs.

### RQ4. Which joint pretraining objective? *(bullets 3, 4, 6 and 7)*
> **Update 2026-11-19:** Pilots A (encoders trained apart) and B (trained together, time-column fusion), spec in Notion Agent Handoff 11-17. Compared frozen on TDOA/T60/DRR + SC2/SID: A, A-concat, B fusion-off, B fusion-on.

Run only after RQ2/RQ3. Start with one small pretraining (WSJ0 simulated set, 10 epochs, the ipd16 recipe) and read it
with the RQ1/RQ2 tables.
- **Architecture, informed by SAIL (arXiv 2609.34347).** SAIL feeds log-mel L/R and cos/sin IPD into separate
  conv-BN-GELU stems and patch embeddings, followed by one shared backbone over the concatenated tokens. Projecting IPD
  onto the same mel grid gives the two streams aligned token grids. The stems match our own finding: the cue needs a
  nonlinear per-bin step before patching. This "two tokenizers, one backbone" design sits between 1 and 2 encoders and
  costs about one encoder. **SAIL's training is supervised** (event class, DoA and distance heads), so it gives us an
  architecture, not an SSL objective.
- **Objectives, cheapest and best understood first:**
  - **(a) Shared masked reconstruction.** Mask both streams and reconstruct log-mel L/R plus IPD (mel grid) from the
    joint tokens. This is bullet 3 and the direct heir of CCSR, with the target moved to inputs where the cues are linear.
  - **(b) JEPA-style latent prediction.** Predict EMA-teacher embeddings of masked tokens instead of inputs, including
    cross-stream prediction (spatial targets from spectral context and vice versa). This is bullet 4. It risks collapse
    and has no SSAMBA precedent, so it is time-boxed to one run after (a) works.
  - **(c) CCSR-style "roles" in token space** (bullet 7). Your concern is valid: CCSR's roles came from *what* was masked
    in the input, and masking embeddings does not by itself make a masking pattern spatial or spectral. A possible
    translation is a different mask per stream (time-frequency blocks for the spectral stream, whole time frames for the
    spatial stream), or dropping one stream entirely so it is predicted from the other. **Parked**: we revisit it only if
    (a) gives one stream a free ride, for example if spatial tokens are ignored on spectral targets.
- **SAC** stays as an optional extra term, not something this sprint tests.

## Not in this sprint
- New mono SAC pretrainings or sweeps (sigma, lambda, features). Running mono jobs finish; their reads go into TODO.md.
- Spatial seeds 2-3 for the standalone ipd16 encoder. The thesis needs them, but not for the sprint's decisions.
- The old binaural `logmel` pair and CCSR on log-mel. RQ4(a) replaces both.
- GRAM-T retraining on our data.

## Pending jobs (11-16)
| Job(s) | What | Verdict |
|---|---|---|
| 23652530/31/32 | ipd16 final encoders + random control on DRR/C50/ABS | **Must-have.** Completes the spatial reference profile for RQ1/RQ2. Also needs SAR-SSL refs for these tasks. |
| 23638875 (running), 23640910, 23652540 + collect_after | `amba` s2 SID / s2 SC2 / s3 IEMOCAP | **Must-have, thesis.** Completes the 3-seed no-SAC mono baseline. The log-mel encoder used in RQ3 is from these runs. |
| 23654005 | mono pretraining probe (1 h) | **Keep.** Cheap, and the probe code is reused in RQ2/RQ3. |
| 23647242 (running, epoch 4/10, ~1 day left) + gate 23647243 + collect 23648997 | mono SAC perplexity k=5 | **Keep.** Mostly paid for. The only test of the sharper SAC targets. |
| 23652538 + gate 23652539 | mono SAC perplexity k=10, not started | **Held 11-16** (`scontrol hold`). ~60 h of pretraining plus 3 downstream jobs, and it spends fairshare the sprint needs. Release if k=5 shows a gain. |
| 23640933/35 (held) + gates 23640934/36 + collect 23640937/39 | mono SAC seeds 2/3, median rule | **Stay held.** Possibly superseded: if k=5 wins, the seeds should use the perplexity rule. Cancel at the k=5 read if so. |
| 23653353 | daily backup + prune | Keep. |

## Schedule
- **Week 1 (11-16 → 11-22):** RQ1 wrapper + jobs; RQ2 ipd16 front end in the probe + probe jobs; RQ3 decoder code.
  Read DRR/C50/ABS.
- **Week 2 (11-23 → 11-30):** RQ3 reads. Choose RQ4 (single vs SAIL-style) and objective (a). Run one small pretraining
  and give it first reads on both task families by 11-30.
- **Sprint output (11-30):** a table with rows GRAM-T, log-mel SSAMBA, ipd16 SSAMBA, frozen concat, joint v0, and
  columns for spectral tasks and SAR-SSL tasks. Plus a go/no-go on the joint design for the next sprint.

## Decision log (this sprint)
- 11-16: sprint set from the supervisor meeting. RQ order: GRAM-T / single encoder / complementarity, then the joint objective.
- 11-16: k=10 held (your OK). RQ1 started: `--upstream gramt` (SAC-SAR-SSL `code/gramt_spatial.py`, `scripts/gramt_spatial_eval.sbatch`), no scratchLOW for GRAM-T. Smoke 23655896, then label `gramt_bf` TDOA/T60/DRR/C50/ABS 23655897-901 (afterok).
- 11-17: RQ1 and RQ2 read (TODO.md). RQ2: ipd16 far below log-mel/GRAM-T on speech probes -> two token streams.
- 11-17 (your call): **RQ3 drops the frozen-concat downstream** (step 1); the frozen binaural spectral SSAMBA was already far below GRAM-T, so concat can't move spectral accuracy meaningfully. RQ3 = reconstruction from frozen embeddings only.
- 11-17 (your call): RQ3 spectral encoder = the binaural 2-ch log-mel `spectral_only_logmel` run (23348511), not mono `amba` s1 on L. L-only would handicap R / level reconstruction from the spectral side and bias the read toward "complementary". `amba` s1 stays an optional control row.
- 11-17: RQ2 started: ipd16 front end in the probe (4.112-s windows, fp32), jobs 23663228 (smoke) -> 23663229.
