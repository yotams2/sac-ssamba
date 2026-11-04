# Spectral-only experiment: is CCSR or the raw STFT holding the spectral encoder back?

Started 2026-11-04. Job ids and results go in [TODO.md](../TODO.md); this file holds the reasoning and how to run it.

## Background

The dual-encoder binaural model is pretrained with CCSR, the setup from SAR-SSL
(`SAC-SAR-SSL/code/model.py`, `in_ver='separate'`). The spectral encoder sees the target ear on visible frames and the
other ear on masked frames. The spatial encoder sees both ears on visible frames only. The decoder uses both to
predict the target ear on masked frames.

Frozen-probe results of the RunAI run (`pretrain_probe.py`, job 23317158, linear probe):

| Model | SC2 | SID | IEMOCAP UA |
|---|---|---|---|
| Random encoder | 3.7 | 1.9 | 39.0 |
| Dual SAC, steps 4k to final | 11-13 | ~16 | ~47 |
| GRAM-T | 71.2 | 62.4 | 62.9 |

The dual model is above random but stops improving before step 4k.

What is known and what is only suspected:

- **Known, now fixed (6606ac8):** the reconstruction loss weights were unrelated to the mask (correlation 0.000), so half of the
  loss fell on visible frames. Single-frame masking also left 16-frame patches about half visible.
- **Suspected, not proven:** CCSR suits the spatial encoder but not the spectral one. SAR-SSL only keeps the spatial encoder. The spectral
  stream is handed the other ear's content on exactly the frames being predicted, so it can relay content instead of modelling it.
  Block masking does not change that: a masked left-ear block is still visible in the right ear.
- **Unknown:** how much raw complex STFT input costs the spectral encoder compared with log-mel. Nothing has tested this alone. Comparing
  the fine-tuned VoxCeleb1 numbers (mono log-mel ~60%, binaural STFT 41.5%) mixes input, objective, architecture and data.

Why the design used STFT ([Adaptation Plan](Binaural%20SSAMBA%20Adaptation%20Plan.md)): phase carries the inter-ear time cues needed for spatial
tasks, and CCSR shares one complex reconstruction target. Both are spatial reasons. The spectral tasks do not need phase.

## Experiment

Two runs with one spectral encoder and no spatial encoder or CCSR, differing only in the input:

| | Run 1: `stft` | Run 2: `logmel` |
|---|---|---|
| Input (both ears) | raw complex STFT, 4 channels, 256 x 1024 | log-mel, 2 channels, 128 x 1024 |
| Token grid | 16 x 64 = 1024 | 8 x 64 = 512 |
| `mask_patch` | 600 | 300 (same ~58.6% as mono SSAMBA's 300 / 512) |
| `recon_lambda` | 1 | 10 (SSAMBA's weight) |

`recon_lambda` differs on purpose: raw STFT MSE starts ~10x larger than normalised log-mel MSE (2.87 vs ~0.25), so these
weights give both runs the same starting recon term. The comparison should not turn on the loss scale.

Shared by both: binaural pretraining set, 6 epochs (~9.4k steps), batch size 32, lr 1e-4, SSAMBA masked patches with mask
tokens and clustered masking, `mpg` + `mpc` (`classif_lambda=1`), spectral SAC on an unmasked pass (`sac_spec_lambda=0.02`, the mono
recipe, temperature 0.3, offline median sigma, the 5 features in the binaural manifests), and the three frozen probes every 1000 steps.

The 1 s SC2 clips and the zero-padding stay as in the rest of the project. Log-mel frames are zero-padded before normalising, like the mono
dataloader, so padding is not silent after normalisation; the probe pools over real time steps only.

## How to read the results

Compare the linear-probe curves with the dual baseline (job 23347188, `...-lossfix_blockmask_baseline`).

| Result | Meaning |
|---|---|
| Both beat the dual baseline | CCSR holds the spectral encoder back: give it its own objective |
| `logmel` clearly beats `stft` | Raw STFT costs quality: use log-mel for the spectral encoder |
| Neither beats the baseline | The CCSR explanation is wrong: look elsewhere, for example at the binaural data |

Differences of about 1 point are noise (roughly 1000 query clips per task). If the spectral encoder leaves CCSR, the spatial encoder needs a small
throwaway content encoder for CCSR (discarded after pretraining, as SAR-SSL discards its spectral encoder), and the spectral encoder keeps its own
masked patches and input. That needs a spatial probe first, since nothing measures the spatial encoder yet (`src/spatial_downstream/` may be a start).

## Running

From the `sac-ssamba` root:

```bash
sbatch --job-name=sac_spec_stft   scripts/spectral_only.sbatch stft
sbatch --job-name=sac_spec_logmel scripts/spectral_only.sbatch logmel
grep "probe:" logs/sac_spec_stft_<jobid>.out | cut -c1-210
```

Outputs: `$SAC_EXP_ROOT/binaural_sac_pretrain/*-spectral_only_{stft,logmel}/` (`probe_metrics.jsonl`, `models/latest.pth`).
Runs resume after preemption. After a time-limit kill, resubmit the same command.

## Code

- `run_binaural_amba.py --spectral_only --input_repr {stft,logmel}`: `BinauralSSAMBASACModel(spectral_only=True)` builds one encoder (kept under
  `dual_encoder.spectral_encoder`, so the checkpoint loaders and probes work) and calls `AMBAModel.mpg` / `mpc`.
- `wav2fbank` in `binaural_dataloader.py`: two-ear log-mel, normalised with `FBANK_MEAN` / `FBANK_STD` in `scripts/spectral_only.sbatch`,
  computed on 2000 training clips (the mono constants came from clean LibriSpeech).
- `AMBAModel.mpc` / `mpg` now support `in_chans > 1`. Mono behaviour is unchanged, apart from `mpc`'s dot product now always running in fp32.

## Limits

- Checkpoints of the `logmel` run work with the probes only. The s3prl and SC2 downstream paths feed the 4-channel STFT, so real downstream
  runs need a log-mel path first. `stft` checkpoints load as they are.
- Standalone probes of a `logmel` checkpoint need the same normalisation:
  `pretrain_probe.py --input_repr logmel --fbank_mean ... --fbank_std ...`
- Comparisons with the mono SSAMBA results are not meaningful: the data differs.
