# Project Overview: SSAMBA with Factorized Soft Acoustic Contrastive (SAC) Loss

## 1. Project Goal
To develop a computationally efficient, self-supervised audio foundation model that learns a highly disentangled and physically grounded latent representation of speech and acoustics. The model aims to achieve this by combining the linear-time sequence modeling of State Space Models (Mamba) with a novel, physics-guided continuous contrastive loss framework.

## 2. Context & Motivation
* **The Efficiency Problem:** Traditional self-supervised audio models (like SSAST) rely on Transformers, which suffer from quadratic computational and memory complexity regarding sequence length.
* **The Geometric Problem:** Standard contrastive learning (like InfoNCE) relies on discrete positive/negative pairs, which fails to capture the continuous nature of audio physics. Furthermore, forcing a network to organize a latent space based on a single concatenated vector of diverse acoustic features leads to "$L_2$ distance dilution" (gradient interference between uncorrelated physical properties like pitch and room noise).

## 3. The Core Architecture (SSAMBA)
The backbone of the project is **Self-Supervised Audio Mamba (SSAMBA)**.
* **Input:** Single-channel audio waveforms converted to 128-dimensional log-Mel spectrograms, split into 16x16 patches.
* **Encoder:** A deep, bidirectional Mamba (State Space Model) encoder. It processes the flattened spectrogram patches sequentially, capturing global audio context with subquadratic complexity.
* **Base Pretraining:** The standard SSAMBA framework masks a portion of the input patches and uses a reconstruction head to minimize Mean Squared Error (MSE) between the predicted and actual masked patches.

## 4. The Novel Contribution: Factorized SAC Loss
The primary research novelty is the replacement of SSAMBA's discrete patch-classification loss with the **Soft Acoustic Contrastive (SAC) Loss**, enhanced by an **Acoustic Group Cross-Attention** bottleneck.
* **Continuous Weights:** SAC uses continuous similarity weights ($w_{ij}$) computed via a Gaussian kernel over the $L_2$ distance of specific analytical acoustic features (e.g., $F_0$, Formants, MFCCs).
* **The Cross-Attention Router:** To solve gradient interference, the model uses learnable "Acoustic Queries" in a PyTorch `MultiheadAttention` layer. These queries attend to the Mamba sequence to extract distinct latent sub-spaces (Feature Groups).
* **Factorized Loss Computation:** The SAC loss is computed independently for each semantic group and then averaged.
  * *Prosody Query:* Regulated by $F_0$ mean/variance.
  * *Vocal Tract Query:* Regulated by Formant frequencies ($F_1, F_2, F_3$).
  * *Timbre Query:* Regulated by early MFCCs.
  * *Voice Quality Query:* Regulated by Harmonics-to-Noise Ratio (HNR).
  * *Scene/Noise Query:* Regulated by Spectral Centroid, Flux, and Zero-Crossing Rate (ZCR).

## 5. Pipeline & Data
* **Pretraining Dataset:** LibriSpeech (960 hours, 1-mic, clean speech). Audio is normalized to 10-second clips at 16kHz.
* **Acoustic Feature Extraction:** Computed on-the-fly or pre-extracted using standard DSP libraries. Features are z-score normalized and bounded to $[-1, 1]$.
* **Compute Environment:** Linux, Slurm workload manager (RunAI), L40 GPUs.

## 6. Downstream Evaluation (Fine-Tuning)
The downstream tasks utilize the raw patch-level hidden states from the Mamba encoder, typically processed through the **S3PRL** framework. Pretraining projection heads and cross-attention queries are discarded.
* **Target Downstream Tasks:**
  * Speaker Identification (VoxCeleb)
  * Emotion Recognition (IEMOCAP)
  * Keyword Spotting (Speech Commands)
  * Environmental Audio Event Detection (ESC-50 / AudioSet)

## 7. Codebase Structure & Current State
* **Pretraining Logic:** Centered in `src/run_amba.py` which calls into the training loops, and `src/sac/sac_model.py` which contains the `SSAMBASACModel` class.
* **SSAMBASACModel:** Implements the dual-objective pretraining (reconstruction + SAC). It features a `Cross-Attention` module to extract feature group latents, and computes the `sac_loss` using sophisticated local sigma calculation modes like `chi2_median`, `offline_global_median`, and `dynamic_batch_median`.
* **Results / Metrics (`src/metrics/downstream_perf`):** 
  * Early metrics show promising results on VoxCeleb1 and IEMOCAP downstream tasks.
  * For instance, `ssamba_sac_feat_sid_lam0_02_sig2_0_1e-4` outperformed the baseline test performance on VoxCeleb1 (`0.614` vs `0.592`). 
  * For IEMOCAP, the `ssamba_sac_feat_emo_lam0_02_sig2_0_1e-5` run outperformed the baseline test performance (`0.605` vs `0.565`).

## 8. Next Steps Focus
Monitoring pretraining stability, ensuring gradient norms are balanced between MSE reconstruction and Factorized SAC, and tracking downstream S3PRL fine-tuning metrics.
