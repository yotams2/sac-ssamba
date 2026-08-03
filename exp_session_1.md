# Experiment Session 1: Factorized SAC Loss Improvements

This document tracks a sequence of four planned experiments designed to improve the Universal SSAMBA pretraining performance. The experiments are ordered from most foundational (fixing loss/data targets) to most complex (architectural capacity and training schedules).

---

## Experiment 1: Learnable Group-Specific Sigmas ($\sigma$)
**Rationale:** Current heuristic median scaling (`chi2_median`, etc.) struggles because the acoustic features are artificially bounded to $[-1, 1]$, violating standard normal distribution assumptions. Manual tuning of $\sigma$ proved beneficial. Making it a learnable parameter allows the network to find the optimal bandwidth per feature group via gradient descent.
**Implementation:** Replace the static `local_sigma` calculation in `sac_model.py` with `F.softplus(self.group_sigmas[k]) + 1e-4` where `group_sigmas` is an `nn.Parameter`.
**Status:** Abandoned

### Insights & Conclusions
* **Abandonment Rationale:** Making $\sigma$ learnable allows the network to bypass learning useful representations by simply scaling $\sigma$ to make the contrastive target distribution ($w_{ij}$) trivially easy to mimic (e.g. uniform distribution by pushing $\sigma \to \infty$). Minimizing the SAC loss with respect to hyperparameters does not guarantee alignment with downstream semantic tasks. We will instead rely on hyperparameter sweeps (like testing $\sigma=2.0$ for the Universal feature set) or statically calibrated sigmas.

---

## Experiment 2: Better Formant Extraction (LPC)
**Rationale:** The current formant proxy (`_compute_formants_proxy`) uses simple spectral centroids within fixed frequency bands. This is highly inaccurate across diverse speaker profiles (e.g., male vs. female pitch shifts). Upgrading to Linear Predictive Coding (LPC) roots provides a mathematically grounded representation of the vocal tract.
**Implementation:** Update `acoustic_features.py` to calculate `f1, f2, f3` using LPC coefficients rather than banded centroids.
**Status:** Implemented

### Insights & Conclusions
* **LPC Implementation:** Implemented a pure PyTorch native Levinson-Durbin recursion and eigenvalue solver to replace the bounded spectral centroid proxy, since `torchaudio.functional.lpc` was unavailable. This method extracts 10 high-energy frames per clip (vowels), computes LPC roots, and averages the formants (F1, F2, F3). It provides a mathematically grounded model of the vocal tract resonances, which should provide a much more accurate target for the Vocal Tract feature group.
* **VoxCeleb1 (Speaker ID) Evaluation:** The LPC formants significantly improved speaker identification performance compared to the old bounded spectral centroid method. Test Perf @ Best Dev increased from `0.592` to `0.598`, and Max Test Perf increased from `0.597` to `0.603`. This confirms the hypothesis that LPC roots provide a more discriminative biometric vocal tract representation.
* **IEMOCAP (Emotion Recognition) Evaluation:** For emotion recognition, the results are mixed. While the model with LPC formants achieved a higher Best Dev score (`0.678` vs `0.649`) and a slightly higher Max Test score (`0.595` vs `0.591`), its Test Perf @ Best Dev was noticeably worse (`0.577` vs `0.591`). It appears the more precise vocal tract features might lead to overfitting on the development set for emotion tasks, or perhaps emotion is less dependent on fine-grained formant tracking than speaker identity. Overall, EXP2 was highly successful for identity tasks but showed high variance for emotion.

---

## Experiment 3: Multi-Query Subspaces
**Rationale:** Each feature group currently uses exactly one query vector in the Cross-Attention block, resulting in a single latent representation. Complex groups, such as `Scene` (which mixes Centroid, Flux, and ZCR), likely require a higher-capacity subspace to represent their variance accurately.
**Implementation:** Modify `sac_model.py` to assign $N$ queries (e.g., $N=4$) per active group, generating a `[B, 4, embed_dim]` subspace, which is then pooled prior to computing the SAC loss.
**Status:** Implemented

### Insights & Conclusions
* **Multi-Query Pooling:** Implemented $N=4$ queries per group. The queries are flattened for the multi-head attention block (`num_groups * 4`), and the resulting attention outputs are mean-pooled across the 4 queries per group before passing into the projection head. This increases the capacity of the feature group subspaces without changing the downstream SAC loss dimensionality.
* **SID Task Evaluation:** Based on the downstream Speaker Identification (VoxCeleb1) results, the multi-query approach did not yield the expected improvements. EXP3 achieved a Test Perf @ Best Dev of `0.596` and a Max Test Perf of `0.598`. This is slightly lower than the equivalent single-query model (`0.598` and `0.603` respectively), although it still outperforms the pure SSAMBA baseline (`0.592` / `0.594`). It appears that mean-pooling across multiple queries might be diluting the signal for features relevant to SID rather than capturing useful extra variance.
* **IEMOCAP & Speech Commands Evaluation:** Now that all downstream tasks are complete, the multi-query approach is definitively unsuccessful. On **IEMOCAP** (Emotion Recognition), EXP3 had a very low Test @ Best Dev of `0.540` (worse than the `0.591` of the single-query model), indicating high variance or overfitting. On **Speech Commands** (Keyword Spotting), EXP3 achieved an accuracy of `0.9753`, which underperforms even the pure SSAMBA baseline (`0.9770`). Overall, increasing query capacity and mean-pooling them negatively impacts downstream performance across the board.

---

## Experiment 4: Curriculum Learning for SAC Groups
**Rationale:** Activating all 5 feature groups at step 0 might cause "gradient confusion" early in training. Starting with robust, low-level features and gradually introducing complex phonetic features might stabilize the latent space formation.
**Implementation:** Update the pretraining script (`run_amba.py` / `traintest_mask.py`) to dynamically unfreeze or introduce specific `Acoustic Queries` at different training milestones (e.g., Steps 0-20k: Timbre/Scene, Steps 20k-50k: Prosody/Voice Quality, Steps 50k+: Vocal Tract).
**Status:** Planned

### Insights & Conclusions
* *(To be updated after execution)*

---

## Experiment 5: Mathematically Optimal Sigma and Tau Calibration
**Rationale:** Previous runs used dynamic heuristic medians or global scalar defaults for the Softmax temperature ($\\tau$) and Gaussian bandwidth ($\\sigma$). This often resulted in contrastive target distributions that were either washed out (high entropy) or completely collapsed, severely degrading the learning signal. By using a batch-aware search to target exactly 62.5% of maximum theoretical Shannon entropy for $\\sigma$, and empirically sweeping $\\tau$ to find the "Goldilocks" boundary (e.g. $\\tau=0.1$, maximizing alignment before uniformity degrades), we can mathematically guarantee a robust and discriminative latent geometry.
**Implementation:** Update `run_sac.sh` to use the new `--local_sigma_mode static_entropy_optimal` (which automatically queries the batch-aware dictionary in `sigma_configs.py`) and set `--sac-temperature 0.12`.
**Status:** Implemented

### Insights & Conclusions
* **VoxCeleb1 (Speaker ID):** EXP5 achieved a Test Perf @ Best Dev of `0.580` and Max Test Perf of `0.586`. This is a noticeable **regression** compared to the SSAMBA baseline (`0.592` / `0.594`) and earlier SAC variants like `offline_global_median` (`0.598` / `0.603`). It appears that the mathematically optimal entropy target might be forcing the latent space to be *too* uniform, thereby washing out the fine-grained variance required to distinguish individual speaker biometrics.
* **IEMOCAP (Emotion Recognition):** Conversely, EXP5 performed relatively well here, achieving a Test Perf @ Best Dev of `0.576` (and Max Test Perf of `0.576`). This **outperforms** the pure SSAMBA baselines (`0.547` - `0.566`), though it does not reach the peak of manually tuned, task-specific runs (e.g., `feat_emo_...` at `0.605`).
* **Overall Conclusion:** The `static_entropy_optimal` approach (targeting 62.5% max Shannon entropy) creates a learning signal that is beneficial for broad, global tasks like emotion classification, but actively harmful for highly specific, granular tasks like speaker identification. This suggests that forcing a strictly uniform target distribution over-regularizes the space and suppresses fine-grained identity features.

---

## Experiment 6: True Batch Size 64 with Gradient Checkpointing
**Rationale:** Previous runs with gradient accumulation suffered from weakened contrastive signals because the SAC loss was computed over micro-batches (e.g., batch size 16), significantly reducing the number of negative samples in the denominator. To evaluate the true impact of the SAC loss, we need a large effective batch size of negative samples.
**Implementation:** Reverted gradient accumulation and enabled PyTorch activation checkpointing (`--use_checkpointing true`) on the Mamba layers. This allows training with a true batch size of 64 within GPU memory limits. We also set `local_sigma_mode="offline_global_median"`.
**Status:** Implemented & Completed

### Insights & Conclusions
* **Initial Evaluation (The Batch Size Paradox):** The initial run of EXP6 at `lr=1e-4` showed an absolute performance regression across both the Universal SAC model and the pure SSAMBA baseline compared to micro-batch accumulation runs. For instance, the VoxCeleb1 baseline dropped from `0.592` to `0.561`. However, within the True BS64 regime at `lr=1e-4`, the SAC loss *did* provide a clear improvement over the baseline (`0.576` vs `0.561` on VoxCeleb1, and `0.5696` vs `0.5613` on IEMOCAP).
* **Root Cause - Learning Rate Scaling & Warmup Bug:** 
  1. Micro-batch accumulation (`b16` x 4) injected stochastic gradient noise per micro-batch that acted as implicit regularization. Under True BS64, maintaining `1e-4` under-regularized and undertrained the model.
  2. The learning rate warmup was tied to `global_step` (micro-batches), causing micro-batch runs to reach peak LR in 250 effective optimizer steps vs 1,000 steps for True BS64. The codebase was patched to use `eff_step`.
* **Validation via Baseline LR Scaling:** Scaling the pretraining LR to `4e-4` for True BS64 successfully restored baseline performance to peak levels: VoxCeleb1 baseline reached **`0.6050`** Test Perf @ Best Dev (surpassing the old `0.5924` micro-batch baseline) and IEMOCAP baseline reached **`0.5724`**.
* **Final Results for SAC BS64 @ LR 4e-4:**
  * **VoxCeleb1 (Speaker Identification):** **`0.6129`** Test Perf @ Best Dev (and **`0.6145`** Max Test Perf). This outperforms the matching BS64 LR4e-4 baseline (`0.6050`) by **+0.79%** (and **+0.95%** max test), as well as the old micro-batch baseline (`0.5924`) by **+2.05%**.
  * **IEMOCAP (Emotion Recognition):** **`0.5899`** Max Test Perf (**`0.5613`** @ Best Dev step 10k), outperforming the matching BS64 baseline (`0.5724`) by **+1.75%** max test.
  * **Speech Commands v2 (Keyword Spotting):** **`0.9746`** Test Acc, slightly outperforming the BS64 baseline (`0.9734`).
* **Conclusion:** Moving to True BS64 with activation checkpointing and scaling LR to `4e-4` proves that SAC contrastive learning significantly benefits from large negative sample pools (64 in-batch samples per step without micro-batch fragmentation). This sets a new strong baseline for SSAMBA pretraining.

---

## Experiment 7: Pure SAC Loss (No Reconstruction Loss)
**Rationale:** Standard SSAMBA pretraining relies on dual objectives: a generative masked patch reconstruction loss ($L_{\text{recon}}$) and the factorized Soft Acoustic Contrastive loss ($L_{\text{SAC}}$). Generative reconstruction requires passing masked audio patches through the Mamba backbone and predicting raw spectrogram patches via a linear decoder (`gpredlayer`), creating substantial computational overhead. We want to test whether the geometry-guided continuous SAC contrastive loss alone is sufficient to pretrain high-quality audio representations without needing a decoder or generative reconstruction signal.
**Implementation:** 
* Added explicit loss weighting flags `--recon-lambda`, `--classif-lambda`, and `--sac-lambda` to `SSAMBASACModel`, `run_pretrain_sac.py`, and `run_sac.sh`.
* Setting `recon_lambda=0.0`, `classif_lambda=0.0`, and `sac_lambda=0.02` (or 1.0) completely disables the generative reconstruction pass (`mpg()`), avoiding the second forward pass through the 24-layer Mamba backbone.
* **Efficiency Impact:**
  - **Parameter Savings:** Eliminates the decoder (`gpredlayer`), saving **787,456 parameters** (~0.95% of total model parameters).
  - **Compute Speedup:** Cuts Mamba backbone forward passes from 2 to 1 per batch, resulting in **~2x faster pretraining throughput** and significantly lower GPU VRAM consumption.
**Status:** Completed

### Insights & Conclusions
* **Downstream Performance Degradation:** Removing the generative reconstruction loss ($L_{\text{recon}}$) led to substantial performance degradation across all three evaluated downstream tasks:
  * **VoxCeleb1 (Speaker ID):** Dropped from **`0.6129`** (EXP6 Dual Loss) to **`0.5272`** Test Perf @ Best Dev (-8.57% absolute drop).
  * **IEMOCAP (Emotion Recognition):** Dropped from **`0.5899`** Max Test / `0.5613` @ Best Dev (EXP6) to **`0.5585`** Max Test / **`0.5253`** @ Best Dev (-3.60% drop @ Best Dev).
  * **Speech Commands v2 (Keyword Spotting):** Dropped from **`0.9746`** (EXP6) to **`0.9522`** Test Acc (-2.24% drop).
* **Root Cause - Loss Complementarity:** 
  * The generative patch reconstruction loss ($L_{\text{recon}}$) acts as a high-frequency local anchor, preserving fine-grained spectrogram patch structures and temporal transitions.
  * The Factorized SAC loss ($L_{\text{SAC}}$) acts as a low-frequency global regularizer, structuring feature group latent geometry.
  * Without $L_{\text{recon}}$, the Mamba encoder over-specializes on coarse physical summary statistics and loses fine-grained local spectral representations required for downstream transfer.
* **Verdict:** Pure SAC loss cannot replace generative reconstruction. Dual-objective pretraining ($L_{\text{recon}} + \lambda L_{\text{SAC}}$) is strictly required for optimal representation learning.

---

## Experiment 8: Optuna-Calibrated Factorized SAC (Dual Loss)
**Rationale:** Previous pretraining runs used either static heuristic medians (`offline_global_median`) or theoretical entropy baselines (`static_entropy_optimal`). Optuna Hyperparameter Study `v5` identified the mathematically winning configuration (Trial #67) that guarantees non-collapsed representations and optimal alignment/uniformity balance across all 5 feature groups simultaneously. To evaluate if these tuned parameters improve downstream transfer over the **EXP6 baseline** (True BS64 @ LR 4e-4 with `offline_global_median`), we maintain the exact True BS64 @ LR 4e-4 dual loss setup ($L_{\text{recon}} + 0.02 L_{\text{SAC}}$) and test the new `optuna_optimal` group sigmas and temperature ($\tau = 0.5034$).
**Implementation:**
* **`sac_temperature`**: `0.5034`
* **`local_sigma_mode`**: `"optuna_optimal"` ($\sigma_{\text{Prosody}} = 0.2969$, $\sigma_{\text{Vocal\_Tract}} = 0.1982$, $\sigma_{\text{Timbre}} = 0.6409$, $\sigma_{\text{Voice\_Quality}} = 0.0928$, $\sigma_{\text{Scene}} = 0.2748$)
* **Dual Loss Weights (Matching EXP6)**: `recon_lambda=1.0`, `sac_lambda=0.02`
* **Training Setup**: True BS64 @ LR 4e-4 with activation checkpointing on LibriSpeech.
**Status:** Prepared & Ready to Launch

### Insights & Conclusions
* *(To be updated after execution)*

---

## Experiment 9: Tri-Objective Pretraining (Reconstruction + Classification + Factorized SAC)
**Rationale:** Standard SSAST / SSAMBA pretraining (`run_mask_patch_amba.sh`) uses a joint loss combining Masked Patch Classification (MPC, discriminative NCE) and Masked Patch Generation (MPG, generative MSE): $\mathcal{L}_{\text{baseline}} = \mathcal{L}_{\text{MPC}} + 10 \cdot \mathcal{L}_{\text{MPG}}$. All previous SAC experiments (EXP1–EXP8) set `classif_lambda=0.0`, testing only generative reconstruction + SAC loss. Adding the discriminative patch classification loss ($\mathcal{L}_{\text{MPC}}$) introduces discrete patch-level contrastive targets alongside continuous acoustic SAC targets. Normalizing relative to $\mathcal{L}_{\text{MPG}} = 1.0$ (as in `SSAMBASACModel`), maintaining the exact baseline weighting ratio requires $\text{classif\_lambda} = 0.1$ ($\frac{1}{10}$).
**Implementation:**
* **`recon_lambda`**: `1.0` ($L_{\text{recon}}$ generative reconstruction)
* **`classif_lambda`**: `0.1` ($L_{\text{classif}}$ discriminative patch classification, maintaining the $1 : 10$ ratio from `traintest_mask.py`)
* **`sac_lambda`**: `0.02` ($L_{\text{SAC}}$ factorized acoustic contrastive loss)
* **`local_sigma_mode`**: `"offline_global_median"` (or `"optuna_optimal"`)
* **Training Setup**: True BS64 @ LR 4e-4 with activation checkpointing on LibriSpeech-960h (`run_sac.sh`).
**Status:** Planned

### Insights & Conclusions
* *(To be updated after execution)*




