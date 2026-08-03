# Comprehensive Architectural Review: Extending SSAMBA & Factorized SAC to Binaural (2-Channel) Audio

> **Document Note for External LLMs (Gemini, ChatGPT, Claude)**: This is a standalone technical review document describing the **SSAMBA** (Self-Supervised Audio Mamba) architecture and **Factorized SAC** (Soft Acoustic Contrastive) loss framework. It presents candidate architectural approaches for extending the model from single-channel (monaural) to two-channel (binaural Left/Right) audio, with a specific focus on **Spatial Audio Tasks** (e.g., Speaker Localization / Sound Source Localization / Direction of Arrival estimation), **Complex STFT vs. Log-Mel Inputs**, and multi-channel input handling inspired by **MC-Conformer (SAR-SSL)** (`sig_shape=(nf, nt, nreim, nmic)`). Use this document to discuss pros, cons, trade-offs, and implementation strategies with the user.

---

## 1. Executive Summary & Objective

The objective of this project is to extend **SSAMBA**—a state-of-the-art self-supervised audio foundation model based on State Space Models (Mamba)—to process **binaural (2-channel, Left/Right) audio recordings**. 

A key focus of this extension is supporting **Spatial Audio Tasks** such as:
1. **Speaker / Sound Source Localization (DoA - Direction of Arrival Estimation)**: Estimating the azimuth and elevation angles of active speech sources.
2. **Sound Event Localization and Detection (SELD)**: Jointly detecting sound classes and tracking their spatial trajectories.
3. **Binaural Speaker Separation & Spatial Scene Classification**.

To learn representations effective for these spatial tasks during self-supervised pretraining, the model must leverage physical spatial features:
* **GCC-PHAT (Generalized Cross-Correlation with Phase Transform)**: The gold-standard DSP feature for time-delay of arrival (TDOA) and speaker localization. It computes frequency-domain cross-power spectrum phase correlations between channels.
* **Interaural Coherence (IC)**: Normalized cross-correlation peak measuring soundfield diffuseness, direct-to-reverberant ratio, and room geometry.
* **Interaural Time Difference (ITD)**: Time/phase arrival delay between Left and Right ears ($\tau^* = \arg\max_\tau R_{LR}(\tau)$).
* **Interaural Level Difference (ILD)**: Log-energy ratio ($10 \log_{10}(E_L / E_R)$) caused by head shadowing.

This document reviews **candidate architectural approaches** for binaural extension, incorporating key insights from multi-channel signal parameterization in **MC-Conformer (SAR-SSL)** (`sig_shape=(nf, nt, nreim, nmic)`), and evaluates how each model fits spatial localization tasks and integrates with GCC-PHAT / Spatial SAC loss.

---

## 2. Baseline Architecture Overview (SSAMBA + Factorized SAC)

To understand how binaural audio impacts the system, the baseline monaural pipeline operates as follows:

```
[1-Channel Audio (10s @ 16kHz)]
              │
              ▼
 [Log-Mel Spectrogram Extraction]
   Shape: [B, 1024 frames, 128 mel bins]
              │
              ▼
   [2D Patch Projection Conv2d]
   Kernel: (16, 16), Stride: (16, 16), In_Chans: 1
   Output: N = (128/16) × (1024/16) = 8 × 64 = 512 Patch Tokens (Dim D = 768)
              │
              ▼
   [24-Layer Bidirectional VisionMamba Encoder]
   Sequence Length: N = 512 tokens + 1 CLS token
              │
      ┌───────┴────────────────────────────────┐
      ▼                                        ▼
[Masked Patch Reconstruction (MPG)]     [Factorized SAC Loss]
 Generative MSE Loss on raw patches      Cross-Attention Bottleneck Queries
 Predicts [B, 400 masked, 256 pixels]   (F0, Formants, MFCCs, HNR, ZCR)
```

### Key Dimensions & Parameters
* **Audio Clip**: 10 seconds at 16,000 Hz sample rate.
* **Log-Mel Spectrogram**: 128 frequency bins, 1024 time frames.
* **Patch Size**: $16 \times 16$ time-frequency tiles $\rightarrow$ $256$ pixels per patch.
* **Sequence Tokens**: $N = 512$ patch tokens.
* **Latent Dimension**: $D = 768$ (Base Mamba model).

---

## 3. Multi-Channel Signal Handling Inspired by MC-Conformer (SAR-SSL)

The **MC-Conformer** architecture (from SAR-SSL) parameterizes multi-channel input audio signals as:
$$\text{sig\_shape} = (n_f, n_t, n_{\text{reim}}, n_{\text{mic}})$$

Where:
* $n_f$: Number of frequency bins (e.g., 256 or 128)
* $n_t$: Number of time frames (e.g., 256 or 1024)
* $n_{\text{reim}}$: 2 (Real and Imaginary components of complex STFT) or 1 (Magnitude / Log-Mel)
* $n_{\text{mic}}$: 2 (Number of channels / microphones, e.g., Left & Right binaural ears)

Total input channels:
$$C_{\text{in}} = n_{\text{reim}} \times n_{\text{mic}}$$

### Application to SSAMBA Input Projection
Adapting this multi-channel input structure allows SSAMBA to support two input feature modes seamlessly:

#### 1. Real Log-Mel Spectrograms ($n_{\text{reim}}=1, n_{\text{mic}}=2 \implies C_{\text{in}}=2$)
- **Tensor Shape**: `[B, 2, F, T]` (Left and Right Log-Mel spectrograms).
- **Physical Information**: Captures spectral energy distribution and Interaural Level Differences (ILD), but discards phase.

#### 2. Complex STFT ($n_{\text{reim}}=2, n_{\text{mic}}=2 \implies C_{\text{in}}=4$)
- **Tensor Shape**: `[B, 4, F, T]` representing $(\text{Real}_L, \text{Imag}_L, \text{Real}_R, \text{Imag}_R)$.
- **Physical Information & Benefit for Speaker Localization**: Complex STFT directly retains raw inter-channel phase differences ($\Delta \phi$). Setting `in_chans = nreim * nmic = 4` in SSAMBA's initial `Conv2d(4, 768, kernel_size=(16,16))` projection allows the network to learn fine interaural time delays (ITD) without requiring external DSP pre-computation.

---

## 4. Review of Candidate Binaural Architectural Approaches

---

### Candidate 1: Early Channel Fusion (`in_chans = nreim * nmic`)

#### Architectural Description
In Early Channel Fusion, multi-channel inputs are concatenated along the channel dimension. The initial patch projection convolution in SSAMBA is updated to `Conv2d(in_chans, 768, ...)` where `in_chans = nreim * nmic`.
* **Log-Mel**: `in_chans = 1 * 2 = 2` channels (`[B, 2, F, T]`).
* **Complex STFT**: `in_chans = 2 * 2 = 4` channels (`[B, 4, F, T]`).

```
Left & Right Audio (STFT/Mel) [B, nreim * nmic, F, T]
                           │
                           ▼
         Conv2d(in_chans=4, embed_dim=768, kernel_size=(16,16))
                           │
                           ▼
          [B, 512 Patch Tokens, 768 Dim]
                           │
                           ▼
         [24-Layer Mamba Encoder (Unchanged)]
```

#### Technical & Mathematical Details
1. **Input Shape**: `[B, C_in, F, T]` where $C_{\text{in}} = n_{\text{reim}} \times n_{\text{mic}}$.
2. **Patch Token Count**: $N = 512$ (Identical to monaural baseline).
3. **Reconstruction Target (`mpg`)**: Maps $768 \rightarrow C_{\text{in}} \times 16 \times 16$ patch values.
4. **Computational Complexity**: 
   - Mamba Encoder FLOPs: **0% increase** ($O(N)$ over 512 tokens).

#### Suitability for Speaker Localization & GCC-PHAT
- **Pros**: Zero increase in sequence length or Mamba FLOPs; minimal refactoring ($\approx 20$ lines of code).
- **Limitations for Speaker Localization**: The input convolution immediately linearly mixes Left and Right channels at Layer 0 ($W_L X_L + W_R X_R$). While Complex STFT ($C_{\text{in}}=4$) mitigates phase loss, separate channel streams are combined into a single token sequence at the input layer.

---

### Candidate 2: Spectrogram Stacking (Frequency or Time Dimension Concatenation)

#### Architectural Description
Left and Right spectrograms are concatenated along the Frequency dimension ($128 + 128 = 256$ Mel bins). The model uses `in_chans=1`.

#### Technical & Mathematical Details
1. **Input Shape**: `[B, 1, 1024, 256]`.
2. **Patch Token Count**: $N = 1024$ tokens (**2x sequence length**).
3. **Computational Complexity**: Mamba Encoder FLOPs: **2x increase**.

---

### Candidate 3: Dual-Stream Siamese Encoder with Spatial Cross-Attention Fusion (Recommended for Speaker Localization)

#### Architectural Description
Left ($L$) and Right ($R$) signals are processed by **two parallel Mamba encoder streams** (sharing weights in a Siamese setup or independent). Each stream outputs 512 patch tokens ($H_L, H_R \in \mathbb{R}^{B \times 512 \times 768}$). 

To optimize this architecture specifically for **Speaker Localization** and **GCC-PHAT Spatial SAC**, the outputs are fused using an **Inter-Stream Cross-Attention / Bilinear Correlation Fusion Layer** prior to the SAC projection head.

```
Left  Spectrogram [B, 1, 1024, 128] ──> Mamba Encoder L ──> H_L [B, 512, 768] ──┐
                                                                                 ├──> Spatial Cross-Attn / Bilinear Fusion ──> H_Binaural
Right Spectrogram [B, 1, 1024, 128] ──> Mamba Encoder R ──> H_R [B, 512, 768] ──┘     (Cross-Correlation Neural GCC-PHAT)
```

#### Technical & Mathematical Details
1. **Parallel Encoders**:
   - $H_L = \text{MambaEncoder}(X_L) \in \mathbb{R}^{B \times 512 \times 768}$
   - $H_R = \text{MambaEncoder}(X_R) \in \mathbb{R}^{B \times 512 \times 768}$
2. **Spatial Cross-Attention Fusion Layer (Neural GCC-PHAT)**:
   $$H_{\text{cross}} = \text{MultiHeadAttention}(Q=H_L, K=H_R, V=H_R)$$
   Querying $H_L$ against key/value $H_R$ performs a high-dimensional cross-correlation across time/frequency patches—**functionally equivalent to a learnable GCC-PHAT operation**.
3. **Bilinear Correlation Option**:
   $$H_{\text{binaural}} = \text{Linear}([H_L; H_R; H_L \odot H_R; |H_L - H_R|])$$
   Explicitly provides difference ($|H_L - H_R|$, capturing ILD) and element-wise product ($H_L \odot H_R$, capturing phase alignment).

#### Suitability for Speaker Localization & GCC-PHAT
- **Highly Suitable**:
  1. **Preserves Channel Independence**: $H_L$ and $H_R$ maintain unpolluted ear-specific acoustic features in separate streams until spatial fusion.
  2. **Direct Mapping to GCC-PHAT & Spatial SAC**: The spatial cross-attention / bilinear fusion output $H_{\text{binaural}}$ aligns naturally with GCC-PHAT feature vectors in Factorized SAC.
  3. **Dual Downstream Compatibility**: Downstream speaker localization tasks evaluate $H_{\text{binaural}}$, while standard monaural tasks (e.g., VoxCeleb1 speaker ID) evaluate $H_L$ or $H_R$ independently.

---

### Candidate 4: Mid-Fusion with Inter-Channel Cross-Attention Blocks

#### Architectural Description
Left and Right channel patches are processed in two parallel token streams ($N_L = 512, N_R = 512$). At every $k$-th Mamba layer (e.g., layers 6, 12, 18, 24), an **Inter-Channel Cross-Attention block** is inserted, allowing continuous interaural information exchange at intermediate depth.

```
Left  Stream ──> Mamba Block ──> [ Cross-Attn: Q=L, K=R, V=R ] ──> Mamba Block ──> H_L
                                         ▲             │
                                         │             ▼
Right Stream ──> Mamba Block ──> [ Cross-Attn: Q=R, K=L, V=L ] ──> Mamba Block ──> H_R
```

---

## 5. Integration with Factorized SAC Loss (Spatial SAC & GCC-PHAT Extension)

The primary novelty of this codebase is **Factorized SAC (Soft Acoustic Contrastive) Loss**, which uses learnable "Acoustic Queries" in a `MultiheadAttention` layer to route Mamba hidden states into separate semantic sub-spaces ($F_0$, Formants, MFCCs, HNR, ZCR).

### Extending SAC for Speaker Localization with GCC-PHAT
To train the network specifically for spatial audio tasks, we expand Factorized SAC by defining a **Spatial / Binaural Feature Query Group**:

1. **Extract Physical Spatial Feature Vectors ($c_{\text{spatial}}$)**:
   - **GCC-PHAT Vector ($c_{\text{GCC-PHAT}}$)**: Compute GCC-PHAT cross-correlation sequence across $M$ delay lags:
     $$R_{LR}^{\text{PHAT}}(\tau) = \mathcal{F}^{-1} \left( \frac{X_L(f) X_R^*(f)}{|X_L(f) X_R^*(f)| + \epsilon} \right), \quad \tau \in [-\tau_{\max}, \tau_{\max}]$$
   - **Interaural Time Difference ($c_{\text{ITD}}$)**: Peak delay index $\tau^* = \arg\max_\tau R_{LR}^{\text{PHAT}}(\tau)$.
   - **Interaural Level Difference ($c_{\text{ILD}}$)**: Sub-band log-energy ratios $10 \log_{10}(E_{L,b} / E_{R,b})$.
   - **Interaural Coherence ($c_{\text{IC}}$)**: Maximum cross-correlation value $\max_\tau R_{LR}(\tau)$.

2. **Acoustic Group Router**:
   Add the 6th feature group query $\mathbf{q}_{\text{spatial}}$ to `SSAMBASACModel`:
   $$\text{Groups} = \{\text{Prosody}, \text{Vocal Tract}, \text{Timbre}, \text{Voice Quality}, \text{Scene/Noise}, \mathbf{\text{Spatial / GCC-PHAT}}\}$$

3. **Continuous Gaussian Distance Kernel ($w_{ij}^{\text{spatial}}$)**:
   $$w_{ij}^{\text{spatial}} = \exp\left( - \left( \frac{\| c_{i,\text{spatial}} - c_{j,\text{spatial}} \|}{\sigma_{\text{spatial}}} \right)^2 \right)$$

This forces the Mamba latent space to organize clips according to spatial directional similarity (GCC-PHAT / ITD / ILD), providing an ideal self-supervised pretraining signal for downstream **Speaker Localization (DoA)** and **SELD** tasks.

---

## 6. Comparative Trade-Off Matrix (Pros & Cons)

The table below evaluates all candidate approaches across computational, architectural, and spatial localization criteria:

| Evaluation Criterion | Approach 1: Early Fusion (`in_chans=nreim*nmic`) | Approach 2: Spectrogram Stacking (Freq Concatenation) | Approach 3: Dual-Stream Siamese Encoder | Approach 4: Mid-Fusion Cross-Attention |
| :--- | :--- | :--- | :--- | :--- |
| **Compute Complexity (FLOPs)** | **Baseline ($1\times$)** | $2\times$ (due to $N=1024$) | $2\times$ (two encoder passes) | $2.3\times$ (encoder + cross-attn) |
| **Memory Footprint (VRAM)** | **Baseline ($1\times$)** | $\approx 1.8\times$ | $\approx 2.0\times$ | $\approx 2.2\times$ |
| **Sequence Token Count** | **512 tokens** | 1024 tokens | $512 + 512$ tokens | $512 + 512$ tokens |
| **Input Feature Support** | Log-Mel ($C_{in}=2$) or Complex STFT ($C_{in}=4$) | Log-Mel only | **Log-Mel ($C_{in}=2$) or Complex STFT ($C_{in}=4$)** | Log-Mel or Complex STFT |
| **Suitability for Speaker Localization (DoA / SELD)** | Moderate (channel mixing at layer 0) | Low | **High** (explicit ear streams + neural GCC-PHAT cross-attention) | **Highest** (continuous multi-depth phase cross-attention) |
| **GCC-PHAT & Spatial SAC Synergy** | Indirect | Indirect | **Direct** (cross-attention fusion matches GCC-PHAT operation) | **Direct** (intermediate layers align with GCC-PHAT) |
| **Downstream Transfer to Mono Tasks** | Requires averaging input weights | Requires spectrogram reshaping | **Native & Seamless** (evaluate single encoder stream) | Requires dummy zero-channel |
| **Codebase Refactoring Effort** | **Minimal** ($\approx 20-30$ lines) | Low | Moderate (new wrapper class + fusion head) | High |

---

## 7. Strategic Recommendation & Thesis Supervisor Pitch

### Position for Speaker Localization & Spatial Audio Tasks (Recommended: Dual-Stream + MC-Conformer Channel Input)
* **Recommendation**: Combine **Dual-Stream Siamese Mamba (Approach 3)** with **MC-Conformer `sig_shape` parameterization** ($C_{\text{in}} = n_{\text{reim}} \times n_{\text{mic}}$).
* **Pitch to Supervisor**:
  > *"Drawing inspiration from MC-Conformer (`sig_shape=(nf, nt, nreim, nmic)`), we adapt SSAMBA for multi-channel spatial audio. By passing complex STFT ($n_{\text{reim}}=2, n_{\text{mic}}=2 \implies C_{\text{in}}=4$) or dual-channel log-Mel into Siamese Mamba streams and combining them with a Spatial Cross-Attention fusion layer, the network explicitly models interaural phase cross-correlations (Neural GCC-PHAT) for downstream speaker localization while integrating seamlessly with Factorized Spatial SAC pretraining."*

---

## 8. Direct Implementation Guide (For Developers & LLMs)

For step-by-step code implementation details and line-by-line modifications for Early Fusion (`in_chans=nreim*nmic`), refer to:
* [binaural_audio_implementation_spec.md](file:///storage/yotam/ssamba/docs/binaural_audio_implementation_spec.md)
