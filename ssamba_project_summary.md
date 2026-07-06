# SSAMBA / SAC Architecture Overview & Progress Report

## 1. The Basics: SSAMBA Architecture
The foundation of this project is **SSAMBA (Self-Supervised Audio Mamba)**. Traditional self-supervised audio models (like SSAST) rely on Transformers, which suffer from quadratic computational and memory complexity as audio sequences get longer. 

SSAMBA solves this by utilizing a deep, bidirectional **Mamba (State Space Model)** encoder. 
- **Input:** Single-channel audio waveforms are converted into 128-dimensional log-Mel spectrograms and split into 16x16 patches.
- **Processing:** The Mamba encoder processes these patches sequentially, capturing the global context of the audio in linear time (subquadratic complexity). 
- **Standard Pretraining:** The base model learns by masking portions of the spectrogram and using a reconstruction head to predict the missing patches (Mean Squared Error).

## 2. The SAC Model (Soft Acoustic Contrastive Loss)
The core novelty of this research is the **Soft Acoustic Contrastive (SAC) Loss**. Standard contrastive learning (like InfoNCE) relies on discrete positive/negative audio pairs, which fails to capture the continuous, nuanced nature of audio physics.

SAC replaces this discrete approach by comparing the latent representations of audio patches using continuous similarity weights. These weights are computed via a Gaussian kernel applied to the $L_2$ distance of specific, mathematically derived acoustic features. If two audio clips have physically similar acoustic properties, the SAC loss forces their high-dimensional latent vectors to be closer together.

## 3. Chosen Acoustic Features
To ground the model in real-world physics, we extract several classical DSP features, grouped into **Feature Groups**:
- **Prosody:** Fundamental Frequency ($F_0$ mean & variance), Rhythm.
- **Vocal Tract:** Formant frequencies ($F_1, F_2, F_3$), computed rigorously via Linear Predictive Coding (LPC).
- **Timbre:** Early Mel-Frequency Cepstral Coefficients (MFCCs).
- **Voice Quality:** Harmonics-to-Noise Ratio (HNR).
- **Scene / Noise Profile:** Spectral Centroid, Spectral Flux, Zero-Crossing Rate (ZCR).

## 4. "Dedicated" vs. "Universal" Models
During development, we explored two distinct ways to train the model:
- **Dedicated Models:** These models were trained with only a small subset of features targeted toward a specific downstream task (e.g., using only Formants and MFCCs for Speaker ID). While effective for their specific niche, they acted as extreme bottlenecks. By forcing the entire global latent vector to align with just one feature group, the model suppressed other useful information (like pitch or temporal dynamics).
- **Universal Models:** The current state-of-the-art for our project. A Universal model trains on *all* acoustic feature groups simultaneously to create a holistic, richly disentangled representation of the audio.

## 5. The Cross-Attention Layer 
Training a Universal model initially suffered from **"$L_2$ distance dilution"** (gradient interference). If you force a single vector to represent both pitch (Prosody) and room noise (Scene), the gradients clash because those physical properties are uncorrelated.

- **The Solution:** We introduced an **Acoustic Group Cross-Attention** bottleneck.
- **The Effect:** We use learnable "Acoustic Queries" in a PyTorch `MultiheadAttention` layer. Instead of one global vector, specific queries attend to the Mamba sequence to extract distinct, independent latent sub-spaces for each Feature Group. 
- **Recent Upgrade:** We recently assigned $N=4$ queries per group, mean-pooling them after the cross-attention block. This dramatically increased the model's capacity to represent complex, multi-dimensional groups (like Scene) without expanding the size of the loss calculation.

## 6. Downstream Tasks & Results
We evaluate the raw Mamba encoder representations (discarding the pretraining heads) on several downstream tasks via the S3PRL framework:
- **Speaker Identification (VoxCeleb1):** Early Dedicated models successfully outperformed baseline test metrics (`0.614` vs `0.592`).
- **Emotion Recognition (IEMOCAP):** Dedicated emotion models also outperformed baselines (`0.605` vs `0.565`).
- **Keyword Spotting (Speech Commands v2):** Dedicated models underperformed the baseline (~97.3% vs 97.7%) because the bottleneck suppressed vital phoneme-level phonetic cues. This directly motivated the switch to the high-capacity Universal Multi-Query model.
- **Vocal Tract Validation:** By extracting formants via LPC, we verified the model's physical grounding. Attention maps show precise localization on high-energy vowel segments, and Representational Similarity Analysis (RSA) proved a highly significant structural alignment between physical formants and the latent space (Spearman $\rho \approx 0.50$, $p \approx 0$).

## 7. Future Experiments
1. **Curriculum Learning:** Instead of activating all 5 feature groups at Step 0 (which can cause gradient confusion), we plan to gradually introduce complex phonetic features (like Vocal Tract) only after the model has learned robust, low-level features (like Scene/Timbre).
2. **Loss Function Ablation:** Insights from previous dual-stream spatial SSL projects suggest the SAC loss is powerful enough to stand alone. We will test running pretraining entirely without the Mamba reconstruction loss (dropping the decoder) to isolate the pure effect of the physical contrastive priors.
