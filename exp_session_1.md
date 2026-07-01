# Experiment Session 1: Factorized SAC Loss Improvements

This document tracks a sequence of four planned experiments designed to improve the Universal SSAMBA pretraining performance. The experiments are ordered from most foundational (fixing loss/data targets) to most complex (architectural capacity and training schedules).

---

## Experiment 1: Learnable Family-Specific Sigmas ($\sigma$)
**Rationale:** Current heuristic median scaling (`chi2_median`, etc.) struggles because the acoustic features are artificially bounded to $[-1, 1]$, violating standard normal distribution assumptions. Manual tuning of $\sigma$ proved beneficial. Making it a learnable parameter allows the network to find the optimal bandwidth per feature family via gradient descent.
**Implementation:** Replace the static `local_sigma` calculation in `sac_model.py` with `F.softplus(self.family_sigmas[k]) + 1e-4` where `family_sigmas` is an `nn.Parameter`.
**Status:** Abandoned

### Insights & Conclusions
* **Abandonment Rationale:** Making $\sigma$ learnable allows the network to bypass learning useful representations by simply scaling $\sigma$ to make the contrastive target distribution ($w_{ij}$) trivially easy to mimic (e.g. uniform distribution by pushing $\sigma \to \infty$). Minimizing the SAC loss with respect to hyperparameters does not guarantee alignment with downstream semantic tasks. We will instead rely on hyperparameter sweeps (like testing $\sigma=2.0$ for the Universal feature set) or statically calibrated sigmas.

---

## Experiment 2: Better Formant Extraction (LPC)
**Rationale:** The current formant proxy (`_compute_formants_proxy`) uses simple spectral centroids within fixed frequency bands. This is highly inaccurate across diverse speaker profiles (e.g., male vs. female pitch shifts). Upgrading to Linear Predictive Coding (LPC) roots provides a mathematically grounded representation of the vocal tract.
**Implementation:** Update `acoustic_features.py` to calculate `f1, f2, f3` using LPC coefficients rather than banded centroids.
**Status:** Implemented

### Insights & Conclusions
* **LPC Implementation:** Implemented a pure PyTorch native Levinson-Durbin recursion and eigenvalue solver to replace the bounded spectral centroid proxy, since `torchaudio.functional.lpc` was unavailable. This method extracts 10 high-energy frames per clip (vowels), computes LPC roots, and averages the formants (F1, F2, F3). It provides a mathematically grounded model of the vocal tract resonances, which should provide a much more accurate target for the Vocal Tract feature family.

---

## Experiment 3: Multi-Query Subspaces
**Rationale:** Each feature family currently uses exactly one query vector in the Cross-Attention block, resulting in a single latent representation. Complex families, such as `Scene` (which mixes Centroid, Flux, and ZCR), likely require a higher-capacity subspace to represent their variance accurately.
**Implementation:** Modify `sac_model.py` to assign $N$ queries (e.g., $N=4$) per active family, generating a `[B, 4, embed_dim]` subspace, which is then pooled prior to computing the SAC loss.
**Status:** Planned

### Insights & Conclusions
* *(To be updated after execution)*

---

## Experiment 4: Curriculum Learning for SAC Families
**Rationale:** Activating all 5 feature families at step 0 might cause "gradient confusion" early in training. Starting with robust, low-level features and gradually introducing complex phonetic features might stabilize the latent space formation.
**Implementation:** Update the pretraining script (`run_amba.py` / `traintest_mask.py`) to dynamically unfreeze or introduce specific `Acoustic Queries` at different training milestones (e.g., Steps 0-20k: Timbre/Scene, Steps 20k-50k: Prosody/Voice Quality, Steps 50k+: Vocal Tract).
**Status:** Planned

### Insights & Conclusions
* *(To be updated after execution)*
