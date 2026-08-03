# Binaural SSAMBA + Factorized SAC: Architectural Adaptation Plan

**Status:** Design proposal, pre-implementation
**Scope:** Extends `AMBAModel` (`both_models.py`) and `SSAMBASACModel` (`sac_model.py`) from single-channel to binaural (2-mic) input, adding a spatial representation stream and a spatial acoustic feature group to Factorized SAC.

---

## 1. Motivation and Constraints Recap

The current system (per `project_context.md`) is a single-channel Mamba encoder (`AMBAModel`) pretrained with a combination of masked-patch reconstruction (`mpg`) and Factorized SAC loss (`SSAMBASACModel.sac_loss`), where acoustic feature groups (Prosody, Vocal Tract, Timbre, Voice Quality, Scene/Noise) are extracted via cross-attention queries and aligned against DSP-derived feature vectors using a CWCL objective.

The goal now is to add a sixth capability — spatial acoustic parameter estimation (TDOA, DRR, T60, DoA) — without disturbing the existing monaural pipeline or the sigma/CWCL machinery that's already been validated. Two hard constraints, established in prior discussion, shape every decision below:

1. **Same input representation for every stream.** Both the "spatial" and "spectral" encoders must consume the *same* complex-STFT tensor. Feeding log-mel to one and complex STFT to the other reintroduces ILD leakage into the "spectral" stream and breaks the shared-reconstruction-target mechanism that makes CCSR-style disentanglement work at all.
2. **No cross-attention between channels.** SSAMBA's value proposition is subquadratic Mamba sequence modeling; introducing inter-channel cross-attention (as some literature does for "neural GCC-PHAT") reopens exactly the quadratic-cost problem the project exists to avoid. Disentanglement must come from **masking**, not attention, and any explicit L/R fusion must be **linear/bilinear**, not attention-based.

Everything that follows is written against the actual classes in `both_models.py` and `sac_model.py`, not a from-scratch design.

---

## 2. Input Representation Change

### 2.1 Current state
`AMBAModel.forward` expects `x = (batch, time_frame_num, freq_bins)`, e.g. `(B, 1024, 128)` — a single-channel log-mel spectrogram — then does `x.unsqueeze(1).transpose(2,3)` to get `[B, 1, F, T]` before `patch_embed`.

### 2.2 Proposed change
Replace the log-mel front end with the complex-STFT, real/imag-concatenated, dual-mic representation used in your own SAC paper (`Learning Self-Supervised Spatial Representations via Soft Acoustic Contrastive Alignment`, Eq. 2):

```
X = [Re(X_L), Im(X_L), Re(X_R), Im(X_R)]  ∈  R^(4 × F × T)
```

- `F, T` become STFT frequency/time bins rather than mel bins — this needs to be picked deliberately (see §7), not inherited from either paper's defaults.
- This tensor **is** the input to both the spatial and spectral encoders (§4) — only the masking applied to it differs, not its content or channel count.
- `in_chans` for `patch_embed` becomes `4` (Candidate 1 / early-fusion from the architectural review) rather than `1`. This keeps the token count identical to the monaural baseline — no Mamba FLOPs increase from tokenization itself.

### 2.3 Code-level touch points
- `AMBAModel.__init__`: `new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, ...)` → `torch.nn.Conv2d(4, self.original_embedding_dim, ...)` (both the pretraining-stage constructor path, line ~143, and the fine-tuning load path, line ~240).
- `AMBAModel.get_shape`: `test_input = torch.randn(1, 1, input_fdim, input_tdim)` → `torch.randn(1, 4, input_fdim, input_tdim)`.
- The fine-tuning weight-transplant trick at line 242 (`new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))`, used when `fstride != p_fshape`) sums over the channel dimension to go from a pretrained multi-channel patch embed to a single-channel one at a new stride. This logic is channel-count-agnostic in principle but was written assuming `in_chans=1` on both ends — needs to be re-audited once `in_chans=4` on both ends, not treated as already-correct.
- Any place upstream that builds `fbank` (the dataloader / feature-extraction pipeline referenced in `project_context.md` §5) needs a parallel STFT-based path; the log-mel extraction code is not reusable as-is since phase must be retained.

---

## 3. Masking Mechanism: Reconciling SSAMBA's MPG with CCSR's Disentanglement Mask

This is the single most important implementation detail, and it's easy to get subtly wrong, so it's worth stating precisely.

### 3.1 What SSAMBA currently does
In `AMBAModel.mpg` / `AMBAModel.mpc`, masking happens **after** `patch_embed`: `x = self.v.patch_embed(input)` first produces `[B, num_patches, embed_dim]`, and *then* `mask_dense` zeroes out selected patch *tokens*, replacing them with a learned `mask_embed` vector. The unmasked "target" used for the loss is the *raw unfolded input patch* (via `self.unfold(input)`), not a token.

### 3.2 What CCSR does (and what disentanglement requires)
In the CCSR paper, masking happens **before** any encoder sees the signal, at the level of raw STFT time frames of a *specific microphone channel*:
- `X̃_spat`: the *same* time-frame mask `W(n,f)` applied identically to **both** channels (Eq. 7) — this denies the spatial encoder any spectral content in the masked region for *either* ear, forcing it to reconstruct that content purely from inter-channel relationships learned from the unmasked frames.
- `X̃_spec`: an *inverse* mask applied to the non-source channel (Eq. 8) — the spectral encoder always sees full content in at least one channel, denying it access to the *simultaneous* dual-channel view needed to extract spatial cues.

This is a fundamentally different masking axis than SSAMBA's: SSAMBA masks *patches* (post-embedding, single-stream); CCSR masks *channels-conditional-on-time-frame* (pre-embedding, two differently-masked copies of the same input feeding two streams).

### 3.3 Proposed reconciliation
Implement channel-conditional masking as a **new preprocessing step** that runs before `patch_embed`, producing two masked tensors from one input:

```python
def apply_ccsr_masking(X, W):
    # X: [B, 4, F, T]  (Re_L, Im_L, Re_R, Im_R)
    # W: [B, 1, 1, T]  binary time-frame mask, 0 = masked
    X_spat = X * W                                  # identical mask, both mics
    X_spec = X.clone()
    X_spec[:, 2:, :, :] = X[:, 2:, :, :] * (1 - W)  # inverse mask on R-channel only
    return X_spat, X_spec
```

`X_spat` and `X_spec` each go through their *own* `patch_embed` (both `in_chans=4`, per §2) and their own Mamba stack (§4). SSAMBA's existing `mask_embed`-token mechanism, `gen_maskid_patch`/`gen_maskid_frame`, and the post-embedding masking in `mpg`/`mpc` are **not** the mechanism doing disentanglement here — they'd need to be either removed from this path or repurposed purely as an auxiliary MSPM objective layered on top (optional; see §5.3).

This also means the reconstruction target changes from "masked patches of the log-mel input" to "masked time-frames of the complex STFT," and the decoder needs to reconstruct across **both** channels' real/imag components — closer to CCSR's Eq. 6 loss than SSAMBA's current `mpg`.

---

## 4. Dual-Encoder Architecture

### 4.1 New class: `BinauralAMBAEncoder`
Rather than modifying `AMBAModel` in place, wrap two instances of it (or two `VisionMamba` stacks sharing the modified `patch_embed`/`in_chans=4` config from §2):

```python
class BinauralAMBAEncoder(nn.Module):
    def __init__(self, shared_config, share_weights=False):
        super().__init__()
        self.spatial_encoder = AMBAModel(**shared_config)   # sees X_spat
        self.spectral_encoder = (
            self.spatial_encoder if share_weights
            else AMBAModel(**shared_config)                  # sees X_spec
        )
```

- **Independent weights (default recommendation):** matches CCSR's own setup (two separate encoders), and is the configuration their Table VII ablation validates as achieving the disentanglement. Doubles encoder parameter count and pretraining compute relative to the current monaural `AMBAModel`, roughly matching CCSR's own cost profile — not more, since we're *not* additionally duplicating per-ear (Candidate 3's cost), we're splitting by role.
- **Shared weights (cheaper option, flagged as riskier):** halves parameter count but forces one set of weights to serve two different masking regimes; not something to default to without an ablation showing it doesn't collapse the spatial/spectral distinction. Worth trying only *after* the independent-weight version's disentanglement is confirmed to hold (analogous to your existing Table VII-style check), as a compute-reduction follow-up, not a first attempt.

### 4.2 Forward pass sketch

```python
def forward(self, X, W, mask_patch, cluster):
    X_spat, X_spec = apply_ccsr_masking(X, W)
    h_spat = self.spatial_encoder._encode_with_mamba(X_spat)   # [B, N+cls, D]
    h_spec = self.spectral_encoder._encode_with_mamba(X_spec)  # [B, N+cls, D]
    return h_spat, h_spec
```

`_encode_with_mamba` already exists in `SSAMBASACModel` (not `AMBAModel` itself) — it's the method that runs patch-embed → cls-token → pos-embed → `_forward_mamba_layers` → norm, *without* doing any masking or loss computation. That method is effectively already the right primitive to reuse per-stream; it just currently assumes a single `self.encoder`.

### 4.3 Shared reconstruction decoder
Per CCSR Fig. 1/Eq. structure and your SAC paper's Fig. 1: concatenate `h_spat` and `h_spec` along the embedding dimension, feed to a small FC decoder (2 layers, matching CCSR's decoder in Doc 2 Fig. 3b) that reconstructs the real/imag STFT coefficients for the masked time-frames of the **first** (masked) channel only — this mirrors CCSR's Eq. 6, where the loss is `(1 - W(n,f))`-weighted MSE over just the masked region of one designated channel, not both. This decoder replaces `AMBAModel.mpg` for the binaural path; `mpg`/`mpc` remain available for monaural-only training runs so nothing breaks for existing single-channel experiments.

---

## 5. SAC Loss Integration

### 5.1 Where the SAC head attaches
Attach the existing `projection_head` / cross-attention `group_queries` mechanism **only to `h_spat`** (spatial encoder output), unchanged from its current form in `SSAMBASACModel`. The spectral encoder's output does not get a SAC head at all in the binaural path — its only training signal is its contribution to the shared reconstruction loss (§4.3), exactly matching CCSR's finding (Doc 2, Table VII) that the spectral encoder shouldn't be pushed to encode spatial information.

### 5.2 New feature group: Spatial
Extend `get_feature_groups` (`sac/acoustic_features.py`, not shown but referenced in `sac_model.py` line 117) with a sixth group, mirroring the existing Prosody/Vocal-Tract/etc. structure:

```
Spatial group ← [TDOA, GCC-PHAT peak magnitude, Coh_low, Coh_mid, Coh_high]
```

This is exactly the 5-feature vector from your own SAC paper (§2.4, Eq. 12) — reuse it verbatim rather than inventing a new one, since it's already validated against the five downstream spatial metrics (TDOA, T60, DRR, C50, ABS) in that paper. The review doc's proposal to add ITD/ILD as *additional* separate features is not necessary at this stage: TDOA already **is** essentially ITD (both are the GCC-PHAT-argmax delay), and adding ILD would reintroduce the same level-difference cue that §2 of the prior discussion flagged as a leakage risk if it ever ends up reachable by the spectral stream — keep it out of scope unless a specific downstream task shows the existing 5 features are insufficient.

### 5.3 Sigma/CWCL implications — direct interaction with `sigma_configs.py`
This is the part most likely to silently break something in your existing pipeline. `SSAMBASACModel.sac_loss` looks up `local_sigma` per group by name, with several modes (`chi2_median`, `optuna_optimal`, `static_entropy_optimal`, etc.), and two of those modes (`optuna_optimal`, `static_entropy_optimal`) key into hardcoded tables (`OPTIMAL_SIGMAS`, `OPTUNA_OPTIMAL_SIGMAS`) that were tuned for the *current five* feature groups at specific batch sizes. Adding a sixth "Spatial" group means:

- `chi2_median` mode is dimension-derived (`chi2_medians.get(D, D - 2/3)` keyed only on feature-vector length `D`), so it degrades gracefully to a reasonable extrapolated value for a new group size — this is the safest mode to use for the first Spatial-group experiments.
- `optuna_optimal` / `static_entropy_optimal` will silently fall back to `self.sac_sigma * math.sqrt(len(indices))` for the unseen `"Spatial"` group name (via the `.get(group_name, ...)` default), which is *not* a tuned value — it'll run without erroring, which is the dangerous case. This connects directly to the open problem in your `[[sac-hyperparameter-selection]]` work: before trusting any Spatial-group result under those two modes, the sigma selection needs to be re-run for the new group rather than assumed to transfer, exactly as flagged when this was discussed for the general σ/τ pipeline.
- Recommendation: run initial binaural experiments with `local_sigma_mode='chi2_median'` specifically *because* it's the mode that doesn't depend on a stale lookup table, then revisit `optuna_optimal` only after generating fresh entries for the six-group configuration.

### 5.4 `SSAMBASACModel` → `BinauralSSAMBASACModel`
Concretely, this is a new class rather than a modified one (to keep the monaural path untouched):

```python
class BinauralSSAMBASACModel(nn.Module):
    def __init__(self, ..., share_weights=False):
        super().__init__()
        self.dual_encoder = BinauralAMBAEncoder(shared_config, share_weights)
        self.decoder = nn.Sequential(...)          # §4.3
        self.group_queries = ...                    # unchanged, attaches to h_spat only
        self.cross_attention = ...                  # unchanged
        self.projection_head = ...                  # unchanged

    def forward(self, X, W, acoustic_features, ...):
        h_spat, h_spec = self.dual_encoder(X, W, ...)
        loss_recon = self._reconstruct(h_spat, h_spec, X, W)       # §4.3
        Z_groups = self._sac_head(h_spat)                          # §5.1, unchanged internals
        loss_sac = self.sac_loss(Z_groups, acoustic_features)      # unchanged
        loss_total = self.recon_lambda * loss_recon + self.sac_lambda * loss_sac
        return {...}
```

Everything downstream of `Z_groups` — `sac_loss`, the `chi2_median` etc. branching, the entropy diagnostics — is reused **verbatim**. This is deliberate: the CWCL math is architecture-agnostic (it operates on whatever `z̃_i` and `c_i` it's given), so the binaural extension should touch the encoder/masking/decoder path and leave `sac_loss` itself alone.

---

## 6. Downstream Fusion (No Attention)

For localization/DoA fine-tuning heads that need an explicit cross-ear comparison, use a linear/bilinear fusion on the two spatial-encoder pooled outputs rather than cross-attention:

```python
h_fused = self.fusion_linear(torch.cat([e_L, e_R, e_L * e_R, torch.abs(e_L - e_R)], dim=-1))
```

where `e_L`, `e_R` are per-ear spatial-encoder embeddings obtained by running the (frozen, fine-tuned) spatial encoder twice at inference/fine-tune time — once per ear — even though pretraining only ever saw the two-mic input jointly. This keeps the fine-tuning-time cost close to today's single-encoder fine-tuning path (`finetuningavgtok`, `finetuningcls` in `AMBAModel`), just called twice plus one small linear layer, and avoids adding attention anywhere in the model.

Monaural downstream tasks (VoxCeleb SID, IEMOCAP ER — the tasks in `project_context.md` §6) should read from the **spectral** encoder only, exactly as today's single-stream fine-tuning does, since that's the stream CCSR's Table VII shows retains content information without spatial contamination.

---

## 7. Open Parameters Needing a Decision

| Parameter | Current (monaural) | Binaural proposal | Notes |
|---|---|---|---|
| Input feature | 128-bin log-mel, 1024 frames | Complex STFT, F×T TBD | Needs its own pick, not inherited from SAC paper's F=256,T=256 or SSAMBA's F=128,T=1024 |
| `in_chans` | 1 | 4 | Real/imag × L/R, early-fused |
| Patch grid / token count | 512 | Depends on F,T pick above | Recompute via `get_shape` once F,T is fixed |
| Encoder count | 1 | 2 (spatial + spectral) | Independent weights recommended initially |
| Masking axis | post-patch-embed, token-level | pre-patch-embed, channel-conditional time-frame | New mechanism, doesn't replace MPG's token-masking utilities, sits before them |
| SAC feature groups | 5 | 6 (+ Spatial) | Reuse existing 5-feature spatial vector from your SAC paper verbatim |
| `local_sigma_mode` for first experiments | `chi2_median` (per project defaults) | `chi2_median` | Avoids stale lookup-table fallback for the new group |
| Fusion mechanism | n/a | linear/bilinear (concat, product, abs-diff) | No attention, preserves subquadratic cost |

---

## 8. Suggested Validation Order

1. **Disentanglement check first, SAC second.** Before attaching any SAC loss, verify the dual-encoder + channel-conditional-masking reconstruction pipeline reproduces CCSR's own Table VII pattern (spectral-only downstream fine-tune ≈ training-from-scratch; spatial-only ≈ full performance) using plain reconstruction loss. This isolates masking/architecture bugs from SAC-specific ones.
2. **Then attach SAC to the spatial stream** with the existing 5-feature spatial vector and `chi2_median` sigma, and confirm it doesn't regress the reconstruction-only disentanglement result from step 1.
3. **Only then** experiment with `optuna_optimal`/entropy-based sigma modes for the 6-group configuration, after generating fresh calibration data for the Spatial group specifically.
4. **Fusion layer and downstream localization heads** last, once the pretrained representation itself is validated — this keeps the fusion-layer choice (or a future attention-based one, if linear fusion underperforms) decoupled from pretraining validity.

---

## 9. Summary of Class/File-Level Changes

- `both_models.py`: `AMBAModel.__init__` and fine-tune-load path — change `in_chans` 1→4 in both `patch_embed.proj` constructions; `get_shape` test tensor updated to match; audit the channel-summing weight-transplant logic at line ~242 for the new channel count.
- New: `BinauralAMBAEncoder` — thin wrapper holding two `AMBAModel` instances, exposes `apply_ccsr_masking` + dual forward.
- New: shared FC reconstruction decoder (§4.3), replacing `mpg` for the binaural path only; `mpg`/`mpc` untouched for monaural runs.
- `sac_model.py`: new `BinauralSSAMBASACModel` (parallel to, not replacing, `SSAMBASACModel`), reusing `sac_loss`/`sac_loss_legacy` unchanged; SAC head attaches to spatial-stream output only.
- `sac/acoustic_features.py`: add `"Spatial"` group with the 5-feature TDOA/GCC-PHAT/coherence vector.
- `sac/sigma_configs.py`: no changes required to use `chi2_median` immediately; `OPTIMAL_SIGMAS`/`OPTUNA_OPTIMAL_SIGMAS` need new entries before those modes are trustworthy for the 6-group case.
