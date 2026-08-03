# Comprehensive Implementation Guide: Binaural SSAMBA + Disentangled SAC (CCSR-Style Pretraining)

> **Document Purpose**: This is a self-contained, line-by-line technical implementation blueprint for extending the **SSAMBA** Mamba encoder and **Factorized SAC** framework to support 2-channel binaural audio. It details the exact code modifications, dataset/dataloader additions, pre-embedding channel-conditional masking, spatial acoustic feature extraction, and downstream task routing. An agent or developer can follow this guide to implement the entire binaural extension without making architectural errors.
>
> **Core References**:
> - [Binaural SSAMBA Adaptation Plan.md](file:///storage/yotam/ssamba/docs/Binaural%20SSAMBA%20Adaptation%20Plan.md)
> - [binaural_architectural_approaches_review.md](file:///storage/yotam/ssamba/docs/binaural_architectural_approaches_review.md)

---

## 1. High-Level Architecture & Signal Flow

```
                     Input Audio (2-Channel Binaural WAV, 10s @ 16kHz)
                                           │
                                           ▼
             [Complex STFT Extraction] -> X ∈ R^(4 × F × T)  (Re_L, Im_L, Re_R, Im_R)
                                           │
                                           ▼
             [CCSR Channel-Conditional Masking (apply_ccsr_masking)]
                ├──> X_spat = X * W                              (Identical Time-Frame Mask W on both mics)
                └──> X_spec = X with Right-mic inverse mask     (Inverse Time-Frame Mask 1-W on Right mic)
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                ▼                                                     ▼
     [Spatial Mamba Encoder]                               [Spectral Mamba Encoder]
     Conv2d(in_chans=4, 768, ...)                          Conv2d(in_chans=4, 768, ...)
     Output: h_spat ∈ R^(B × N × D)                         Output: h_spec ∈ R^(B × N × D)
                │                                                     │
                ├─────────────────────────────────────────────────────┤
                ▼                                                     ▼
     [Factorized SAC Projection Head]                     [Shared FC Reconstruction Decoder]
     Attaches ONLY to h_spat                              Concat(h_spat, h_spec) -> R^(B × N × 2D)
     Groups: Prosody, VocalTract, Timbre,                 Predicts Re_L, Im_L for masked time frames
     VoiceQuality, SceneNoise, SPATIAL                    MSE Loss over masked region of Channel 1
```

---

## 2. Critical Fallpoints & Edge Cases (MUST READ BEFORE IMPLEMENTING)

Before writing code, be aware of these **5 common implementation mistakes**:

1. **Fallpoint 1: Confusing Pre-Embedding Channel Masking with Post-Embedding Patch Masking**
   - *Mistake*: Trying to use SSAMBA's existing post-embedding patch token masking (`mask_dense`, `mask_tokens`) for spatial disentanglement.
   - *Fix*: Disentanglement requires **pre-embedding time-frame channel masking** (`apply_ccsr_masking`) on the raw input STFT tensor $X \in \mathbb{R}^{B \times 4 \times F \times T}$ *before* `patch_embed`. Do NOT use `AMBAModel.mpg` token-level zeroing for the binaural CCSR path.

2. **Fallpoint 2: STFT Frequency/Time Dimension Grid Misalignment**
   - *Mistake*: Using an STFT $n_{\text{fft}}$ that produces a frequency dimension $F$ not divisible by `fshape=16`. For instance, standard $n_{\text{fft}}=512$ yields $F=257$ bins.
   - *Fix*: You MUST slice/pad $F \rightarrow 256$ (or $F=128$) so that $F \bmod 16 = 0$ and $T \bmod 16 = 0$.

3. **Fallpoint 3: Silent Fallback in `sac_loss` Sigma Lookup Tables**
   - *Mistake*: Running `optuna_optimal` or `static_entropy_optimal` sigma mode with the new `"Spatial"` group without updating lookup tables in `sac/sigma_configs.py`.
   - *Fix*: Force `local_sigma_mode = 'chi2_median'` for initial binaural pretraining runs. `chi2_median` dynamically computes local sigma from feature dimension $D$, avoiding stale hardcoded table fallbacks.

4. **Fallpoint 4: Attaching SAC Spatial Head to the Spectral Encoder**
   - *Mistake*: Attaching the Factorized SAC head or spatial queries to `h_spec` (spectral encoder).
   - *Fix*: `h_spec` MUST NOT be pushed to encode spatial information. Attach SAC queries and projection head **ONLY to `h_spat`**.

5. **Fallpoint 5: Downstream Fine-Tuning Task Stream Routing**
   - *Mistake*: Evaluating monaural speech tasks (VoxCeleb SID, IEMOCAP ER) on `h_spat` or spatial tasks (DoA / Speaker Localization) on `h_spec`.
   - *Fix*: Monaural tasks read from `h_spec` ONLY. Spatial localization tasks read from `h_spat` (or dual-stream linear/bilinear fusion `[e_L; e_R; e_L * e_R; |e_L - e_R|]`).

---

## 3. Step-by-Step Codebase Implementation Blueprint

### Module 1: `AMBAModel` Updates ([src/models/both_models.py](file:///storage/yotam/ssamba/src/models/both_models.py))

Update `AMBAModel` to support arbitrary input channels (`in_chans=4`):

```python
# In AMBAModel.__init__ (line 63)
def __init__(self, label_dim=527,
             fshape=128, tshape=2, fstride=128, tstride=2,
             input_fdim=128, input_tdim=1024, model_size='base',
             in_chans=1, # <-- Add in_chans parameter (default 1, set 4 for complex STFT)
             pretrain_stage=True, load_pretrained_mdl_path=None, vision_mamba_config=None):

    self.in_chans = in_chans
    # ...
    # Update patch projection (line 143):
    new_proj = torch.nn.Conv2d(in_chans, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
    self.v.patch_embed.proj = new_proj

# In AMBAModel.get_shape (line 261):
def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape, in_chans=1):
    test_input = torch.randn(1, in_chans, input_fdim, input_tdim)
    test_proj = nn.Conv2d(in_chans, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
    test_out = test_proj(test_input)
    return test_out.shape[2], test_out.shape[3]
```

Add an explicit helper method `_encode_with_mamba` inside `AMBAModel`:

```python
def _encode_with_mamba(self, x):
    """
    Forward pass through patch_embed -> cls_token -> pos_embed -> mamba_layers -> norm.
    Expects x of shape [B, in_chans, F, T].
    Returns hidden_states of shape [B, num_patches + cls_token_num, embed_dim].
    """
    B = x.shape[0]
    x_embed = self.v.patch_embed(x)
    cls_tokens = self.v.cls_token.expand(B, -1, -1)
    x_embed = torch.cat((cls_tokens, x_embed), dim=1)
    x_embed = x_embed + self.v.pos_embed
    x_embed = self.v.pos_drop(x_embed)

    residual = None
    hidden_states, residual = self._forward_mamba_layers(x_embed, residual)

    if not self.v.fused_add_norm:
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.v.drop_path(hidden_states)
        hidden_states = self.v.norm_f(residual.to(dtype=self.v.norm_f.weight.dtype))
    else:
        try:
            from mamba_ssm.ops.triton.layernorm import rms_norm_fn, layer_norm_fn, RMSNorm
            fused_add_norm_fn = rms_norm_fn if (RMSNorm is not None and isinstance(self.v.norm_f, RMSNorm)) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.v.drop_path(hidden_states),
                self.v.norm_f.weight,
                self.v.norm_f.bias,
                eps=self.v.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.v.residual_in_fp32,
            )
        except (ImportError, TypeError):
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + hidden_states
            hidden_states = self.v.norm_f(residual)

    return hidden_states
```

---

### Module 2: CCSR Masking & Dual Encoder ([src/models/binaural_amba.py](file:///storage/yotam/ssamba/src/models/binaural_amba.py))

Create a new file `src/models/binaural_amba.py`:

```python
import torch
import torch.nn as nn
from models.both_models import AMBAModel

def apply_ccsr_masking(X, W):
    """
    Pre-embedding channel-conditional time-frame masking.
    
    Args:
        X: [B, 4, F, T] tensor where dim 1 contains (Re_L, Im_L, Re_R, Im_R)
        W: [B, 1, 1, T] binary time-frame mask (1 = unmasked, 0 = masked)
        
    Returns:
        X_spat: [B, 4, F, T] - Same time-frame mask applied to both channels
        X_spec: [B, 4, F, T] - Inverse mask applied to Right channel only
    """
    # X_spat gets identical time-frame mask W on both Left and Right channels
    X_spat = X * W
    
    # X_spec keeps full Left channel, applies inverse mask (1-W) to Right channel
    X_spec = X.clone()
    X_spec[:, 2:, :, :] = X[:, 2:, :, :] * (1.0 - W)
    
    return X_spat, X_spec


class BinauralAMBAEncoder(nn.Module):
    """
    Dual-stream Mamba encoder for CCSR-style spatial/spectral disentanglement.
    Holds spatial_encoder and spectral_encoder.
    """
    def __init__(self, encoder_config, share_weights=False):
        super().__init__()
        self.share_weights = share_weights
        encoder_config['in_chans'] = 4 # Ensure in_chans=4 for complex STFT
        
        self.spatial_encoder = AMBAModel(**encoder_config)
        if share_weights:
            self.spectral_encoder = self.spatial_encoder
        else:
            self.spectral_encoder = AMBAModel(**encoder_config)

    def forward(self, X, W):
        """
        Args:
            X: [B, 4, F, T] complex STFT tensor
            W: [B, 1, 1, T] binary time-frame mask
            
        Returns:
            h_spat: [B, num_patches + cls, D] spatial encoder representations
            h_spec: [B, num_patches + cls, D] spectral encoder representations
        """
        X_spat, X_spec = apply_ccsr_masking(X, W)
        h_spat = self.spatial_encoder._encode_with_mamba(X_spat)
        h_spec = self.spectral_encoder._encode_with_mamba(X_spec)
        return h_spat, h_spec
```

---

### Module 3: Binaural SAC Model Wrapper ([src/sac/binaural_sac_model.py](file:///storage/yotam/ssamba/src/sac/binaural_sac_model.py))

Create a new file `src/sac/binaural_sac_model.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.binaural_amba import BinauralAMBAEncoder
from sac.sac_model import SSAMBASACModel

class BinauralSSAMBASACModel(nn.Module):
    """
    Binaural SSAMBA + Disentangled Factorized SAC Model.
    Integrates BinauralAMBAEncoder, CCSR Time-Frame Reconstruction, and Spatial SAC Loss.
    """
    def __init__(
        self,
        fshape=16, tshape=16,
        input_fdim=256, input_tdim=1024,
        model_size='base',
        embed_dim=768,
        proj_dim=128,
        sac_temperature=0.3,
        sac_sigma=1.0,
        sac_lambda=1.0,
        recon_lambda=1.0,
        sac_features="f0_mean,hnr,centroid,flux,zcr_mean,spatial",
        vision_mamba_config=None,
        local_sigma_mode='chi2_median',
        share_encoder_weights=False,
    ):
        super().__init__()
        self.sac_temperature = sac_temperature
        self.sac_sigma = sac_sigma
        self.sac_lambda = sac_lambda
        self.recon_lambda = recon_lambda
        self.local_sigma_mode = local_sigma_mode
        self.fshape = fshape
        self.tshape = tshape

        encoder_config = {
            'fshape': fshape, 'tshape': tshape,
            'fstride': fshape, 'tstride': tshape,
            'input_fdim': input_fdim, 'input_tdim': input_tdim,
            'model_size': model_size,
            'in_chans': 4,
            'pretrain_stage': True,
            'vision_mamba_config': vision_mamba_config,
        }

        # Dual Encoder
        self.dual_encoder = BinauralAMBAEncoder(encoder_config, share_weights=share_encoder_weights)

        # Shared Reconstruction Decoder (reconstructs Re_L, Im_L for masked time frames)
        # Maps concat([h_spat, h_spec]) -> 2 * fshape * tshape patch values
        patch_dim = 2 * fshape * tshape # Re_L and Im_L for 1st mic
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, patch_dim)
        )

        # SAC Head (Attaches ONLY to spatial encoder h_spat)
        from sac.acoustic_features import get_feature_groups
        self.group_indices, self.num_active_groups = get_feature_groups(sac_features)
        
        self.group_queries = nn.Parameter(torch.empty(self.num_active_groups * 4, embed_dim))
        nn.init.normal_(self.group_queries, std=0.02)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def _reconstruct(self, h_spat, h_spec, X, W):
        """
        CCSR Eq. 6 Reconstruction Loss: MSE over masked time-frames of Channel 1 (Re_L, Im_L).
        """
        B = X.shape[0]
        # Exclude CLS token
        h_spat_patch = h_spat[:, 1:, :] # [B, num_patches, D]
        h_spec_patch = h_spec[:, 1:, :] # [B, num_patches, D]
        
        # Concat along embedding dimension
        h_concat = torch.cat([h_spat_patch, h_spec_patch], dim=-1) # [B, num_patches, 2D]
        pred_patches = self.reconstruction_decoder(h_concat) # [B, num_patches, 2 * fshape * tshape]
        
        # Target: unfold Channel 1 (Re_L, Im_L) raw STFT patches
        X_ch1 = X[:, :2, :, :] # [B, 2, F, T]
        unfold = torch.nn.Unfold(kernel_size=(self.fshape, self.tshape), stride=(self.fshape, self.tshape))
        target_patches = unfold(X_ch1).transpose(1, 2) # [B, num_patches, 2 * fshape * tshape]
        
        # Masked region weights (where W == 0)
        # Expand W [B, 1, 1, T] to patch level
        masked_weights = (1.0 - W).squeeze(1).squeeze(1) # [B, T]
        # Average mask weight per patch
        patch_w = F.adaptive_avg_pool1d(masked_weights.unsqueeze(1), pred_patches.shape[1]).transpose(1, 2)
        
        # MSE Loss weighted by (1 - W)
        loss_recon = torch.mean(patch_w * (pred_patches - target_patches) ** 2)
        return loss_recon

    def forward(self, X, W, acoustic_features=None):
        B = X.shape[0]
        
        # 1. Dual Encoder Pass
        h_spat, h_spec = self.dual_encoder(X, W)
        
        # 2. Reconstruction Loss (CCSR Eq. 6)
        loss_recon = self._reconstruct(h_spat, h_spec, X, W)
        
        # 3. Factorized SAC Loss (Attaches ONLY to h_spat)
        if acoustic_features is not None:
            Q = self.group_queries.unsqueeze(0).expand(B, -1, -1)
            attn_out, _ = self.cross_attention(query=Q, key=h_spat[:, 1:, :], value=h_spat[:, 1:, :])
            attn_out = attn_out.view(B, self.num_active_groups, 4, -1).mean(dim=2)
            Z_groups = self.projection_head(attn_out)
            
            # Reuse sac_loss from SSAMBASACModel logic
            loss_sac = SSAMBASACModel.sac_loss(self, Z_groups, acoustic_features)
        else:
            loss_sac = torch.tensor(0.0, device=X.device)
            
        loss_total = self.recon_lambda * loss_recon + self.sac_lambda * loss_sac
        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_sac': loss_sac,
        }
```

---

### Module 4: Data Loader & Complex STFT Frontend ([src/dataloader.py](file:///storage/yotam/ssamba/src/dataloader.py))

Add `_wav2stft` to `AudioDataset` in `src/dataloader.py`:

```python
# In AudioDataset class in src/dataloader.py
def _wav2stft(self, filename):
    """
    Loads 2-channel audio and computes 4-channel Complex STFT: [4, F, T].
    """
    waveform, sr = torchaudio.load(filename) # [channels, num_samples]
    
    # Ensure 2 channels
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]
        
    # STFT settings (n_fft=512, hop_length=160 -> 10ms frame shift at 16kHz)
    n_fft = 512
    hop_length = 160
    
    stft_L = torch.stft(waveform[0], n_fft=n_fft, hop_length=hop_length, return_complex=True) # [257, T]
    stft_R = torch.stft(waveform[1], n_fft=n_fft, hop_length=hop_length, return_complex=True) # [257, T]
    
    # Stack [Re_L, Im_L, Re_R, Im_R] -> [4, 257, T]
    stft_4ch = torch.stack([
        stft_L.real, stft_L.imag,
        stft_R.real, stft_R.imag
    ], dim=0)
    
    # Truncate frequency bins from 257 to 256 for patch divisibility (256 % 16 == 0)
    stft_4ch = stft_4ch[:, :256, :]
    
    # Cut/pad time frames to target_length (e.g. 1024 frames)
    target_length = self.audio_conf.get('target_length', 1024)
    T = stft_4ch.shape[2]
    if T < target_length:
        stft_4ch = F.pad(stft_4ch, (0, target_length - T))
    elif T > target_length:
        stft_4ch = stft_4ch[:, :, :target_length]
        
    return stft_4ch

def generate_ccsr_time_mask(self, T, mask_ratio=0.5):
    """
    Generates binary time-frame mask W of shape [1, 1, 1, T].
    1 = unmasked, 0 = masked.
    """
    num_mask = int(T * mask_ratio)
    W = torch.ones(T, dtype=torch.float32)
    mask_indices = torch.randperm(T)[:num_mask]
    W[mask_indices] = 0.0
    return W.view(1, 1, 1, T)
```

---

### Module 5: Spatial Feature Extraction Pipeline ([src/sac/acoustic_features.py](file:///storage/yotam/ssamba/src/sac/acoustic_features.py))

Add `"spatial"` feature group definition in `sac/acoustic_features.py`:

```python
# In sac/acoustic_features.py
FEATURE_GROUPS = {
    'prosody': ['f0_mean', 'f0_std'],
    'vocal_tract': ['f1_mean', 'f2_mean', 'f3_mean'],
    'timbre': ['mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean'],
    'voice_quality': ['hnr'],
    'scene_noise': ['centroid', 'flux', 'zcr_mean'],
    'spatial': ['tdoa', 'gcc_phat_peak', 'coh_low', 'coh_mid', 'coh_high'], # <-- 5-feature Spatial Vector
}

def extract_spatial_features(waveform_2ch, sr=16000):
    """
    Extracts 5 spatial DSP features from 2-channel audio waveform [2, N]:
    1. TDOA (GCC-PHAT peak lag index)
    2. GCC-PHAT peak magnitude
    3. Coh_low (0-1kHz interaural coherence)
    4. Coh_mid (1-4kHz interaural coherence)
    5. Coh_high (4-8kHz interaural coherence)
    """
    import numpy as np
    from scipy.signal import coherence, correlation
    
    wav_L = waveform_2ch[0].cpu().numpy()
    wav_R = waveform_2ch[1].cpu().numpy()
    
    # 1. GCC-PHAT TDOA & Peak Magnitude
    n = len(wav_L)
    X_L = np.fft.rfft(wav_L)
    X_R = np.fft.rfft(wav_R)
    R = X_L * np.conj(X_R)
    R_phat = R / (np.abs(R) + 1e-8)
    gcc_phat = np.fft.irfft(R_phat, n=n)
    
    max_lag = int(0.001 * sr) # +/- 1ms max delay
    gcc_phat_window = np.concatenate([gcc_phat[-max_lag:], gcc_phat[:max_lag+1]])
    
    tdoa = (np.argmax(gcc_phat_window) - max_lag) / max_lag # Scale to [-1, 1]
    gcc_phat_peak = np.max(gcc_phat_window)
    
    # 2. Interaural Coherence (Sub-band Coherence)
    f, Cxy = coherence(wav_L, wav_R, fs=sr, nperseg=512)
    coh_low = np.mean(Cxy[(f >= 0) & (f < 1000)])
    coh_mid = np.mean(Cxy[(f >= 1000) & (f < 4000)])
    coh_high = np.mean(Cxy[(f >= 4000) & (f <= 8000)])
    
    return np.array([tdoa, gcc_phat_peak, coh_low, coh_mid, coh_high], dtype=np.float32)
```

---

### Module 6: Pretraining Script ([src/run_binaural_amba.py](file:///storage/yotam/ssamba/src/run_binaural_amba.py))

Create a dedicated pretraining script `src/run_binaural_amba.py`:

```python
import argparse
import torch
from torch.utils.data import DataLoader
from sac.binaural_sac_model import BinauralSSAMBASACModel
from dataloader import AudioDataset

def main():
    parser = argparse.ArgumentParser(description="Binaural SSAMBA + SAC Pretraining")
    parser.add_argument('--dataset_json', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sac_lambda', type=float, default=1.0)
    parser.add_argument('--recon_lambda', type=float, default=1.0)
    parser.add_argument('--local_sigma_mode', type=str, default='chi2_median')
    parser.add_argument('--share_encoder_weights', action='store_true', default=False)
    args = parser.parse_args()

    # Build model
    model = BinauralSSAMBASACModel(
        input_fdim=256, input_tdim=1024,
        sac_lambda=args.sac_lambda,
        recon_lambda=args.recon_lambda,
        local_sigma_mode=args.local_sigma_mode,
        share_encoder_weights=args.share_encoder_weights,
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Synthetic batch verification
    print("Verifying forward pass with synthetic binaural tensor...")
    dummy_X = torch.randn(args.batch_size, 4, 256, 1024).cuda() # [B, 4, F, T]
    dummy_W = (torch.rand(args.batch_size, 1, 1, 1024) > 0.5).float().cuda()
    dummy_c = torch.randn(args.batch_size, 20).cuda() # 20 acoustic features (15 monaural + 5 spatial)

    out = model(dummy_X, dummy_W, dummy_c)
    print(f"Success! Loss Total: {out['loss_total'].item():.4f}, Recon: {out['loss_recon'].item():.4f}, SAC: {out['loss_sac'].item():.4f}")

if __name__ == '__main__':
    main()
```

---

### Module 7: Downstream Fine-Tuning Task Stream Routing ([src/finetune/](file:///storage/yotam/ssamba/src/finetune/))

When fine-tuning on downstream tasks, route embeddings based on task type:

#### A. Monaural Speech & Emotion Tasks (VoxCeleb SID, IEMOCAP ER)
Read from `spectral_encoder` ONLY:
```python
# In monaural downstream evaluation wrapper:
spectral_encoder = checkpoint['dual_encoder.spectral_encoder']
mono_X = X_mono.repeat(1, 4, 1, 1) # Duplicate mono to 4 channels
hidden_states = spectral_encoder._encode_with_mamba(mono_X)
features = hidden_states[:, 1:, :].mean(dim=1) # Mean pool patch tokens
output = mlp_head(features)
```

#### B. Spatial Audio Tasks (Speaker Localization / DoA / SELD)
Read from `spatial_encoder` using **Linear/Bilinear Fusion**:
```python
# In spatial downstream evaluation wrapper:
spatial_encoder = checkpoint['dual_encoder.spatial_encoder']

# Run spatial encoder for Left-ear and Right-ear views:
e_L = spatial_encoder._encode_with_mamba(X_Left_view)[:, 1:, :].mean(dim=1)
e_R = spatial_encoder._encode_with_mamba(X_Right_view)[:, 1:, :].mean(dim=1)

# Bilinear non-attention fusion:
h_fused = torch.cat([e_L, e_R, e_L * e_R, torch.abs(e_L - e_R)], dim=-1)
h_projected = linear_fusion_layer(h_fused)
prediction_doa = localization_head(h_projected)
```

---

## 4. Step-by-Step Validation & Verification Protocol

Follow this 3-step protocol to verify correctness before running full pretraining:

### Step 1: Forward Shape & Tensor Flow Test
Run `python src/run_binaural_amba.py` to confirm synthetic tensor shapes:
- Input `dummy_X` shape `[32, 4, 256, 1024]`
- Spatial encoder output `h_spat` shape `[32, 513, 768]`
- Spectral encoder output `h_spec` shape `[32, 513, 768]`
- Concat representation `h_concat` shape `[32, 512, 1536]`
- Prediction output `pred_patches` shape `[32, 512, 512]`
- Ensure no NaNs appear in `loss_recon` or `loss_sac`.

### Step 2: Disentanglement Sanity Check (Reconstruction-Only)
Train `BinauralSSAMBASACModel` with `sac_lambda=0.0` (Reconstruction only) for 5 epochs:
- Fine-tune a frozen `spectral_encoder` on a speech recognition/speaker ID downstream task.
- Confirm downstream performance matches training from scratch (proving content is preserved in `spectral_encoder`).
- Confirm `spatial_encoder` fails on speech tasks but excels on TDOA estimation (proving spatial features are isolated in `spatial_encoder`).

### Step 3: Full Binaural SAC Pretraining
Set `sac_lambda=1.0`, `recon_lambda=1.0`, and `local_sigma_mode='chi2_median'`:
- Verify gradient norms between `loss_recon` and `loss_sac` are balanced (ratio $\approx 1:1$).
- Monitor entropy diagnostics for the 6th `"spatial"` query group to ensure non-degenerate weight distributions ($H_{\text{spatial}} \in [1.5, 3.0]$).
