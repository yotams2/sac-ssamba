# Technical Specification: Binaural Audio (2-Channel) Extension for SSAMBA & Factorized SAC

## 1. Executive Summary & Purpose
This document provides a comprehensive, self-contained implementation blueprint for adapting the **Self-Supervised Audio Mamba (SSAMBA)** model and its **Factorized Soft Acoustic Contrastive (SAC)** loss framework to support **binaural (2-channel) audio recordings**.

An agent or developer reading this specification can directly implement the required changes across the codebase without needing additional context or external research.

---

## 2. Background & Architecture Overview

### Baseline Monaural Architecture
1. **Input Signal**: Single-channel audio waveforms normalized to 10-second clips at 16kHz.
2. **Spectrogram**: $128$-dimensional log-Mel filterbanks over $1024$ time frames $\rightarrow$ shape `[B, 1024, 128]`.
3. **Patch Embedding**: `Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))` converts input `[B, 1, 128, 1024]` into 512 patch tokens of dimension $D = 768$.
4. **Encoder Backbone**: Bidirectional VisionMamba (`VisionMamba`) sequence model processing the sequence of patch tokens.
5. **Dual Pretraining Objectives**:
   - **Generative Masked Reconstruction (`mpg`)**: Predicts raw spectrogram pixel values for masked patches.
   - **Factorized SAC Loss**: Uses acoustic feature cross-attention queries ($F_0$, Formants, MFCCs, HNR, ZCR) to route encoder hidden states into specialized latent sub-spaces and compute continuous contrastive loss.

---

## 3. Design Strategy: Early Channel Fusion

To support 2-channel binaural audio, we adopt **Early Channel Fusion**:
* **Mechanism**: Update the input patch embedding layer `Conv2d` from `in_chans=1` to `in_chans=2`.
* **Input Tensor Shape**: `[B, 2, 1024, 128]` (Left and Right channel log-Mel spectrograms).
* **Key Advantages**:
  1. **Constant Sequence Length**: Preserves `num_patches = 512`. Sequence length and Mamba computation complexity remain identical ($O(N)$).
  2. **Early Inter-aural Feature Learning**: Fuses Left/Right channel spectral differences (Interaural Level Differences - ILD and Interaural Time Differences - ITD within time-frequency patches) directly at the first projection layer.
  3. **Backward Compatibility**: Setting `in_chans=1` preserves existing monaural functionality.

---

## 4. File-by-File Implementation Blueprint

### File 1: [both_models.py](file:///storage/yotam/ssamba/src/models/both_models.py) (`AMBAModel`)

#### 4.1 Update `AMBAModel.__init__` Signature and Convolution Projection
- Add `in_chans=1` to `AMBAModel.__init__` parameters.
- Store `self.in_chans = in_chans`.
- Update `PatchEmbed` initialization to use `in_chans`:

```python
# In AMBAModel.__init__ (line 63)
def __init__(self, label_dim=527,
             fshape=128, tshape=2, fstride=128, tstride=2,
             input_fdim=128, input_tdim=1024, model_size='base',
             in_chans=1, # <-- Add in_chans parameter
             pretrain_stage=True, load_pretrained_mdl_path=None, vision_mamba_config=None):

    self.in_chans = in_chans
    # ...
    # Update patch projection (line 143):
    new_proj = torch.nn.Conv2d(in_chans, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
    self.v.patch_embed.proj = new_proj
```

#### 4.2 Update `get_shape` Method
Update `get_shape` to construct `test_input` with `in_chans` channels:

```python
# In AMBAModel.get_shape (line 261)
def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape, in_chans=1):
    test_input = torch.randn(1, in_chans, input_fdim, input_tdim)
    test_proj = nn.Conv2d(in_chans, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
    test_out = test_proj(test_input)
    return test_out.shape[2], test_out.shape[3]
```

#### 4.3 Update Reconstruction Head (`gpredlayer`) and `mpg` Method
- `gpredlayer` must map from `original_embedding_dim` to `in_chans * fshape * tshape` (line 125):

```python
# In AMBAModel.__init__
self.gpredlayer = nn.Sequential(
    nn.Linear(self.original_embedding_dim, self.original_embedding_dim),
    nn.ReLU(),
    nn.Linear(self.original_embedding_dim, in_chans * fshape * tshape)
)
```

- In `mpg()` (lines 604-675):
  - `self.unfold` on `[B, in_chans, F, T]` with kernel `(fshape, tshape)` produces patch targets of size `in_chans * fshape * tshape`.
  - Update `pred` and `target` tensor initialization:
    ```python
    patch_pixels = self.in_chans * self.fshape * self.tshape
    pred = torch.empty((B, mask_patch, patch_pixels), device=x.device).float()
    target = torch.empty((B, mask_patch, patch_pixels), device=x.device).float()
    ```

#### 4.4 Update `AMBAModel.forward`
Handle both 3D (`[B, T, F]`) and 4D (`[B, C, T, F]`) inputs dynamically:

```python
# In AMBAModel.forward (line 678)
def forward(self, x, task, cluster=True, mask_patch=400):
    if x.dim() == 3:
        # Single-channel input [B, T, F] -> [B, 1, F, T]
        x = x.unsqueeze(1).transpose(2, 3)
    elif x.dim() == 4:
        # Multi-channel input [B, C, T, F] -> [B, C, F, T]
        x = x.transpose(2, 3)
    else:
        raise ValueError(f"Expected input x of dim 3 or 4, got shape {x.shape}")
    # ... proceed with downstream tasks (ft_avgtok, pretrain_mpg, etc.)
```

---

### File 2: [sac_model.py](file:///storage/yotam/ssamba/src/sac/sac_model.py) (`SSAMBASACModel`)

#### 4.1 Update `SSAMBASACModel.__init__`
Add `in_chans=1` parameter and pass it to `AMBAModel`:

```python
# In SSAMBASACModel.__init__ (line 59)
def __init__(
    self,
    fshape=16, tshape=16,
    input_fdim=128, input_tdim=1024,
    model_size='base',
    in_chans=1, # <-- Add in_chans parameter
    embed_dim=768,
    depth=24,
    # ...
):
    self.in_chans = in_chans
    # ...
    self.encoder = AMBAModel(
        fshape=fshape, tshape=tshape,
        fstride=fshape, tstride=tshape,
        input_fdim=input_fdim, input_tdim=input_tdim,
        model_size=model_size,
        in_chans=in_chans, # <-- Pass down in_chans
        pretrain_stage=True,
        load_pretrained_mdl_path=None,
        vision_mamba_config=vision_mamba_config,
    )
```

#### 4.2 Update `forward` Pass Tensor Reshaping
In `SSAMBASACModel.forward` (line 404):

```python
# In SSAMBASACModel.forward
if fbank.dim() == 3:
    # [B, T, F] -> [B, 1, F, T]
    x = fbank.unsqueeze(1).transpose(2, 3)
elif fbank.dim() == 4:
    # [B, C, T, F] -> [B, C, F, T]
    x = fbank.transpose(2, 3)
```

---

### File 3: [dataloader.py](file:///storage/yotam/ssamba/src/dataloader.py) (`AudioDataset`)

#### 4.1 Multi-channel Spectrogram Extraction in `_wav2fbank`
Modify `_wav2fbank` to support 2-channel audio extraction:

```python
# In AudioDataset._wav2fbank (line 99)
def _wav2fbank(self, filename, filename2=None):
    waveform, sr = torchaudio.load(filename) # waveform shape: [num_channels, num_samples]
    
    if self.audio_conf.get('in_chans', 1) == 2:
        # Ensure audio has 2 channels (duplicate mono if needed)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        # Compute log-Mel filterbank per channel
        fbanks = []
        for c in range(2):
            w_c = waveform[c:c+1, :] - waveform[c:c+1, :].mean()
            fb = torchaudio.compliance.kaldi.fbank(
                w_c, htk_compat=True, sample_frequency=sr, use_energy=False,
                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10
            )
            # Pad / cut to target_length
            target_length = self.audio_conf.get('target_length')
            n_frames = fb.shape[0]
            p = target_length - n_frames
            if p > 0:
                fb = torch.nn.ZeroPad2d((0, 0, 0, p))(fb)
            elif p < 0:
                fb = fb[:target_length, :]
            fbanks.append(fb)
            
        # Stack channels -> shape: [2, target_length, melbins]
        fbank = torch.stack(fbanks, dim=0)
        return fbank, 0
```

---

### File 4: [acoustic_features.py](file:///storage/yotam/ssamba/src/sac/acoustic_features.py) (Optional Spatial SAC Extension)

To leverage binaural recordings in the Factorized SAC loss framework, extract spatial acoustic features:
1. **Interaural Time Difference (ITD)**: Cross-correlation peak lag between Left and Right audio signals $\Delta t = \arg\max_\tau R_{LR}(\tau)$.
2. **Interaural Level Difference (ILD)**: Log power ratio $10 \log_{10} (E_L / E_R)$.
3. **Interaural Coherence (IC)**: Maximum normalized cross-correlation value.

Add a **"Spatial / Binaural Feature Group"** (`spatial_query`) to `get_feature_groups()`:

```python
# In sac/acoustic_features.py
FEATURE_GROUPS = {
    'prosody': ['f0_mean', 'f0_std'],
    'vocal_tract': ['f1_mean', 'f2_mean', 'f3_mean'],
    'timbre': ['mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean'],
    'voice_quality': ['hnr'],
    'scene_noise': ['centroid', 'flux', 'zcr_mean'],
    'spatial_binaural': ['itd', 'ild', 'ic'], # <-- Spatial Group
}
```

---

## 5. Verification & Testing Protocol

### 5.1 Shape Verification Test
Execute a quick PyTorch synthetic forward pass test:
```python
import torch
from models.both_models import AMBAModel
from sac.sac_model import SSAMBASACModel

# Test AMBAModel with in_chans=2
model = AMBAModel(fshape=16, tshape=16, fstride=16, tstride=16, in_chans=2, pretrain_stage=True)
dummy_input = torch.randn(4, 2, 1024, 128) # [B, C, T, F]
loss_mpg = model(dummy_input, task='pretrain_mpg')
print("AMBAModel Binaural MPG Loss Shape/Value:", loss_mpg)

# Test SSAMBASACModel with in_chans=2
sac_model = SSAMBASACModel(fshape=16, tshape=16, in_chans=2, sac_lambda=1.0, recon_lambda=1.0)
dummy_c = torch.randn(4, 5) # Dummy acoustic features
output = sac_model(dummy_input, task='pretrain_joint', acoustic_features=dummy_c)
print("SSAMBASACModel Total Loss:", output['loss_total'])
```

### 5.2 Pretraining Script Call (`run_amba.py`)
Add `--in_chans 2` command-line argument to `src/run_amba.py` and pass to model initialization.

---

## 6. Summary of Architectural Impact
- **Compute Cost**: 0% increase in sequence tokens or Mamba layer compute.
- **Parameters**: Negligible increase (only $768$ additional parameters in `Conv2d(2, 768, ...)` input layer).
- **Functionality**: Seamless dual-channel processing while preserving full backward compatibility with monaural models.
