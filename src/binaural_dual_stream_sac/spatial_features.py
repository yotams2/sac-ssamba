import numpy as np
import torch
import sys
import os

# Ensure parent directory imports work for importing sac.acoustic_features
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sac.acoustic_features import get_feature_groups, extract_acoustic_features

# Spatial feature list
SPATIAL_FEATURES = ['tdoa', 'gcc_phat_peak', 'coh_low', 'coh_mid', 'coh_high']


def get_spatial_feature_groups(features_str="spatial"):
    """
    Returns column indices for spatial acoustic feature groups (5 spatial features).
    """
    feature_to_idx = {name: idx for idx, name in enumerate(SPATIAL_FEATURES)}
    group_indices = {
        'spatial': [feature_to_idx[feat] for feat in SPATIAL_FEATURES]
    }
    return group_indices, len(group_indices)


def extract_spatial_features(waveform_2ch, sr=16000, coh_freq_bands=(1000.0, 4000.0)):
    """
    Extracts 5 spatial DSP features from a 2-channel audio waveform tensor/ndarray [2, N]:
    
    1. TDOA (GCC-PHAT peak lag index normalized to [-1, 1])
    2. GCC-PHAT peak magnitude [0, 1]
    3. Coh_low  (0 - f_low Hz interaural coherence; default 0 - 1000Hz for ITD/phase range)
    4. Coh_mid  (f_low - f_high Hz interaural coherence; default 1000 - 4000Hz for formant transition)
    5. Coh_high (> f_high Hz interaural coherence; default 4000 - 8000Hz for ILD/head-shadow range)
    
    Args:
        waveform_2ch: [2, N] 2-channel audio signal
        sr: sample rate (default 16000)
        coh_freq_bands: tuple (f_low, f_high) defining sub-band boundaries.
            Default (1000.0, 4000.0) aligns with Duplex Theory & SSAMBA fshape=16 grid (Bins 32 & 128).
            Pass (500.0, 2500.0) to use SAR-SSL room acoustics cutoffs.
    """
    from scipy.signal import coherence

    if isinstance(waveform_2ch, torch.Tensor):
        waveform_2ch = waveform_2ch.detach().cpu().numpy()

    if waveform_2ch.ndim == 1 or waveform_2ch.shape[0] < 2:
        return np.zeros(5, dtype=np.float32)

    wav_L = waveform_2ch[0].astype(np.float32)
    wav_R = waveform_2ch[1].astype(np.float32)

    n = len(wav_L)
    if n == 0:
        return np.zeros(5, dtype=np.float32)

    # 1. GCC-PHAT & TDOA
    X_L = np.fft.rfft(wav_L)
    X_R = np.fft.rfft(wav_R)
    R = X_L * np.conj(X_R)
    denom = np.abs(R) + 1e-8
    R_phat = R / denom
    gcc_phat = np.fft.irfft(R_phat, n=n)

    max_lag = int(0.001 * sr) # +/- 1ms max delay (~34cm mic distance limit)
    if max_lag > 0 and len(gcc_phat) > 2 * max_lag:
        gcc_phat_window = np.concatenate([gcc_phat[-max_lag:], gcc_phat[:max_lag + 1]])
        best_idx = np.argmax(np.abs(gcc_phat_window))
        tdoa = (best_idx - max_lag) / float(max_lag)
        gcc_phat_peak = float(np.abs(gcc_phat_window[best_idx]))
    else:
        tdoa = 0.0
        gcc_phat_peak = float(np.max(np.abs(gcc_phat)))

    # 2. Sub-band Interaural Coherence
    f_low, f_high = coh_freq_bands
    try:
        f, Cxy = coherence(wav_L, wav_R, fs=sr, nperseg=min(512, len(wav_L)))
        mask_low = (f < f_low)
        mask_mid = (f >= f_low) & (f <= f_high)
        mask_high = (f > f_high)

        coh_low = float(np.mean(Cxy[mask_low])) if np.any(mask_low) else 0.0
        coh_mid = float(np.mean(Cxy[mask_mid])) if np.any(mask_mid) else 0.0
        coh_high = float(np.mean(Cxy[mask_high])) if np.any(mask_high) else 0.0
    except Exception:
        coh_low, coh_mid, coh_high = 0.0, 0.0, 0.0

    return np.array([tdoa, gcc_phat_peak, coh_low, coh_mid, coh_high], dtype=np.float32)
