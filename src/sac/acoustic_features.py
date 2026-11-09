"""
Step 1: Single-Channel Acoustic Feature Extraction for SAC Loss.

Features extracted per clip:
    - f0_mean / f0_var: median and interquartile range of F0 over voiced frames, in semitones above 100 Hz
    - hnr: mean over voiced frames, in dB
    - centroid
    - flux
    - zcr_mean / zcr_var
    - formants (f1, f2, f3)
    - mfcc (mean of first 5 mfccs)

Output: normalized feature vector c_i ∈ [-1, 1]^K for each sample in the batch.
"""

import torch
import torch.nn.functional as F
import math

# Pitch tracking, used by F0 and HNR
PITCH_FMIN = 70.0
PITCH_FMAX = 500.0
PITCH_FRAME_SEC = 0.04            # at least two periods at PITCH_FMIN
PITCH_HOP_SEC = 0.01
PITCH_VOICING_THRESHOLD = 0.45    # Praat's default
PITCH_SILENCE_RATIO = 0.03        # frame RMS relative to the loudest frame of the clip
PITCH_OCTAVE_COST = 0.01          # Praat's default
F0_NO_VOICE_SEMITONES = 10.0      # about 178 Hz

# LPC formants, with Praat's default formant settings (5 formants up to 5500 Hz)
FORMANT_SAMPLE_RATE = 11000       # twice the 5500 Hz ceiling
LPC_ORDER = 10                    # two poles per formant
LPC_PRE_EMPHASIS = 0.97
FORMANT_MIN_HZ = 50.0

def _safe_log(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return torch.log(x.clamp(min=eps))

def _pitch_track(waveform: torch.Tensor, sample_rate: int) -> tuple:
    """
    Frame-level pitch by normalised autocorrelation (Boersma 1993, the method in Praat).

    Dividing by the window's own autocorrelation removes the bias towards short lags. Among the autocorrelation peaks in
    the pitch range, a small octave cost prefers the shortest period, so that 2T doesn't beat T.

    Returns: f0 [B, N] in Hz, strength [B, N] (autocorrelation at the chosen lag, 1 = periodic), voiced [B, N] bool.
    """
    B, T = waveform.shape
    device, dtype = waveform.device, waveform.dtype
    win_length = int(sample_rate * PITCH_FRAME_SEC)
    hop_length = int(sample_rate * PITCH_HOP_SEC)
    if T < win_length:
        waveform = F.pad(waveform, (0, win_length - T))

    frames = waveform.unfold(-1, win_length, hop_length)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    rms = frames.pow(2).mean(dim=-1).sqrt()
    loud = rms > PITCH_SILENCE_RATIO * rms.amax(dim=-1, keepdim=True)

    window = torch.hann_window(win_length, device=device, dtype=dtype)
    n_fft = int(2 ** math.ceil(math.log2(win_length * 2)))

    def autocorr(x):
        return torch.fft.irfft(torch.fft.rfft(x, n=n_fft, dim=-1).abs() ** 2, n=n_fft, dim=-1)

    frame_ac = autocorr(frames * window)
    window_ac = autocorr(window)
    r = (frame_ac / frame_ac[..., :1].clamp(min=1e-10)) * (window_ac[0] / window_ac.clamp(min=1e-10))

    min_lag = int(sample_rate / PITCH_FMAX)
    max_lag = int(sample_rate / PITCH_FMIN)
    r = r[..., min_lag - 1:max_lag + 2]  # one neighbour on each side, for the peak test
    centre = r[..., 1:-1]
    is_peak = (centre > r[..., :-2]) & (centre >= r[..., 2:])

    lags = torch.arange(min_lag, max_lag + 1, device=device, dtype=dtype)
    score = centre - PITCH_OCTAVE_COST * torch.log2(PITCH_FMIN * lags / sample_rate)
    score = score.masked_fill(~is_peak, float('-inf'))
    best_score, best_idx = score.max(dim=-1)

    strength = centre.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    voiced = loud & torch.isfinite(best_score) & (strength > PITCH_VOICING_THRESHOLD)
    return sample_rate / lags[best_idx], strength, voiced


def _compute_f0_stats(waveform: torch.Tensor, sample_rate: int, track: tuple = None) -> tuple:
    """Clip-level pitch over voiced frames: median and interquartile range of F0, in semitones above 100 Hz."""
    f0, _, voiced = track if track is not None else _pitch_track(waveform, sample_rate)
    semitones = (12.0 * torch.log2(f0 / 100.0)).masked_fill(~voiced, float('nan'))
    q25, median, q75 = torch.nanquantile(semitones, torch.tensor([0.25, 0.5, 0.75], device=f0.device, dtype=f0.dtype), dim=-1)
    # Clips with no voiced frame get a mid-range pitch rather than NaN
    return torch.nan_to_num(median, nan=F0_NO_VOICE_SEMITONES), torch.nan_to_num(q75 - q25, nan=0.0)


def _compute_hnr(waveform: torch.Tensor, sample_rate: int, track: tuple = None) -> torch.Tensor:
    """Mean harmonics-to-noise ratio over voiced frames, in dB, from each frame's periodicity (Boersma 1993)."""
    _, strength, voiced = track if track is not None else _pitch_track(waveform, sample_rate)
    r = strength.clamp(1e-3, 0.999)
    hnr = 10.0 * torch.log10(r / (1.0 - r))
    voiced = voiced.to(hnr.dtype)
    return (hnr * voiced).sum(dim=-1) / voiced.sum(dim=-1).clamp(min=1.0)

def _compute_spectral_centroid(waveform: torch.Tensor, sample_rate: int, n_fft: int = 1024, hop_length: int = 512) -> torch.Tensor:
    B, T = waveform.shape
    device = waveform.device
    window = torch.hann_window(n_fft, device=device)
    if T < n_fft: waveform = F.pad(waveform, (0, n_fft - T))
    
    X = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True, center=True)
    mag = X.abs()
    freqs = torch.linspace(0, sample_rate / 2, mag.shape[1], device=device)
    
    mag_sum = mag.sum(dim=1, keepdim=True).clamp(min=1e-10)
    centroid_frames = (freqs.unsqueeze(0).unsqueeze(-1) * mag).sum(dim=1) / mag_sum.squeeze(1)
    return centroid_frames.mean(dim=-1)

def _compute_spectral_flux_var(waveform: torch.Tensor, sample_rate: int, n_fft: int = 1024, hop_length: int = 512) -> torch.Tensor:
    B, T = waveform.shape
    device = waveform.device
    window = torch.hann_window(n_fft, device=device)
    if T < n_fft: waveform = F.pad(waveform, (0, n_fft - T))
    
    X = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True, center=True)
    mag = X.abs()
    if mag.shape[-1] < 2: return torch.zeros(B, device=device)
    
    diff = mag[:, :, 1:] - mag[:, :, :-1]
    flux = torch.norm(diff, p=2, dim=1)
    return flux.var(dim=-1)

def _compute_zcr_stats(waveform: torch.Tensor) -> tuple:
    signs = torch.sign(waveform)
    sign_diff = torch.abs(signs[:, 1:] - signs[:, :-1]) / 2.0
    
    # We want frame level variance to capture cadence/rhythm
    # Let's chunk into 0.5s windows
    chunk_size = 8000 # 0.5s at 16k
    B, T_diff = sign_diff.shape
    if T_diff < chunk_size:
        return sign_diff.mean(dim=-1), torch.zeros(B, device=waveform.device)
        
    chunks = sign_diff.unfold(-1, chunk_size, chunk_size) # [B, num_chunks, chunk_size]
    chunk_means = chunks.mean(dim=-1)
    
    mean_zcr = sign_diff.mean(dim=-1)
    var_zcr = chunk_means.var(dim=-1)
    return mean_zcr, var_zcr

def _autocorr(x, order):
    n_fft = 2 ** math.ceil(math.log2(2 * x.shape[-1] - 1))
    X = torch.fft.rfft(x, n=n_fft, dim=-1)
    S = X.abs().pow(2)
    R = torch.fft.irfft(S, n=n_fft, dim=-1)
    return R[..., :order + 1]

def _levinson_durbin(R, order):
    a = torch.zeros_like(R)
    a[..., 0] = 1.0
    E = R[..., 0].clone()
    
    for i in range(1, order + 1):
        a_prev = a[..., :i]
        R_rev = R[..., 1:i+1].flip(dims=[-1])
        acc = (a_prev * R_rev).sum(dim=-1)
        
        k = -acc / (E + 1e-8)
        
        if i > 1:
            a_prev_no_0 = a[..., 1:i]
            a_prev_no_0_rev = a_prev_no_0.flip(dims=[-1])
            a[..., 1:i] = a_prev_no_0 + k.unsqueeze(-1) * a_prev_no_0_rev
        
        a[..., i] = k
        E = E * (1.0 - k**2)
        
    return a

def _compute_formants_lpc(waveform: torch.Tensor, sample_rate: int) -> tuple:
    """F1-F3 by LPC root solving (autocorrelation method) with Praat's default settings: resample to 11 kHz,
    pre-emphasis, order 10, the three lowest roots above 50 Hz. Averaged over the 10 highest-energy 30 ms frames
    (chosen on the raw signal, so they are mostly vowels). On LibriSpeech it agrees with Praat's Burg formants at
    the same frames to a median of 3 / 5 / 19 Hz (scripts/check_formant_features.py)."""
    import torchaudio.functional as AF
    B = waveform.shape[0]
    device = waveform.device
    sr = FORMANT_SAMPLE_RATE
    x = AF.resample(waveform, sample_rate, sr).double()   # float64: in float32 Levinson-Durbin is unstable here

    win_length = int(sr * 0.03)
    hop_length = int(sr * 0.02)
    if x.shape[-1] < win_length:
        x = F.pad(x, (0, win_length - x.shape[-1]))
    window = torch.hann_window(win_length, device=device, dtype=x.dtype)
    emphasized = torch.cat([x[..., :1], x[..., 1:] - LPC_PRE_EMPHASIS * x[..., :-1]], dim=-1)
    raw_frames = x.unfold(-1, win_length, hop_length) * window
    frames = emphasized.unfold(-1, win_length, hop_length) * window   # [B, num_frames, win_length]

    num_f = min(10, frames.shape[1])
    _, top_indices = raw_frames.pow(2).sum(dim=-1).topk(num_f, dim=-1)
    top_frames = torch.gather(frames, 1, top_indices.unsqueeze(-1).expand(-1, -1, win_length))

    a = _levinson_durbin(_autocorr(top_frames, LPC_ORDER), LPC_ORDER).reshape(-1, LPC_ORDER + 1)

    # Roots of the LPC polynomial = eigenvalues of its companion matrix
    companion = torch.zeros((B * num_f, LPC_ORDER, LPC_ORDER), device=device, dtype=x.dtype)
    idx = torch.arange(LPC_ORDER - 1, device=device)
    companion[:, idx + 1, idx] = 1.0
    companion[:, :, -1] = -a[:, 1:].flip(dims=[1])
    roots = torch.linalg.eigvals(companion)

    freqs = torch.angle(roots) * (sr / (2 * math.pi))
    mask = (roots.imag > 0) & (freqs > FORMANT_MIN_HZ) & (freqs < sr / 2 - FORMANT_MIN_HZ)
    valid_f, _ = torch.where(mask, freqs, torch.full_like(freqs, 1e5)).sort(dim=-1)

    # Fallbacks for frames with fewer than 3 roots (e.g. silence)
    f1s = torch.where(valid_f[:, 0] < 1e5, valid_f[:, 0], torch.full_like(valid_f[:, 0], 500.0))
    f2s = torch.where(valid_f[:, 1] < 1e5, valid_f[:, 1], f1s + 1000.0)
    f3s = torch.where(valid_f[:, 2] < 1e5, valid_f[:, 2], f2s + 1000.0)

    return tuple(f.view(B, num_f).mean(dim=1).to(waveform.dtype) for f in (f1s, f2s, f3s))

def _compute_mfccs(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    import torchaudio.transforms as T
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=5,
        melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 40, "center": True}
    ).to(waveform.device)
    mfcc = mfcc_transform(waveform) # [B, n_mfcc, T_frames]
    # We return the mean of the first 5 coefficients
    return mfcc[:, :5, :].mean(dim=-1) # [B, 5]


def extract_acoustic_features(
    audio_tensor: torch.Tensor,
    sample_rate: int = 16000,
    feature_stats: dict = None,
    features_list: str = "f0,hnr,centroid,flux,zcr",
    normalize: bool = True
) -> torch.Tensor:
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(1)
    
    device = audio_tensor.device
    selected = [f.strip().lower() for f in features_list.split(',')]
    
    computed_feats = []
    
    with torch.no_grad():
        track = None
        if any(f in selected for f in ['f0', 'f0_mean', 'f0_var', 'hnr']):
            track = _pitch_track(audio_tensor, sample_rate)
        if any(f in selected for f in ['f0', 'f0_mean', 'f0_var']):
            m_f0, v_f0 = _compute_f0_stats(audio_tensor, sample_rate, track)
        if any(f in selected for f in ['zcr', 'zcr_mean', 'zcr_var', 'rhythm']):
            m_zcr, v_zcr = _compute_zcr_stats(audio_tensor)
        if any(f in selected for f in ['formants', 'f1', 'f2', 'f3']):
            f1, f2, f3 = _compute_formants_lpc(audio_tensor, sample_rate)
            
        for f in selected:
            if f == 'f0' or f == 'f0_mean': computed_feats.append(m_f0)
            elif f == 'f0_var': computed_feats.append(v_f0)
            elif f == 'hnr': computed_feats.append(_compute_hnr(audio_tensor, sample_rate, track))
            elif f == 'centroid': computed_feats.append(_compute_spectral_centroid(audio_tensor, sample_rate))
            elif f == 'flux' or f == 'flux_var': computed_feats.append(_compute_spectral_flux_var(audio_tensor, sample_rate))
            elif f == 'zcr' or f == 'zcr_mean': computed_feats.append(m_zcr)
            elif f == 'zcr_var' or f == 'rhythm': computed_feats.append(v_zcr)
            elif f == 'formants': 
                computed_feats.extend([f1, f2, f3])
            elif f == 'f1': computed_feats.append(f1)
            elif f == 'f2': computed_feats.append(f2)
            elif f == 'f3': computed_feats.append(f3)
            elif f == 'mfcc': 
                mfccs = _compute_mfccs(audio_tensor, sample_rate)
                for i in range(mfccs.shape[1]):
                    computed_feats.append(mfccs[:, i])
                    
    features = torch.stack(computed_feats, dim=1) # [B, K]

    # Normalize
    if normalize:
        if feature_stats is not None and 'mean' in feature_stats and 'std' in feature_stats:
            mean = feature_stats['mean'].to(device)
            std = feature_stats['std'].to(device)
            # Handle shape mismatch if global stats were computed with different K
            if mean.shape[0] == features.shape[1]:
                features = (features - mean.unsqueeze(0)) / (3.0 * std.unsqueeze(0) + 1e-6)
            else:
                mean = features.mean(dim=0, keepdim=True)
                std = features.std(dim=0, keepdim=True)
                features = (features - mean) / (3.0 * std + 1e-6)
        else:
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            features = (features - mean) / (3.0 * std + 1e-6)

        features = features.clamp(-1.0, 1.0)
        
    return features


def calculate_acoustic_feature_stats(
    dataloader,
    sample_rate: int = 16000,
    num_batches: int = 50,
    device: torch.device = None,
    features_list: str = "f0,hnr,centroid,flux,zcr"
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_features = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            if len(batch) >= 3:
                _, waveform, _ = batch
            else:
                waveform = batch[0]
                
            waveform = waveform.to(device)
            feats = extract_acoustic_features(waveform, sample_rate, feature_stats=None, features_list=features_list, normalize=False)
            all_features.append(feats.cpu())

    all_features = torch.cat(all_features, dim=0)
    
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)
    
    # Normalize features to match what the model will see during training
    norm_features = (all_features - mean.unsqueeze(0)) / (3.0 * std.unsqueeze(0) + 1e-6)
    norm_features = norm_features.clamp(-1.0, 1.0)
    
    # Helper to parse feature groups (shared with sac_model.py)
    active_groups, _ = get_feature_groups(features_list)
    
    # Subsample to avoid memory explosion with large num_batches
    if norm_features.shape[0] > 2000:
        rand_idx = torch.randperm(norm_features.shape[0])[:2000]
        norm_features = norm_features[rand_idx]
        
    group_medians = {}
    for group_name, idxs in active_groups.items():
        group_feats = norm_features[:, idxs]
        dist_matrix = torch.cdist(group_feats, group_feats, p=2)
        
        N = dist_matrix.shape[0]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=dist_matrix.device)
        off_diag_dists = dist_matrix[off_diag_mask]
        
        median_dist = off_diag_dists.median().item()
        # Prevent division by zero if a group is completely collapsed
        group_medians[group_name] = max(median_dist, 1e-4)

    print("\n--- Offline Global Medians Computed ---")
    for group_name, m_val in group_medians.items():
        print(f"  {group_name}: {m_val:.6f}")
    print("---------------------------------------\n")

    return {
        'mean': mean,
        'std': std,
        'group_medians': group_medians
    }


def get_feature_groups(features_list: str):
    """
    Parses a comma-separated list of acoustic features and maps them to their
    respective feature groups. Used to ensure consistency across the codebase.
    """
    groups = {
        'Prosody': ['f0', 'f0_mean', 'f0_var'],
        'Vocal_Tract': ['formants', 'f1', 'f2', 'f3'],
        'Timbre': ['mfcc'],
        'Voice_Quality': ['hnr'],
        'Scene': ['centroid', 'flux', 'flux_var', 'zcr', 'zcr_mean', 'zcr_var', 'rhythm']
    }
    selected = [f.strip().lower() for f in features_list.split(',')]
    group_indices = {k: [] for k in groups}
    current_idx = 0
    for f in selected:
        num_feats = 3 if f == 'formants' else 5 if f == 'mfcc' else 1
        for group_name, group_members in groups.items():
            if f in group_members:
                group_indices[group_name].extend(list(range(current_idx, current_idx + num_feats)))
                break
        current_idx += num_feats
    active_groups = {k: v for k, v in group_indices.items() if len(v) > 0}
    return active_groups, len(active_groups)
