"""
Step 1: Single-Channel Acoustic Feature Extraction for SAC Loss.

Features extracted per clip:
    - f0_mean / f0_var
    - hnr
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

def _safe_log(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return torch.log(x.clamp(min=eps))

def _compute_f0_stats(waveform: torch.Tensor, sample_rate: int, fmin: float = 80.0, fmax: float = 600.0, hop_length: int = 512) -> tuple:
    B, T = waveform.shape
    device = waveform.device
    min_lag = int(sample_rate / fmax)
    max_lag = int(sample_rate / fmin)
    
    pad_len = max_lag
    padded = F.pad(waveform, (pad_len, pad_len))
    
    frames = padded.unfold(-1, max_lag * 2 + 1, hop_length)
    frames = frames * torch.hann_window(frames.shape[-1], device=device)
    
    fft_size = int(2 ** math.ceil(math.log2(frames.shape[-1] * 2)))
    S = torch.fft.rfft(frames, n=fft_size, dim=-1)
    power = torch.abs(S) ** 2
    auto_corr = torch.fft.irfft(power, n=fft_size, dim=-1)
    
    auto_corr = auto_corr[..., :max_lag + 1]
    auto_corr = auto_corr[..., min_lag:]
    
    val, idx = torch.max(auto_corr, dim=-1)
    lags = idx + min_lag
    f0 = sample_rate / lags.float()
    
    # Simple unvoiced masking (if autocorrelation peak is very small)
    max_corr_val = auto_corr[..., 0] if auto_corr.shape[-1] > 0 else 1.0
    voiced_mask = val > (0.2 * max_corr_val.clamp(min=1e-6))
    f0 = f0 * voiced_mask.float()
    
    # We want mean and var, but ignore exactly 0 (unvoiced) frames
    f0_sum = f0.sum(dim=-1)
    f0_count = voiced_mask.float().sum(dim=-1).clamp(min=1.0)
    mean_f0 = f0_sum / f0_count
    
    # Var
    diff_sq = ((f0 - mean_f0.unsqueeze(-1)) ** 2) * voiced_mask.float()
    var_f0 = diff_sq.sum(dim=-1) / f0_count
    
    return mean_f0, var_f0

def _compute_hnr(waveform: torch.Tensor, sample_rate: int, fmin: float = 80.0) -> torch.Tensor:
    B, T = waveform.shape
    device = waveform.device
    max_lag = int(sample_rate / fmin)
    if T <= max_lag: return torch.zeros(B, device=device)
    
    fft_size = int(2 ** math.ceil(math.log2(T * 2)))
    S = torch.fft.rfft(waveform, n=fft_size, dim=-1)
    power = torch.abs(S) ** 2
    auto_corr = torch.fft.irfft(power, n=fft_size, dim=-1)
    auto_corr = auto_corr[..., :max_lag + 1]
    
    R0 = auto_corr[:, 0].clamp(min=1e-8)
    R_max, _ = torch.max(auto_corr[:, int(sample_rate / 600.0):], dim=-1)
    
    hnr = R_max / (R0 - R_max).clamp(min=1e-8)
    return 10.0 * torch.log10(hnr.clamp(min=1e-5))

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

def _compute_formants_proxy(waveform: torch.Tensor, sample_rate: int, n_fft=1024, hop_length=512) -> tuple:
    B, T = waveform.shape
    device = waveform.device
    window = torch.hann_window(n_fft, device=device)
    if T < n_fft: waveform = F.pad(waveform, (0, n_fft - T))
    
    X = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True, center=True)
    mag = X.abs().mean(dim=-1) # Mean over time to get global spectral envelope
    freqs = torch.linspace(0, sample_rate / 2, mag.shape[1], device=device)
    
    # F1 proxy: centroid in 300-1000 Hz
    mask1 = (freqs >= 300) & (freqs <= 1000)
    f1 = (freqs[mask1].unsqueeze(0) * mag[:, mask1]).sum(dim=1) / mag[:, mask1].sum(dim=1).clamp(min=1e-10)
    
    # F2 proxy: centroid in 1000-2500 Hz
    mask2 = (freqs > 1000) & (freqs <= 2500)
    f2 = (freqs[mask2].unsqueeze(0) * mag[:, mask2]).sum(dim=1) / mag[:, mask2].sum(dim=1).clamp(min=1e-10)
    
    # F3 proxy: centroid in 2500-3500 Hz
    mask3 = (freqs > 2500) & (freqs <= 3500)
    f3 = (freqs[mask3].unsqueeze(0) * mag[:, mask3]).sum(dim=1) / mag[:, mask3].sum(dim=1).clamp(min=1e-10)
    
    return f1, f2, f3

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
        if any(f in selected for f in ['f0', 'f0_mean', 'f0_var']):
            m_f0, v_f0 = _compute_f0_stats(audio_tensor, sample_rate)
        if any(f in selected for f in ['zcr', 'zcr_mean', 'zcr_var', 'rhythm']):
            m_zcr, v_zcr = _compute_zcr_stats(audio_tensor)
        if any(f in selected for f in ['formants', 'f1', 'f2', 'f3']):
            f1, f2, f3 = _compute_formants_proxy(audio_tensor, sample_rate)
            
        for f in selected:
            if f == 'f0' or f == 'f0_mean': computed_feats.append(m_f0)
            elif f == 'f0_var': computed_feats.append(v_f0)
            elif f == 'hnr': computed_feats.append(_compute_hnr(audio_tensor, sample_rate))
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
    
    # Helper to parse feature families (shared with sac_model.py)
    active_families, _ = get_feature_families(features_list)
    
    # Subsample to avoid memory explosion with large num_batches
    if norm_features.shape[0] > 2000:
        rand_idx = torch.randperm(norm_features.shape[0])[:2000]
        norm_features = norm_features[rand_idx]
        
    family_medians = {}
    for fam_name, idxs in active_families.items():
        fam_feats = norm_features[:, idxs]
        dist_matrix = torch.cdist(fam_feats, fam_feats, p=2)
        
        N = dist_matrix.shape[0]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=dist_matrix.device)
        off_diag_dists = dist_matrix[off_diag_mask]
        
        median_dist = off_diag_dists.median().item()
        # Prevent division by zero if a family is completely collapsed
        family_medians[fam_name] = max(median_dist, 1e-4)

    print("\n--- Offline Global Medians Computed ---")
    for fam_name, m_val in family_medians.items():
        print(f"  {fam_name}: {m_val:.6f}")
    print("---------------------------------------\n")

    return {
        'mean': mean,
        'std': std,
        'family_medians': family_medians
    }


def get_feature_families(features_list: str):
    """
    Parses a comma-separated list of acoustic features and maps them to their
    respective feature families. Used to ensure consistency across the codebase.
    """
    families = {
        'Prosody': ['f0', 'f0_mean', 'f0_var'],
        'Vocal_Tract': ['formants', 'f1', 'f2', 'f3'],
        'Timbre': ['mfcc'],
        'Voice_Quality': ['hnr'],
        'Scene': ['centroid', 'flux', 'flux_var', 'zcr', 'zcr_mean', 'zcr_var', 'rhythm']
    }
    selected = [f.strip().lower() for f in features_list.split(',')]
    family_indices = {k: [] for k in families}
    current_idx = 0
    for f in selected:
        num_feats = 3 if f == 'formants' else 5 if f == 'mfcc' else 1
        for fam_name, fam_members in families.items():
            if f in fam_members:
                family_indices[fam_name].extend(list(range(current_idx, current_idx + num_feats)))
                break
        current_idx += num_feats
    active_families = {k: v for k, v in family_indices.items() if len(v) > 0}
    return active_families, len(active_families)
