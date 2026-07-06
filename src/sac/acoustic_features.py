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
    B, T = waveform.shape
    device = waveform.device
    
    # 30ms window, 20ms hop
    win_length = int(sample_rate * 0.03)
    hop_length = int(sample_rate * 0.02)
    window = torch.hann_window(win_length, device=device)
    
    if T < win_length:
        waveform = F.pad(waveform, (0, win_length - T))
        
    frames = waveform.unfold(-1, win_length, hop_length) # [B, num_frames, win_length]
    frames = frames * window
    
    # Select top 10 highest energy frames to represent the vowel sounds
    energy = frames.pow(2).sum(dim=-1)
    num_f = min(10, frames.shape[1])
    _, top_indices = energy.topk(num_f, dim=-1)
    
    top_frames = torch.gather(frames, 1, top_indices.unsqueeze(-1).expand(-1, -1, win_length)) # [B, num_f, win_length]
    
    # LPC order: typically 2 + sample_rate / 1000
    order = int(2 + sample_rate / 1000) # For 16k -> 18
    
    R = _autocorr(top_frames, order)
    a_coeffs = _levinson_durbin(R, order) # [B, num_f, order+1]
    
    B_size = B
    a_coeffs_flat = a_coeffs.reshape(-1, order + 1)
    
    # Companion matrix
    companion = torch.zeros((B_size * num_f, order, order), device=device)
    idx = torch.arange(order - 1, device=device)
    companion[:, idx + 1, idx] = 1.0
    companion[:, :, -1] = -a_coeffs_flat[:, 1:].flip(dims=[1])
    
    # Eigenvalues (roots of LPC polynomial)
    roots = torch.linalg.eigvals(companion) # [B*num_f, order]
    
    # Frequencies from roots
    angles = torch.angle(roots)
    freqs = angles * (sample_rate / (2 * math.pi))
    
    # Filter valid formants (positive frequencies, upper half of complex plane)
    mask = (freqs > 50) & (roots.imag > 0)
    valid_f = torch.where(mask, freqs, torch.full_like(freqs, 1e5))
    valid_f, _ = valid_f.sort(dim=-1)
    
    f1s = valid_f[:, 0]
    f2s = valid_f[:, 1]
    f3s = valid_f[:, 2]
    
    # Fallbacks for failed frames
    f1s = torch.where(f1s < 1e5, f1s, torch.tensor(500.0, device=device))
    f2s = torch.where(f2s < 1e5, f2s, f1s + 1000.0)
    f3s = torch.where(f3s < 1e5, f3s, f2s + 1000.0)
    
    f1_mean = f1s.view(B_size, num_f).mean(dim=1)
    f2_mean = f2s.view(B_size, num_f).mean(dim=1)
    f3_mean = f3s.view(B_size, num_f).mean(dim=1)
    
    return f1_mean, f2_mean, f3_mean

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
            f1, f2, f3 = _compute_formants_lpc(audio_tensor, sample_rate)
            
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
