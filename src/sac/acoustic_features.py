"""
Clip-level acoustic features for the SAC loss, following the eGeMAPS definitions (Eyben et al., 2016) where one exists:
each low-level descriptor is computed every 10 ms (the pitch-track grid) and summarised over voiced frames (voice
features) or non-silent frames (others), so pauses and zero padding do not enter the averages.

    - f0_mean / f0_var: mean and coefficient of variation of F0 over voiced frames, in semitones from 27.5 Hz
    - voiced_rate, voiced_len, unvoiced_len: voiced segments per second, mean voiced and unvoiced segment length (s),
      within the span from the first to the last non-silent frame
    - formant_stats: mean and coefficient of variation of F1-F3 over voiced frames (LPC, Praat's default settings)
    - formants: mean F1-F3 over the 10 loudest frames (older definition, kept for comparison)
    - mfcc: mean of MFCC 1-4 over voiced frames (no c0, which is log energy)
    - hnr: mean harmonics-to-noise ratio over voiced frames, in dB
    - loudness_cov: coefficient of variation of loudness over non-silent frames (level-invariant loudness dynamics)
    - centroid: mean spectral centroid over non-silent frames
    - flux / flux_cov: mean and coefficient of variation over voiced frames of the spectral flux between
      level-normalised magnitude spectra
    - zcr_mean: mean zero-crossing rate over non-silent frames

Output: normalized feature vector c_i in [-1, 1]^K for each sample in the batch.
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
F0_REFERENCE_HZ = 27.5           # eGeMAPS: F0 in semitones from 27.5 Hz
F0_NO_VOICE_SEMITONES = 32.3      # about 178 Hz from 27.5 Hz, for clips without voice
MIN_VOICED_FRAMES = 5             # fewer voiced frames than this counts as no voice

# Spectral descriptors: 25 ms Hann windows centred on the pitch-track frames (eGeMAPS uses 25 ms for these)
SPECTRAL_FRAME_SEC = 0.025
SPECTRAL_N_FFT = 512
MFCC_N_MELS = 26
LOUDNESS_EXPONENT = 0.33          # intensity-to-loudness power law (Stevens), applied to mel-band powers

# LPC formants, with Praat's default formant settings (5 formants up to 5500 Hz)
FORMANT_SAMPLE_RATE = 11000       # twice the 5500 Hz ceiling
LPC_ORDER = 10                    # two poles per formant
LPC_PRE_EMPHASIS = 0.97
FORMANT_MIN_HZ = 50.0
FORMANT_NO_VOICE = (500.0, 1500.0, 2500.0)   # formant_stats: means for clips without voice (neutral vowel)

def _safe_log(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return torch.log(x.clamp(min=eps))

def _loud(frames: torch.Tensor) -> torch.Tensor:
    """Non-silent frames: RMS above PITCH_SILENCE_RATIO of the clip's loudest frame. frames [B, N, L] -> [B, N]."""
    rms = frames.pow(2).mean(dim=-1).sqrt()
    return rms > PITCH_SILENCE_RATIO * rms.amax(dim=-1, keepdim=True)


def _loud_frames(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """The pitch track's non-silent mask [B, N] (same frames), for features that don't need voicing."""
    win_length = int(sample_rate * PITCH_FRAME_SEC)
    if waveform.shape[-1] < win_length:
        waveform = F.pad(waveform, (0, win_length - waveform.shape[-1]))
    frames = waveform.unfold(-1, win_length, int(sample_rate * PITCH_HOP_SEC))
    return _loud(frames - frames.mean(dim=-1, keepdim=True))


def _masked_mean_std(values: torch.Tensor, mask: torch.Tensor):
    """Mean and standard deviation (n - 1) of values [B, N] or [B, N, K] over frames where mask [B, N] is true and
    the value is not NaN. Returns mean, std, count, each [B] or [B, K]."""
    if values.dim() == 3:
        mask = mask.unsqueeze(-1)
    use = mask & ~torch.isnan(values)
    count = use.sum(dim=1).to(values.dtype)
    x = torch.nan_to_num(values) * use
    mean = x.sum(dim=1) / count.clamp(min=1)
    var = ((x - mean.unsqueeze(1)) * use).pow(2).sum(dim=1) / (count - 1).clamp(min=1)
    return mean, var.sqrt(), count


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
    loud = _loud(frames)

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
    """eGeMAPS F0: mean and coefficient of variation over voiced frames, in semitones from 27.5 Hz."""
    f0, _, voiced = track if track is not None else _pitch_track(waveform, sample_rate)
    semitones = 12.0 * torch.log2(f0 / F0_REFERENCE_HZ)
    mean, std, count = _masked_mean_std(semitones, voiced)
    enough = count >= MIN_VOICED_FRAMES
    mean_out = torch.where(enough, mean, torch.full_like(mean, F0_NO_VOICE_SEMITONES))
    return mean_out, torch.where(enough, std / mean.clamp(min=1.0), torch.zeros_like(mean))


def _compute_hnr(waveform: torch.Tensor, sample_rate: int, track: tuple = None) -> torch.Tensor:
    """Mean harmonics-to-noise ratio over voiced frames, in dB, from each frame's periodicity (Boersma 1993)."""
    _, strength, voiced = track if track is not None else _pitch_track(waveform, sample_rate)
    r = strength.clamp(1e-3, 0.999)
    hnr = 10.0 * torch.log10(r / (1.0 - r))
    voiced = voiced.to(hnr.dtype)
    return (hnr * voiced).sum(dim=-1) / voiced.sum(dim=-1).clamp(min=1.0)

def _spectral_frames(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """25 ms frames, 10 ms hop, centred on the pitch-track frames, so frame i lines up with voiced[:, i]. Unwindowed."""
    win = int(sample_rate * SPECTRAL_FRAME_SEC)
    offset = (int(sample_rate * PITCH_FRAME_SEC) - win) // 2
    x = waveform[..., offset:]
    if x.shape[-1] < win:
        x = F.pad(x, (0, win - x.shape[-1]))
    return x.unfold(-1, win, int(sample_rate * PITCH_HOP_SEC))


def _magnitude_spectra(frames: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(frames.shape[-1], device=frames.device, dtype=frames.dtype)
    return torch.fft.rfft(frames * window, n=SPECTRAL_N_FFT, dim=-1).abs()   # [B, N, n_fft/2 + 1]


def _trim(x: torch.Tensor, mask: torch.Tensor):
    n = min(x.shape[1], mask.shape[1])
    return x[:, :n], mask[:, :n]


def _compute_spectral_centroid(mag: torch.Tensor, loud: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Mean magnitude-weighted spectral centroid (Hz) over non-silent frames."""
    freqs = torch.linspace(0, sample_rate / 2, mag.shape[-1], device=mag.device, dtype=mag.dtype)
    centroid = (mag * freqs).sum(dim=-1) / mag.sum(dim=-1).clamp(min=1e-10)
    return _masked_mean_std(*_trim(centroid, loud))[0]


def _compute_spectral_flux(mag: torch.Tensor, voiced: torch.Tensor) -> tuple:
    """Spectral flux between consecutive level-normalised magnitude spectra (L2), mean and coefficient of variation
    over voiced frames (eGeMAPS: voiced-region flux)."""
    p = mag / mag.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    flux = F.pad((p[:, 1:] - p[:, :-1]).norm(dim=-1), (1, 0), value=float('nan'))   # no flux for the first frame
    mean, std, count = _masked_mean_std(*_trim(flux, voiced))
    enough = count >= MIN_VOICED_FRAMES
    return torch.where(enough, mean, torch.zeros_like(mean)), torch.where(enough, std / mean.clamp(min=1e-6), torch.zeros_like(mean))


def _compute_zcr(frames: torch.Tensor, loud: torch.Tensor) -> torch.Tensor:
    """Mean zero-crossing rate (crossings per sample) over non-silent frames."""
    positive = frames >= 0
    zcr = (positive[..., 1:] != positive[..., :-1]).to(frames.dtype).mean(dim=-1)
    return _masked_mean_std(*_trim(zcr, loud))[0]


def _compute_mfccs(mag: torch.Tensor, voiced: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Mean of MFCC 1-4 over voiced frames (26 mel bands, log power in dB, orthonormal DCT; c0 is dropped)."""
    import torchaudio.functional as AF
    fb = AF.melscale_fbanks(mag.shape[-1], 0.0, sample_rate / 2, MFCC_N_MELS, sample_rate).to(mag.device, mag.dtype)
    dct = AF.create_dct(5, MFCC_N_MELS, norm='ortho').to(mag.device, mag.dtype)
    mfcc = (10.0 * torch.log10((mag.pow(2) @ fb).clamp(min=1e-10))) @ dct     # [B, N, 5]
    mean, _, count = _masked_mean_std(*_trim(mfcc[..., 1:], voiced))
    return torch.where((count >= MIN_VOICED_FRAMES), mean, torch.zeros_like(mean))   # [B, 4]


def _compute_loudness_cov(mag: torch.Tensor, loud: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Coefficient of variation over non-silent frames of a loudness estimate: sum of the 26 mel-band powers, each
    compressed by the 0.33 power law. A gain change scales every frame equally, so the CoV does not depend on level."""
    import torchaudio.functional as AF
    fb = AF.melscale_fbanks(mag.shape[-1], 0.0, sample_rate / 2, MFCC_N_MELS, sample_rate).to(mag.device, mag.dtype)
    loudness = (mag.pow(2) @ fb).clamp(min=0).pow(LOUDNESS_EXPONENT).sum(dim=-1)
    mean, std, count = _masked_mean_std(*_trim(loudness, loud))
    return torch.where(count >= MIN_VOICED_FRAMES, std / mean.clamp(min=1e-10), torch.zeros_like(mean))


def _compute_temporal(voiced: torch.Tensor, loud: torch.Tensor) -> torch.Tensor:
    """eGeMAPS temporal features within the span from the first to the last non-silent frame: voiced segments per
    second, mean voiced segment length (s), mean unvoiced segment length (s). Returns [B, 3]."""
    hop = PITCH_HOP_SEC
    idx = torch.arange(loud.shape[1], device=loud.device)
    first = torch.where(loud, idx, loud.shape[1]).amin(dim=1, keepdim=True)
    last = torch.where(loud, idx, -1).amax(dim=1, keepdim=True)
    span = (idx >= first) & (idx <= last)
    v = voiced & span
    u = span & ~voiced

    def segments(m):
        return (m & ~F.pad(m[:, :-1], (1, 0), value=False)).sum(dim=1).float()

    seconds = span.sum(dim=1).float() * hop
    n_v, n_u = segments(v), segments(u)
    rate = n_v / seconds.clamp(min=hop)
    v_len = v.sum(dim=1).float() * hop / n_v.clamp(min=1)
    u_len = u.sum(dim=1).float() * hop / n_u.clamp(min=1)
    return torch.stack([rate, v_len, u_len], dim=1)


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

def _lpc_formant_frames(waveform: torch.Tensor, sample_rate: int, hop_sec: float):
    """Per-frame F1-F3 by LPC root solving (autocorrelation method) with Praat's default settings: resample to
    11 kHz, pre-emphasis, order 10, the three lowest roots above 50 Hz, 30 ms Hann windows.
    Returns formants [B, N, 3] (NaN where a frame has fewer than 3 roots) and raw-signal frame energy [B, N]."""
    import torchaudio.functional as AF
    device = waveform.device
    sr = FORMANT_SAMPLE_RATE
    x = AF.resample(waveform, sample_rate, sr).double()   # float64: in float32 Levinson-Durbin is unstable here

    win_length = int(sr * 0.03)
    hop_length = int(round(sr * hop_sec))
    if x.shape[-1] < win_length:
        x = F.pad(x, (0, win_length - x.shape[-1]))
    window = torch.hann_window(win_length, device=device, dtype=x.dtype)
    emphasized = torch.cat([x[..., :1], x[..., 1:] - LPC_PRE_EMPHASIS * x[..., :-1]], dim=-1)
    energy = (x.unfold(-1, win_length, hop_length) * window).pow(2).sum(dim=-1)
    frames = emphasized.unfold(-1, win_length, hop_length) * window   # [B, N, win_length]
    B, N, _ = frames.shape

    a = _levinson_durbin(_autocorr(frames, LPC_ORDER), LPC_ORDER).reshape(-1, LPC_ORDER + 1)
    # Roots of the LPC polynomial = eigenvalues of its companion matrix
    companion = torch.zeros((B * N, LPC_ORDER, LPC_ORDER), device=device, dtype=x.dtype)
    idx = torch.arange(LPC_ORDER - 1, device=device)
    companion[:, idx + 1, idx] = 1.0
    companion[:, :, -1] = -a[:, 1:].flip(dims=[1])
    roots = torch.linalg.eigvals(companion)

    freqs = torch.angle(roots) * (sr / (2 * math.pi))
    mask = (roots.imag > 0) & (freqs > FORMANT_MIN_HZ) & (freqs < sr / 2 - FORMANT_MIN_HZ)
    valid_f, _ = torch.where(mask, freqs, torch.full_like(freqs, float('inf'))).sort(dim=-1)
    formants = valid_f[:, :3].reshape(B, N, 3)
    return torch.where(torch.isfinite(formants), formants, torch.full_like(formants, float('nan'))), energy


def _compute_formants_lpc(waveform: torch.Tensor, sample_rate: int) -> tuple:
    """Mean F1-F3 over the 10 highest-energy frames (chosen on the raw signal, so mostly vowels). On LibriSpeech it
    agrees with Praat's Burg formants at the same frames to a median of 3 / 5 / 19 Hz (scripts/check_formant_features.py)."""
    formants, energy = _lpc_formant_frames(waveform, sample_rate, hop_sec=0.02)
    num_f = min(10, formants.shape[1])
    top = energy.topk(num_f, dim=-1).indices
    f = torch.gather(formants, 1, top.unsqueeze(-1).expand(-1, -1, 3))   # [B, num_f, 3]

    # Fallbacks for frames with fewer than 3 roots (e.g. silence)
    f1s = torch.nan_to_num(f[..., 0], nan=500.0)
    f2s = torch.where(torch.isnan(f[..., 1]), f1s + 1000.0, f[..., 1])
    f3s = torch.where(torch.isnan(f[..., 2]), f2s + 1000.0, f[..., 2])
    return tuple(v.mean(dim=1).to(waveform.dtype) for v in (f1s, f2s, f3s))


def _compute_formant_stats(waveform: torch.Tensor, sample_rate: int, track: tuple = None) -> torch.Tensor:
    """eGeMAPS-style formant functionals (Eyben et al., 2016): mean and coefficient of variation of F1-F3 over
    voiced frames. Formants per 10 ms frame as in _lpc_formant_frames; voicing from _pitch_track (same 10 ms hop;
    frame centres differ by 5 ms). Clips with fewer than MIN_VOICED_FRAMES voiced frames get
    FORMANT_NO_VOICE means and zero variation. Returns [B, 6]: mean F1, F2, F3, CoV F1, F2, F3."""
    formants, _ = _lpc_formant_frames(waveform, sample_rate, hop_sec=PITCH_HOP_SEC)
    voiced = (track if track is not None else _pitch_track(waveform, sample_rate))[2]
    n = min(formants.shape[1], voiced.shape[1])
    formants, voiced = formants[:, :n], voiced[:, :n]

    mean, std, count = _masked_mean_std(formants, voiced)
    cov = std / mean.clamp(min=1.0)

    enough = count >= MIN_VOICED_FRAMES
    no_voice = torch.tensor(FORMANT_NO_VOICE, device=mean.device, dtype=mean.dtype).expand_as(mean)
    mean = torch.where(enough, mean, no_voice)
    cov = torch.where(enough, cov, torch.zeros_like(cov))
    return torch.cat([mean, cov], dim=-1).to(waveform.dtype)

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
    
    legacy = {'zcr_var', 'rhythm', 'flux_var'} & set(selected)
    if legacy:
        raise ValueError(f'{sorted(legacy)} were replaced by voiced_rate / voiced_len / unvoiced_len and flux_cov')

    with torch.no_grad():
        x = audio_tensor
        voiced_feats = {'f0', 'f0_mean', 'f0_var', 'hnr', 'formant_stats', 'mfcc', 'flux', 'flux_cov',
                        'voiced_rate', 'voiced_len', 'unvoiced_len'}
        track = _pitch_track(x, sample_rate) if voiced_feats & set(selected) else None
        voiced = track[2] if track is not None else None
        loud = _loud_frames(x, sample_rate) if {'centroid', 'zcr', 'zcr_mean', 'loudness_cov', 'voiced_rate', 'voiced_len', 'unvoiced_len'} & set(selected) else None
        frames = _spectral_frames(x, sample_rate) if {'centroid', 'flux', 'flux_cov', 'mfcc', 'loudness_cov', 'zcr', 'zcr_mean'} & set(selected) else None
        mag = _magnitude_spectra(frames) if {'centroid', 'flux', 'flux_cov', 'mfcc', 'loudness_cov'} & set(selected) else None
        if {'f0', 'f0_mean', 'f0_var'} & set(selected):
            m_f0, v_f0 = _compute_f0_stats(x, sample_rate, track)
        if {'flux', 'flux_cov'} & set(selected):
            m_flux, cov_flux = _compute_spectral_flux(mag, voiced)
        if {'voiced_rate', 'voiced_len', 'unvoiced_len'} & set(selected):
            temporal = _compute_temporal(*_trim(voiced, loud))
        if {'formants', 'f1', 'f2', 'f3'} & set(selected):
            f1, f2, f3 = _compute_formants_lpc(x, sample_rate)

        for f in selected:
            if f in ('f0', 'f0_mean'): computed_feats.append(m_f0)
            elif f == 'f0_var': computed_feats.append(v_f0)
            elif f == 'voiced_rate': computed_feats.append(temporal[:, 0])
            elif f == 'voiced_len': computed_feats.append(temporal[:, 1])
            elif f == 'unvoiced_len': computed_feats.append(temporal[:, 2])
            elif f == 'hnr': computed_feats.append(_compute_hnr(x, sample_rate, track))
            elif f == 'loudness_cov': computed_feats.append(_compute_loudness_cov(mag, loud, sample_rate))
            elif f == 'centroid': computed_feats.append(_compute_spectral_centroid(mag, loud, sample_rate))
            elif f == 'flux': computed_feats.append(m_flux)
            elif f == 'flux_cov': computed_feats.append(cov_flux)
            elif f in ('zcr', 'zcr_mean'): computed_feats.append(_compute_zcr(frames, loud))
            elif f == 'formants': computed_feats.extend([f1, f2, f3])
            elif f == 'formant_stats':
                computed_feats.extend(_compute_formant_stats(x, sample_rate, track).unbind(dim=1))
            elif f == 'f1': computed_feats.append(f1)
            elif f == 'f2': computed_feats.append(f2)
            elif f == 'f3': computed_feats.append(f3)
            elif f == 'mfcc': computed_feats.extend(_compute_mfccs(mag, voiced, sample_rate).unbind(dim=1))
            else: raise ValueError(f'unknown acoustic feature {f!r}')
        computed_feats = [c.to(x.dtype) for c in computed_feats]

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
        'Prosody': ['f0', 'f0_mean', 'f0_var', 'voiced_rate', 'voiced_len', 'unvoiced_len'],
        'Vocal_Tract': ['formants', 'formant_stats', 'f1', 'f2', 'f3'],
        'Timbre': ['mfcc'],
        'Voice_Quality': ['hnr', 'loudness_cov'],
        'Scene': ['centroid', 'flux', 'flux_cov', 'zcr', 'zcr_mean']
    }
    selected = [f.strip().lower() for f in features_list.split(',')]
    group_indices = {k: [] for k in groups}
    current_idx = 0
    for f in selected:
        num_feats = {'formants': 3, 'formant_stats': 6, 'mfcc': 4}.get(f, 1)
        for group_name, group_members in groups.items():
            if f in group_members:
                group_indices[group_name].extend(list(range(current_idx, current_idx + num_feats)))
                break
        current_idx += num_feats
    active_groups = {k: v for k, v in group_indices.items() if len(v) > 0}
    return active_groups, len(active_groups)
