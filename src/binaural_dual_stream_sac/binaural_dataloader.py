import json
import torchaudio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from binaural_dual_stream_sac.spatial_features import extract_spatial_features


def generate_ccsr_time_mask(T, mask_ratio=0.5):
    """
    Generates a binary time-frame mask W of shape [1, 1, T].
    1 = unmasked, 0 = masked.
    """
    num_mask = int(T * mask_ratio)
    W = torch.ones(T, dtype=torch.float32)
    mask_indices = torch.randperm(T)[:num_mask]
    W[mask_indices] = 0.0
    return W.view(1, 1, T)


import logging


def wav2stft(waveform, n_fft=512, hop_length=160, target_freq_bins=256, target_length=1024):
    """
    Extracts 4-channel Complex STFT: [4, F, T] from a 1D or 2D waveform tensor.
    If waveform is mono (1 channel), duplicates to 2 channels and logs a warning.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] == 1:
        logging.warning("Mono waveform encountered; duplicating to 2 channels for binaural STFT.")
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    # STFT extraction for Left and Right channels
    stft_L = torch.stft(waveform[0], n_fft=n_fft, hop_length=hop_length, return_complex=True)
    stft_R = torch.stft(waveform[1], n_fft=n_fft, hop_length=hop_length, return_complex=True)

    # Stack [Re_L, Im_L, Re_R, Im_R] -> [4, F, T]
    stft_4ch = torch.stack([
        stft_L.real, stft_L.imag,
        stft_R.real, stft_R.imag
    ], dim=0)

    # Truncate frequency dimension to target_freq_bins (e.g. 256) for patch divisibility
    if stft_4ch.shape[1] > target_freq_bins:
        stft_4ch = stft_4ch[:, :target_freq_bins, :]
    elif stft_4ch.shape[1] < target_freq_bins:
        stft_4ch = F.pad(stft_4ch, (0, 0, 0, target_freq_bins - stft_4ch.shape[1]))

    # Cut or pad time dimension to target_length (e.g. 1024 frames)
    T = stft_4ch.shape[2]
    if T < target_length:
        stft_4ch = F.pad(stft_4ch, (0, target_length - T))
    elif T > target_length:
        stft_4ch = stft_4ch[:, :, :target_length]

    return stft_4ch


class BinauralAudioDataset(Dataset):
    """
    Dataset for loading 2-channel binaural audio, extracting 4-channel Complex STFT,
    generating pre-embedding CCSR time-frame masks, and returning monaural (15) and spatial (5) feature vectors.
    """
    def __init__(self, dataset_json_file, audio_conf=None):
        super().__init__()
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf if audio_conf is not None else {}
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.n_fft = self.audio_conf.get('n_fft', 512)
        self.hop_length = self.audio_conf.get('hop_length', 160)
        self.mask_ratio = self.audio_conf.get('mask_ratio', 0.5)
        self.target_freq_bins = self.audio_conf.get('target_freq_bins', 256)

    def _wav2stft(self, filename):
        """
        Loads WAV audio and extracts 4-channel Complex STFT: [4, F, T].
        """
        waveform, sr = torchaudio.load(filename)
        stft_4ch = wav2stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            target_freq_bins=self.target_freq_bins,
            target_length=self.target_length
        )
        return stft_4ch, waveform, sr

    def __getitem__(self, index):
        datum = self.data[index]
        wav_path = datum['wav']

        stft_4ch, waveform, sr = self._wav2stft(wav_path)
        T = stft_4ch.shape[2]
        W = generate_ccsr_time_mask(T, mask_ratio=self.mask_ratio)

        coh_freq_bands = self.audio_conf.get('coh_freq_bands', (1000.0, 4000.0))
        
        # 1. Spatial features
        if 'spatial_features' in datum and datum['spatial_features'] is not None:
            spatial_feats = np.array(datum['spatial_features'], dtype=np.float32)
        else:
            # Live fallback
            spatial_feats = extract_spatial_features(waveform, sr=sr, coh_freq_bands=coh_freq_bands)

        # 2. Monaural features
        if 'acoustic_features' in datum and datum['acoustic_features'] is not None:
            mono_feats = np.array(datum['acoustic_features'], dtype=np.float32)
        else:
            # We strictly enforce precomputed features to avoid silent zero-vector collapses.
            # If they are missing or null (due to extraction failure), we explicitly error out
            # rather than falling back to zeros.
            raise RuntimeError(f"Missing or null 'acoustic_features' for {wav_path}. Please run precompute_acoustic_features.py.")

        return stft_4ch, W, torch.tensor(mono_feats, dtype=torch.float32), torch.tensor(spatial_feats, dtype=torch.float32), wav_path

    def __len__(self):
        return len(self.data)
