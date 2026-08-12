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
        Loads 2-channel WAV audio and extracts 4-channel Complex STFT: [4, F, T].
        """
        waveform, sr = torchaudio.load(filename)
        
        # Ensure 2 channels
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        # STFT extraction for Left and Right channels
        stft_L = torch.stft(waveform[0], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        stft_R = torch.stft(waveform[1], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)

        # Stack [Re_L, Im_L, Re_R, Im_R] -> [4, F, T]
        stft_4ch = torch.stack([
            stft_L.real, stft_L.imag,
            stft_R.real, stft_R.imag
        ], dim=0)

        # Truncate frequency dimension to target_freq_bins (e.g. 256) for patch divisibility
        if stft_4ch.shape[1] > self.target_freq_bins:
            stft_4ch = stft_4ch[:, :self.target_freq_bins, :]
        elif stft_4ch.shape[1] < self.target_freq_bins:
            stft_4ch = F.pad(stft_4ch, (0, 0, 0, self.target_freq_bins - stft_4ch.shape[1]))

        # Cut or pad time dimension to target_length (e.g. 1024 frames)
        T = stft_4ch.shape[2]
        if T < self.target_length:
            stft_4ch = F.pad(stft_4ch, (0, self.target_length - T))
        elif T > self.target_length:
            stft_4ch = stft_4ch[:, :, :self.target_length]

        return stft_4ch, waveform, sr

    def __getitem__(self, index):
        datum = self.data[index]
        wav_path = datum['wav']

        stft_4ch, waveform, sr = self._wav2stft(wav_path)
        T = stft_4ch.shape[2]
        W = generate_ccsr_time_mask(T, mask_ratio=self.mask_ratio)

        coh_freq_bands = self.audio_conf.get('coh_freq_bands', (1000.0, 4000.0))
        # Extract 5 spatial features
        spatial_feats = extract_spatial_features(waveform, sr=sr, coh_freq_bands=coh_freq_bands)

        # 15 monaural features
        if 'acoustic_features' in datum:
            mono_feats = np.array(datum['acoustic_features'], dtype=np.float32)
        else:
            mono_feats = np.zeros(15, dtype=np.float32)

        return stft_4ch, W, torch.tensor(mono_feats, dtype=torch.float32), torch.tensor(spatial_feats, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
