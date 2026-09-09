import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed):
    """
    Fix random seeds across numpy, random, and PyTorch for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class STFT(nn.Module):
    """
    Get STFT coefficients of multi-channel microphone signals.
    Args:
        win_len: length of window / frame
        win_shift_ratio: ratio between frame shift and frame length
        nfft: number of FFT points
        win: window type ('hann' or 'boxcar')
    Input:
        signal: time-domain microphone signals of shape (nbatch, nsample, nch)
    Returns:
        stft: complex STFT tensor of shape (nbatch, nf, nt, nch)
    """
    def __init__(self, win_len, win_shift_ratio, nfft, win='hann', inv=False):
        super(STFT, self).__init__()
        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft
        self.win = win
        self.inv = inv

    def forward(self, signal):
        nb = signal.shape[0]
        nsample = signal.shape[-2]
        nch = signal.shape[-1]
        win_shift = int(self.win_len * self.win_shift_ratio)
        nf = int(self.nfft / 2) + 1

        if self.inv:
            nt = int(nsample / win_shift) + 1
        else:
            nt = int(np.floor((nsample - self.win_len) / win_shift + 1))

        stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64, device=signal.device)

        if self.win == 'hann':
            window = torch.hann_window(window_length=self.win_len, device=signal.device)
        else:
            window = None

        for ch_idx in range(nch):
            if self.inv:
                stft[:, :, :, ch_idx] = torch.stft(
                    signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift,
                    win_length=self.win_len, window=window, center=True,
                    normalized=False, return_complex=True
                )
            else:
                stft[:, :, :, ch_idx] = torch.stft(
                    signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift,
                    win_length=self.win_len, window=window, center=False,
                    normalized=False, return_complex=True
                )
        return stft


class ISTFT(nn.Module):
    """
    Get inverse STFT of multi-channel STFT coefficients.
    Input:
        stft: complex STFT tensor of shape (nbatch, nf, nt, nch)
    Returns:
        signal: time-domain microphone signals of shape (nbatch, nsample, nch)
    """
    def __init__(self, win_len, win_shift_ratio, nfft, inv=False):
        super(ISTFT, self).__init__()
        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft
        self.inv = inv

    def forward(self, stft):
        nb, nf, nt, nch = stft.shape
        win_shift = int(self.win_len * self.win_shift_ratio)
        if self.inv:
            nsample = (nt - 1) * win_shift
        else:
            nsample = (nt - 1) * win_shift + self.win_len

        window = torch.hann_window(window_length=self.win_len, device=stft.device)
        signal = torch.zeros((nb, nsample, nch), dtype=torch.float32, device=stft.device)

        for ch_idx in range(nch):
            signal[:, :, ch_idx] = torch.istft(
                stft[:, :, :, ch_idx], n_fft=self.nfft, hop_length=win_shift,
                win_length=self.win_len, window=window, center=not self.inv,
                length=nsample
            )
        return signal


class AddChToBatch(nn.Module):
    """
    Formats multi-channel STFT tensor into complex 4-channel format for SSAMBA:
    Input stft: (nb, nch, nf, nt) complex tensor (where nch=2 for binaural L/R)
    Returns: (nb, 2, nf, nt) complex STFT representing binaural channel pair.
    """
    def __init__(self, ch_mode='M'):
        super(AddChToBatch, self).__init__()
        self.ch_mode = ch_mode

    def forward(self, stft):
        # stft: (nb, nch, nf, nt)
        nb, nch, nf, nt = stft.shape
        if nch == 2:
            return stft
        elif nch > 2:
            # Take first 2 channels by default for binaural processing
            return stft[:, :2, :, :]
        else:
            raise ValueError(f"AddChToBatch expected nch >= 2, got nch={nch}")


def get_nparams(model, param_key_list=[]):
    """
    Counts total parameters (in Millions) and parameters per specified key.
    """
    nparam_sum = 0.0
    nparam = {key: 0.0 for key in param_key_list}
    for key, value in model.named_parameters():
        num_m = value.numel() / 1e6
        nparam_sum += num_m
        for param_key_specified in param_key_list:
            if param_key_specified in key:
                nparam[param_key_specified] += num_m
    return nparam, nparam_sum
