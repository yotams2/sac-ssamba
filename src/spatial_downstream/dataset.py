import numpy as np
import scipy.signal
import soundfile
from pathlib import Path
from torch.utils.data import Dataset


class Selecting:
    """
    Transform to truncate or pad audio signal array to target range [st_sample, ed_sample].
    """
    def __init__(self, select_range):
        self.st_sample = select_range[0]
        self.ed_sample = select_range[1]
        self.target_len = self.ed_sample - self.st_sample

    def __call__(self, sig):
        nsample = sig.shape[0]
        if nsample >= self.target_len:
            sig = sig[self.st_sample:self.ed_sample, :]
        else:
            pad_len = self.target_len - nsample
            sig = np.pad(sig, ((0, pad_len), (0, 0)), mode='constant')
        return sig


class FixMicSigDataset(Dataset):
    """
    Loads fixed presaved multi-channel microphone signals and ground-truth acoustic annotations.
    """
    def __init__(self, data_dir, load_anno=True, load_dp=False, fs=16000, dataset_sz=None, transforms=None):
        self.data_paths = []
        if isinstance(data_dir, list):
            self.files = []
            for d in data_dir:
                if Path(d).exists():
                    self.files.extend(list(Path(d).rglob('*.wav')))
            np.random.shuffle(self.files)
        else:
            if Path(data_dir).exists():
                self.files = list(Path(data_dir).rglob('*.wav'))
            else:
                self.files = []

        # Filter out _dp.wav files
        self.files = [f for f in self.files if not str(f).endswith('_dp.wav')]

        if dataset_sz is not None and len(self.files) > 0:
            self.dataset_sz = min(len(self.files), dataset_sz)
        else:
            self.dataset_sz = len(self.files)

        self.fs = fs
        self.load_anno = load_anno
        self.load_dp = load_dp
        self.transforms = transforms

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):
        if len(self.files) == 0:
            raise RuntimeError("FixMicSigDataset has no files available.")

        file_name = str(self.files[idx])
        mic_sig, fs = soundfile.read(file_name)

        if self.fs != fs:
            mic_sig = scipy.signal.resample_poly(mic_sig, self.fs, fs)

        if self.transforms is not None:
            for t in self.transforms:
                mic_sig = t(mic_sig)

        return_data = [mic_sig.astype(np.float32)]

        if self.load_anno:
            info_file_name = file_name.replace('.wav', '_info.npz')
            if Path(info_file_name).exists():
                info = dict(np.load(info_file_name))
                vol = info['room_sz'][0] * info['room_sz'][1] * info['room_sz'][2]
                sur = (info['room_sz'][0] * info['room_sz'][1] +
                       info['room_sz'][0] * info['room_sz'][2] +
                       info['room_sz'][1] * info['room_sz'][2])
                
                t60_val = info['T60_edc'].astype(np.float32)
                abs_val = np.array(0.161 * vol / sur / (t60_val + 1e-8)).astype(np.float32)

                annos = {
                    'TDOA': info['TDOA'].astype(np.float32),
                    'T60': t60_val,
                    'DRR': info['DRR'].astype(np.float32),
                    'C50': info['C50'].astype(np.float32),
                    'ABS': abs_val,
                }
            else:
                annos = {
                    'TDOA': np.float32(0.0),
                    'T60': np.float32(0.0),
                    'DRR': np.float32(0.0),
                    'C50': np.float32(0.0),
                    'ABS': np.float32(0.0),
                }
            return_data.append(annos)

        return_data.append(idx)
        return return_data
