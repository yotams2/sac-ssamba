# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py.py

# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import os
import sys
import logging
from data_paths import remap_wav_path

logging.basicConfig(filename='audio_loading.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

try:
    from binaural_dual_stream_sac.binaural_dataloader import wav2stft
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
    from binaural_dual_stream_sac.binaural_dataloader import wav2stft


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        try:
            if filename2 == None:
                waveform, sr = torchaudio.load(filename)
                waveform = waveform - waveform.mean()
            # mixup
            else:
                waveform1, sr = torchaudio.load(filename)
                waveform2, _ = torchaudio.load(filename2)

                waveform1 = waveform1 - waveform1.mean()
                waveform2 = waveform2 - waveform2.mean()

                if waveform1.shape[1] != waveform2.shape[1]:
                    if waveform1.shape[1] > waveform2.shape[1]:
                        # padding
                        temp_wav = torch.zeros(1, waveform1.shape[1])
                        temp_wav[0, 0:waveform2.shape[1]] = waveform2
                        waveform2 = temp_wav
                    else:
                        # cutting
                        waveform2 = waveform2[0, 0:waveform1.shape[1]]

                # sample lambda from uniform distribution
                #mix_lambda = random.random()
                # sample lambda from beta distribtion
                mix_lambda = np.random.beta(10, 10)

                mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
                waveform = mix_waveform - mix_waveform.mean()
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            logging.error(f"Error loading file {filename}: {e}")
            raise e  

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices

    def __len__(self):
        return len(self.data)


class BinauralSTFTClassificationDataset(Dataset):
    """
    Dataset for binaural audio classification returning 4-channel complex STFT:
    (x_4ch [4, 256, 1024], label_vector)
    Matches pretraining input representation without log, mel, or dataset_mean/std normalization.
    """
    def __init__(self, dataset_json_file, audio_conf=None, label_csv=None):
        if isinstance(audio_conf, str) and (label_csv is None or isinstance(label_csv, dict)):
            # Handle swapped positional arguments (dataset_json_file, label_csv, audio_conf)
            label_csv, audio_conf = audio_conf, label_csv

        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.audio_conf = audio_conf if audio_conf is not None else {}
        
        self.target_length = self.audio_conf.get('target_length', 1024)
        if self.target_length is None or self.target_length < 256:
            self.target_length = 1024
        self.target_freq_bins = self.audio_conf.get('target_freq_bins', 256)
        self.n_fft = self.audio_conf.get('n_fft', 512)
        self.hop_length = self.audio_conf.get('hop_length', 160)
        self.mode = self.audio_conf.get('mode', 'train')
        self.mixup = self.audio_conf.get('mixup', 0.0)

        self.index_dict = make_index_dict(label_csv) if label_csv else {}
        self.label_num = len(self.index_dict)
        print(f'BinauralSTFTClassificationDataset: {len(self.data)} samples, {self.label_num} classes, mode={self.mode}, mixup={self.mixup}')

    def _get_label_vector(self, datum):
        label_indices = np.zeros(self.label_num, dtype=np.float32)
        if 'labels' in datum and datum['labels']:
            for label_str in str(datum['labels']).split(','):
                label_str = label_str.strip()
                if label_str in self.index_dict:
                    label_indices[int(self.index_dict[label_str])] = 1.0
        elif 'label' in datum and datum['label'] is not None:
            lbl = int(datum['label'])
            if 0 <= lbl < self.label_num:
                label_indices[lbl] = 1.0
        return label_indices

    def __getitem__(self, index):
        datum = self.data[index]
        
        # Mixup on raw waveforms before STFT if enabled during training
        if self.mode == 'train' and self.mixup > 0 and random.random() < self.mixup:
            mix_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_idx]
            
            w1, sr1 = torchaudio.load(remap_wav_path(datum['wav']))
            w2, sr2 = torchaudio.load(remap_wav_path(mix_datum['wav']))
            
            if w1.shape[0] == 1:
                logging.warning("Mono waveform encountered in mixup w1; duplicating to 2 channels.")
                w1 = w1.repeat(2, 1)
            if w2.shape[0] == 1:
                logging.warning("Mono waveform encountered in mixup w2; duplicating to 2 channels.")
                w2 = w2.repeat(2, 1)
            w1 = w1[:2, :]
            w2 = w2[:2, :]
            
            # Align lengths
            if w1.shape[1] > w2.shape[1]:
                temp = torch.zeros(2, w1.shape[1], dtype=w1.dtype)
                temp[:, :w2.shape[1]] = w2
                w2 = temp
            elif w1.shape[1] < w2.shape[1]:
                temp = torch.zeros(2, w2.shape[1], dtype=w2.dtype)
                temp[:, :w1.shape[1]] = w1
                w1 = temp
                
            mix_lambda = float(np.random.beta(10, 10))
            mix_w = mix_lambda * w1 + (1.0 - mix_lambda) * w2
            
            stft_4ch = wav2stft(
                mix_w,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                target_freq_bins=self.target_freq_bins,
                target_length=self.target_length
            )
            
            lbl1 = self._get_label_vector(datum)
            lbl2 = self._get_label_vector(mix_datum)
            label_vector = torch.tensor(mix_lambda * lbl1 + (1.0 - mix_lambda) * lbl2, dtype=torch.float32)
            return stft_4ch, label_vector
        else:
            w, sr = torchaudio.load(remap_wav_path(datum['wav']))
            stft_4ch = wav2stft(
                w,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                target_freq_bins=self.target_freq_bins,
                target_length=self.target_length
            )
            label_vector = torch.tensor(self._get_label_vector(datum), dtype=torch.float32)
            return stft_4ch, label_vector

    def __len__(self):
        return len(self.data)