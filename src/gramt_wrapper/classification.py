"""
GRAM-T clip classifier + binaural waveform dataset for the run_amba.py fine-tuning pipeline
(Speech Commands v2). Drop-in for dataloader.AudioDataset / AMBAModel: the dataset returns
(waveform [C, N], label_vector) and the model is called as model(x, task).
"""
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from data_paths import remap_wav_path
from .gramt_encoder import GRAMTEncoder, DEFAULT_MODEL_ID


class BinauralWaveformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_json_file, label_csv, target_secs=1.0, sample_rate=16000, mixup=0.0):
        import dataloader
        with open(dataset_json_file, "r") as fp:
            self.data = json.load(fp)["data"]
        self.index_dict = dataloader.make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        self.sample_rate = sample_rate
        self.num_samples = int(target_secs * sample_rate)
        self.mixup = mixup
        print("---------------GRAM-T binaural waveform dataloader---------------")
        print("{:d} clips, {:.2f} s, mixup {:.2f}".format(len(self.data), target_secs, mixup))

    def _load(self, path):
        wav, sr = torchaudio.load(remap_wav_path(path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav - wav.mean(dim=-1, keepdim=True)
        if wav.shape[-1] < self.num_samples:
            wav = torch.nn.functional.pad(wav, (0, self.num_samples - wav.shape[-1]))
        return wav[:, :self.num_samples]

    def _labels(self, datum, weight, label_indices):
        for label_str in datum["labels"].split(","):
            label_indices[int(self.index_dict[label_str])] += weight

    def __getitem__(self, index):
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        wav = self._load(datum["wav"])
        if random.random() < self.mixup:
            mix_datum = self.data[random.randint(0, len(self.data) - 1)]
            mix_lambda = np.random.beta(10, 10)
            wav = mix_lambda * wav + (1 - mix_lambda) * self._load(mix_datum["wav"])
            self._labels(datum, mix_lambda, label_indices)
            self._labels(mix_datum, 1.0 - mix_lambda, label_indices)
        else:
            self._labels(datum, 1.0, label_indices)
        return wav, torch.FloatTensor(label_indices)

    def __len__(self):
        return len(self.data)


class GRAMTClassifier(nn.Module):
    def __init__(self, label_dim, model_id=DEFAULT_MODEL_ID, sample_rate=16000):
        super().__init__()
        self.encoder = GRAMTEncoder(model_id=model_id, input_sample_rate=sample_rate)
        self.encoder_dim = self.encoder.output_dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, label_dim)
        )

    def forward(self, x, task=None, cluster=None, mask_patch=None):
        # x is [B, C, N] waveforms
        if task != 'ft_avgtok':
            raise ValueError(f"Task {task} not supported for GRAMTClassifier (use ft_avgtok)")
        trainable = any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and trainable):
            hidden_states, n_tokens = self.encoder(x)
        h = hidden_states[-1]  # [B, T_tokens, grid_f * D]
        # "raw" strategy: average over the time tokens that cover real audio (not the 2 s chunk padding)
        mask = torch.zeros(h.shape[:2], device=h.device, dtype=h.dtype)
        for i, n in enumerate(n_tokens):
            mask[i, :n] = 1.0
        h_pool = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.mlp_head(h_pool)
