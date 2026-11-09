#!/usr/bin/env python3
"""Is 'mean + spread of F1-F3 over voiced frames' (eGeMAPS-style) a more stable clip feature than the current
'mean F1-F3 over the 10 loudest frames'? On LibriSpeech dev-clean (40 speakers):
  - split-half reliability: Spearman between the two halves of each clip, per dimension
  - speaker information: share of variance between speakers (eta^2) per dimension, and leave-one-out
    1-nearest-neighbour speaker accuracy on the z-scored group vector (chance 1/40)
F0 median is included as a reference for a feature known to carry speaker information.
The functionals use Praat (Burg formants, default settings; voicing from Praat's pitch, default settings).
Run on a CPU job: sbatch scripts/check_formant_functionals.sbatch"""
import json, os, sys
import numpy as np
import parselmouth
import soundfile as sf
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))
from sac import acoustic_features as af

SR = 16000
DATA = os.environ['SAC_DATA_ROOT']


def spearman(a, b):
    rank = lambda v: np.argsort(np.argsort(v))
    return np.corrcoef(rank(a), rank(b))[0, 1]


def functionals(x):
    """Mean and coefficient of variation of F1-F3 over voiced frames (Praat), or NaNs if < 5 voiced frames."""
    snd = parselmouth.Sound(x.numpy().astype(np.float64), SR)
    formant, pitch = snd.to_formant_burg(), snd.to_pitch()
    times = [t for t in pitch.xs() if pitch.get_value_at_time(t) == pitch.get_value_at_time(t)]   # voiced: not NaN
    v = np.array([[formant.get_value_at_time(k, t) for k in (1, 2, 3)] for t in times]) if times else np.zeros((0, 3))
    v = v[~np.isnan(v).any(1)] if len(v) else v
    if len(v) < 5:
        return np.full(6, np.nan)
    return np.concatenate([v.mean(0), v.std(0) / v.mean(0)])


def features(x):
    current = [t.item() for t in af._compute_formants_lpc(x.unsqueeze(0), SR)]
    f0 = af._compute_f0_stats(x.unsqueeze(0), SR)[0].item()
    return np.concatenate([current, functionals(x), [f0]])   # 3 + 6 + 1


NAMES = ['cur F1', 'cur F2', 'cur F3', 'mean F1', 'mean F2', 'mean F3', 'CoV F1', 'CoV F2', 'CoV F3', 'F0 median']
GROUPS = {'current (3)': [0, 1, 2], 'mean only (3)': [3, 4, 5], 'mean + CoV (6)': [3, 4, 5, 6, 7, 8], 'F0 median (1)': [9]}

if __name__ == '__main__':
    items = json.load(open(os.path.join(DATA, 'librispeech_eval.tau.json')))['data']
    speakers, whole, first, second = [], [], [], []
    for item in items:
        x, sr = sf.read(item['wav'], dtype='float32')
        assert sr == SR
        x = torch.from_numpy(x)
        speakers.append(os.path.basename(item['wav']).split('-')[0])
        whole.append(features(x))
        first.append(features(x[:len(x) // 2]))
        second.append(features(x[len(x) // 2:]))
    whole, first, second, speakers = map(np.array, (whole, first, second, speakers))
    ok = ~np.isnan(whole).any(1) & ~np.isnan(first).any(1) & ~np.isnan(second).any(1)
    whole, first, second, speakers = whole[ok], first[ok], second[ok], speakers[ok]
    print(f'{len(whole)} clips kept of {len(items)}, {len(set(speakers))} speakers, '
          f'mean clip length {np.mean([sf.info(i["wav"]).duration for i in items]):.1f} s')

    print(f'{"dimension":12s} {"split-half rho":>15s} {"speaker eta^2":>14s}')
    for i, name in enumerate(NAMES):
        v = whole[:, i]
        between = sum((speakers == s).sum() * (v[speakers == s].mean() - v.mean()) ** 2 for s in set(speakers))
        eta2 = between / ((v - v.mean()) ** 2).sum()
        print(f'{name:12s} {spearman(first[:, i], second[:, i]):15.3f} {eta2:14.3f}')

    print('leave-one-out 1-NN speaker accuracy on the z-scored group vector (chance', round(1 / len(set(speakers)), 3), ')')
    for group, cols in GROUPS.items():
        z = (whole[:, cols] - whole[:, cols].mean(0)) / whole[:, cols].std(0)
        d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        print(f'{group:16s} {np.mean(speakers[d.argmin(1)] == speakers):.3f}')
