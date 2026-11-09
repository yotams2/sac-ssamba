#!/usr/bin/env python3
"""Check the LPC formants in src/sac/acoustic_features.py: synthetic vowels with known formants, edge cases, and
agreement with Praat (Burg, default settings, via parselmouth) on LibriSpeech clips at the same frames.
Run on a CPU job: sbatch scripts/check_formant_features.sbatch [old commit]"""
import json, os, subprocess, sys, types
import numpy as np
import parselmouth
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy.signal import lfilter

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))
from sac import acoustic_features as new

SR = 16000
OLD_COMMIT = sys.argv[1] if len(sys.argv) > 1 else 'de50fc3'
DATA = os.environ['SAC_DATA_ROOT']
N_CLIPS = 300
FAILURES = []

# Peterson & Barney adult male (F1, F2, F3); F4, F5 fixed at 3500 / 4500 Hz.
VOWELS = {'i': (270, 2290, 3010), 'ae': (660, 1720, 2410), 'a': (730, 1090, 2440),
          'u': (300, 870, 2240), 'er': (490, 1350, 1690)}
BANDWIDTHS = (60, 90, 120, 150, 200)


def check(ok, message):
    print('PASS' if ok else 'FAIL', message)
    if not ok:
        FAILURES.append(message)


def vowel(formants, f0=110.0, seconds=0.6):
    """Impulse train, -12 dB/oct glottal tilt, 5 two-pole resonators, +6 dB/oct lip radiation."""
    x = np.zeros(int(SR * seconds))
    x[::int(round(SR / f0))] = 1.0
    x = lfilter([1], [1, -0.97], lfilter([1], [1, -0.97], x))
    for f, bw in zip(list(formants) + [3500, 4500], BANDWIDTHS):
        r = np.exp(-np.pi * bw / SR)
        x = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r], x)
    x = lfilter([1, -1], [1], x)
    return torch.tensor(x / np.abs(x).max(), dtype=torch.float32)


def load_old():
    source = subprocess.check_output(['git', '-C', REPO, 'show', f'{OLD_COMMIT}:src/sac/acoustic_features.py'], text=True)
    module = types.ModuleType('old_features')
    exec(source, module.__dict__)
    return module


def spearman(a, b):
    rank = lambda v: np.argsort(np.argsort(v))
    return np.corrcoef(rank(a), rank(b))[0, 1]


def synthetic():
    torch.manual_seed(0)
    for label, f0, snr in (('F0 110 Hz', 110, None), ('F0 220 Hz', 220, None), ('F0 110 Hz, noise 30 dB', 110, 30)):
        errors = []
        for target in VOWELS.values():
            x = vowel(target, f0)
            if snr:
                x = x + torch.randn_like(x) * x.pow(2).mean().sqrt() * 10 ** (-snr / 20)
            est = np.array([v.item() for v in new._compute_formants_lpc(x.unsqueeze(0), SR)])
            errors.append(np.abs(est - np.array(target)) / np.array(target))
        e = np.array(errors)
        print(f'synthetic vowels, {label}: mean relative error F1-F3 {e.mean(0).round(3)}, max {e.max(0).round(3)}')
        if snr is None:
            check(e.max() < 0.1, f'synthetic vowels, {label}: every formant within 10%')

    for label, x in (('silence', torch.zeros(1, SR)), ('noise', torch.randn(1, SR)), ('short clip', torch.randn(1, 200))):
        check(bool(torch.isfinite(torch.stack(new._compute_formants_lpc(x, SR))).all()), f'finite formants on {label}')
    batch = torch.stack([vowel(VOWELS['i']), vowel(VOWELS['a'])])
    together = torch.stack(new._compute_formants_lpc(batch, SR))[:, 1]
    alone = torch.stack(new._compute_formants_lpc(batch[1:], SR))[:, 0]
    check(bool(torch.allclose(together, alone, atol=1e-2)), 'batch result equals single-clip result')


def frame_times(x):
    """Centres (s) of the 10 frames the module averages over: highest energy on the raw signal at 11 kHz."""
    sr = new.FORMANT_SAMPLE_RATE
    y = AF.resample(x, SR, sr).double()
    w, h = int(sr * 0.03), int(sr * 0.02)
    energy = (y.unfold(-1, w, h) * torch.hann_window(w, dtype=y.dtype)).pow(2).sum(-1)
    return (energy.topk(min(10, energy.shape[-1]))[1].numpy() * h + w / 2) / sr


def against_praat(old):
    rows = []
    for item in json.load(open(os.path.join(DATA, 'librispeech_eval.tau.json')))['data'][:N_CLIPS]:
        x, sr = sf.read(item['wav'], dtype='float32')
        assert sr == SR
        x = torch.from_numpy(x)
        formant = parselmouth.Sound(x.numpy().astype(np.float64), SR).to_formant_burg()   # Praat defaults
        praat = np.nanmean([[formant.get_value_at_time(k, t) for k in (1, 2, 3)] for t in frame_times(x)], 0)
        rows.append(np.concatenate([praat] + [[v.item() for v in m._compute_formants_lpc(x.unsqueeze(0), SR)] for m in (new, old)]))
    r = np.array(rows)
    r = r[~np.isnan(r).any(1)]
    praat, ours, previous = r[:, :3], r[:, 3:6], r[:, 6:9]
    print(f'--- {len(r)} LibriSpeech dev-clean clips, median F1-F3: Praat {np.median(praat, 0).round(0)}, '
          f'new {np.median(ours, 0).round(0)}, old {np.median(previous, 0).round(0)}')
    for name, est in (('new', ours), ('old', previous)):
        diff = np.median(np.abs(est - praat), 0).round(0)
        rho = [round(spearman(est[:, i], praat[:, i]), 3) for i in range(3)]
        print(f'{name} vs Praat: median |difference| F1-F3 {diff} Hz, Spearman {rho}')
    diff = np.median(np.abs(ours - praat), 0)
    rho = [spearman(ours[:, i], praat[:, i]) for i in range(3)]
    check(bool((diff < [50, 75, 100]).all()), 'new vs Praat: median |difference| < 50 / 75 / 100 Hz')
    check(min(rho) > 0.9, 'new vs Praat: Spearman > 0.9 on F1-F3')


if __name__ == '__main__':
    print(f'old = {OLD_COMMIT}, new = working tree')
    synthetic()
    against_praat(load_old())
    if FAILURES:
        sys.exit('FAILED: ' + '; '.join(FAILURES))
    print('OK')
