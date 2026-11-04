#!/usr/bin/env python3
"""Check F0 and HNR in src/sac/acoustic_features.py against signals with a known answer, then compare with the previous
version on real binaural clips. Run on a CPU job: sbatch scripts/check_pitch_features.sbatch [old git commit]"""
import json, os, subprocess, sys, types, wave
import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))
from sac import acoustic_features as new

SR = 16000
OLD_COMMIT = sys.argv[1] if len(sys.argv) > 1 else '0c3f0dc'
MANIFEST = os.path.join(os.environ['SAC_DATA_ROOT'], 'binaural_val.json')
N_CLIPS = 400


def hz(semitones):
    return 100.0 * 2 ** (semitones / 12.0)


def harmonic(f0, seconds=2.0, n_harmonics=12, formant=500.0):
    """Voice-like harmonic complex, with most energy near a formant, so that the formant can't be mistaken for pitch."""
    t = torch.arange(int(SR * seconds), dtype=torch.float32) / SR
    x = torch.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        f = k * f0
        if f < SR / 2:
            x += torch.sin(2 * torch.pi * f * t) * (1.0 / (1.0 + ((f - formant) / 300.0) ** 2))
    return x / x.abs().max()


def known_signals():
    print('--- F0 of harmonic signals')
    for f0 in (85, 100, 120, 150, 200, 250, 300, 400):
        median, spread = new._compute_f0_stats(harmonic(f0).unsqueeze(0), SR)
        est = hz(median.item())
        print(f'true {f0:3d} Hz  estimated {est:6.1f} Hz  error {100 * (est / f0 - 1):+5.1f}%  iqr {spread.item():.2f} st')
        assert abs(est / f0 - 1) < 0.03, f'F0 wrong at {f0} Hz'

    print('--- voicing: speech-like, then silence and noise')
    half = harmonic(150, seconds=1.0)
    with_gap = torch.cat([half, torch.zeros(SR)]).unsqueeze(0)
    frames = new._pitch_track(with_gap, SR)[2].float()
    n = frames.shape[1] // 2
    print(f'voiced fraction: harmonic half {frames[0, :n].mean():.2f}, silent half {frames[0, n + 10:].mean():.2f}')
    assert frames[0, :n].mean() > 0.9 and frames[0, n + 10:].mean() == 0
    noise = torch.randn(1, 2 * SR)
    print(f'voiced fraction of white noise: {new._pitch_track(noise, SR)[2].float().mean():.3f}')
    assert new._pitch_track(noise, SR)[2].float().mean() < 0.02
    print('no voiced frame:', new._compute_f0_stats(noise, SR), new._compute_hnr(noise, SR))

    print('--- HNR of a harmonic signal with added noise')
    torch.manual_seed(0)
    x = harmonic(150)
    previous = float('inf')
    for snr in (20, 10, 5, 0):
        noisy = x + torch.randn_like(x) * x.pow(2).mean().sqrt() * 10 ** (-snr / 20)
        value = new._compute_hnr(noisy.unsqueeze(0), SR).item()
        print(f'added noise at {snr:2d} dB: HNR {value:5.1f} dB')
        assert value < previous, 'HNR must fall as noise rises'
        previous = value

    batch = torch.stack([harmonic(110), harmonic(220)])
    single = new._compute_f0_stats(batch[1:], SR)[0]
    assert abs(new._compute_f0_stats(batch, SR)[0][1] - single) < 1e-4, 'batch result differs from single clip'


def load_old():
    source = subprocess.check_output(['git', '-C', REPO, 'show', f'{OLD_COMMIT}:src/sac/acoustic_features.py'], text=True)
    module = types.ModuleType('old_features')
    exec(source, module.__dict__)
    return module


def read_mono(path):
    with wave.open(path) as w:
        channels, width = w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    x = np.frombuffer(data, {2: '<i2', 4: '<f4'}[width]).astype(np.float32)
    if width == 2:
        x = x / 32768.0
    return torch.from_numpy(x.reshape(-1, channels).mean(1).copy())


def spearman(a, b):
    rank = lambda v: np.argsort(np.argsort(v))
    return np.corrcoef(rank(a), rank(b))[0, 1]


def real_clips():
    data = json.load(open(MANIFEST))['data'][:N_CLIPS]
    old = load_old()
    rows = []
    for item in data:
        x = read_mono(item['wav'].replace('/storage/yotam/ssamba/data', os.environ['SAC_DATA_ROOT']))
        first, second = x[:len(x) // 2].unsqueeze(0), x[len(x) // 2:].unsqueeze(0)
        rows.append([
            old._compute_f0_stats(first, SR)[0].item(), old._compute_f0_stats(second, SR)[0].item(),
            old._compute_hnr(first, SR).item(), old._compute_hnr(second, SR).item(),
            new._compute_f0_stats(first, SR)[0].item(), new._compute_f0_stats(second, SR)[0].item(),
            new._compute_hnr(first, SR).item(), new._compute_hnr(second, SR).item(),
            new._compute_f0_stats(x.unsqueeze(0), SR)[0].item(), new._pitch_track(x.unsqueeze(0), SR)[2].float().mean().item(),
        ])
    r = np.array(rows)
    print(f'--- {len(r)} real clips: agreement between the two halves of each clip (Spearman, higher = more reliable)')
    print(f'F0   old {spearman(r[:, 0], r[:, 1]):.3f}   new {spearman(r[:, 4], r[:, 5]):.3f}')
    print(f'HNR  old {spearman(r[:, 2], r[:, 3]):.3f}   new {spearman(r[:, 6], r[:, 7]):.3f}')
    f0 = hz(r[:, 8])
    print(f'--- new clip F0 (Hz): p1 {np.percentile(f0, 1):.0f}  p25 {np.percentile(f0, 25):.0f}  p50 {np.median(f0):.0f}  p75 {np.percentile(f0, 75):.0f}  p99 {np.percentile(f0, 99):.0f}')
    print('histogram', np.histogram(f0, bins=[60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 280, 340, 500]))
    print(f'voiced fraction: mean {r[:, 9].mean():.2f}, min {r[:, 9].min():.2f}; clips with none: {(r[:, 9] == 0).sum()}')
    hnr = (r[:, 6] + r[:, 7]) / 2
    print(f'new HNR (dB): mean {hnr.mean():.1f}, p1 {np.percentile(hnr, 1):.1f}, p99 {np.percentile(hnr, 99):.1f}')
    stored = np.array([d['acoustic_features'] for d in data])
    print('corr(stored manifest f0, new f0):', round(spearman(stored[:, 0], r[:, 8]), 3), ' corr(stored hnr, new hnr):', round(spearman(stored[:, 1], hnr), 3))
    spatial = np.array([d['spatial_features'] for d in data])
    print('new HNR vs spatial tdoa/gccmax/cohL/cohM/cohH:', [round(spearman(hnr, spatial[:, i]), 2) for i in range(5)])
    print('new F0  vs spatial tdoa/gccmax/cohL/cohM/cohH:', [round(spearman(r[:, 8], spatial[:, i]), 2) for i in range(5)])


if __name__ == '__main__':
    known_signals()
    real_clips()
    print('OK')
