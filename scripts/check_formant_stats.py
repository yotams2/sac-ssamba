#!/usr/bin/env python3
"""Check the `formant_stats` feature (mean + coefficient of variation of F1-F3 over voiced frames) in
src/sac/acoustic_features.py, on LibriSpeech dev-clean:
  - `formants` (10 loudest frames) is unchanged by the refactor (same values as on main)
  - agreement with the same functionals computed with Praat (scripts/check_formant_functionals.py)
  - split-half reliability, speaker eta^2, leave-one-out 1-NN speaker accuracy (as in check_formant_functionals)
  - edge cases and run time
Run on a CPU job: sbatch scripts/check_formant_stats.sbatch [main commit]"""
import json, os, subprocess, sys, time, types
import numpy as np
import soundfile as sf
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from sac import acoustic_features as af
from check_formant_functionals import functionals, spearman

SR = 16000
MAIN = sys.argv[1] if len(sys.argv) > 1 else 'main'
DATA = os.environ['SAC_DATA_ROOT']
NAMES = ['mean F1', 'mean F2', 'mean F3', 'CoV F1', 'CoV F2', 'CoV F3']
FAILURES = []


def check(ok, message):
    print('PASS' if ok else 'FAIL', message)
    if not ok:
        FAILURES.append(message)


def load_main():
    source = subprocess.check_output(['git', '-C', REPO, 'show', f'{MAIN}:src/sac/acoustic_features.py'], text=True)
    module = types.ModuleType('main_features')
    exec(source, module.__dict__)
    return module


def stats(x):
    return af._compute_formant_stats(x.unsqueeze(0), SR)[0].numpy()


if __name__ == '__main__':
    main = load_main()
    items = json.load(open(os.path.join(DATA, 'librispeech_eval.tau.json')))['data']
    speakers, ours, praat, first, second, same = [], [], [], [], [], []
    for item in items:
        x, sr = sf.read(item['wav'], dtype='float32')
        assert sr == SR
        x = torch.from_numpy(x)
        speakers.append(os.path.basename(item['wav']).split('-')[0])
        ours.append(stats(x))
        praat.append(functionals(x))
        first.append(stats(x[:len(x) // 2]))
        second.append(stats(x[len(x) // 2:]))
        if len(same) < 300:
            a = torch.stack(af._compute_formants_lpc(x.unsqueeze(0), SR))
            b = torch.stack(main._compute_formants_lpc(x.unsqueeze(0), SR))
            same.append((a - b).abs().max().item())
    ours, praat, first, second, speakers = map(np.array, (ours, praat, first, second, speakers))

    check(max(same) < 1e-3, f'`formants` unchanged vs {MAIN} on 300 clips (max |difference| {max(same):.2e} Hz)')

    ok = ~np.isnan(praat).any(1)
    print(f'--- {len(items)} clips, {len(set(speakers))} speakers; ours vs Praat on {ok.sum()} clips with voice in Praat')
    for i, name in enumerate(NAMES):
        diff = np.median(np.abs(ours[ok, i] - praat[ok, i]))
        print(f'{name:8s} median ours {np.median(ours[ok, i]):8.3f}  Praat {np.median(praat[ok, i]):8.3f}  '
              f'median |diff| {diff:7.3f}  Spearman {spearman(ours[ok, i], praat[ok, i]):.3f}')
    rho = [spearman(ours[ok, i], praat[ok, i]) for i in range(6)]
    check(min(rho[:3]) > 0.9, 'means agree with Praat: Spearman > 0.9')
    check(min(rho[3:]) > 0.7, 'CoV agrees with Praat: Spearman > 0.7')

    print(f'{"dimension":10s} {"split-half rho":>15s} {"speaker eta^2":>14s}')
    for i, name in enumerate(NAMES):
        v = ours[:, i]
        between = sum((speakers == s).sum() * (v[speakers == s].mean() - v.mean()) ** 2 for s in set(speakers))
        print(f'{name:10s} {spearman(first[:, i], second[:, i]):15.3f} {between / ((v - v.mean()) ** 2).sum():14.3f}')
    z = (ours - ours.mean(0)) / ours.std(0)
    d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    acc = np.mean(speakers[d.argmin(1)] == speakers)
    print(f'leave-one-out 1-NN speaker accuracy: {acc:.3f} (Praat functionals 0.445, current `formants` 0.122)')
    check(acc > 0.3, '1-NN speaker accuracy > 0.3')

    print('--- edge cases')
    for label, x in (('silence', torch.zeros(1, SR)), ('noise', torch.randn(1, SR)), ('short clip', torch.randn(1, 200))):
        out = af._compute_formant_stats(x, SR)
        print(label, out.numpy().round(3))
        check(bool(torch.isfinite(out).all()), f'finite formant_stats on {label}')
    x = torch.from_numpy(sf.read(items[0]['wav'], dtype='float32')[0])
    batch = torch.stack([x, x.flip(0)])
    check(bool(torch.allclose(af._compute_formant_stats(batch, SR)[1], af._compute_formant_stats(batch[1:], SR)[0], atol=1e-3)),
          'batch result equals single-clip result')
    feats = af.extract_acoustic_features(batch, SR, features_list='f0_mean,formant_stats,mfcc', normalize=False)
    groups, _ = af.get_feature_groups('f0_mean,formant_stats,mfcc')
    check(feats.shape[1] == 12 and groups['Vocal_Tract'] == list(range(1, 7)), 'feature vector and groups: 1 + 6 + 5')

    batch = torch.randn(64, int(SR * 10.24))
    t0 = time.time(); af._compute_formant_stats(batch, SR); t1 = time.time()
    af._compute_formants_lpc(batch, SR); t2 = time.time()
    print(f'CPU time for a batch of 64 x 10.24 s ({torch.get_num_threads()} threads): formant_stats {t1 - t0:.2f} s, formants {t2 - t1:.2f} s')

    if FAILURES:
        sys.exit('FAILED: ' + '; '.join(FAILURES))
    print('OK')
