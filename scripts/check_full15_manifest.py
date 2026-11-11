#!/usr/bin/env python3
"""Check binaural_*.full15.json against the pitchfix manifests and print each feature's distribution.

Exits 1 if the manifest is not usable: failed clips, changed order or spatial features, non-finite values, or a change in
the 5 features both manifests hold. Formants and MFCC have only been used on clean mono audio, so the output is also there to be read.
"""
import json, os, sys
import numpy as np

ROOT = os.environ['SAC_DATA_ROOT']
NAMES = ['f0_mean', 'f0_var', 'f1', 'f2', 'f3', 'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
         'hnr', 'centroid', 'flux', 'zcr_mean', 'rhythm']
GROUPS = {'Prosody': [0, 1], 'Vocal_Tract': [2, 3, 4], 'Timbre': [5, 6, 7, 8, 9], 'Voice_Quality': [10], 'Scene': [11, 12, 13, 14]}
SHARED = {0: 0, 10: 1, 11: 2, 12: 3, 13: 4}  # index here -> index in the 5-feature pitchfix manifest
ok = True

for split in ('val', 'train'):
    old = json.load(open(f'{ROOT}/binaural_{split}.pitchfix.json'))['data']
    new = json.load(open(f'{ROOT}/binaural_{split}.full15.json'))['data']
    failed = sum(d['acoustic_features'] is None for d in new)
    same_order = len(old) == len(new) and all(a['wav'] == b['wav'] for a, b in zip(old, new))
    same_spatial = same_order and all(a['spatial_features'] == b['spatial_features'] for a, b in zip(old, new))
    print(f'--- {split}: {len(new)} clips, {failed} failed, same wav order {same_order}, same spatial {same_spatial}')
    ok &= failed == 0 and same_order and same_spatial
    if failed:
        continue
    a = np.array([d['acoustic_features'] for d in old], float)
    b = np.array([d['acoustic_features'] for d in new], float)
    shape_ok = b.ndim == 2 and b.shape[1] == len(NAMES)
    finite = shape_ok and bool(np.isfinite(b).all())
    print(f'shape {b.shape}, all finite: {finite}')
    ok &= finite
    if not finite:
        continue
    for i, j in SHARED.items():
        if np.abs(b[:, i] - a[:, j]).max() > 1e-3 * a[:, j].std():
            print(f'CHANGED: {NAMES[i]} differs from the pitchfix manifest')
            ok = False

    print(f'{"feature":9} {"mean":>11} {"std":>10} {"p1":>10} {"p50":>10} {"p99":>10} {"min":>10} {"max":>10} {"top value %":>11} {"clipped %":>9}')
    z = (b - b.mean(0)) / (3 * b.std(0) + 1e-6)  # the training normalisation, which clamps to [-1, 1]
    for i, name in enumerate(NAMES):
        x = b[:, i]
        top = np.unique(np.round(x, 6), return_counts=True)[1].max() / len(x) * 100  # a pile-up means a fallback value
        print(f'{name:9} {x.mean():11.3f} {x.std():10.3f} {np.percentile(x, 1):10.3f} {np.percentile(x, 50):10.3f} {np.percentile(x, 99):10.3f}'
              f' {x.min():10.3f} {x.max():10.3f} {top:11.2f} {(np.abs(z[:, i]) >= 1).mean() * 100:9.2f}')

    print(f'formant order F1<F2<F3: {((b[:, 2] < b[:, 3]) & (b[:, 3] < b[:, 4])).mean() * 100:.2f}% of clips')
    z = np.clip(z, -1, 1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(z), min(2000, len(z)), replace=False)
    for group, cols in GROUPS.items():
        d = np.linalg.norm(z[idx][:, None, cols] - z[idx][None, :, cols], axis=-1)
        print(f'{group:14} median pair distance {np.median(d[~np.eye(len(idx), dtype=bool)]):.4f}')
    print('largest |correlation| between features:')
    c = np.corrcoef(b.T) - np.eye(len(NAMES))
    pairs = sorted(((abs(c[i, j]), NAMES[i], NAMES[j]) for i in range(len(NAMES)) for j in range(i)), reverse=True)
    for v, n1, n2 in pairs[:6]:
        print(f'  {n1} / {n2}: {v:.3f}')

print('OK: usable manifest' if ok else 'NOT usable')
sys.exit(0 if ok else 1)
