#!/usr/bin/env python3
"""Compare binaural_*.pitchfix.json with the originals: only F0 and HNR may change. Exits 1 if they aren't a clean replacement."""
import json, os, sys
import numpy as np

ROOT = os.environ['SAC_DATA_ROOT']
NAMES = ['f0', 'hnr', 'centroid', 'flux', 'zcr']
ok = True


def rank(x):
    return np.argsort(np.argsort(x))


for split in ('val', 'train'):
    old = json.load(open(f'{ROOT}/binaural_{split}.json'))['data']
    new = json.load(open(f'{ROOT}/binaural_{split}.pitchfix.json'))['data']
    failed = sum(d['acoustic_features'] is None for d in new)
    same_order = len(old) == len(new) and all(a['wav'] == b['wav'] for a, b in zip(old, new))
    same_spatial = same_order and all(a['spatial_features'] == b['spatial_features'] for a, b in zip(old, new))
    print(f'--- {split}: {len(new)} clips, {failed} failed, same wav order {same_order}, same spatial {same_spatial}')
    ok &= failed == 0 and same_order and same_spatial
    if failed:
        continue
    a = np.array([d['acoustic_features'] for d in old], float)
    b = np.array([d['acoustic_features'] for d in new], float)
    finite = np.isfinite(b).all()
    print('all finite:', finite)
    ok &= bool(finite) and a.shape == b.shape
    for i, name in enumerate(NAMES):
        tol = np.abs(a[:, i] - b[:, i]).max() / (a[:, i].std() + 1e-12)
        print(f'{name:9} old mean {a[:, i].mean():10.3f} new mean {b[:, i].mean():10.3f} new std {b[:, i].std():9.3f}'
              f'  largest change / old std {tol:8.4f}  rank corr {np.corrcoef(rank(a[:, i]), rank(b[:, i]))[0, 1]:.3f}')
    for i in (2, 3, 4):  # same code as before, so these must not move
        if np.abs(a[:, i] - b[:, i]).max() > 1e-3 * a[:, i].std():
            print(f'CHANGED: {NAMES[i]} should be unchanged')
            ok = False
    hz = 100 * 2 ** (b[:, 0] / 12)
    print('F0 Hz percentiles 1/25/50/75/99:', np.round(np.percentile(hz, [1, 25, 50, 75, 99])).astype(int))
    print('F0 Hz histogram:', np.histogram(hz, bins=[60, 100, 120, 140, 160, 180, 200, 220, 260, 340, 500])[0])
    print(f'clips with no voiced frame: {(b[:, 0] == 10.0).sum()}')

print('OK: clean replacement' if ok else 'NOT a clean replacement')
sys.exit(0 if ok else 1)
