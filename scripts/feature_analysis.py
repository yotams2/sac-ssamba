#!/usr/bin/env python3
"""Per-feature analysis of the SAC acoustic features, old (`main`) vs new (this branch). For each feature:
  - reliability: Spearman between the two halves of each clip (LibriSpeech dev-clean)
  - information, as rank-based eta^2 (share of rank variance between classes): speaker (LibriSpeech dev-clean,
    40 speakers), word (Speech Commands v2 train split, 100 clips x 35 words), emotion (IEMOCAP sessions 2-5, i.e. the
    fold-1 training set, 4 classes, features z-scored per speaker first)
  - level sensitivity: median |change| when the clip is played 20 dB quieter, in units of the feature's IQR
  - padding sensitivity: median |change| when a 4 s excerpt is zero-padded to the 11.24 s training window, same units
  - redundancy: largest |Spearman| with another feature of the same set
plus leave-one-out 1-NN accuracy of the whole z-scored vector for each task. No downstream test data is used.
LibriSpeech clips are prepared as in mono pretraining: first 11.24 s, zero-padded to 11.24 s.
Run on a CPU job: sbatch scripts/feature_analysis.sbatch [old commit]. Writes logs/feature_analysis.json."""
import json, os, random, subprocess, sys, time, types
import numpy as np
import soundfile as sf
import torch
from scipy.stats import rankdata

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
from sac import acoustic_features as new

SR = 16000
WINDOW = int(SR * 11.24)
OLD_COMMIT = sys.argv[1] if len(sys.argv) > 1 else 'main'
DATA = os.environ['SAC_DATA_ROOT']
SETS = {
    'old': ('f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm',
            ['F0 median', 'F0 IQR', 'F1 (loudest)', 'F2 (loudest)', 'F3 (loudest)', 'MFCC0', 'MFCC1', 'MFCC2',
             'MFCC3', 'MFCC4', 'HNR', 'Centroid', 'Flux var', 'ZCR mean', 'ZCR var ("rhythm")']),
    'new': ('f0_mean,f0_var,voiced_rate,voiced_len,unvoiced_len,formant_stats,mfcc,hnr,centroid,flux,flux_cov,zcr_mean',
            ['F0 mean', 'F0 CoV', 'Voiced segments/s', 'Voiced length', 'Unvoiced length', 'F1 mean', 'F2 mean',
             'F3 mean', 'F1 CoV', 'F2 CoV', 'F3 CoV', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'HNR', 'Centroid',
             'Flux mean', 'Flux CoV', 'ZCR mean']),
}


def load_old():
    source = subprocess.check_output(['git', '-C', REPO, 'show', f'{OLD_COMMIT}:src/sac/acoustic_features.py'], text=True)
    module = types.ModuleType('old_features')
    exec(source, module.__dict__)
    return module


MODULES = {'old': load_old(), 'new': new}


def feats(x):
    """{'old': [K_old], 'new': [K_new]} for a 1-D waveform."""
    return {k: MODULES[k].extract_acoustic_features(x.unsqueeze(0), SR, features_list=SETS[k][0], normalize=False)[0].double().numpy()
            for k in SETS}


def pad(x, n=WINDOW):
    x = x[:n]
    return torch.nn.functional.pad(x, (0, n - len(x)))


def load(path):
    x, sr = sf.read(path, dtype='float32')
    assert sr == SR, path
    if x.ndim > 1:
        x = x.mean(1)
    x = torch.from_numpy(x)
    return x - x.mean()


def rank(v):
    return rankdata(v)   # ties get their average rank


def spearman(a, b):
    return np.corrcoef(rank(a), rank(b))[0, 1]


def eta2(values, labels):
    r = rank(values)
    between = sum((labels == c).sum() * (r[labels == c].mean() - r.mean()) ** 2 for c in set(labels))
    return between / ((r - r.mean()) ** 2).sum()


def knn(x, labels):
    z = (x - x.mean(0)) / x.std(0).clip(min=1e-9)
    sq = (z ** 2).sum(1)
    d = sq[:, None] + sq[None, :] - 2 * z @ z.T
    np.fill_diagonal(d, np.inf)
    return float(np.mean(labels[d.argmin(1)] == labels))


def collect(items, prepare, tests=False):
    out = {k: {'whole': [], 'a': [], 'b': [], 'quiet': [], 'short': [], 'short_pad': []} for k in SETS}
    for n, path in enumerate(items):
        x = load(path)
        for k, v in feats(prepare(x)).items():
            out[k]['whole'].append(v)
        if tests:
            half = min(len(x), WINDOW) // 2
            for part, y in (('a', x[:half]), ('b', x[half:2 * half])):
                for k, v in feats(y).items():
                    out[k][part].append(v)
            if n < 500:
                for k, v in feats(prepare(x) * 0.1).items():
                    out[k]['quiet'].append(v)
                if len(x) >= 4 * SR:
                    for k, v in feats(x[:4 * SR]).items():
                        out[k]['short'].append(v)
                    for k, v in feats(pad(x[:4 * SR])).items():
                        out[k]['short_pad'].append(v)
    arrays = {k: {p: np.array(v) for p, v in d.items()} for k, d in out.items()}
    for k, d in arrays.items():
        for p, v in d.items():
            if v.size and np.isnan(v).any():
                cols = np.isnan(v).any(0)
                print(f'NaN in {k} set, {p}: {[SETS[k][1][i] for i in np.where(cols)[0]]}, {np.isnan(v).any(1).sum()} of {len(v)} clips; replaced by the column median')
                d[p] = np.where(np.isnan(v), np.nanmedian(v, 0), v)
    return arrays


def sanity():
    checks = []
    for label, x in (('silence', torch.zeros(1, SR)), ('noise', torch.randn(1, SR)), ('short', torch.randn(1, 200))):
        f = new.extract_acoustic_features(x, SR, features_list=SETS['new'][0], normalize=False)
        checks.append((f'finite on {label}', bool(torch.isfinite(f).all())))
    x = load(json.load(open(os.path.join(DATA, 'librispeech_eval.tau.json')))['data'][0]['wav'])
    batch = torch.stack([pad(x), pad(x.flip(0))])
    f2 = new.extract_acoustic_features(batch, SR, features_list=SETS['new'][0], normalize=False)
    f1 = new.extract_acoustic_features(batch[1:], SR, features_list=SETS['new'][0], normalize=False)
    checks.append(('batch = single clip', bool(torch.allclose(f2[1], f1[0], rtol=1e-4, atol=1e-4))))
    groups, _ = new.get_feature_groups(SETS['new'][0])
    checks.append(('20 values in 5 groups', f2.shape[1] == len(SETS['new'][1]) == 20 and len(groups) == 5))
    t0 = time.time()
    new.extract_acoustic_features(torch.randn(64, WINDOW), SR, features_list=SETS['new'][0], normalize=False)
    t_new = time.time() - t0
    t0 = time.time()
    MODULES['old'].extract_acoustic_features(torch.randn(64, WINDOW), SR, features_list=SETS['old'][0], normalize=False)
    t_old = time.time() - t0
    return checks, {'new': t_new, 'old': t_old}, {g: [SETS['new'][1][i] for i in idx] for g, idx in groups.items()}


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    random.seed(0)
    checks, timing, groups = sanity()
    for name, ok in checks:
        print('PASS' if ok else 'FAIL', name)
    print(f'CPU time, batch of 64 x 11.24 s: new {timing["new"]:.2f} s, old {timing["old"]:.2f} s')

    libri = json.load(open(os.path.join(DATA, 'librispeech_eval.tau.json')))['data']
    libri_paths = [d['wav'] for d in libri]
    speakers = np.array([os.path.basename(p).split('-')[0] for p in libri_paths])

    sc = json.load(open(os.path.join(DATA, 'mono', 'speechcommands_datafiles', 'speechcommand_train_data.json')))['data']
    by_word = {}
    for d in sc:
        by_word.setdefault(d['wav'].split('/')[-2], []).append(d['wav'])
    sc_paths, words = [], []
    for w, paths in sorted(by_word.items()):
        for p in random.sample(paths, min(100, len(paths))):
            sc_paths.append(p)
            words.append(w)
    words = np.array(words)

    meta_dir = os.path.join(os.path.dirname(REPO), 'sac-s3prl', 's3prl', 'downstream', 'emotion', 'meta_data')
    iemo = json.load(open(os.path.join(meta_dir, 'Session1', 'train_meta_data.json')))['meta_data']   # sessions 2-5
    assert not any(d['path'].startswith('Session1') for d in iemo)
    root = os.path.join(DATA, 'mono', 'data', 'IEMOCAP_audio_only', 'IEMOCAP_full_release')
    iemo_paths = [os.path.join(root, d['path']) for d in iemo]
    emotions = np.array([d['label'] for d in iemo])
    iemo_speakers = np.array([d['speaker'] for d in iemo])

    t0 = time.time()
    L = collect(libri_paths, pad, tests=True)
    S = collect(sc_paths, lambda x: x)
    E = collect(iemo_paths, lambda x: x)
    print(f'features computed in {(time.time() - t0) / 60:.1f} min: {len(libri_paths)} LibriSpeech, {len(sc_paths)} SC2, {len(iemo_paths)} IEMOCAP')

    results = {'checks': checks, 'timing': timing, 'groups': groups, 'features': {}, 'knn': {}, 'redundant_pairs': {}}
    for k, (_, names) in SETS.items():
        e = E[k]['whole'].copy()
        for s in set(iemo_speakers):
            m = iemo_speakers == s
            e[m] = (e[m] - e[m].mean(0)) / e[m].std(0).clip(min=1e-9)
        lw = L[k]['whole']
        iqr = np.subtract(*np.percentile(lw, [75, 25], axis=0)).clip(min=1e-9)
        corr = np.array([[spearman(lw[:, i], lw[:, j]) for j in range(lw.shape[1])] for i in range(lw.shape[1])])
        np.fill_diagonal(corr, 0)
        rows = []
        for i, name in enumerate(names):
            rows.append({
                'name': name,
                'median': float(np.median(lw[:, i])),
                'split_half': float(spearman(L[k]['a'][:, i], L[k]['b'][:, i])),
                'speaker': float(eta2(lw[:, i], speakers)),
                'word': float(eta2(S[k]['whole'][:, i], words)),
                'emotion': float(eta2(e[:, i], emotions)),
                'level': float(np.median(np.abs(L[k]['quiet'][:, i] - lw[:len(L[k]['quiet']), i])) / iqr[i]),
                'padding': float(np.median(np.abs(L[k]['short_pad'][:, i] - L[k]['short'][:, i])) / iqr[i]),
                'max_corr': float(np.abs(corr[i]).max()),
                'max_corr_with': names[int(np.abs(corr[i]).argmax())],
            })
        results['features'][k] = rows
        results['redundant_pairs'][k] = [(names[i], names[j], float(corr[i, j])) for i in range(len(names))
                                         for j in range(i + 1, len(names)) if abs(corr[i, j]) > 0.7]
        results['knn'][k] = {'speaker': knn(lw, speakers), 'word': knn(S[k]['whole'], words), 'emotion': knn(e, emotions)}

    for k in SETS:
        print(f'=== {k}: 1-NN speaker / word / emotion', {t: round(v, 3) for t, v in results['knn'][k].items()})
        print(f'{"feature":20s} {"median":>9s} {"split":>6s} {"spk":>6s} {"word":>6s} {"emo":>6s} {"level":>6s} {"pad":>6s} {"maxcorr":>8s}')
        for r in results['features'][k]:
            print(f'{r["name"]:20s} {r["median"]:9.3f} {r["split_half"]:6.2f} {r["speaker"]:6.2f} {r["word"]:6.2f} '
                  f'{r["emotion"]:6.2f} {r["level"]:6.2f} {r["padding"]:6.2f} {r["max_corr"]:8.2f} ({r["max_corr_with"]})')
        print('pairs with |rho| > 0.7:', [(a, b, round(c, 2)) for a, b, c in results['redundant_pairs'][k]])
    json.dump(results, open(os.path.join(REPO, 'logs', 'feature_analysis.json'), 'w'), indent=1)
    print('chance 1-NN: speaker', round(1 / len(set(speakers)), 3), 'word', round(1 / len(set(words)), 3), 'emotion (majority share)',
          round(max((emotions == c).mean() for c in set(emotions)), 3))
