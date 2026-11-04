"""Frozen-embedding probes for tracking binaural SAC pretraining without running the downstream jobs.

Fixed, small labeled subsets of the downstream sets are loaded into RAM once. At each call the
spectral encoder (the only stream the SC2 / SID / ER downstream runs use) embeds them without
gradients, pooled the way the downstream heads pool, and these metrics are computed:

  SC2 (35 keywords): reference = train split, query = test split
      knn20_acc   20-NN vote, cosine similarity, weights exp(sim / 0.07)
      ncm_acc     nearest class mean (cosine to the class centroid)
      linear_acc  logistic regression on the reference set (full batch, fixed recipe)
  VoxCeleb1 SID (closed set, the same speakers in both splits): reference = train, query = test
      ncm_acc     nearest speaker centroid, i.e. enrollment from n_ref utterances
      linear_acc  logistic regression on the reference set
      eer         equal error rate of cosine scoring over all query-query pairs (lower is better)
  IEMOCAP ER (4 classes), s3prl fold split (default fold1): reference = up to 250 per class of the
  fold's train list, query = the whole fold test session; wavs from the regenerated (v2) binaural manifests
      knn20_ua, ncm_ua, linear_ua   unweighted accuracy (mean per-class recall)
      linear_acc                    plain accuracy

Features are standardised with the reference-set mean/std before any metric.

How each task embeds a clip (mirrors the downstream code):
  SC2  run_amba.py BinauralSACClassifier, ft_avgtok, pool_real_time_only=True:
       STFT of the 1 s clip, frames zero-padded to 1024; mean of the 768-d tokens over the
       16 frequency patches x the time patches that hold audio.
  SID, IEMOCAP  sac-s3prl upstream/ssamba/expert.py + MeanPooling: waveform zero-padded to 1024 * 160
       samples, STFT; the [64 time, 16 * 768] features averaged over the first
       round(len / 2560) time steps. Only the first 10.24 s window is used (s3prl concatenates
       every window of longer utterances), to keep the cost bounded.
  GRAM-T anchor: last layer of gramt_wrapper.GRAMTEncoder, mean over valid tokens (as GRAMTClassifier).

The numbers are proxies for comparing checkpoints of one run (and against the random-init and
GRAM-T anchors), not estimates of the downstream accuracies.

Use from the pretraining loop (run_binaural_amba.py --probe_every N), or standalone on saved checkpoints:
  python pretrain_probe.py --models random gramt <exp>/models/audio_model.*.pth --out <exp>/probe_metrics.jsonl
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
from data_paths import remap_wav_path  # noqa: E402
from binaural_dual_stream_sac.binaural_dataloader import wav2fbank, wav2stft  # noqa: E402

SAMPLE_RATE = 16000
HOP = 160
N_FRAMES = 1024
F_PATCHES, T_PATCHES, PATCH_T = 16, 64, 16
CHUNK_SAMPLES = N_FRAMES * HOP          # 10.24 s, one encoder window
SID_DOWNSAMPLE = HOP * PATCH_T          # samples per time step in the s3prl upstream (2560)
DEFAULT_DATA_ROOT = '/storage/yotam/ssamba/data'
DEFAULT_LABEL_CSV = os.path.join(SRC, 'finetune/speechcommands_v2/data/speechcommands_class_labels_indices.csv')
# sibling repo sac-s3prl, next to sac-ssamba
DEFAULT_IEMOCAP_META = os.path.join(os.path.dirname(os.path.dirname(SRC)), 'sac-s3prl/s3prl/downstream/emotion/meta_data')
DEFAULT_TASKS = ('sc2', 'sid', 'iemocap')


# ----------------------------------------------------------------------------- subsets

def _read_json(path):
    with open(path) as f:
        return json.load(f)['data']


def _sc2_subset(json_path, mid_to_idx, per_class, rng):
    by_class = defaultdict(list)
    for d in _read_json(json_path):
        by_class[mid_to_idx[d['labels'].split(',')[0].strip()]].append(d['wav'])
    items = []
    for c in sorted(by_class):
        paths = sorted(by_class[c])
        rng.shuffle(paths)
        items += [(p, c) for p in paths[:per_class]]
    return items


def _sid_subsets(train_json, test_json, n_speakers, n_ref, n_query, rng):
    def group(path):
        g = defaultdict(list)
        for d in _read_json(path):
            g[int(d['label'])].append(d['wav'])
        return g
    tr, te = group(train_json), group(test_json)
    eligible = sorted(s for s in tr if len(tr[s]) >= n_ref and len(te.get(s, [])) >= n_query)
    rng.shuffle(eligible)
    speakers = sorted(eligible[:n_speakers])
    ref, query = [], []
    for new_label, s in enumerate(speakers):
        a, b = sorted(tr[s]), sorted(te[s])
        rng.shuffle(a)
        rng.shuffle(b)
        ref += [(p, new_label) for p in a[:n_ref]]
        query += [(p, new_label) for p in b[:n_query]]
    return ref, query, len(speakers)


def _iemocap_subsets(manifest_dir, meta_dir, fold, ref_per_class, rng):
    """s3prl emotion split (meta_data/Session<fold>): train -> reference, test -> query (all of it).
    Wavs come from the binaural manifests, matched on orig_rel_path as in s3prl emotion/dataset.py."""
    rel2wav = {}
    for split in ('pretrain', 'preval', 'pretest'):
        for d in _read_json(os.path.join(manifest_dir, f'iemocap_spatial_{split}.json')):
            if d['orig_rel_path'] in rel2wav:
                raise ValueError(f"{d['orig_rel_path']} is simulated twice in {manifest_dir}: old (v1) binaural IEMOCAP?")
            rel2wav[d['orig_rel_path']] = d['wav']

    def load_meta(name):
        with open(os.path.join(meta_dir, f'Session{fold}', f'{name}_meta_data.json')) as f:
            meta = json.load(f)
        missing = [m['path'] for m in meta['meta_data'] if m['path'] not in rel2wav]
        if missing:
            raise ValueError(f'{len(missing)} IEMOCAP {name} utterances have no simulation in {manifest_dir}, e.g. {missing[0]}')
        return meta['labels'], [(rel2wav[m['path']], meta['labels'][m['label']]) for m in meta['meta_data']]

    class_dict, train = load_meta('train')
    _, test = load_meta('test')
    by_class = defaultdict(list)
    for item in sorted(train):
        by_class[item[1]].append(item)
    ref = []
    for c in sorted(by_class):
        items = by_class[c]
        rng.shuffle(items)
        ref += items[:ref_per_class]
    return ref, test, len(class_dict)


def _load_wav(path, max_samples):
    wav, sr = torchaudio.load(remap_wav_path(path))
    if sr != SAMPLE_RATE:
        raise ValueError(f'{path}: sample rate {sr}, expected {SAMPLE_RATE}')
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav[:2, :max_samples].to(torch.float16)  # fp16 halves the RAM; quantisation noise is ~-66 dB


# ----------------------------------------------------------------------------- metrics

def _standardize(ref, query):
    mu, sd = ref.mean(0, keepdim=True), ref.std(0, keepdim=True) + 1e-6
    return (ref - mu) / sd, (query - mu) / sd


def accuracy(pred, y):
    return (pred == y).float().mean().item()


def unweighted_accuracy(pred, y, n_class):
    """Mean per-class recall (UA), the usual IEMOCAP metric next to plain accuracy."""
    recalls = [(pred[y == c] == c).float().mean() for c in range(n_class) if (y == c).any()]
    return torch.stack(recalls).mean().item()


def knn_predict(ref, y_ref, query, n_class, k=20, temperature=0.07):
    sim = F.normalize(query, dim=1) @ F.normalize(ref, dim=1).T
    top_sim, top_idx = sim.topk(min(k, ref.shape[0]), dim=1)
    votes = torch.zeros(query.shape[0], n_class, device=ref.device)
    votes.scatter_add_(1, y_ref[top_idx], (top_sim / temperature).exp())
    return votes.argmax(1)


def ncm_predict(ref, y_ref, query, n_class):
    ref_n = F.normalize(ref, dim=1)
    centroids = torch.zeros(n_class, ref.shape[1], device=ref.device).index_add_(0, y_ref, ref_n)
    return (F.normalize(query, dim=1) @ F.normalize(centroids, dim=1).T).argmax(1)


def linear_probe_predict(ref, y_ref, query, n_class, steps=300, lr=1e-3, weight_decay=1e-4):
    with torch.enable_grad():
        with torch.random.fork_rng(devices=[]):  # fixed init, without touching the training RNG
            torch.manual_seed(0)
            head = nn.Linear(ref.shape[1], n_class).to(ref.device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(steps):
            loss = F.cross_entropy(head(ref), y_ref)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return head(query).argmax(1)


def cosine_eer(emb, labels):
    e = F.normalize(emb, dim=1)
    i, j = torch.triu_indices(len(e), len(e), offset=1, device=e.device)
    scores = (e @ e.T)[i, j]
    target = (labels[i] == labels[j]).float()
    order = scores.argsort(descending=True)
    target = target[order]
    n_tgt, n_non = target.sum(), (1 - target).sum()
    fnr = 1 - target.cumsum(0) / n_tgt            # targets below the threshold
    fpr = (1 - target).cumsum(0) / n_non           # non-targets above it
    idx = (fnr - fpr).abs().argmin()
    return ((fnr[idx] + fpr[idx]) / 2).item()


# ----------------------------------------------------------------------------- probe

class PretrainProbe:
    def __init__(self, data_root=None, label_csv=DEFAULT_LABEL_CSV, tasks=DEFAULT_TASKS,
                 sc2_per_class=30, sid_speakers=200, sid_ref=8, sid_query=4,
                 iemocap_manifest_dir=None, iemocap_meta_dir=DEFAULT_IEMOCAP_META, iemocap_fold=1, emo_ref_per_class=250,
                 batch_size=32, seed=0, device='cuda', num_threads=8,
                 input_repr='stft', fbank_mean=None, fbank_std=None):
        data_root = data_root or os.environ.get('SAC_DATA_ROOT', DEFAULT_DATA_ROOT)
        self.device = device
        self.batch_size = batch_size
        self.input_repr, self.fbank_mean, self.fbank_std = input_repr, fbank_mean, fbank_std
        self.sets = {}
        t0 = time.time()
        rng = random.Random(seed)
        jobs = {}
        if 'sc2' in tasks:
            with open(label_csv) as f:
                mid_to_idx = {row['mid']: int(row['index']) for row in csv.DictReader(f)}
            jobs['sc2'] = (
                _sc2_subset(os.path.join(data_root, 'speechcommands_spatial_pretrain.json'), mid_to_idx, sc2_per_class, rng),
                _sc2_subset(os.path.join(data_root, 'speechcommands_spatial_pretest.json'), mid_to_idx, sc2_per_class, rng),
                len(mid_to_idx),
            )
        if 'sid' in tasks:
            jobs['sid'] = _sid_subsets(os.path.join(data_root, 'voxceleb1_spatial_pretrain.json'),
                                       os.path.join(data_root, 'voxceleb1_spatial_pretest.json'),
                                       sid_speakers, sid_ref, sid_query, rng)
        if 'iemocap' in tasks:  # built last, so adding it leaves the sc2 / sid subsets unchanged
            if iemocap_manifest_dir is None:
                v2 = os.path.join(data_root, 'IEMOCAP', 'simu_v2', 'manifests')
                iemocap_manifest_dir = v2 if os.path.isdir(v2) else data_root
            jobs['iemocap'] = _iemocap_subsets(iemocap_manifest_dir, iemocap_meta_dir, iemocap_fold, emo_ref_per_class, rng)
        unknown = set(tasks) - set(jobs)
        if unknown:
            raise ValueError(f'unknown probe tasks: {sorted(unknown)} (choose from sc2, sid, iemocap)')

        with ThreadPoolExecutor(num_threads) as pool:
            for task, (ref, query, n_class) in jobs.items():
                splits = {}
                for name, items in (('ref', ref), ('query', query)):
                    wavs = list(pool.map(lambda it: _load_wav(it[0], CHUNK_SAMPLES), items))
                    splits[name] = (wavs, torch.tensor([it[1] for it in items], device=device))
                self.sets[task] = {'n_class': n_class, **splits}
                print(f'[probe] {task}: {len(ref)} reference / {len(query)} query clips, {n_class} classes', flush=True)
        print(f'[probe] subsets loaded from {data_root} in {time.time() - t0:.0f} s', flush=True)

    # --- embeddings

    @torch.no_grad()
    def _sac_batch(self, encoder, wavs, task):
        xs, lengths = [], []
        for w in wavs:
            w = w.to(self.device).float()
            lengths.append(w.shape[-1])
            if self.input_repr == 'logmel':  # frames zero-padded to 1024 for every task, as in the mono dataloader
                xs.append(wav2fbank(w.cpu(), self.fbank_mean, self.fbank_std).to(self.device))
                continue
            if task != 'sc2':  # s3prl (SID, ER) pads the waveform, then takes the STFT
                w = F.pad(w, (0, CHUNK_SAMPLES - w.shape[-1]))
            xs.append(wav2stft(w))  # SC2 dataset: STFT of the clip, frames zero-padded to 1024
        x = torch.stack(xs)
        B = x.shape[0]
        with torch.cuda.amp.autocast(enabled=self.device != 'cpu'):
            h = encoder._encode_with_mamba(x)[:, 1:, :]
        h = h.float().reshape(B, -1, T_PATCHES, h.shape[-1])  # token order: frequency-major (16 rows for STFT, 8 for log-mel)
        if task == 'sc2':
            if self.input_repr == 'logmel':  # padded frames are not 0 after normalising, so use the clip length
                n_frames = torch.tensor([(n - 400) // HOP + 1 for n in lengths], device=self.device)
                t_mask = (torch.arange(T_PATCHES, device=self.device)[None] * PATCH_T < n_frames[:, None]).float()
            else:
                patch_energy = x.abs().sum(dim=(1, 2)).view(B, T_PATCHES, PATCH_T).sum(-1)
                t_mask = (patch_energy > 0).float()
            t_mask[t_mask.sum(-1) == 0] = 1.0
            return (h * t_mask[:, None, :, None]).sum(dim=(1, 2)) / (h.shape[1] * t_mask.sum(-1, keepdim=True))
        n_t = torch.tensor([min(T_PATCHES, max(1, round(n / SID_DOWNSAMPLE))) for n in lengths], device=self.device)
        t_mask = (torch.arange(T_PATCHES, device=self.device)[None] < n_t[:, None]).float()
        h = h.permute(0, 2, 1, 3).reshape(B, T_PATCHES, -1)  # [B, 64, 16 * D]
        return (h * t_mask[..., None]).sum(1) / t_mask.sum(-1, keepdim=True)

    @torch.no_grad()
    def _gramt_batch(self, gramt, wavs, task):
        ws = []
        for w in wavs:
            w = w.to(self.device).float()
            if task == 'sc2':  # as gramt_wrapper.BinauralWaveformDataset: DC removed, 1 s
                w = w - w.mean(dim=-1, keepdim=True)
                w = F.pad(w, (0, max(0, SAMPLE_RATE - w.shape[-1])))[:, :SAMPLE_RATE]
            ws.append(w)
        with torch.cuda.amp.autocast(enabled=self.device != 'cpu'):
            hidden, n_tokens = gramt(ws)
        h = hidden[-1].float()
        mask = torch.zeros(h.shape[:2], device=h.device)
        for i, n in enumerate(n_tokens):
            mask[i, :n] = 1.0
        return (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

    def _embed(self, batch_fn, wavs):
        out = [batch_fn(wavs[i:i + self.batch_size]) for i in range(0, len(wavs), self.batch_size)]
        return torch.cat(out)

    # --- evaluation

    def _metrics(self, task, ref, query):
        s = self.sets[task]
        y_ref, y_q, C = s['ref'][1], s['query'][1], s['n_class']
        ref, query = _standardize(ref, query)
        if task == 'sc2':
            return {'sc2/knn20_acc': accuracy(knn_predict(ref, y_ref, query, C, k=20), y_q),
                    'sc2/ncm_acc': accuracy(ncm_predict(ref, y_ref, query, C), y_q),
                    'sc2/linear_acc': accuracy(linear_probe_predict(ref, y_ref, query, C), y_q)}
        if task == 'sid':
            return {'sid/ncm_acc': accuracy(ncm_predict(ref, y_ref, query, C), y_q),
                    'sid/linear_acc': accuracy(linear_probe_predict(ref, y_ref, query, C), y_q),
                    'sid/eer': cosine_eer(query, y_q)}
        lin = linear_probe_predict(ref, y_ref, query, C)
        return {'emo/knn20_ua': unweighted_accuracy(knn_predict(ref, y_ref, query, C, k=20), y_q, C),
                'emo/ncm_ua': unweighted_accuracy(ncm_predict(ref, y_ref, query, C), y_q, C),
                'emo/linear_ua': unweighted_accuracy(lin, y_q, C),
                'emo/linear_acc': accuracy(lin, y_q)}

    def evaluate(self, batch_fn_for_task):
        """batch_fn_for_task(task) -> fn(list of [2, N] wavs) -> [B, D] embeddings."""
        res = {}
        t0 = time.time()
        for task, s in self.sets.items():
            fn = batch_fn_for_task(task)
            ref, query = self._embed(fn, s['ref'][0]), self._embed(fn, s['query'][0])
            res.update(self._metrics(task, ref, query))
        res['probe_sec'] = time.time() - t0
        return res

    def evaluate_sac(self, spectral_encoder):
        was_training = spectral_encoder.training
        spectral_encoder.eval()  # also turns off gradient checkpointing in AMBAModel
        try:
            return self.evaluate(lambda task: lambda w: self._sac_batch(spectral_encoder, w, task))
        finally:
            spectral_encoder.train(was_training)

    def evaluate_gramt(self, gramt_encoder):
        gramt_encoder.eval()
        return self.evaluate(lambda task: lambda w: self._gramt_batch(gramt_encoder, w, task))

    def chance(self):
        return {f'{t}/chance': 1.0 / s['n_class'] for t, s in self.sets.items()}


def format_metrics(res):
    return ' | '.join(f'{k} {v * 100:.2f}' if k != 'probe_sec' else f'{k} {v:.0f}' for k, v in res.items()
                      if isinstance(v, float))


# ----------------------------------------------------------------------------- standalone

def _expand_models(specs):
    """Checkpoint paths / globs / 'random' / 'gramt[:model_id]' -> ordered list of specs."""
    def ckpt_key(p):
        m = re.search(r'audio_model\.(\d+)\.pth$', p)
        return (0, int(m.group(1)), p) if m else (1, 0, p)
    out = []
    for s in specs:
        if s == 'random' or s.startswith('gramt'):
            out.append(s)
        else:
            paths = sorted(glob.glob(s), key=ckpt_key)
            if not paths:
                raise FileNotFoundError(f'no checkpoint matches {s}')
            out += paths
    return out


def main():
    p = argparse.ArgumentParser(description='Frozen-embedding probes (SC2, VoxCeleb1 SID, IEMOCAP ER) on saved pretraining checkpoints')
    p.add_argument('--models', nargs='+', required=True,
                   help="checkpoint paths or globs (audio_model.*.pth), 'random', 'gramt' or 'gramt:<hf id or dir>'")
    p.add_argument('--out', default=None, help='append one json line per model here')
    p.add_argument('--epoch_iter', type=int, default=4000, help='steps between checkpoints (audio_model.N = step (N-1) * epoch_iter)')
    p.add_argument('--data_root', default=None, help='default: $SAC_DATA_ROOT, else /storage/yotam/ssamba/data')
    p.add_argument('--tasks', default=','.join(DEFAULT_TASKS))
    p.add_argument('--sc2_per_class', type=int, default=30)
    p.add_argument('--sid_speakers', type=int, default=200)
    p.add_argument('--sid_ref', type=int, default=8)
    p.add_argument('--sid_query', type=int, default=4)
    p.add_argument('--iemocap_manifest_dir', default=None,
                   help='iemocap_spatial_*.json dir (default: <data_root>/IEMOCAP/simu_v2/manifests if it exists, else <data_root>)')
    p.add_argument('--iemocap_meta_dir', default=DEFAULT_IEMOCAP_META, help='s3prl emotion meta_data dir')
    p.add_argument('--iemocap_fold', type=int, default=1)
    p.add_argument('--emo_ref_per_class', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--input_repr', default='stft', choices=['stft', 'logmel'], help='input of the SAC checkpoints; logmel needs --fbank_mean/--fbank_std')
    p.add_argument('--fbank_mean', type=float, default=None)
    p.add_argument('--fbank_std', type=float, default=None)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    probe = PretrainProbe(args.data_root, tasks=tuple(args.tasks.split(',')), sc2_per_class=args.sc2_per_class,
                          sid_speakers=args.sid_speakers, sid_ref=args.sid_ref, sid_query=args.sid_query,
                          iemocap_manifest_dir=args.iemocap_manifest_dir, iemocap_meta_dir=args.iemocap_meta_dir,
                          iemocap_fold=args.iemocap_fold, emo_ref_per_class=args.emo_ref_per_class,
                          batch_size=args.batch_size, device=device,
                          input_repr=args.input_repr, fbank_mean=args.fbank_mean, fbank_std=args.fbank_std)
    print(f'[probe] chance: {format_metrics(probe.chance())}', flush=True)

    rows = []
    for spec in _expand_models(args.models):
        row = {'model': spec}
        if spec.startswith('gramt'):
            from gramt_wrapper.gramt_encoder import GRAMTEncoder, DEFAULT_MODEL_ID
            model_id = spec.split(':', 1)[1] if ':' in spec else DEFAULT_MODEL_ID
            enc = GRAMTEncoder(model_id=model_id).to(device)
            row.update(probe.evaluate_gramt(enc))
        else:
            from binaural_dual_stream_sac.binaural_sac_model import BinauralSSAMBASACModel
            from binaural_dual_stream_sac.downstream_experts import load_binaural_checkpoint
            torch.manual_seed(0)
            logmel = args.input_repr == 'logmel'
            model = BinauralSSAMBASACModel(use_checkpointing=False, spectral_only=logmel,
                                           input_fdim=128 if logmel else 256, in_chans=2 if logmel else 4)
            load_binaural_checkpoint(model, spec, 'dual_encoder.spectral_encoder.')
            enc = model.dual_encoder.spectral_encoder.to(device)
            m = re.search(r'audio_model\.(\d+)\.pth$', spec)
            if m:
                row['step'] = (int(m.group(1)) - 1) * args.epoch_iter
            row.update(probe.evaluate_sac(enc))
            del model
        del enc
        torch.cuda.empty_cache()
        print(f'[probe] {spec}: {format_metrics(row)}', flush=True)
        rows.append(row)
        if args.out:
            with open(args.out, 'a') as f:
                f.write(json.dumps(row) + '\n')

    keys = [k for k in rows[0] if k not in ('model', 'step', 'probe_sec')] if rows else []
    print('\n' + ' '.join(f'{k:>15}' for k in keys) + '  model')
    for r in rows:
        print(' '.join(f'{r.get(k, float("nan")) * 100:15.2f}' for k in keys) + f'  {os.path.basename(r["model"])}')


if __name__ == '__main__':
    main()
