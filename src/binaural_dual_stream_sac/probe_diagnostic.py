"""Sanity check for the frozen binaural-SAC linear probe on Speech Commands v2.

The frozen SC2 probe reached 13.5% (GRAM-T: 89.6%). The SC2 runs already pooled only over
time patches that hold audio (pool_real_time_only=True), so pooling over padding is ruled out.
This script tests the remaining suspects with a linear probe on cached frozen features:

  encoder: pretrained checkpoint vs random init
      pretrained ~= random -> the pretrained weights add nothing for keywords
      (or did not load: the loader now raises / logs that)
  input:   'zero_pad' = the 1 s clip zero-padded to 1024 frames (10.24 s, ~90% zeros), as in the runs
           'tile'     = the clip repeated to fill the 1024 frames (no padding)
      tile >> zero_pad -> the long zero padding (unlike the pretraining clips) hurts the encoder
Features are mean-pooled over the time patches that hold audio (all patches for 'tile').
It also prints the input level (mean |STFT| over real frames) for SC2 and for the pretraining
set, to spot a scale mismatch: the encoder gets the raw STFT, without log or normalisation.

Usage: python probe_diagnostic.py --ckpt <.../models/best_audio_model.pth> [--n_train 4000 --n_test 2000]
"""
import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn as nn

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import dataloader  # noqa: E402
from binaural_dual_stream_sac.downstream_experts import MonauralSpectralUpstreamExpert  # noqa: E402

N_FRAMES, T_PATCHES, F_PATCHES, PATCH_T = 1024, 64, 16, 16


def subset_json(path, n, seed, out):
    with open(path) as f:
        data = json.load(f)['data']
    random.Random(seed).shuffle(data)
    with open(out, 'w') as f:
        json.dump({'data': data[:n]}, f)
    return out


def real_frame_counts(x):
    """Number of leading STFT frames that hold audio, per clip. x: [B, 4, F, T]."""
    frame_has_audio = x.abs().sum(dim=(1, 2)) > 0  # [B, T]
    idx = torch.arange(x.shape[-1], device=x.device)
    return torch.where(frame_has_audio, idx + 1, torch.zeros_like(idx)).max(dim=-1).values.clamp(min=1)


def tile_to_full(x, n_real):
    out = torch.empty_like(x)
    for b in range(x.shape[0]):
        n = int(n_real[b])
        out[b] = x[b, ..., :n].repeat(1, 1, math.ceil(N_FRAMES / n))[..., :N_FRAMES]
    return out


@torch.no_grad()
def extract(encoder, loader, device):
    """Pooled features for both input variants, labels, and the real-audio fraction."""
    feats = {'zero_pad': [], 'tile': []}
    labels, real_frac = [], []
    for x, y in loader:
        x = x.to(device)
        B = x.shape[0]
        n_real = real_frame_counts(x)
        # time patches holding audio (same rule as BinauralSACClassifier pool_real_time_only)
        t_mask = (torch.arange(T_PATCHES, device=device)[None] * PATCH_T < n_real[:, None]).float()  # [B, 64]
        real_frac.append(t_mask.mean(-1).cpu())

        h = encoder._encode_with_mamba(x)[:, 1:, :]  # [B, 1024 tokens, D], CLS dropped
        D = h.shape[-1]
        hg = h.reshape(B, F_PATCHES, T_PATCHES, D) * t_mask[:, None, :, None]
        feats['zero_pad'].append((hg.sum(dim=(1, 2)) / (F_PATCHES * t_mask.sum(-1, keepdim=True))).float().cpu())

        h = encoder._encode_with_mamba(tile_to_full(x, n_real))[:, 1:, :]
        feats['tile'].append(h.mean(dim=1).float().cpu())
        labels.append(y.argmax(-1))
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(labels), torch.cat(real_frac).mean().item()


@torch.no_grad()
def input_level(loader, max_batches=8):
    vals = []
    for i, (x, _) in enumerate(loader):
        if i == max_batches:
            break
        n_real = real_frame_counts(x)
        for b in range(x.shape[0]):
            vals.append(x[b, ..., :int(n_real[b])].abs().mean().item())
    v = torch.tensor(vals)
    return v.mean().item(), v.median().item()


def linear_probe(xtr, ytr, xte, yte, n_class, device, epochs=200, lr=1e-3):
    mean, std = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xte = ((xtr - mean) / std).to(device), ((xte - mean) / std).to(device)
    ytr, yte = ytr.to(device), yte.to(device)
    head = nn.Linear(xtr.shape[1], n_class).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), 256):
            idx = perm[i:i + 256]
            loss = nn.functional.cross_entropy(head(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (head(xte).argmax(-1) == yte).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_root', default=os.environ.get('SAC_DATA_ROOT', '/storage/yotam/ssamba/data'))
    p.add_argument('--label_csv', default=os.path.join(SRC, 'finetune/speechcommands_v2/data/speechcommands_class_labels_indices.csv'))
    p.add_argument('--n_train', type=int, default=4000)
    p.add_argument('--n_test', type=int, default=2000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--workdir', default=os.environ.get('TMPDIR', '/tmp'))
    args = p.parse_args()
    device = 'cuda'

    conf = {'mode': 'evaluation', 'mixup': 0, 'target_length': N_FRAMES}
    loaders = {}
    for split, n in [('pretrain', args.n_train), ('pretest', args.n_test)]:
        js = subset_json(os.path.join(args.data_root, f'speechcommands_spatial_{split}.json'), n, 0,
                         os.path.join(args.workdir, f'probe_diag_sc_{split}.json'))
        ds = dataloader.BinauralSTFTClassificationDataset(js, label_csv=args.label_csv, audio_conf=conf)
        loaders[split] = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=8)
    n_class = loaders['pretrain'].dataset.label_num

    pt_json = subset_json(os.path.join(args.data_root, 'binaural_train.json'), 256, 0,
                          os.path.join(args.workdir, 'probe_diag_pretrain_subset.json'))
    pt_loader = torch.utils.data.DataLoader(
        dataloader.BinauralSTFTClassificationDataset(pt_json, label_csv=args.label_csv, audio_conf=conf),
        batch_size=args.batch_size, num_workers=8)
    for name, ld in [('SC2 train', loaders['pretrain']), ('pretraining set', pt_loader)]:
        mean, med = input_level(ld)
        print(f'input level, {name}: mean |STFT| over real frames: mean {mean:.4g}, median {med:.4g}', flush=True)

    results = []
    for name, ckpt in [('pretrained', args.ckpt), ('random', 'random')]:
        torch.manual_seed(0)
        expert = MonauralSpectralUpstreamExpert(ckpt).to(device).eval()
        enc = expert.spectral_encoder
        tr, ytr, frac = extract(enc, loaders['pretrain'], device)
        te, yte, _ = extract(enc, loaders['pretest'], device)
        print(f'[{name}] fraction of time patches holding audio: {frac:.3f}', flush=True)
        for variant in ('zero_pad', 'tile'):
            torch.manual_seed(0)
            acc = linear_probe(tr[variant], ytr, te[variant], yte, n_class, device)
            results.append((name, variant, acc))
            print(f'[{name}] input={variant}: test acc {acc * 100:.2f}%', flush=True)
        del expert, enc
        torch.cuda.empty_cache()

    print('\nencoder      input        test acc')
    for name, variant, acc in results:
        print(f'{name:<12} {variant:<12} {acc * 100:6.2f}%')
    print(f'(chance = {100 / n_class:.2f}%, {args.n_train} train / {args.n_test} test clips)')


if __name__ == '__main__':
    main()
