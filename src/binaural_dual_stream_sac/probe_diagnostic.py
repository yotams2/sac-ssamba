"""Sanity check for the frozen binaural-SAC linear probe on Speech Commands v2.

Extracts frozen spectral-encoder features once for a subset of SC2 clips, then trains a
linear probe on the cached features for each combination of
  encoder: pretrained checkpoint vs random init
  pooling: mean over all 1024 time frames (current default) vs only frames that hold audio
SC2 clips are 1 s, but the STFT is padded to 1024 frames (10.24 s), so ~90% of the tokens
are padding in the default pooling.

If pretrained ~= random, the probe is not using the pretrained weights (or they carry no
keyword information). If real-time pooling >> all-frame pooling, padding is the problem.

Usage: python probe_diagnostic.py --ckpt <.../models/best_audio_model.pth> [--n_train 4000 --n_test 2000]
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import dataloader  # noqa: E402
from binaural_dual_stream_sac.downstream_experts import MonauralSpectralUpstreamExpert  # noqa: E402


def subset_json(path, n, seed, out):
    with open(path) as f:
        data = json.load(f)['data']
    random.Random(seed).shuffle(data)
    with open(out, 'w') as f:
        json.dump({'data': data[:n]}, f)
    return out


@torch.no_grad()
def extract(encoder, loader, device):
    """Return pooled features for both pooling modes, and labels."""
    feats_all, feats_real, labels = [], [], []
    real_frac = []
    for x, y in loader:
        x = x.to(device)
        h = encoder._encode_with_mamba(x)[:, 1:, :]  # [B, 1024, D], CLS dropped
        B, _, D = h.shape
        feats_all.append(h.mean(dim=1).float().cpu())
        # Same logic as BinauralSACClassifier(pool_real_time_only=True): 16 freq x 64 time patches.
        frame_energy = x.abs().sum(dim=(1, 2))  # [B, 1024 frames]
        time_mask = (frame_energy.view(B, 64, 16).sum(-1) > 0).float()  # [B, 64]
        time_mask[time_mask.sum(-1) == 0] = 1.0
        real_frac.append(time_mask.mean(-1).cpu())
        hg = h.reshape(B, 16, 64, D) * time_mask[:, None, :, None]
        feats_real.append((hg.sum(dim=(1, 2)) / (16.0 * time_mask.sum(-1, keepdim=True))).float().cpu())
        labels.append(y.argmax(-1))
    return torch.cat(feats_all), torch.cat(feats_real), torch.cat(labels), torch.cat(real_frac).mean().item()


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
    torch.manual_seed(0)

    conf = {'mode': 'evaluation', 'mixup': 0, 'target_length': 1024}
    loaders = {}
    for split, n in [('pretrain', args.n_train), ('pretest', args.n_test)]:
        js = subset_json(os.path.join(args.data_root, f'speechcommands_spatial_{split}.json'), n, 0,
                         os.path.join(args.workdir, f'probe_diag_sc_{split}.json'))
        ds = dataloader.BinauralSTFTClassificationDataset(js, label_csv=args.label_csv, audio_conf=conf)
        loaders[split] = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=8)
    n_class = loaders['pretrain'].dataset.label_num

    results = []
    for name, ckpt in [('pretrained', args.ckpt), ('random', 'random')]:
        torch.manual_seed(0)
        expert = MonauralSpectralUpstreamExpert(ckpt).to(device).eval()
        enc = expert.spectral_encoder
        tr_all, tr_real, ytr, frac = extract(enc, loaders['pretrain'], device)
        te_all, te_real, yte, _ = extract(enc, loaders['pretest'], device)
        print(f'[{name}] real-audio fraction of time patches: {frac:.3f}; '
              f'feature std across clips (all-frame pool): {tr_all.std(0).mean():.4f}', flush=True)
        for pool, xtr, xte in [('all_frames', tr_all, te_all), ('real_frames', tr_real, te_real)]:
            acc = linear_probe(xtr, ytr, xte, yte, n_class, device)
            results.append((name, pool, acc))
            print(f'[{name}] pool={pool}: test acc {acc * 100:.2f}%', flush=True)
        del expert, enc
        torch.cuda.empty_cache()

    print('\nencoder      pooling      test acc')
    for name, pool, acc in results:
        print(f'{name:<12} {pool:<12} {acc * 100:6.2f}%')
    print(f'(chance = {100 / n_class:.2f}%, {args.n_train} train / {args.n_test} test clips)')


if __name__ == '__main__':
    main()
