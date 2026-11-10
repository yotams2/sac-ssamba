"""CPU test of the step-exact resume in src/sac/run_pretrain_sac.py (train_sac).

Runs the real train_sac with a tiny linear "model" (task pretrain_mpg, no SAC) on a toy dataset of 23 items,
batch 4 (5 batches per epoch, 3 epochs = 15 steps, eval every 4 steps, latest.pth every step). The model records
which sample ids it sees. A model that raises at a chosen call simulates a preemption.

  A: uninterrupted
  B: killed at step 7 (mid epoch 2), resumed from latest.pth          -> same samples, order, weights as A
  C: killed at step 10 (epoch boundary), resumed                       -> same as A
  D: killed twice (steps 3 and 12)                                     -> same as A
  E: killed at step 7, checkpoint stripped to the old format (no shuffle_seed, wrong epoch) -> 15 steps total,
     each epoch's batches distinct within the resumed part

Usage: python -I scripts/test_pretrain_resume.py  (needs torch; run via sbatch, not on the Jupyter node)
"""
import copy
import os
import sys
import tempfile
import types
from argparse import Namespace

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'src')
sys.path.insert(0, SRC)

# Stub the SAC model/feature modules (they need the Mamba CUDA kernels); load the real run_pretrain_sac.
pkg = types.ModuleType('sac')
pkg.__path__ = [os.path.join(SRC, 'sac')]
sys.modules['sac'] = pkg
for name, attrs in [('sac.acoustic_features', ['extract_acoustic_features', 'calculate_acoustic_feature_stats']),
                    ('sac.sac_model', ['SSAMBASACModel'])]:
    m = types.ModuleType(name)
    for a in attrs:
        setattr(m, a, None)
    sys.modules[name] = m
_dl = types.ModuleType('dataloader')  # the real one pulls in the binaural package; only AudioDataset is used
_dl.AudioDataset = torch.utils.data.Dataset
sys.modules['dataloader'] = _dl
import sac.run_pretrain_sac as rp  # noqa: E402

N, B, EPOCHS = 23, 4, 3
SPE = N // B
TOTAL = EPOCHS * SPE


class Toy(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        x = torch.full((3,), float(i))
        return x, torch.zeros(1)


class Killed(Exception):
    pass


class ToyModel(nn.Module):
    def __init__(self, kill_at=None):
        super().__init__()
        self.lin = nn.Linear(3, 1)
        g = torch.Generator().manual_seed(123)  # fixed init without touching the global RNG
        with torch.no_grad():  # small weights, so the tiny warm-up lr changes are resolved in fp32
            self.lin.weight.copy_(torch.randn(self.lin.weight.shape, generator=g) * 1e-3)
            self.lin.bias.copy_(torch.randn(self.lin.bias.shape, generator=g) * 1e-3)
        self.seen = []          # sample ids of each training batch
        self.calls = 0
        self.kill_at = kill_at  # training call index (0-based) that raises

    def forward(self, fbank, task, mask_patch=None, cluster=None):
        if self.training:
            if self.kill_at is not None and self.calls == self.kill_at:
                raise Killed()
            self.calls += 1
            self.seen.append(fbank[:, 0].long().tolist())
        return ((self.lin(fbank / N) - 1.0) ** 2).mean()


def make_args(exp_dir, resume=''):
    return Namespace(sac_loss=False, diagnostic_steps=0, task='pretrain_mpg', lr=1e-2, lr_patience=2,
                     use_wandb=False, n_epochs=EPOCHS, n_print_steps=1000, epoch_iter=4, latest_every=1,
                     exp_dir=exp_dir, mask_patch=1, num_mel_bins=128, fshape=16, sample_rate=16000,
                     sac_features='', sac_lambda=0.0, sac_temperature=0.1, sac_sigma=1.0, resume=resume)


def loaders():
    tr = torch.utils.data.DataLoader(Toy(), batch_size=B, shuffle=True, num_workers=2, drop_last=True)
    va = torch.utils.data.DataLoader(Toy(), batch_size=B, shuffle=False, num_workers=0, drop_last=True)
    return tr, va


def run(exp_dir, kill_at=None, resume='', strip_ckpt=False):
    """One job segment. Returns (seen batches, final state_dict or None if killed)."""
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    latest = os.path.join(exp_dir, 'models', 'latest.pth')
    if resume and strip_ckpt:
        ck = torch.load(latest, map_location='cpu')
        ck.pop('shuffle_seed')
        ck['epoch'] = 1  # what the old code could leave behind
        torch.save(ck, latest)
    if not resume:
        torch.manual_seed(0)  # fixes the fresh run's shuffle_seed, so runs are comparable
    model = ToyModel(kill_at)
    tr, va = loaders()
    try:
        rp.train_sac(model, tr, va, make_args(exp_dir, latest if resume else ''), torch.device('cpu'))
    except Killed:
        return model.seen, None
    return model.seen, copy.deepcopy(model.state_dict())


def run_with_kills(kills, strip=False):
    """Segments: fresh run killed at kills[0], resumes killed at kills[1:], final resume. strip: convert the
    checkpoint to the old format before the first resume."""
    d = tempfile.mkdtemp()
    seen, prev = [], 0
    for j, k in enumerate(kills + [None]):
        s, st = run(d, kill_at=None if k is None else k - prev, resume=j > 0, strip_ckpt=strip and j == 1)
        assert (st is None) == (k is not None), 'kill did not happen as planned'
        seen += s
        prev = k
    return seen, st


def same_weights(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


fails = 0


def check(name, ok, detail=''):
    global fails
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
    fails += not ok


d = tempfile.mkdtemp()
seenA, stA = run(d)
check('A: total steps', len(seenA) == TOTAL, f'{len(seenA)} vs {TOTAL}')
for e in range(EPOCHS):
    ids = sum(seenA[e * SPE:(e + 1) * SPE], [])
    check(f'A: epoch {e + 1} has {SPE * B} distinct samples', len(set(ids)) == SPE * B)
check('A: epochs shuffled differently', seenA[:SPE] != seenA[SPE:2 * SPE])
check('A: audio_model.final.pth written', os.path.exists(os.path.join(d, 'models', 'audio_model.final.pth')))

for name, kills in [('B (kill at 7)', [7]), ('C (kill at 10, epoch boundary)', [10]), ('D (kills at 3, 12)', [3, 12])]:
    seen, st = run_with_kills(kills)
    check(f'{name}: same batches as A', seen == seenA, f'{len(seen)} steps')
    check(f'{name}: same final weights as A', same_weights(st, stA))

seenE, stE = run_with_kills([7], strip=True)
check('E (old-format ckpt): total steps', len(seenE) == TOTAL, f'{len(seenE)} vs {TOTAL}')
check('E: first 7 batches as A', seenE[:7] == seenA[:7])
rest = sum(seenE[7:SPE * 2], [])  # resumed part of epoch 2
check('E: resumed part of epoch 2 has distinct samples', len(set(rest)) == len(rest))

print('ALL PASS' if fails == 0 else f'{fails} FAILED')
sys.exit(1 if fails else 0)
