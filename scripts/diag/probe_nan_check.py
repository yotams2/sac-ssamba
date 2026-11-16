"""Why does k=5 audio_model.2 probe at exactly chance? Per checkpoint: weight NaN/inf + max |w|, then SC2-subset
embeddings (pretrain_probe._sac_batch) under fp16 autocast vs fp32: non-finite fraction, max |h|, mean std over clips,
and SC2 kNN / class-mean accuracy in each precision. Run from sac-ssamba/src (see probe_nan_check.sbatch)."""
import sys, os, math, contextlib
import torch
from binaural_dual_stream_sac import pretrain_probe as pp

ckpts = sys.argv[1:]
dev = 'cuda'
import pickle
with open(os.path.join(pp._mono_exp_dir(ckpts[0]), 'args.pkl'), 'rb') as f:
    ra = pickle.load(f)
probe = pp.PretrainProbe(tasks=('sc2',), sc2_per_class=10, device=dev, input_repr='logmel',
                         fbank_mean=ra.dataset_mean, fbank_std=ra.dataset_std, audio='mono')
s = probe.sets['sc2']
real_autocast = torch.cuda.amp.autocast


def embed(enc, fp16):
    torch.cuda.amp.autocast = real_autocast if fp16 else (lambda enabled=True, **k: contextlib.nullcontext())
    try:
        fn = lambda w: probe._sac_batch(enc, w, 'sc2')
        return probe._embed(fn, s['ref'][0]), probe._embed(fn, s['query'][0])
    finally:
        torch.cuda.amp.autocast = real_autocast


for spec in ckpts + ['random']:
    print(f'\n=== {spec}', flush=True)
    if spec != 'random':
        sd = torch.load(spec, map_location='cpu')
        sd = sd.get('state_dict', sd)
        bad = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
        mx = max((v.abs().max().item() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point() and v.numel()), default=0)
        print(f'weights: {len(sd)} tensors, non-finite in {len(bad)} {bad[:3]}, max |w| {mx:.3g}', flush=True)
        del sd
    enc, _ = pp.load_mono_model(spec, dev, ckpts[0])
    enc.eval()
    for fp16 in (True, False):
        ref, query = embed(enc, fp16)
        h = torch.cat([ref, query])
        fin = torch.isfinite(h)
        hf = torch.where(fin, h, torch.zeros_like(h))
        line = (f'{"fp16" if fp16 else "fp32"}: shape {tuple(h.shape)}, non-finite {1 - fin.float().mean().item():.4f}, '
                f'max |h| {hf.abs().max().item():.3g}, mean per-dim std over clips {hf.std(0).mean().item():.3g}')
        try:
            m = probe._metrics('sc2', ref, query)
            line += f", sc2 knn {m['sc2/knn20_acc']*100:.1f} ncm {m['sc2/ncm_acc']*100:.1f} lin {m['sc2/linear_acc']*100:.1f}"
        except Exception as e:
            line += f', metrics failed: {e!r}'
        print(line, flush=True)
    del enc
    torch.cuda.empty_cache()
print('\nchance', 100 / s['n_class'])
