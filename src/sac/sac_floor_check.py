"""How far is the SAC loss of a finished mono pretraining from its floor?

The SAC loss per group is the cross-entropy H(P_i, Q_i) = H(P_i) + KL(P_i || Q_i), so H(P) is the lowest value any
encoder can reach, and a uniform Q gives ln(B-1). For each feature group this prints, on fixed training batches:
  H(P)   the floor,   L   the trained encoder's loss,   ln(B-1)   the uniform-guess loss,
  closed = (ln(B-1) - L) / (ln(B-1) - H(P)), the fraction of the reachable gap the encoder closed.
If H(P) is close to ln(B-1), the targets are nearly uniform and a larger SAC weight cannot help.

Also repeats the table for sharper kernels (--sigma_scales, multiplying the training sigma), with the same encoder.

Usage (GPU job, from src/sac):  python sac_floor_check.py --exp_dir <pretrain exp dir> [...] --n_batches 40
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pretrain_sac import AudioDatasetWithWaveform  # noqa: E402  (also sets the import paths)
from sac.acoustic_features import extract_acoustic_features, calculate_acoustic_feature_stats  # noqa: E402
from sac.sac_model import SSAMBASACModel, perplexity_targets  # noqa: E402

BOOL_ATTRS = ['rms_norm', 'residual_in_fp32', 'fused_add_norm', 'if_rope', 'if_rope_residual', 'if_bidirectional',
              'if_abs_pos_embed', 'if_bimamba', 'if_cls_token', 'if_devide_out', 'use_double_cls_token',
              'use_middle_cls_token', 'use_cross_attention', 'use_checkpointing']


def build_model(a):
    vm = dict(img_size=(a.num_mel_bins, a.target_length), patch_size=a.patch_size, stride=a.stride,
              embed_dim=a.embed_dim, depth=a.depth, channels=a.channels, num_classes=a.num_classes,
              drop_rate=a.drop_rate, drop_path_rate=a.drop_path_rate, norm_epsilon=a.norm_epsilon,
              rms_norm=a.rms_norm, residual_in_fp32=a.residual_in_fp32, fused_add_norm=a.fused_add_norm,
              if_rope=a.if_rope, if_rope_residual=a.if_rope_residual, bimamba_type=a.bimamba_type,
              if_bidirectional=a.if_bidirectional, final_pool_type=a.final_pool_type,
              if_abs_pos_embed=a.if_abs_pos_embed, if_bimamba=a.if_bimamba, if_cls_token=a.if_cls_token,
              if_devide_out=a.if_devide_out, use_double_cls_token=a.use_double_cls_token,
              use_middle_cls_token=a.use_middle_cls_token)
    return SSAMBASACModel(
        fshape=a.fshape, tshape=a.tshape, input_fdim=a.num_mel_bins, input_tdim=a.target_length,
        model_size=a.model_size, embed_dim=a.embed_dim, depth=a.depth, proj_dim=a.proj_dim,
        sac_temperature=a.sac_temperature, sac_sigma=a.sac_sigma, sac_lambda=a.sac_lambda,
        recon_lambda=a.recon_lambda, classif_lambda=a.classif_lambda, sac_features=a.sac_features,
        mask_patch=a.mask_patch, vision_mamba_config=vm, local_sigma_mode=a.local_sigma_mode,
        use_cross_attention=a.use_cross_attention, num_queries_per_group=a.num_queries_per_group,
        use_checkpointing=False)


def group_sigma(model, name, n_idx, scale=1.0):
    """Same sigma rule as SSAMBASACModel.sac_loss for the modes used in the mono runs."""
    fs = model.feature_stats
    if model.local_sigma_mode == 'offline_global_median' and fs and name in fs.get('group_medians', {}):
        return scale * model.sac_sigma * fs['group_medians'][name] / math.sqrt(math.log(2.0))
    if model.local_sigma_mode == 'offline_global_median':
        return scale * model.sac_sigma * math.sqrt(n_idx)
    raise ValueError(f'sigma mode {model.local_sigma_mode} not handled here')


@torch.no_grad()
def per_group(model, Z, c, scales=(1.0,), eps=1e-8):
    B = Z.shape[0]
    off = ~torch.eye(B, dtype=torch.bool, device=Z.device)
    out = {}
    for k, (name, idx) in enumerate(model.group_indices.items()):
        cg = c[:, idx]
        d = torch.cdist(cg, cg)
        z = F.normalize(Z[:, k, :], dim=-1)
        sim = z @ z.T / model.sac_temperature
        logq = sim - torch.log((torch.exp(sim) * off.float()).sum(1, keepdim=True) + eps)
        for s in scales:
            if isinstance(s, str):  # 'k<perplexity>': SNE calibration per anchor
                P = perplexity_targets(d, float(s[1:]))
                logP = torch.log(P.clamp_min(1e-30)).masked_fill(~off, 0.0)
                H = -(P * logP).sum(1)
                L = -(P * logq.masked_fill(~off, 0.0)).sum(1).mean().item()
                out[(name, s)] = dict(H=H.mean().item(), L=L, eff=H.exp().mean().item(),
                                      pmax=P.max(1).values.mean().item(), eff_sd=H.exp().std().item())
                continue
            # log-domain row softmax: stays exact when a small sigma underflows exp()
            logits = (-(d / group_sigma(model, name, len(idx), s)) ** 2).masked_fill(~off, float('-inf'))
            P = torch.softmax(logits, dim=1)
            logP = torch.log_softmax(logits, dim=1).masked_fill(~off, 0.0)
            H = -(P * logP).sum(1).mean().item()
            L = -(P * logq.masked_fill(~off, 0.0)).sum(1).mean().item()
            # effective number of neighbours exp(H) and the largest target weight, to read how peaked P is
            out[(name, s)] = dict(H=H, L=L, eff=math.exp(H), pmax=P.max(1).values.mean().item())
    return out


def run(exp_dir, n_batches, device, seed, scales):
    with open(os.path.join(exp_dir, 'args.pkl'), 'rb') as f:
        a = pickle.load(f)
    for attr in BOOL_ATTRS:
        v = getattr(a, attr)
        if isinstance(v, str):
            setattr(a, attr, v == 'true')
    from sac.acoustic_features import get_feature_groups
    feats = [f.strip() for f in a.sac_features.split(',')]
    grouped = {f for f in feats if get_feature_groups(f)[1] > 0}
    dropped = [f for f in feats if f not in grouped]
    if dropped:
        a.sac_features = ','.join(f for f in feats if f in grouped)
    print(f'\n=== {os.path.basename(exp_dir)}' + (f'\n  dropped (in no SAC group, never in the loss): {dropped}' if dropped else '') + f'\n  features: {a.sac_features}\n  lambda {a.sac_lambda}, '
          f'tau {a.sac_temperature}, sigma mode {a.local_sigma_mode}, batch {a.batch_size}', flush=True)

    conf = dict(num_mel_bins=a.num_mel_bins, target_length=a.target_length, freqm=0, timem=0, mixup=0,
                dataset=a.dataset, mode='evaluation', mean=a.dataset_mean, std=a.dataset_std, noise=False)
    ds = AudioDatasetWithWaveform(a.data_train, label_csv=a.label_csv, audio_conf=conf, target_sr=a.sample_rate)
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True, generator=g,
                                         num_workers=8, drop_last=True)

    model = build_model(a)
    ckpt_path = os.path.join(exp_dir, 'models', 'audio_model.final.pth')
    ck = torch.load(ckpt_path, map_location='cpu')
    state = ck.get('model', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
    if isinstance(ck, dict):
        print(f'  checkpoint keys (top level, first 8): {list(ck.keys())[:8]}')
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  load: {len(missing)} missing, {len(unexpected)} unexpected'
          + (f' (missing e.g. {missing[:3]})' if missing else ''))
    model = model.to(device).eval()

    fs = ck.get('feature_stats') if isinstance(ck, dict) else None
    if fs is None:
        # Not saved with the final weights: recompute exactly as training did (50 shuffled batches).
        print('  feature_stats not in checkpoint: recomputing (50 batches, as at training start)')
        sl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed + 1), num_workers=8)
        fs = calculate_acoustic_feature_stats(sl, sample_rate=a.sample_rate, num_batches=50, device=device,
                                              features_list=a.sac_features)
    else:
        print('  feature_stats from checkpoint')
    fs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in fs.items()}
    model.feature_stats = fs

    acc = {}
    max_diff = 0.0
    for i, (fbank, wav, _) in enumerate(loader):
        if i >= n_batches:
            break
        fbank, wav = fbank.to(device), wav.to(device)
        with torch.no_grad():
            c = extract_acoustic_features(wav, sample_rate=a.sample_rate, feature_stats=fs,
                                          features_list=a.sac_features)
            x = fbank.unsqueeze(1).transpose(2, 3)
            hs = model._encode_with_mamba(x)
            Q = model.group_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
            n = model.encoder.cls_token_num
            att, _ = model.cross_attention(query=Q, key=hs[:, n:, :], value=hs[:, n:, :])
            att = att.view(x.shape[0], model.num_active_groups, model.num_queries_per_group, -1).mean(2)
            Z = model.projection_head(att)
            ref = model.sac_loss(Z, c).item()
        r = per_group(model, Z, c, scales)
        max_diff = max(max_diff, abs(np.mean([v['L'] for (n_, s_), v in r.items() if s_ == 1.0]) - ref))
        for name, v in r.items():
            for key, val in v.items():
                acc.setdefault(name, {}).setdefault(key, []).append(val)

    B = a.batch_size
    chance = math.log(B - 1)
    print(f'  {i} batches of {B}; check vs model.sac_loss: max |diff| = {max_diff:.2e}')
    print(f'  ln(B-1) = {chance:.4f}')
    for s in scales:
        print(f'  -- perplexity {s[1:]} (per anchor)' if isinstance(s, str) else f'  -- sigma x {s} (1.0 = training rule)')
        print(f'  {"group":<14}{"dims":>5}{"H(P)":>9}{"loss L":>9}{"KL":>8}{"closed":>8}{"eff.nbrs":>10}{"max P":>8}')
        tot = {'H': [], 'L': []}
        for name in model.group_indices:
            d = acc[(name, s)]
            H, L = np.mean(d['H']), np.mean(d['L'])
            tot['H'].append(H)
            tot['L'].append(L)
            closed = (chance - L) / (chance - H) if chance - H > 1e-6 else float('nan')
            print(f'  {name:<14}{len(model.group_indices[name]):>5}{H:>9.4f}{L:>9.4f}{L - H:>8.4f}{closed:>8.1%}'
                  f'{np.mean(d["eff"]):>10.1f}{np.mean(d["pmax"]):>8.3f}'
                  + (f'   (sd of eff.nbrs across anchors {np.mean(d["eff_sd"]):.3f})' if 'eff_sd' in d else ''))
        H, L = np.mean(tot['H']), np.mean(tot['L'])
        print(f'  {"MEAN":<14}{"":>5}{H:>9.4f}{L:>9.4f}{L - H:>8.4f}{(chance - L) / (chance - H):>8.1%}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', nargs='+', required=True)
    p.add_argument('--n_batches', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sigma_scales', type=float, nargs='+', default=[1.0, 0.5, 0.3, 0.2, 0.1])
    p.add_argument('--perplexities', type=float, nargs='*', default=[],
                   help='also score SNE per-anchor targets at these perplexities (training scale 1.0 is always kept)')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for d in args.exp_dir:
        scales = list(args.sigma_scales) + [f'k{k:g}' for k in args.perplexities]
        if 1.0 not in scales:
            scales = [1.0] + scales
        run(d, args.n_batches, device, args.seed, scales)


if __name__ == '__main__':
    main()
