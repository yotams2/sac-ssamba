"""
Optuna Hyperparameter Search for SSAMBA Factorized SAC Loss (Bandwidth \u03c3 and Temperature \u03c4).

This script performs self-supervised optimization of SAC loss hyperparameters without
relying on downstream task labels, following representation geometry metrics:
  - Alignment: Cosine similarity of top 10% highest target-similarity sample pairs.
  - Uniformity: Logarithmic pairwise distance exponent (Wang & Isola, 2020).
  - Effective Rank: Entropy of singular values of normalized embeddings (Roy & Vetterli, 2007).
  - Weight Entropy Ratio: H(w_norm) / ln(B - 1).

Key Features:
  - Zero-shot / offline baseline calibration of per-group feature bandwidths (\u03c3_k).
  - Persistent SQLite storage for seamless resumption if interrupted.
  - WANDB disabled to avoid database overloading.
  - Generates comprehensive visualizations, CSV tables, JSON results, and markdown summaries.

Output Directory:
  /storage/yotam/ssamba/src/metrics/optuna_hyperparameters_search/
"""

import os
import sys
import gc
import json
import time
import math
import argparse
import datetime
import builtins
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

# Add SSAMBA source paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/storage/yotam/ssamba/src')
sys.path.insert(0, '/storage/yotam/ssamba/Vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/mamba-1p1p1')

from sac.sac_model import SSAMBASACModel
from sac.acoustic_features import (
    extract_acoustic_features,
    calculate_acoustic_feature_stats,
    get_feature_groups
)
from sac.run_pretrain_sac import AudioDatasetWithWaveform


def compute_sac_geometry_metrics(z_norm: torch.Tensor, w: torch.Tensor, group_names: list, eps: float = 1e-8):
    """
    Computes per-group and overall representation geometry metrics:
      - Alignment: Top 10% target similarity cosine similarity
      - Uniformity: Wang & Isola (2020) logarithmic pairwise repulsion
      - Effective Rank: Roy & Vetterli (2007) singular value entropy
      - Weight Entropy Ratio: H(w_norm) / ln(B - 1)

    Args:
        z_norm: [B, K, proj_dim] normalized latent embeddings
        w: [K, B, B] Gaussian target similarity weights
        group_names: list of active group names
    """
    B = z_norm.shape[0]
    K = z_norm.shape[1]
    device = z_norm.device
    mask = ~torch.eye(B, dtype=torch.bool, device=device)

    alignments = []
    uniformities = []
    ranks = []
    entropies = []
    group_breakdown = {}

    for k in range(K):
        g_name = group_names[k] if k < len(group_names) else f"Group_{k}"
        z_k = z_norm[:, k, :] # [B, proj_dim]
        w_k = w[k]            # [B, B]

        # 1. Normalized Weight Entropy Ratio
        w_masked = w_k * mask.float()
        w_sum = w_masked.sum(dim=1, keepdim=True) + eps
        w_norm = w_masked / w_sum
        H_i = -(w_norm * torch.log(w_norm + eps)).sum(dim=1)
        H_mean = H_i.mean().item()
        H_max = np.log(max(B - 1, 1))
        H_ratio = H_mean / H_max if H_max > 0 else 0.0

        # 2. Latent Cosine Similarity & Distance
        cos_sim = torch.matmul(z_k, z_k.T)
        sq_dist = (2.0 - 2.0 * cos_sim).clamp(min=0.0)
        sq_dist_flat = sq_dist[mask]

        # 3. Uniformity (Wang & Isola, 2020)
        uni = torch.log(torch.exp(-2.0 * sq_dist_flat).mean() + eps).item()

        # 4. Alignment (Top 10% highest target similarity pairs)
        w_flat = w_k[mask]
        topk = max(1, int(0.10 * w_flat.numel()))
        _, topk_idx = torch.topk(w_flat, topk)
        cos_flat = cos_sim[mask]
        ali = cos_flat[topk_idx].mean().item()

        # 5. Effective Rank (Roy & Vetterli, 2007)
        try:
            _, S, _ = torch.linalg.svd(z_k)
            p = S / S.sum().clamp(min=eps)
            p = p[p > 1e-12]
            er = torch.exp(-(p * torch.log(p)).sum()).item()
        except Exception:
            er = 1.0

        group_breakdown[g_name] = {
            'alignment': float(ali),
            'uniformity': float(uni),
            'effective_rank': float(er),
            'weight_entropy_ratio': float(H_ratio)
        }

        alignments.append(ali)
        uniformities.append(uni)
        ranks.append(er)
        entropies.append(H_ratio)

    return {
        'alignment_avg': float(np.mean(alignments)),
        'uniformity_avg': float(np.mean(uniformities)),
        'effective_rank_avg': float(np.mean(ranks)),
        'weight_entropy_ratio_avg': float(np.mean(entropies)),
        'group_breakdown': group_breakdown
    }


def run_probe_trial(trial, args, feature_stats, train_loader, device):
    """
    Executes a single short-probe pretraining trial for Optuna optimization.
    """
    # 1. Suggest Hyperparameters
    tau = trial.suggest_float('tau', args.tau_min, args.tau_max)

    active_groups, num_active_groups = get_feature_groups(args.sac_features)
    group_names = list(active_groups.keys())

    if args.mode == 'shared_sigma':
        sigma_scale = trial.suggest_float('sigma_scale', args.sigma_scale_min, args.sigma_scale_max, log=True)
        sigmas_dict = {g: sigma_scale for g in group_names}
    else: # per_group_sigma
        sigmas_dict = {}
        for g in group_names:
            s_val = trial.suggest_float(f'sigma_scale_{g}', args.sigma_scale_min, args.sigma_scale_max, log=True)
            sigmas_dict[g] = s_val
        sigma_scale = np.mean(list(sigmas_dict.values()))

    print(f"\n--- Optuna Trial {trial.number}: Tau = {tau:.4f}, Sigma Scale(s) = {sigmas_dict} ---")

    # 2. Vision Mamba Config matching run_sac.sh
    vision_mamba_config = {
        'img_size': (args.num_mel_bins, args.target_length),
        'patch_size': args.patch_size,
        'stride': args.stride,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'channels': args.channels,
        'num_classes': args.num_classes,
        'drop_rate': args.drop_rate,
        'drop_path_rate': args.drop_path_rate,
        'norm_epsilon': args.norm_epsilon,
        'rms_norm': args.rms_norm,
        'residual_in_fp32': args.residual_in_fp32,
        'fused_add_norm': args.fused_add_norm,
        'if_rope': args.if_rope,
        'if_rope_residual': args.if_rope_residual,
        'bimamba_type': args.bimamba_type,
        'if_bidirectional': args.if_bidirectional,
        'final_pool_type': args.final_pool_type,
        'if_abs_pos_embed': args.if_abs_pos_embed,
        'if_bimamba': args.if_bimamba,
        'if_cls_token': args.if_cls_token,
        'if_devide_out': args.if_devide_out,
        'use_double_cls_token': args.use_double_cls_token,
        'use_middle_cls_token': args.use_middle_cls_token,
    }

    # 3. Instantiate Model
    model = SSAMBASACModel(
        fshape=args.fshape, tshape=args.tshape,
        input_fdim=args.num_mel_bins, input_tdim=args.target_length,
        model_size=args.model_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        proj_dim=args.proj_dim,
        sac_temperature=tau,
        sac_sigma=sigma_scale,
        sac_lambda=args.sac_lambda,
        sac_features=args.sac_features,
        mask_patch=args.mask_patch,
        local_sigma_mode='offline_global_median',
        use_cross_attention=True,
        num_queries_per_group=args.num_queries_per_group,
        use_checkpointing=args.use_checkpointing,
        vision_mamba_config=vision_mamba_config,
    )
    model.feature_stats = feature_stats

    # If per_group_sigma, adjust offline medians by group scale
    if args.mode == 'per_group_sigma' and feature_stats is not None and 'group_medians' in feature_stats:
        adjusted_stats = dict(feature_stats)
        adjusted_medians = {}
        for g, base_m in feature_stats['group_medians'].items():
            scale_g = sigmas_dict.get(g, 1.0)
            adjusted_medians[g] = base_m * scale_g
        adjusted_stats['group_medians'] = adjusted_medians
        model.feature_stats = adjusted_stats

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-8, betas=(0.95, 0.999))

    # 3. Probe Training Loop
    probe_metrics_history = []
    loader_iter = iter(train_loader)

    try:
        for step in range(args.probe_steps):
            try:
                batch_data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch_data = next(loader_iter)

            fbank, waveform, _ = batch_data
            fbank = fbank.to(device, non_blocking=True)
            waveform = waveform.to(device, non_blocking=True)

            with torch.no_grad():
                acoustic_feats = extract_acoustic_features(
                    waveform, sample_rate=args.sample_rate,
                    feature_stats=feature_stats, features_list=args.sac_features
                )

            cluster = (args.num_mel_bins != args.fshape)
            result = model(
                fbank, task='pretrain_joint',
                mask_patch=args.mask_patch, cluster=cluster,
                acoustic_features=acoustic_feats,
                return_diagnostics=True
            )

            loss = result['loss_total'].mean() if isinstance(result['loss_total'], torch.Tensor) else result['loss_total']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record metrics for the last N probe evaluation steps
            if step >= (args.probe_steps - args.eval_last_steps):
                z_norm = result['z_norm'] # [B, K, proj_dim]
                w = result['w']           # [K, B, B]
                metrics = compute_sac_geometry_metrics(z_norm, w, group_names)
                probe_metrics_history.append(metrics)

    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(probe_metrics_history) == 0:
        raise optuna.exceptions.TrialPruned()

    # 4. Average Metrics Over Evaluation Window
    avg_alignment = float(np.mean([m['alignment_avg'] for m in probe_metrics_history]))
    avg_uniformity = float(np.mean([m['uniformity_avg'] for m in probe_metrics_history]))
    avg_effective_rank = float(np.mean([m['effective_rank_avg'] for m in probe_metrics_history]))
    avg_entropy_ratio = float(np.mean([m['weight_entropy_ratio_avg'] for m in probe_metrics_history]))

    # Average group breakdowns
    group_avg_breakdown = {}
    for g in group_names:
        g_ali = np.mean([m['group_breakdown'][g]['alignment'] for m in probe_metrics_history])
        g_uni = np.mean([m['group_breakdown'][g]['uniformity'] for m in probe_metrics_history])
        g_rank = np.mean([m['group_breakdown'][g]['effective_rank'] for m in probe_metrics_history])
        g_ent = np.mean([m['group_breakdown'][g]['weight_entropy_ratio'] for m in probe_metrics_history])
        group_avg_breakdown[g] = {
            'alignment': float(g_ali),
            'uniformity': float(g_uni),
            'effective_rank': float(g_rank),
            'weight_entropy_ratio': float(g_ent)
        }

    # 5. Store User Attributes for Constraints & Analysis
    trial.set_user_attr('uniformity', avg_uniformity)
    trial.set_user_attr('effective_rank', avg_effective_rank)
    trial.set_user_attr('weight_entropy_ratio', avg_entropy_ratio)
    trial.set_user_attr('group_breakdown', json.dumps(group_avg_breakdown))

    print(f"Trial {trial.number} Result: Alignment={avg_alignment:.4f}, Uniformity={avg_uniformity:.4f}, Rank={avg_effective_rank:.4f}, Entropy Ratio={avg_entropy_ratio:.4f}")

    return avg_alignment


def generate_visualizations_and_reports(study, out_dir, args):
    """
    Generates insightful charts, tables, and reports for all completed trials.
    """
    trials = study.trials
    if len(trials) == 0:
        return

    data = []
    for t in trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        params = t.params
        attrs = t.user_attrs
        uni = attrs.get('uniformity', np.nan)
        rank = attrs.get('effective_rank', np.nan)
        ent = attrs.get('weight_entropy_ratio', np.nan)
        feasible = (uni <= args.max_uniformity) and (rank >= args.min_rank)

        row = {
            'trial_number': t.number,
            'alignment': t.value,
            'uniformity': uni,
            'effective_rank': rank,
            'weight_entropy_ratio': ent,
            'tau': params.get('tau', np.nan),
            'sigma_scale': params.get('sigma_scale', np.nan),
            'feasible': feasible
        }
        for k, v in params.items():
            if k not in row:
                row[k] = v
        data.append(row)

    if len(data) == 0:
        return

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir, 'optuna_trials_summary.csv'), index=False)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Pareto Frontier: Alignment vs Uniformity
    fig, ax = plt.subplots(figsize=(9, 6))
    feasible_df = df[df['feasible']]
    infeasible_df = df[~df['feasible']]

    if len(infeasible_df) > 0:
        ax.scatter(infeasible_df['uniformity'], infeasible_df['alignment'], color='red', alpha=0.5, label='Infeasible (Violates Constraints)', s=60)
    if len(feasible_df) > 0:
        ax.scatter(feasible_df['uniformity'], feasible_df['alignment'], color='green', alpha=0.8, label='Feasible Candidate', s=90)
        # Highlight Pareto Frontier
        sorted_feasible = feasible_df.sort_values(by='uniformity')
        pareto_x, pareto_y = [], []
        max_ali = -float('inf')
        for _, r in sorted_feasible.iterrows():
            if r['alignment'] > max_ali:
                max_ali = r['alignment']
                pareto_x.append(r['uniformity'])
                pareto_y.append(r['alignment'])
        ax.plot(pareto_x, pareto_y, color='darkgreen', linestyle='--', linewidth=2, label='Pareto Boundary')

    ax.axvline(x=args.max_uniformity, color='black', linestyle=':', label=f'Max Uniformity Limit ({args.max_uniformity})')
    ax.set_xlabel('Uniformity (Lower / More Negative is Better)', fontsize=12)
    ax.set_ylabel('Alignment (Higher is Better)', fontsize=12)
    ax.set_title('Factorized SAC Loss Hyperparameter Search: Alignment vs Uniformity', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pareto_alignment_vs_uniformity.png'), dpi=300)
    plt.close()

    # 2. Contour/Scatter: Tau vs Sigma_Scale colored by Alignment
    if 'sigma_scale' in df.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(
            df['tau'], df['sigma_scale'], c=df['alignment'],
            cmap='viridis', s=100, edgecolors='k', alpha=0.9
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Alignment', fontsize=12)
        ax.set_yscale('log')
        ax.set_xlabel('Softmax Temperature (\u03c4)', fontsize=12)
        ax.set_ylabel('Gaussian Bandwidth Multiplier (\u03c3 scale)', fontsize=12)
        ax.set_title('Hyperparameter Space: \u03c4 vs \u03c3 Multiplier', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'contour_tau_sigma.png'), dpi=300)
        plt.close()

    # 3. Effective Rank Analysis
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df['trial_number'], df['effective_rank'], c=df['feasible'].map({True: 'green', False: 'red'}), s=70)
    ax.axhline(y=args.min_rank, color='blue', linestyle='--', label=f'Min Rank Floor ({args.min_rank})')
    ax.set_xlabel('Optuna Trial Number', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Representation Dimensionality (Effective Rank) Across Trials', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'effective_rank_analysis.png'), dpi=300)
    plt.close()

    # 4. Save Top Feasible Summary Table in Markdown & JSON
    if len(feasible_df) > 0:
        top_feasible = feasible_df.sort_values(by='alignment', ascending=False).head(10)
        top_feasible.to_csv(os.path.join(out_dir, 'top_feasible_trials.csv'), index=False)

        best_row = top_feasible.iloc[0]
        best_trial_data = {
            'trial_number': int(best_row['trial_number']),
            'alignment': float(best_row['alignment']),
            'uniformity': float(best_row['uniformity']),
            'effective_rank': float(best_row['effective_rank']),
            'weight_entropy_ratio': float(best_row['weight_entropy_ratio']),
            'tau': float(best_row['tau']),
            'sigma_scale': float(best_row['sigma_scale']) if 'sigma_scale' in best_row else None
        }
        with open(os.path.join(out_dir, 'best_hyperparameters.json'), 'w') as f:
            json.dump(best_trial_data, f, indent=4)

        # Markdown Report
        md_content = f"""# Optuna Hyperparameter Optimization Report: Factorized SAC Loss

## Executive Summary
- **Total Completed Trials:** {len(df)}
- **Feasible Candidates (Satisfied Rank & Uniformity):** {len(feasible_df)}
- **Best Feasible Trial:** Trial #{int(best_row['trial_number'])}

### Optimal Hyperparameters Found
- **Softmax Temperature (\\tau):** `{best_row['tau']:.4f}`
- **Gaussian Bandwidth Scale (\\sigma_scale):** `{best_row['sigma_scale']:.4f}`
- **Alignment (Objective):** `{best_row['alignment']:.4f}`
- **Uniformity:** `{best_row['uniformity']:.4f}` (Constraint: <= {args.max_uniformity})
- **Effective Rank:** `{best_row['effective_rank']:.4f}` (Constraint: >= {args.min_rank})

---

## Top Feasible Trials
| Trial # | Tau (\\tau) | Sigma Scale (\\sigma) | Alignment (\\uparrow) | Uniformity (\\downarrow) | Effective Rank (\\uparrow) | Entropy Ratio |
|---|---|---|---|---|---|---|
"""
        for _, r in top_feasible.iterrows():
            md_content += f"| {int(r['trial_number'])} | {r['tau']:.4f} | {r['sigma_scale']:.4f} | {r['alignment']:.4f} | {r['uniformity']:.4f} | {r['effective_rank']:.4f} | {r['weight_entropy_ratio']:.4f} |\n"

        with open(os.path.join(out_dir, 'optuna_summary_report.md'), 'w') as f:
            f.write(md_content)

    print(f"\n[Visualizations & Reports Generated] Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search for SSAMBA Factorized SAC Loss")
    
    # Dataset & Paths
    parser.add_argument('--data-train', type=str, default='/storage/yotam/ssamba/librispeech_train.json')
    parser.add_argument('--data-val', type=str, default='/storage/yotam/ssamba/librispeech_eval.json')
    parser.add_argument('--label-csv', type=str, default='/storage/yotam/ssamba/src/finetune/audioset/data/class_labels_indices.csv')
    parser.add_argument('--dataset', type=str, default='librispeech')
    parser.add_argument('--dataset-mean', type=float, default=-4.2677393)
    parser.add_argument('--dataset-std', type=float, default=4.5689974)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--target-length', type=int, default=1024)
    parser.add_argument('--num-mel-bins', type=int, default=128)

    # Model & Loss Config
    parser.add_argument('--model-size', type=str, default='base')
    parser.add_argument('--fshape', type=int, default=16)
    parser.add_argument('--tshape', type=int, default=16)
    parser.add_argument('--embed-dim', type=int, default=768)
    parser.add_argument('--proj-dim', type=int, default=128)
    parser.add_argument('--mask-patch', type=int, default=300)
    parser.add_argument('--sac-lambda', type=float, default=0.02)
    parser.add_argument('--sac-features', type=str, default='f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm')
    parser.add_argument('--num-queries-per-group', type=int, default=1)
    parser.add_argument('--use-checkpointing', type=bool, default=True)

    # Vision Mamba Config (matching run_sac.sh)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--depth', type=int, default=24)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--drop-rate', type=float, default=0.0)
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    parser.add_argument('--norm-epsilon', type=float, default=1e-5)
    parser.add_argument('--rms-norm', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--residual-in-fp32', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--fused-add-norm', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--if-rope', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--if-rope-residual', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--bimamba-type', type=str, default='v2')
    parser.add_argument('--if-bidirectional', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--final-pool-type', type=str, default='none')
    parser.add_argument('--if-abs-pos-embed', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--if-bimamba', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--if-cls-token', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--if-devide-out', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use-double-cls-token', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use-middle-cls-token', type=lambda x: (str(x).lower() == 'true'), default=False)

    # Search Space & Mode
    parser.add_argument('--mode', type=str, choices=['shared_sigma', 'per_group_sigma'], default='shared_sigma')
    parser.add_argument('--tau-min', type=float, default=0.05)
    parser.add_argument('--tau-max', type=float, default=1.00)
    parser.add_argument('--sigma-scale-min', type=float, default=0.2)
    parser.add_argument('--sigma-scale-max', type=float, default=5.0)

    # Optuna Probe Protocol
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--probe-steps', type=int, default=25)
    parser.add_argument('--eval-last-steps', type=int, default=5)
    parser.add_argument('--n-trials', type=int, default=30)
    parser.add_argument('--min-rank', type=float, default=3.2)
    parser.add_argument('--max-uniformity', type=float, default=-1.5)

    # Output & Persistence
    parser.add_argument('--out-dir', type=str, default='/storage/yotam/ssamba/src/metrics/optuna_hyperparameters_search')
    parser.add_argument('--study-name', type=str, default='ssamba_sac_hyperparameter_tuning')
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 65)
    print("SSAMBA Factorized SAC Loss Hyperparameter Optimization")
    print(f"Device: {device} | Batch Size: {args.batch_size} | Mode: {args.mode}")
    print(f"Features: {args.sac_features}")
    print(f"Output Directory: {args.out_dir}")
    print("=" * 65)

    # 1. Dataset & Dataloader
    audio_conf = {
        'num_mel_bins': args.num_mel_bins,
        'target_length': args.target_length,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'dataset': args.dataset, 'mode': 'train',
        'mean': args.dataset_mean, 'std': args.dataset_std,
        'noise': False
    }

    print("Loading training dataset...")
    train_dataset = AudioDatasetWithWaveform(
        args.data_train, audio_conf=audio_conf,
        label_csv=args.label_csv, target_sr=args.sample_rate
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    # 2. Compute Offline Feature Statistics & Zero-Shot Medians
    print("\nComputing offline feature statistics and zero-shot median distances...")
    feature_stats = calculate_acoustic_feature_stats(
        train_loader, sample_rate=args.sample_rate,
        num_batches=40, device=device, features_list=args.sac_features
    )
    if feature_stats:
        feature_stats['mean'] = feature_stats['mean'].to(device)
        feature_stats['std'] = feature_stats['std'].to(device)

    # 3. Setup Persistent Optuna Study
    db_path = os.path.join(args.out_dir, "optuna_study.db")
    storage_url = f"sqlite:///{db_path}"

    def constraints(trial):
        uni = trial.user_attrs.get('uniformity', float('inf'))
        rank = trial.user_attrs.get('effective_rank', 0.0)
        return (uni - args.max_uniformity, args.min_rank - rank)

    sampler = optuna.samplers.TPESampler(constraints_func=constraints, seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True
    )

    print(f"\nPersistent Optuna Study initialized.")
    print(f"  Storage: {db_path}")
    print(f"  Existing Trials: {len(study.trials)}")
    print(f"  Target Trials: {args.n_trials}")

    # Callback to save CSV and visualizations after every trial
    def per_trial_callback(study, trial):
        df = study.trials_dataframe()
        df.to_csv(os.path.join(args.out_dir, 'optuna_trials_raw.csv'), index=False)
        try:
            generate_visualizations_and_reports(study, args.out_dir, args)
        except Exception as e:
            print(f"Warning: Could not update visualizations: {e}")

    # 4. Optimize
    study.optimize(
        lambda t: run_probe_trial(t, args, feature_stats, train_loader, device),
        n_trials=args.n_trials,
        callbacks=[per_trial_callback]
    )

    # 5. Final Report Generation
    generate_visualizations_and_reports(study, args.out_dir, args)

    print("\n" + "=" * 65)
    print("OPTUNA SEARCH COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == '__main__':
    main()
