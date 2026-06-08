"""
Step 3: Training Loop for SSAMBA + SAC Loss Pretraining.

This script wires together:
  - The SSAMBA dataloader (log-Mel spectrograms)
  - Raw waveform loading for acoustic feature extraction
  - The SSAMBASACModel (encoder + projection head + SAC loss)
  - Total loss: L_total = L_recon + λ * L_SAC

Usage:
    python run_pretrain_sac.py --sac-loss \\
        --data-train /path/to/train.json \\
        --data-val /path/to/val.json \\
        ...

The --sac-loss flag activates the SAC loss branch.
Without it, the script falls back to standard SSAMBA pretraining.
"""

import argparse
import os
import sys
import time
import pickle
import json
import ast

import torch
import torch.nn as nn
import numpy as np
import torchaudio

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/storage/yotam/ssamba/Vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/mamba-1p1p1')

import dataloader as ssamba_dataloader
from sac.acoustic_features import extract_acoustic_features, calculate_acoustic_feature_stats
from sac.sac_model import SSAMBASACModel


# ============================================================
# Extended Dataloader: returns both fbank AND raw waveform
# ============================================================
class AudioDatasetWithWaveform(ssamba_dataloader.AudioDataset):
    """
    Extends SSAMBA's AudioDataset to also return the raw waveform
    for on-the-fly acoustic feature extraction.
    """

    def __init__(self, dataset_json_file, audio_conf, label_csv=None, target_sr=16000):
        super().__init__(dataset_json_file, audio_conf, label_csv)
        self.target_sr = target_sr
        # Compute target waveform length from target_length (number of mel frames)
        # Each mel frame corresponds to ~10ms (frame_shift=10 in kaldi fbank)
        target_length = self.audio_conf.get('target_length', 1024)
        self.target_waveform_samples = int(target_length * 0.01 * target_sr) + target_sr  # slight overallocation

    def __getitem__(self, index):
        """
        Returns:
            fbank: [T_frames, F_bins] log-Mel spectrogram
            waveform: [T_samples] raw mono waveform (mean-subtracted)
            label_indices: [num_classes] label vector
        """
        datum = self.data[index]

        # Load waveform
        try:
            waveform, sr = torchaudio.load(datum['wav'])
            waveform = waveform.mean(dim=0)  # mono: [T]
            waveform = waveform - waveform.mean()

            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            # Pad or trim to target length
            T = waveform.shape[0]
            target_T = self.target_waveform_samples
            if T < target_T:
                waveform = torch.nn.functional.pad(waveform, (0, target_T - T))
            elif T > target_T:
                waveform = waveform[:target_T]

        except Exception as e:
            print(f"Error loading waveform {datum['wav']}: {e}")
            waveform = torch.zeros(self.target_waveform_samples)

        # Get fbank and labels from parent class (handles mixup, augmentation, normalization)
        fbank, label_indices = super().__getitem__(index)

        return fbank, waveform, label_indices


# ============================================================
# Utilities
# ============================================================
class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================
# Training & Validation
# ============================================================
def train_sac(model, train_loader, test_loader, args, device):
    """
    Main training loop for SSAMBA + SAC pretraining.
    
    When --sac-loss is set:
        L_total = L_recon + λ * L_SAC
        
    Otherwise:
        Falls back to standard SSAMBA pretraining (L_recon only or joint MPC+MPG).
    """
    print(f"Starting SAC pretraining on {device}")
    print(f"  SAC enabled: {args.sac_loss}")
    print(f"  SAC lambda: {args.sac_lambda}")
    print(f"  SAC temperature: {args.sac_temperature}")
    print(f"  SAC sigma: {args.sac_sigma}")

    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    trainables = [p for p in model.parameters() if p.requires_grad]
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M')
    print(f'Trainable parameters: {sum(p.numel() for p in trainables) / 1e6:.3f}M')
    optimizer = torch.optim.AdamW(trainables, args.lr, weight_decay=5e-8, betas=(0.95, 0.999))

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True
    )

    # WandB
    if args.use_wandb:
        import wandb
        wandb.init(project='SSAMBA', config=vars(args), name=args.exp_name if hasattr(args, 'exp_name') else None)

    # Pre-compute acoustic feature statistics
    if args.sac_loss:
        print("Computing acoustic feature statistics...")
        feature_stats = calculate_acoustic_feature_stats(
            train_loader, sample_rate=args.sample_rate,
            num_batches=50, device=device, features_list=args.sac_features
        )
        if feature_stats:
            # Convert to tensors
            feature_stats = {
                'mean': feature_stats['mean'].to(device),
                'std': feature_stats['std'].to(device),
            }
            # Set on model
            model_core = model.module if hasattr(model, 'module') else model
            model_core.feature_stats = feature_stats
            print("Feature stats set on model.")
        else:
            print("Warning: Could not compute feature stats. Using per-batch normalization.")
            feature_stats = None
    else:
        feature_stats = None

    if args.sac_loss and args.diagnostic_steps > 0:
        from sac.sac_diagnostics import SACDebugger
        feature_names = []
        for f in args.sac_features.split(','):
            f = f.strip().lower()
            if f in ['f0', 'f0_mean']: feature_names.append('μ_F0')
            elif f == 'f0_var': feature_names.append('σ_F0')
            elif f == 'hnr': feature_names.append('HNR')
            elif f == 'centroid': feature_names.append('μ_Centroid')
            elif f in ['flux', 'flux_var']: feature_names.append('σ_Flux')
            elif f in ['zcr', 'zcr_mean']: feature_names.append('ZCR')
            elif f in ['zcr_var', 'rhythm']: feature_names.append('σ_ZCR')
            elif f == 'formants': feature_names.extend(['F1', 'F2', 'F3'])
            elif f == 'f1': feature_names.append('F1')
            elif f == 'f2': feature_names.append('F2')
            elif f == 'f3': feature_names.append('F3')
            elif f == 'mfcc': feature_names.extend([f'MFCC_{i+1}' for i in range(5)])
        debugger = SACDebugger(feature_names=feature_names)
    else:
        debugger = None

    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()
    progress = []

    loss_meter = AverageMeter()
    loss_recon_meter = AverageMeter()
    loss_sac_meter = AverageMeter()
    acc_meter = AverageMeter()

    epoch = 1
    while epoch < args.n_epochs + 1:
        model.train()
        begin_time = time.time()

        for i, batch_data in enumerate(train_loader):
            if args.sac_loss and len(batch_data) >= 3:
                fbank, waveform, _ = batch_data
                fbank = fbank.to(device, non_blocking=True)
                waveform = waveform.to(device, non_blocking=True)

                # Extract acoustic features on-the-fly
                with torch.no_grad():
                    acoustic_feats = extract_acoustic_features(
                        waveform, sample_rate=args.sample_rate,
                        feature_stats=feature_stats, features_list=args.sac_features
                    )  # [B, K]
            else:
                fbank, _ = batch_data[0].to(device, non_blocking=True), batch_data[-1]
                if isinstance(batch_data[0], torch.Tensor):
                    fbank = batch_data[0].to(device, non_blocking=True)
                acoustic_feats = None

            B = fbank.size(0)

            # Warm-up learning rate
            if global_step <= 1000 and global_step % 50 == 0:
                warm_lr = max(1e-7, (global_step / 1000) * args.lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr

            # Forward pass
            cluster = (args.num_mel_bins != args.fshape)

            if args.sac_loss and acoustic_feats is not None:
                do_diagnostics = args.diagnostic_steps > 0 and global_step > 0 and global_step % args.diagnostic_steps == 0
                
                # SAC-augmented forward
                result = model(
                    fbank, task='pretrain_joint',
                    mask_patch=args.mask_patch, cluster=cluster,
                    acoustic_features=acoustic_feats,
                    return_diagnostics=do_diagnostics
                )

                # Handle DataParallel output (list of dicts or single dict)
                if isinstance(result, dict):
                    loss = result['loss_total'].mean()
                    loss_recon = result['loss_recon'].mean()
                    loss_sac = result['loss_sac'].mean()
                    acc = result['acc'].mean() if isinstance(result['acc'], torch.Tensor) else result['acc']
                    diag_dict = result if do_diagnostics else None
                elif isinstance(result, list):
                    # DataParallel returns list of outputs from each GPU
                    loss = torch.stack([r['loss_total'] for r in result]).mean()
                    loss_recon = torch.stack([r['loss_recon'] for r in result]).mean()
                    loss_sac = torch.stack([r['loss_sac'] for r in result]).mean()
                    acc = torch.stack([r['acc'] for r in result]).mean()
                    diag_dict = result[0] if do_diagnostics else None
                else:
                    # Fallback: treat as tuple
                    loss = result[0].mean() if hasattr(result[0], 'mean') else result[0]
                    loss_recon = loss
                    loss_sac = torch.tensor(0.0)
                    acc = torch.tensor(0.0)
                    diag_dict = None
            else:
                do_diagnostics = False
                diag_dict = None
                # Standard SSAMBA pretraining
                if args.task == 'pretrain_joint':
                    acc, loss_mpc = model(fbank, 'pretrain_mpc', mask_patch=args.mask_patch, cluster=cluster)
                    loss_mpg = model(fbank, 'pretrain_mpg', mask_patch=args.mask_patch, cluster=cluster)
                    acc, loss_mpc = acc.mean(), loss_mpc.mean()
                    loss_mpg = loss_mpg.mean()
                    loss = loss_mpc + 10 * loss_mpg
                    loss_recon = loss_mpg
                    loss_sac = torch.tensor(0.0)
                elif args.task == 'pretrain_mpg':
                    loss = model(fbank, 'pretrain_mpg', mask_patch=args.mask_patch, cluster=cluster)
                    loss = loss.mean()
                    acc = loss
                    loss_recon = loss
                    loss_sac = torch.tensor(0.0)
                elif args.task == 'pretrain_mpc':
                    acc, loss = model(fbank, 'pretrain_mpc', mask_patch=args.mask_patch, cluster=cluster)
                    acc, loss = acc.mean(), loss.mean()
                    loss_recon = loss
                    loss_sac = torch.tensor(0.0)
                else:
                    raise ValueError(f"Unknown task: {args.task}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update meters
            loss_meter.update(loss.item(), B)
            loss_recon_meter.update(loss_recon.item() if isinstance(loss_recon, torch.Tensor) else loss_recon, B)
            loss_sac_meter.update(loss_sac.item() if isinstance(loss_sac, torch.Tensor) else loss_sac, B)
            if isinstance(acc, torch.Tensor):
                acc_meter.update(acc.item(), B)

            # Print progress
            if global_step % args.n_print_steps == 0 and global_step > 0:
                print(f'Epoch [{epoch}][{i}/{len(train_loader)}] '
                      f'Loss: {loss_meter.avg:.4f} '
                      f'Recon: {loss_recon_meter.avg:.4f} '
                      f'SAC: {loss_sac_meter.avg:.4f} '
                      f'Acc: {acc_meter.avg:.4f} '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

                if args.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss_total': loss_meter.avg,
                        'train/loss_recon': loss_recon_meter.avg,
                        'train/loss_sac': loss_sac_meter.avg,
                        'train/acc': acc_meter.avg,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'step': global_step,
                        'epoch': epoch,
                    })

            if do_diagnostics and diag_dict is not None and debugger is not None:
                # Generate diagnostics
                model.eval()
                z = diag_dict['z']
                z_norm = diag_dict['z_norm']
                sim = diag_dict['sim']
                w = diag_dict['w']
                
                chunk_size = z.shape[0]
                c = acoustic_feats[:chunk_size]
                
                fig_feat_hist, fig_feat_corr = debugger.plot_feature_distributions(c)
                fig_kernel = debugger.plot_kernel_and_similarity(sim, w)
                uniformity, alignment = debugger.compute_alignment_uniformity(z_norm, w)
                fig_manifold = debugger.plot_latent_manifold(z, c)
                
                print(f"  [Diagnostics] Uniformity: {uniformity:.4f}, Alignment_Top10: {alignment:.4f}")
                
                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "diagnostics/features_hist": wandb.Image(fig_feat_hist),
                        "diagnostics/features_corr": wandb.Image(fig_feat_corr),
                        "diagnostics/kernel_sim": wandb.Image(fig_kernel),
                        "diagnostics/manifold_proj": wandb.Image(fig_manifold),
                        "metrics/uniformity": uniformity,
                        "metrics/alignment_top10": alignment,
                        "step": global_step,
                    })
                    
                import matplotlib.pyplot as plt
                plt.close(fig_feat_hist)
                plt.close(fig_feat_corr)
                plt.close(fig_kernel)
                plt.close(fig_manifold)
                model.train()

            global_step += 1

            # Periodic evaluation & model saving
            if global_step % args.epoch_iter == 0:
                print(f'---- Step {global_step} evaluation ----')
                equ_epoch = int(global_step / args.epoch_iter) + 1
                val_loss = validate_sac(model, test_loader, args, device, feature_stats)

                print(f'  Train Loss: {loss_meter.avg:.4f}, Val Loss: {val_loss:.4f}')

                # Save model
                save_path = os.path.join(args.exp_dir, 'models', f'audio_model.{equ_epoch}.pth')
                torch.save(model.state_dict(), save_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.exp_dir, 'models', 'best_audio_model.pth')
                    torch.save(model.state_dict(), best_path)
                    print(f'  New best model saved (val_loss={val_loss:.4f})')

                scheduler.step(val_loss)

                # Save progress
                progress.append([epoch, global_step, equ_epoch, time.time() - start_time, val_loss])
                with open(os.path.join(args.exp_dir, 'progress.pkl'), 'wb') as f:
                    pickle.dump(progress, f)

                # Reset meters
                loss_meter.reset()
                loss_recon_meter.reset()
                loss_sac_meter.reset()
                acc_meter.reset()

                model.train()
                print('---- evaluation finished ----')

        epoch += 1


def validate_sac(model, val_loader, args, device, feature_stats=None):
    """Validation loop for SAC pretraining."""
    model.eval()
    losses = []

    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if args.sac_loss and len(batch_data) >= 3:
                fbank, waveform, _ = batch_data
                fbank = fbank.to(device)
                waveform = waveform.to(device)
                acoustic_feats = extract_acoustic_features(
                    waveform, sample_rate=args.sample_rate,
                    feature_stats=feature_stats, features_list=args.sac_features
                )
            else:
                fbank = batch_data[0].to(device)
                acoustic_feats = None

            cluster = (args.num_mel_bins != args.fshape)

            if args.sac_loss and acoustic_feats is not None:
                result = model(
                    fbank, task='pretrain_joint',
                    mask_patch=400, cluster=cluster,
                    acoustic_features=acoustic_feats
                )
                if isinstance(result, dict):
                    loss = result['loss_total'].mean()
                else:
                    loss = result[0].mean() if hasattr(result[0], 'mean') else result[0]
            else:
                if args.task == 'pretrain_joint':
                    _, loss_mpc = model(fbank, 'pretrain_mpc', mask_patch=400, cluster=cluster)
                    loss_mpg = model(fbank, 'pretrain_mpg', mask_patch=400, cluster=cluster)
                    loss = loss_mpc.mean() + 10 * loss_mpg.mean()
                elif args.task == 'pretrain_mpg':
                    loss = model(fbank, 'pretrain_mpg', mask_patch=400, cluster=cluster).mean()
                else:
                    _, loss = model(fbank, 'pretrain_mpc', mask_patch=400, cluster=cluster)
                    loss = loss.mean()

            losses.append(loss.item())

    return np.mean(losses)


# ============================================================
# Argument Parser
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(
        description='SSAMBA + SAC Loss Pretraining',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--data-train', type=str, required=True, help='Training data JSON')
    parser.add_argument('--data-val', type=str, required=True, help='Validation data JSON')
    parser.add_argument('--label-csv', type=str, default='', help='CSV with class labels')
    parser.add_argument('--dataset', type=str, default='librispeech', help='Dataset name')
    parser.add_argument('--dataset_mean', type=float, default=-4.2677393, help='Dataset spectrogram mean')
    parser.add_argument('--dataset_std', type=float, default=4.5689974, help='Dataset spectrogram std')
    parser.add_argument('--target_length', type=int, default=1024, help='Input length in frames')
    parser.add_argument('--num_mel_bins', type=int, default=128, help='Number of mel bins')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')

    # Training
    parser.add_argument('--exp-dir', type=str, default='./exp/sac_pretrain', help='Experiment directory')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for wandb')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_patience', type=int, default=2, help='LR scheduler patience')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='Batch size')
    parser.add_argument('-w', '--num-workers', default=16, type=int, help='Dataloader workers')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--n-print-steps', type=int, default=100, help='Print frequency')
    parser.add_argument('--epoch_iter', type=int, default=4000, help='Steps between evaluations')
    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging')

    # Task
    parser.add_argument('--task', type=str, default='pretrain_joint',
                        choices=['pretrain_mpc', 'pretrain_mpg', 'pretrain_joint'],
                        help='Pretraining task')
    parser.add_argument('--mask_patch', type=int, default=300, help='Number of patches to mask')

    # Patch / Model
    parser.add_argument('--fshape', type=int, default=16, help='Patch freq shape')
    parser.add_argument('--tshape', type=int, default=16, help='Patch time shape')
    parser.add_argument('--model_size', type=str, default='base', help='Model size')

    # Vision Mamba config
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=24)
    parser.add_argument('--rms_norm', type=str, choices=['true', 'false'], default='false')
    parser.add_argument('--residual_in_fp32', type=str, choices=['true', 'false'], default='false')
    parser.add_argument('--fused_add_norm', type=str, choices=['true', 'false'], default='false')
    parser.add_argument('--if_rope', type=str, choices=['true', 'false'], default='false')
    parser.add_argument('--if_rope_residual', type=str, choices=['true', 'false'], default='false')
    parser.add_argument('--bimamba_type', type=str, default='v2')
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--norm_epsilon', type=float, default=1e-5)
    parser.add_argument('--if_bidirectional', type=str, choices=['true', 'false'], default='true')
    parser.add_argument('--final_pool_type', type=str, default='none')
    parser.add_argument('--if_abs_pos_embed', type=str, choices=['true', 'false'], default='true')
    parser.add_argument('--if_bimamba', type=str, choices=['true', 'false'], default='false')
    parser.add_argument('--if_cls_token', type=str, choices=['true', 'false'], default='true')
    parser.add_argument('--if_devide_out', type=str, choices=['true', 'false'], default='true')
    parser.add_argument('--use_double_cls_token', type=str, choices=['true', 'false'], default='false')
    parser.add_argument('--use_middle_cls_token', type=str, choices=['true', 'false'], default='false')

    # ==== SAC Loss Arguments ====
    parser.add_argument('--diagnostic_steps', type=int, default=500,
                        help='Steps between generating SAC diagnostics (0 to disable)')
    parser.add_argument('--sac-loss', action='store_true',
                        help='Enable SAC (Soft Acoustic Contrastive) loss')
    parser.add_argument('--sac-lambda', type=float, default=1.0,
                        help='Weight λ for SAC loss: L_total = L_recon + λ*L_SAC')
    parser.add_argument('--sac-temperature', type=float, default=0.3,
                        help='Temperature τ for cosine similarity scaling in SAC loss')
    parser.add_argument('--sac-sigma', type=float, default=1.0,
                        help='Bandwidth σ for Gaussian kernel weights in SAC loss')
    parser.add_argument('--sac_features', type=str, default='f0_mean,hnr,centroid,flux,zcr_mean',
                        help='Comma separated list of acoustic features to extract (e.g. formants,mfcc,f0_var,rhythm)')
    parser.add_argument('--proj-dim', type=int, default=128,
                        help='Output dimension of the projection head g(·)')

    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = get_args()

    # Convert string bools
    for attr in ['rms_norm', 'residual_in_fp32', 'fused_add_norm', 'if_rope',
                 'if_rope_residual', 'if_bidirectional', 'if_abs_pos_embed',
                 'if_bimamba', 'if_cls_token', 'if_devide_out',
                 'use_double_cls_token', 'use_middle_cls_token']:
        setattr(args, attr, getattr(args, attr) == 'true')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create experiment directory
    os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)
    with open(os.path.join(args.exp_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Audio config
    audio_conf = {
        'num_mel_bins': args.num_mel_bins,
        'target_length': args.target_length,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'dataset': args.dataset,
        'mode': 'train',
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': False,
    }
    val_audio_conf = {**audio_conf, 'mode': 'evaluation'}

    # Dataloader
    if args.sac_loss:
        # Use extended dataloader that also returns raw waveform
        train_dataset = AudioDatasetWithWaveform(
            args.data_train, label_csv=args.label_csv,
            audio_conf=audio_conf, target_sr=args.sample_rate
        )
        val_dataset = AudioDatasetWithWaveform(
            args.data_val, label_csv=args.label_csv,
            audio_conf=val_audio_conf, target_sr=args.sample_rate
        )
    else:
        train_dataset = ssamba_dataloader.AudioDataset(
            args.data_train, label_csv=args.label_csv, audio_conf=audio_conf
        )
        val_dataset = ssamba_dataloader.AudioDataset(
            args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')

    # Vision Mamba config
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

    # Build model
    model = SSAMBASACModel(
        fshape=args.fshape, tshape=args.tshape,
        input_fdim=args.num_mel_bins, input_tdim=args.target_length,
        model_size=args.model_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        proj_dim=args.proj_dim,
        sac_temperature=args.sac_temperature,
        sac_sigma=args.sac_sigma,
        sac_lambda=args.sac_lambda,
        mask_patch=args.mask_patch,
        vision_mamba_config=vision_mamba_config,
    )

    print(f'\nModel built: SSAMBASACModel')
    print(f'  Encoder embed_dim: {model.encoder.original_embedding_dim}')
    print(f'  Projection head output: {args.proj_dim}')
    print(f'  SAC loss: λ={args.sac_lambda}, τ={args.sac_temperature}, σ={args.sac_sigma}')

    # Train
    train_sac(model, train_loader, val_loader, args, device)


if __name__ == '__main__':
    main()
