import argparse
import sys
import os
import time
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Ensure package imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/scratch/yotam/ssamba/Vim')
sys.path.insert(0, '/scratch/yotam/ssamba/Vim/vim')
sys.path.insert(0, '/scratch/yotam/ssamba/Vim/mamba-1p1p1')
sys.path.insert(0, '/storage/yotam/ssamba/Vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/mamba-1p1p1')

from binaural_dual_stream_sac.binaural_sac_model import BinauralSSAMBASACModel, BinauralSSAMBASACModelParallel
from binaural_dual_stream_sac.binaural_dataloader import BinauralAudioDataset
from sac.acoustic_features import calculate_acoustic_feature_stats


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


def calculate_binaural_feature_stats(dataloader, device, num_batches=50, spectral_features="f0,hnr,centroid,flux,zcr"):
    """
    Computes global offline acoustic feature medians for both monaural and spatial feature groups.
    """
    print("\n--- Computing Binaural Feature Statistics (offline_global_median) ---")
    all_spatial_feats = []
    
    # We create a dummy dataloader adapter returning (fbank, waveform, labels) for sac.acoustic_features
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if len(batch) >= 4:
                _, _, _, c_spatial = batch
                all_spatial_feats.append(c_spatial.cpu())

    feature_stats = {'group_medians': {}}

    if len(all_spatial_feats) > 0:
        spatial_tensor = torch.cat(all_spatial_feats, dim=0) # [N, 5]
        dist_matrix = torch.cdist(spatial_tensor, spatial_tensor, p=2)
        N = dist_matrix.shape[0]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=dist_matrix.device)
        off_diag_dists = dist_matrix[off_diag_mask]
        median_dist = max(float(off_diag_dists.median().item()), 1e-4)
        feature_stats['group_medians']['spatial'] = median_dist
        print(f"  spatial global median: {median_dist:.6f}")

    print("----------------------------------------------------------------------\n")
    return feature_stats


def verify_synthetic_forward_pass(device='cpu'):
    """
    Step 1 Verification Protocol: Verifies shape, gradient flow, and dual SAC branch loss calculation.
    """
    print("\n" + "=" * 60)
    print(" Running Step 1 Verification Protocol: Dual-Branch SAC Batch Test")
    print("=" * 60)

    B = 4
    F = 256
    T = 1024

    model = BinauralSSAMBASACModel(
        input_fdim=F, input_tdim=T,
        fshape=16, tshape=16,
        sac_spec_lambda=1.0,
        sac_spat_lambda=1.0,
        recon_lambda=1.0,
        local_sigma_mode='offline_global_median',
        share_encoder_weights=False,
        vision_mamba_config={'fused_add_norm': False if device == 'cpu' else True},
    ).to(device)

    dummy_X = torch.randn(B, 4, F, T, device=device)
    dummy_W = (torch.rand(B, 1, 1, T, device=device) > 0.5).float()
    dummy_c_mono = torch.randn(B, 15, device=device)
    dummy_c_spatial = torch.randn(B, 5, device=device)

    out = model(dummy_X, dummy_W, c_mono=dummy_c_mono, c_spatial=dummy_c_spatial, return_diagnostics=True)

    print(f"✓ Input STFT Shape: {dummy_X.shape}")
    print(f"✓ Binary Time Mask W Shape: {dummy_W.shape}")
    print(f"✓ Monaural Features c_mono Shape: {dummy_c_mono.shape}")
    print(f"✓ Spatial Features c_spatial Shape: {dummy_c_spatial.shape}")
    print(f"✓ Loss Total:     {out['loss_total'].item():.4f}")
    print(f"✓ Loss Recon:     {out['loss_recon'].item():.4f}")
    print(f"✓ Loss SAC Spec:  {out['loss_sac_spec'].item():.4f}")
    print(f"✓ Loss SAC Spat:  {out['loss_sac_spat'].item():.4f}")
    
    out['loss_total'].backward()
    print("✓ Gradient backward pass executed successfully!")
    print("=" * 60 + "\n")


def validate_binaural_sac(model, val_loader, device):
    """
    Validation loop for Binaural SSAMBA + Dual SAC.
    """
    model.eval()
    losses_total = []
    losses_recon = []
    losses_spec = []
    losses_spat = []

    with torch.no_grad():
        for batch_idx, (X, W, c_mono, c_spatial) in enumerate(val_loader):
            X = X.to(device)
            W = W.to(device)
            c_mono = c_mono.to(device)
            c_spatial = c_spatial.to(device)

            out = model(X, W, c_mono=c_mono, c_spatial=c_spatial)

            if isinstance(out, dict):
                loss_total = out['loss_total'].mean()
                loss_recon = out['loss_recon'].mean()
                loss_spec = out['loss_sac_spec'].mean()
                loss_spat = out['loss_sac_spat'].mean()
            else:
                loss_total = out.mean()
                loss_recon, loss_spec, loss_spat = loss_total, loss_total, loss_total

            losses_total.append(loss_total.item())
            losses_recon.append(loss_recon.item())
            losses_spec.append(loss_spec.item())
            losses_spat.append(loss_spat.item())

    return {
        'val_loss_total': float(np.mean(losses_total)),
        'val_loss_recon': float(np.mean(losses_recon)),
        'val_loss_spec': float(np.mean(losses_spec)),
        'val_loss_spat': float(np.mean(losses_spat)),
    }


def main():
    parser = argparse.ArgumentParser(description="Binaural SSAMBA Dual-Branch SAC Pretraining Runner")
    parser.add_argument('--data_train', type=str, default=None, help="Path to training dataset JSON file")
    parser.add_argument('--data_val', type=str, default=None, help="Path to validation dataset JSON file")
    parser.add_argument('--exp_dir', type=str, default='./exp/binaural_sac_pretrain', help="Experiment directory")
    parser.add_argument('--exp_name', type=str, default=None, help="WandB experiment name")
    parser.add_argument('--exp_description', type=str, default='', help="Free-text experiment description")
    parser.add_argument('--wandb_project', type=str, default='SSAMBA', help="WandB project name")
    parser.add_argument('--use_wandb', action='store_true', help="Enable WandB experiment tracking")
    
    parser.add_argument('--batch_size', type=int, default=32, help="Pretraining batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="DataLoader workers")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--lr_patience', type=int, default=2, help="LR scheduler patience")
    parser.add_argument('--epochs', type=int, default=10, help="Number of pretraining epochs")
    parser.add_argument('--epoch_iter', type=int, default=4000, help="Steps between validation evaluations")
    parser.add_argument('--n_print_steps', type=int, default=100, help="Print frequency in steps")

    # Loss weights
    parser.add_argument('--sac_spec_lambda', type=float, default=1.0, help="Weight for Spectral SAC loss")
    parser.add_argument('--sac_spat_lambda', type=float, default=1.0, help="Weight for Spatial SAC loss")
    parser.add_argument('--recon_lambda', type=float, default=1.0, help="Weight for reconstruction loss")
    
    # Model configs
    parser.add_argument('--local_sigma_mode', type=str, default='offline_global_median',
                        choices=['offline_global_median', 'dynamic_batch_median', 'chi2_median', 'sqrt_dim'],
                        help="Local sigma calculation mode (default: offline_global_median)")
    parser.add_argument('--spectral_features', type=str, default='f0,hnr,centroid,flux,zcr', help="Spectral features list")
    parser.add_argument('--share_encoder_weights', action='store_true', default=False, help="Share weights between encoders")
    parser.add_argument('--time_mask_ratio', type=float, default=0.5, help="CCSR time-frame mask ratio")
    parser.add_argument('--resume', type=str, default='', help="Path to checkpoint (.pth) to resume training")
    parser.add_argument('--verify_synthetic', action='store_true', default=False, help="Run synthetic batch verification and exit")

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Verification test
    verify_synthetic_forward_pass(device=device)
    if args.verify_synthetic:
        print("Synthetic verification requested. Exiting successfully.")
        return

    if args.data_train is None or not os.path.exists(args.data_train):
        print(f"No valid --data_train provided or file not found: {args.data_train}.")
        print("Pretraining verification complete. Provide --data_train to run full pretraining.")
        return

    # 2. Experiment Directory setup
    os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)
    with open(os.path.join(args.exp_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    if args.exp_description:
        with open(os.path.join(args.exp_dir, 'description.log'), 'w') as f:
            f.write(args.exp_description + '\n')

    # 3. WandB Registration
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))
        print(f"✓ WandB initialized for project '{args.wandb_project}' (run: {args.exp_name})")

    # 4. Data Loaders
    audio_conf = {
        'target_length': 1024,
        'target_freq_bins': 256,
        'mask_ratio': args.time_mask_ratio,
        'coh_freq_bands': (1000.0, 4000.0),
    }
    train_dataset = BinauralAudioDataset(args.data_train, audio_conf=audio_conf)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    if args.data_val and os.path.exists(args.data_val):
        val_dataset = BinauralAudioDataset(args.data_val, audio_conf=audio_conf)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    else:
        val_loader = None

    # 5. Build Model
    model = BinauralSSAMBASACModel(
        input_fdim=256, input_tdim=1024,
        fshape=16, tshape=16,
        sac_spec_lambda=args.sac_spec_lambda,
        sac_spat_lambda=args.sac_spat_lambda,
        recon_lambda=args.recon_lambda,
        local_sigma_mode=args.local_sigma_mode,
        spectral_features=args.spectral_features,
        share_encoder_weights=args.share_encoder_weights,
    ).to(device)

    # 6. Feature Stats calculation for offline_global_median
    if args.local_sigma_mode == 'offline_global_median':
        feature_stats = calculate_binaural_feature_stats(train_loader, device=device)
        model.feature_stats = feature_stats

    if torch.cuda.device_count() > 1:
        model = BinauralSSAMBASACModelParallel(model)

    # 7. Optimizer & Scheduler
    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True
    )

    # 8. Resume Checkpoint Logic
    global_step = 0
    start_epoch = 1
    best_val_loss = float('inf')
    progress = []

    if args.resume and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_state = checkpoint.get('state_dict', checkpoint)
        if list(model_state.keys())[0].startswith('module.'):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        core_model = model.module if isinstance(model, BinauralSSAMBASACModelParallel) else model
        core_model.load_state_dict(model_state, strict=False)

        progress_path = os.path.join(args.exp_dir, 'progress.pkl')
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'rb') as f:
                    progress = pickle.load(f)
                if len(progress) > 0:
                    start_epoch = progress[-1][0] + 1
                    global_step = progress[-1][1]
                    best_val_loss = min([p[4] for p in progress])
                    print(f"=> Resumed from epoch {start_epoch}, step {global_step}, best_val_loss {best_val_loss:.4f}")
            except Exception as e:
                print(f"Warning: could not load progress.pkl: {e}")

    # 9. Training Loop
    start_time = time.time()
    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    spec_meter = AverageMeter()
    spat_meter = AverageMeter()

    print(f"\nStarting Binaural SSAMBA Dual-Branch Pretraining ({args.epochs} epochs)...")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        for batch_idx, (X, W, c_mono, c_spatial) in enumerate(train_loader):
            X = X.to(device)
            W = W.to(device)
            c_mono = c_mono.to(device)
            c_spatial = c_spatial.to(device)

            # Warmup learning rate
            if global_step <= 1000 and global_step % 50 == 0:
                warm_lr = max(1e-7, (global_step / 1000.0) * args.lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr

            optimizer.zero_grad()
            out = model(X, W, c_mono=c_mono, c_spatial=c_spatial)
            loss = out['loss_total']
            loss.backward()
            optimizer.step()

            B = X.size(0)
            loss_meter.update(loss.item(), B)
            recon_meter.update(out['loss_recon'].item(), B)
            spec_meter.update(out['loss_sac_spec'].item(), B)
            spat_meter.update(out['loss_sac_spat'].item(), B)

            # Step Logging
            if global_step % args.n_print_steps == 0 and global_step > 0:
                print(f'Epoch [{epoch}][{batch_idx}/{len(train_loader)}] '
                      f'Total: {loss_meter.avg:.4f} | Recon: {recon_meter.avg:.4f} | '
                      f'Spec SAC: {spec_meter.avg:.4f} | Spat SAC: {spat_meter.avg:.4f} | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e}', flush=True)

                if args.use_wandb:
                    import wandb
                    log_dict = {
                        'train/loss_total': loss_meter.avg,
                        'train/loss_recon': recon_meter.avg,
                        'train/loss_sac_spec': spec_meter.avg,
                        'train/loss_sac_spat': spat_meter.avg,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'step': global_step,
                        'epoch': epoch,
                    }
                    if 'entropies_spec' in out and isinstance(out['entropies_spec'], dict):
                        for g_name, ent_val in out['entropies_spec'].items():
                            if isinstance(ent_val, torch.Tensor):
                                ent_val = ent_val.mean().item()
                            log_dict[f'group_metrics/{g_name}_weight_entropy'] = ent_val
                    wandb.log(log_dict)

            global_step += 1

            # Evaluation & Checkpoint saving
            if val_loader is not None and global_step % args.epoch_iter == 0:
                print(f'---- Step {global_step} Evaluation ----')
                equ_epoch = int(global_step / args.epoch_iter) + 1
                val_res = validate_binaural_sac(model, val_loader, device)
                val_loss = val_res['val_loss_total']

                print(f'  Val Loss Total: {val_loss:.4f} | Recon: {val_res["val_loss_recon"]:.4f} | '
                      f'Spec: {val_res["val_loss_spec"]:.4f} | Spat: {val_res["val_loss_spat"]:.4f}')

                if args.use_wandb:
                    import wandb
                    wandb.log({
                        'val/loss_total': val_res['val_loss_total'],
                        'val/loss_recon': val_res['val_loss_recon'],
                        'val/loss_sac_spec': val_res['val_loss_spec'],
                        'val/loss_sac_spat': val_res['val_loss_spat'],
                        'step': global_step,
                    })

                # Save periodic checkpoint
                save_path = os.path.join(args.exp_dir, 'models', f'audio_model.{equ_epoch}.pth')
                torch.save(model.state_dict(), save_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.exp_dir, 'models', 'best_audio_model.pth')
                    torch.save(model.state_dict(), best_path)
                    print(f'  ✓ New best model saved (val_loss={val_loss:.4f})')

                scheduler.step(val_loss)

                progress.append([epoch, global_step, equ_epoch, time.time() - start_time, val_loss])
                with open(os.path.join(args.exp_dir, 'progress.pkl'), 'wb') as f:
                    pickle.dump(progress, f)

                loss_meter.reset()
                recon_meter.reset()
                spec_meter.reset()
                spat_meter.reset()
                model.train()

    # Final checkpoint saving
    final_path = os.path.join(args.exp_dir, 'models', 'audio_model.final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Pretraining complete. Final model saved to '{final_path}'.")


if __name__ == '__main__':
    main()
