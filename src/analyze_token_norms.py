import os
import argparse
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import sys

# Ensure proper paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'sac')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

from sac.sac_model import SSAMBASACModel

matplotlib.rcParams.update({'font.size': 12, 'savefig.dpi': 300})

def load_audio_batch(dataset_path, num_samples=256, device='cuda', length_samples=48000):
    wav_files = glob.glob(os.path.join(dataset_path, "**/*.flac"), recursive=True)
    if len(wav_files) == 0:
        raise ValueError("No flac files found.")
        
    sample_files = random.sample(wav_files, min(num_samples, len(wav_files)))
    
    audio_list = []
    for f in sample_files:
        wav, sr = torchaudio.load(f)
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            wav = resampler(wav)
        if wav.shape[1] > length_samples:
            wav = wav[:, :length_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, length_samples - wav.shape[1]))
        audio_list.append(wav.squeeze(0))
        
    return torch.stack(audio_list).to(device)

def get_fbank_batch(audio_batch, device='cuda', print_diagnostic=False):
    fbank_list = []
    dataset_mean = -4.2677393
    dataset_std = 4.5689974
    for i, wav in enumerate(audio_batch):
        wav = wav - wav.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0), htk_compat=True, sample_frequency=16000, 
            use_energy=False, window_type='hanning', num_mel_bins=128, 
            dither=0.0, frame_shift=10
        )
        
        if print_diagnostic and i == 0:
            # Check zeros before padding and normalization
            # A row is considered zero if its absolute sum is exactly 0
            row_sums = fbank.abs().sum(dim=1)
            zero_rows = (row_sums == 0).sum().item()
            non_zero_rows = fbank.shape[0] - zero_rows
            
            p = 1024 - fbank.shape[0]
            print("\n--- Fbank Diagnostic (First Sample) ---")
            print(f"Original fbank shape (before padding): {fbank.shape}")
            print(f"Non-zero rows (real audio): {non_zero_rows}")
            print(f"Zero rows: {zero_rows}")
            print(f"Zero-padding needed to reach 1024: {p if p > 0 else 0}")
            print("---------------------------------------")
            
        p = 1024 - fbank.shape[0]
        if p > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
        elif p < 0:
            fbank = fbank[0:1024, :]
            
        fbank = (fbank - dataset_mean) / (dataset_std * 2)
        fbank_list.append(fbank)
    return torch.stack(fbank_list).to(device)

def compute_token_norm_map(model, fbank_batch):
    B = fbank_batch.shape[0]
    with torch.no_grad():
        x = fbank_batch.unsqueeze(1).transpose(2, 3)
        hidden_states = model._encode_with_mamba(x) # [B, seq_len, D_model]
        
        cls_token_num = model.encoder.cls_token_num
        patch_tokens = hidden_states[:, cls_token_num:, :] # [B, 512, D_model]
        
        # Norm over hidden dimension
        token_norms = torch.norm(patch_tokens, p=2, dim=-1) # [B, 512]
        
        # Mean over batch
        mean_token_norms = token_norms.mean(dim=0).cpu().numpy() # [512]
        
        # Reshape to [8, 64]
        token_norm_map = mean_token_norms.reshape(8, 64)
        
    return token_norm_map, hidden_states

def main():
    parser = argparse.ArgumentParser(description="Analyze Encoder Token Norms")
    parser.add_argument("exp_dir", type=str, help="Path to experiment directory (e.g. sac/exp/...)")
    parser.add_argument("--sac_features", type=str, default="f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm", help="SAC features string")
    parser.add_argument("--dataset_path", type=str, default="/storage/data/LibriSpeech/test-clean/", help="Path to reference dataset")
    parser.add_argument("--out_dir", type=str, default="metrics/analyze_pretrained_sac", help="Output directory for plots")
    parser.add_argument("--num_samples", type=int, default=256, help="Number of samples to run analysis on")
    args = parser.parse_args()

    model_name = os.path.basename(os.path.normpath(args.exp_dir))
    args.out_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(args.out_dir, exist_ok=True)
    
    ckpt_path = os.path.join(args.exp_dir, "models", "best_audio_model.pth")
    if not os.path.exists(ckpt_path):
        pths = glob.glob(os.path.join(args.exp_dir, "models", "*.pth"))
        if not pths:
            raise FileNotFoundError(f"No checkpoint found in {os.path.join(args.exp_dir, 'models')}")
        ckpt_path = pths[0]
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading model...")
    vision_mamba_config = {
        'img_size': (128, 1024), 'patch_size': 16, 'stride': 16, 'embed_dim': 768, 'depth': 24,
        'channels': 1, 'num_classes': 1000, 'drop_rate': 0.0, 'drop_path_rate': 0.1,
        'norm_epsilon': 1e-5, 'rms_norm': False, 'residual_in_fp32': False,
        'fused_add_norm': False, 'if_rope': False, 'if_rope_residual': False,
        'bimamba_type': 'v2', 'if_bidirectional': True, 'final_pool_type': 'none',
        'if_abs_pos_embed': True, 'if_bimamba': False, 'if_cls_token': True,
        'if_devide_out': True, 'use_double_cls_token': False, 'use_middle_cls_token': False,
    }
    
    model = SSAMBASACModel(
        fshape=16, tshape=16, input_fdim=128, input_tdim=1024,
        model_size='base', embed_dim=768, depth=24, proj_dim=128,
        sac_temperature=0.3, sac_sigma=1.0, sac_lambda=1.0, mask_patch=300,
        sac_features=args.sac_features,
        vision_mamba_config=vision_mamba_config,
    )
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if not isinstance(state_dict, dict):
        state_dict = state_dict.state_dict()
    if 'module' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"\n[1] Testing with 3s (buggy) audio length...")
    audio_3s = load_audio_batch(args.dataset_path, num_samples=args.num_samples, device=device, length_samples=48000)
    fbank_3s = get_fbank_batch(audio_3s, device=device, print_diagnostic=False)
    norm_map_3s, _ = compute_token_norm_map(model, fbank_3s)

    print(f"\n[2] Testing with 10.25s (fixed) audio length...")
    audio_4s = load_audio_batch(args.dataset_path, num_samples=args.num_samples, device=device, length_samples=164080)
    fbank_4s = get_fbank_batch(audio_4s, device=device, print_diagnostic=True)
    norm_map_4s, hidden_states_4s = compute_token_norm_map(model, fbank_4s)

    # Print token norm diagnostics
    for label, norm_map in [("3s input (buggy)", norm_map_3s), ("4s input (fixed)", norm_map_4s)]:
        early_norm = norm_map[:, :20].mean()
        late_norm = norm_map[:, 20:].mean()
        ratio = early_norm / (late_norm + 1e-10)
        print(f"\n--- Token Norm Stats: {label} ---")
        print(f"Mean norm patches 0-19 (early): {early_norm:.4f}")
        print(f"Mean norm patches 20-63 (late): {late_norm:.4f}")
        print(f"Ratio (Early / Late):           {ratio:.4f}")

    # Plot norm maps comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    sns.heatmap(norm_map_3s, ax=axes[0], cmap='viridis')
    axes[0].set_title("3s input (buggy)")
    axes[0].set_ylabel('Freq Patch (0-7)')
    axes[0].set_xlabel('Time Patch (0-63)')
    
    sns.heatmap(norm_map_4s, ax=axes[1], cmap='viridis')
    axes[1].set_title("4s input (fixed)")
    axes[1].set_ylabel('Freq Patch (0-7)')
    axes[1].set_xlabel('Time Patch (0-63)')
    
    plt.suptitle("Mean Encoder Token L2 Norm (freq x time)", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'token_norm_map_comparison.pdf'))
    plt.close()

    # Generate Attention Maps for fixed 4s audio
    print("\nExtracting Attention Maps for fixed 4s audio...")
    B = fbank_4s.shape[0]
    K = model.num_active_families
    family_names = list(model.family_indices.keys())
    
    with torch.no_grad():
        Q = model.family_queries.unsqueeze(0).expand(B, -1, -1)
        cls_token_num = model.encoder.cls_token_num
        attn_output, attn_weights = model.cross_attention(
            query=Q, 
            key=hidden_states_4s[:, cls_token_num:, :], 
            value=hidden_states_4s[:, cls_token_num:, :]
        )
        
    mean_attn = attn_weights.mean(dim=0).cpu().numpy() # [K, 512]
    mean_attn_map = mean_attn.reshape(K, 8, 64)
    
    fig, axes = plt.subplots(K, 1, figsize=(10, 2*K))
    if K == 1: axes = [axes]
    for k in range(K):
        sns.heatmap(mean_attn_map[k], ax=axes[k], cmap='viridis')
        axes[k].set_title(family_names[k])
        axes[k].set_ylabel('Freq (0-7)')
        axes[k].set_xlabel('Time (0-63)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'attention_maps_fixed.pdf'))
    plt.close()
    
    print(f"\nPlots saved to {args.out_dir}:")
    print(" - token_norm_map_comparison.pdf")
    print(" - attention_maps_fixed.pdf")

if __name__ == '__main__':
    main()
