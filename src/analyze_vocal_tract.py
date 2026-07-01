import os
import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.interpolate
import glob
from scipy.stats import spearmanr

# Ensure proper paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'sac')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

from sac.sac_model import SSAMBASACModel
from sac.acoustic_features import extract_acoustic_features

def get_fbank(wav):
    wav = wav - wav.mean()
    dataset_mean = -4.2677393
    dataset_std = 4.5689974
    fbank = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0), htk_compat=True, sample_frequency=16000, 
        use_energy=False, window_type='hanning', num_mel_bins=128, 
        dither=0.0, frame_shift=10
    )
    p = 1024 - fbank.shape[0]
    if p > 0:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
    elif p < 0:
        fbank = fbank[0:1024, :]
        
    fbank = (fbank - dataset_mean) / (dataset_std * 2)
    return fbank

def load_and_preprocess(audio_path):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    wav = wav.squeeze()
    
    orig_len = len(wav)
    max_len = 164080
    if len(wav) > max_len:
        wav = wav[:max_len]
        orig_len = max_len
    else:
        wav = torch.nn.functional.pad(wav, (0, max_len - len(wav)))
        
    fbank = get_fbank(wav)
    return wav, fbank, orig_len

def main():
    parser = argparse.ArgumentParser(description="Analyze Vocal Tract Attention and Latent Space (RSA)")
    parser.add_argument("--exp_dir", type=str, default="/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam0.02-sig1.0-feat_universal-mode_offline_global_median-librispeech-new_LPC", help="Path to experiment directory")
    parser.add_argument("--out_dir", type=str, default="/storage/yotam/ssamba/src/metrics/vocal_tract", help="Output directory for plots")
    parser.add_argument("--num_rsa_samples", type=int, default=200, help="Number of audio samples for RSA")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = os.path.join(args.exp_dir, "models", "best_audio_model.pth")
    if not os.path.exists(ckpt_path):
        pths = glob.glob(os.path.join(args.exp_dir, "models", "*.pth"))
        if pths:
            ckpt_path = pths[0]
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.exp_dir}/models")

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
        sac_features="f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm",
        vision_mamba_config=vision_mamba_config,
    )
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if not isinstance(state_dict, dict):
        state_dict = state_dict.state_dict()
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    family_names = list(model.family_indices.keys())
    if 'Vocal_Tract' not in family_names:
        raise ValueError("Vocal_Tract family not found in model!")
    vt_idx = family_names.index('Vocal_Tract')

    libri_files = glob.glob("/storage/data/LibriSpeech/test-clean/**/*.flac", recursive=True)
    if not libri_files:
        raise ValueError("Could not find LibriSpeech files in /storage/data/LibriSpeech/test-clean/")

    np.random.seed(42)
    selected_files = np.random.choice(libri_files, min(args.num_rsa_samples, len(libri_files)), replace=False)

    print(f"\n--- Part 1: Generating Attention Plot for a single audio ---")
    wav, fbank, orig_len = load_and_preprocess(selected_files[0])
    fbank = fbank.to(device)
    
    with torch.no_grad():
        x = fbank.unsqueeze(0).unsqueeze(1).transpose(2, 3)
        hidden_states = model._encode_with_mamba(x)
        Q = model.family_queries.unsqueeze(0)
        cls_token_num = model.encoder.cls_token_num
        attn_output, attn_weights = model.cross_attention(
            query=Q, 
            key=hidden_states[:, cls_token_num:, :], 
            value=hidden_states[:, cls_token_num:, :]
        )
        vt_attn = attn_weights[0, vt_idx, :].cpu().numpy()
    
    vt_attn_map = vt_attn.reshape(8, 64)
    vt_attn_time = vt_attn_map.sum(axis=0)
    vt_attn_time = (vt_attn_time - vt_attn_time.min()) / (vt_attn_time.max() - vt_attn_time.min() + 1e-8)
    
    time_x = np.linspace(0, len(wav)/16000, len(vt_attn_time))
    wav_time_x = np.linspace(0, len(wav)/16000, len(wav))
    
    interp_func = scipy.interpolate.interp1d(time_x, vt_attn_time, kind='cubic', fill_value="extrapolate")
    attn_upsampled = interp_func(wav_time_x)
    attn_upsampled = np.clip(attn_upsampled, 0, 1)

    wav_plot = wav[:orig_len].numpy()
    wav_time_x_plot = wav_time_x[:orig_len]
    attn_upsampled_plot = attn_upsampled[:orig_len]
    
    orig_frames = min(1024, int((orig_len / 16000) * 100))
    fbank_plot = fbank[:orig_frames, :].cpu().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(wav_time_x_plot, wav_plot, color='black', alpha=0.6)
    axs[0].set_title("Audio Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(0, orig_len/16000)
    
    axs[1].imshow(fbank_plot.T, aspect='auto', origin='lower', extent=[0, orig_len/16000, 0, 128])
    axs[1].set_title("Log-Mel Spectrogram")
    axs[1].set_ylabel("Mel bins")
    
    axs[2].plot(wav_time_x_plot, attn_upsampled_plot, color='red', linewidth=2)
    axs[2].fill_between(wav_time_x_plot, 0, attn_upsampled_plot, color='red', alpha=0.3)
    axs[2].set_title("Vocal Tract Query Attention Weight (Summed across frequency)")
    axs[2].set_ylabel("Normalized Attention")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].set_ylim(0, 1.1)
    
    plt.tight_layout()
    attn_out_file = os.path.join(args.out_dir, "vocal_tract_attention.png")
    plt.savefig(attn_out_file)
    print(f"Attention plot saved to {attn_out_file}")
    plt.close()

    print(f"\n--- Part 2: Representational Similarity Analysis (RSA) over {len(selected_files)} samples ---")
    all_latents = []
    all_formants = []

    for idx, f in enumerate(selected_files):
        if idx % 20 == 0:
            print(f"Processing sample {idx}/{len(selected_files)}...")
            
        wav, fbank, _ = load_and_preprocess(f)
        wav = wav.to(device)
        fbank = fbank.to(device)

        # 1. Get Latent
        with torch.no_grad():
            x = fbank.unsqueeze(0).unsqueeze(1).transpose(2, 3)
            hidden_states = model._encode_with_mamba(x)
            Q = model.family_queries.unsqueeze(0)
            attn_output, _ = model.cross_attention(
                query=Q, 
                key=hidden_states[:, cls_token_num:, :], 
                value=hidden_states[:, cls_token_num:, :]
            )
            # attn_output: [1, num_families, embed_dim]
            vt_latent = attn_output[0, vt_idx, :]
            all_latents.append(vt_latent.cpu())

            # 2. Get Formants (ground truth)
            feats = extract_acoustic_features(wav.unsqueeze(0), sample_rate=16000, features_list='formants', normalize=False)
            all_formants.append(feats[0].cpu())
            
    all_latents = torch.stack(all_latents, dim=0) # [N, embed_dim]
    all_formants = torch.stack(all_formants, dim=0) # [N, 3]

    print("Computing distance matrices...")
    dist_latents = torch.cdist(all_latents, all_latents, p=2)
    dist_formants = torch.cdist(all_formants, all_formants, p=2)

    # Get upper triangular part
    N_samples = all_latents.shape[0]
    triu_idx = torch.triu_indices(N_samples, N_samples, offset=1)
    
    d_latents_flat = dist_latents[triu_idx[0], triu_idx[1]].numpy()
    d_formants_flat = dist_formants[triu_idx[0], triu_idx[1]].numpy()

    # Spearman Correlation
    corr, p_value = spearmanr(d_formants_flat, d_latents_flat)
    print(f"RSA Spearman Correlation: {corr:.4f} (p={p_value:.4e})")

    # Scatter plot
    plt.figure(figsize=(8, 6))
    if len(d_latents_flat) > 5000:
        plot_idx = np.random.choice(len(d_latents_flat), 5000, replace=False)
        plt.scatter(d_formants_flat[plot_idx], d_latents_flat[plot_idx], alpha=0.3, s=5)
    else:
        plt.scatter(d_formants_flat, d_latents_flat, alpha=0.3, s=5)
        
    m, b = np.polyfit(d_formants_flat, d_latents_flat, 1)
    x_line = np.linspace(min(d_formants_flat), max(d_formants_flat), 100)
    plt.plot(x_line, m*x_line + b, color='red', linewidth=2, label=f'Linear Fit')

    plt.title(f"Representational Similarity Analysis (Vocal Tract)\nSpearman $\\rho$ = {corr:.3f}")
    plt.xlabel("LPC Formants L2 Distance (Ground Truth)")
    plt.ylabel("Latent Subspace L2 Distance (Model)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    rsa_out_file = os.path.join(args.out_dir, "rsa_vocal_tract.png")
    plt.savefig(rsa_out_file, dpi=150)
    print(f"RSA plot saved to {rsa_out_file}")

if __name__ == '__main__':
    main()
