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
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import matplotlib
import sys

# Ensure proper paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'sac')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

from sac.sac_model import SSAMBASACModel
from sac.acoustic_features import extract_acoustic_features

matplotlib.rcParams.update({'font.size': 12, 'savefig.dpi': 300})

def load_audio_batch(dataset_path, num_samples=256, device='cuda'):
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
        if wav.shape[1] > 164080:
            wav = wav[:, :164080]
        else:
            wav = torch.nn.functional.pad(wav, (0, 164080 - wav.shape[1]))
        audio_list.append(wav.squeeze(0))
        
    return torch.stack(audio_list).to(device)

def get_fbank_batch(audio_batch, device='cuda'):
    fbank_list = []
    dataset_mean = -4.2677393
    dataset_std = 4.5689974
    for wav in audio_batch:
        wav = wav - wav.mean()
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
        fbank_list.append(fbank)
    return torch.stack(fbank_list).to(device)

def main():
    random.seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser(description="Analyze Pretrained SAC Model")
    parser.add_argument("exp_dir", type=str, help="Path to experiment directory (e.g. sac/exp/...)")
    parser.add_argument("--sac_features", type=str, default="f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm", help="SAC features string")
    parser.add_argument("--dataset_path", type=str, default="/storage/data/LibriSpeech/test-clean/", help="Path to reference dataset")
    parser.add_argument("--out_dir", type=str, default="metrics/analyze_pretrained_sac", help="Output directory for plots")
    parser.add_argument("--num_samples", type=int, default=256, help="Number of samples to run analysis on")
    args = parser.parse_args()

    # Extract model name from exp_dir
    model_name = os.path.basename(os.path.normpath(args.exp_dir))
    
    # Update out_dir to include model name
    args.out_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Resolve checkpoint path
    ckpt_path = os.path.join(args.exp_dir, "models", "best_audio_model.pth")
    if not os.path.exists(ckpt_path):
        # Fallback to any .pth file if best_audio_model.pth doesn't exist
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

    print(f"Loading {args.num_samples} audio samples...")
    audio_batch = load_audio_batch(args.dataset_path, num_samples=args.num_samples, device=device)
    fbank_batch = get_fbank_batch(audio_batch, device=device)
    features = extract_acoustic_features(audio_batch, sample_rate=16000, features_list=args.sac_features)

    # Extract single scalar per group
    c_scalars = []
    group_names = list(model.group_indices.keys())
    K = model.num_active_groups
    
    for k in range(K):
        group_name = group_names[k]
        indices = model.group_indices[group_name]
        c_group = features[:, indices]
        c_scalars.append(c_group.mean(dim=1).cpu().numpy())
    c_scalars = np.stack(c_scalars, axis=1)

    print("Running forward pass...")
    with torch.no_grad():
        x = fbank_batch.unsqueeze(1).transpose(2, 3)
        B = x.shape[0]
        hidden_states = model._encode_with_mamba(x)
        
        Q = model.group_queries.unsqueeze(0).expand(B, -1, -1)
        cls_token_num = model.encoder.cls_token_num
        
        attn_output, attn_weights = model.cross_attention(
            query=Q, 
            key=hidden_states[:, cls_token_num:, :], 
            value=hidden_states[:, cls_token_num:, :]
        )
        
        attn_output_flat = attn_output.reshape(B * K, -1)
        z_flat = model.projection_head(attn_output_flat)
        Z_groups = z_flat.reshape(B, K, -1)

    # --- Analysis 1: Attention Map Visualization ---
    print("Running Analysis 1: Attention Maps...")
    mean_attn = attn_weights.mean(dim=0).cpu().numpy() # [K, 512]
    mean_attn_map = mean_attn.reshape(K, 8, 64)
    
    fig, axes = plt.subplots(K, 1, figsize=(10, 2*K))
    if K == 1: axes = [axes]
    for k in range(K):
        sns.heatmap(mean_attn_map[k], ax=axes[k], cmap='viridis')
        axes[k].set_title(group_names[k])
        axes[k].set_ylabel('Freq (0-7)')
        axes[k].set_xlabel('Time (0-63)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'attention_maps_per_group.pdf'))
    plt.close()

    # --- Analysis 2: Query Orthogonality ---
    print("Running Analysis 2: Query Orthogonality...")
    Q_params = model.group_queries.detach().cpu()
    Q_norm = F.normalize(Q_params, dim=-1)
    S = torch.matmul(Q_norm, Q_norm.T).numpy()
    
    plt.figure(figsize=(8, 6))
    mask = np.eye(K, dtype=bool)
    sns.heatmap(S, annot=True, fmt=".2f", mask=mask, cmap='coolwarm', center=0, 
                xticklabels=group_names, yticklabels=group_names)
    plt.title("Query Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'query_cosine_similarity.pdf'))
    plt.close()
    
    off_diag = S[~mask]
    mean_ortho = off_diag.mean() if len(off_diag) > 0 else 0
    max_ortho = off_diag.max() if len(off_diag) > 0 else 0

    # --- Analysis 2.5: Acoustic Feature Correlation ---
    print("Running Analysis 2.5: Acoustic Feature Correlation...")
    c_corr = np.corrcoef(c_scalars, rowvar=False)
    
    plt.figure(figsize=(8, 6))
    mask_corr = np.eye(K, dtype=bool)
    sns.heatmap(c_corr, annot=True, fmt=".2f", mask=mask_corr, cmap='coolwarm', center=0, 
                xticklabels=group_names, yticklabels=group_names)
    plt.title("Acoustic Feature Inter-Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'acoustic_feature_correlation.pdf'))
    plt.close()

    off_diag_corr = c_corr[~mask_corr]
    mean_abs_off_diag_corr = np.abs(off_diag_corr).mean() if len(off_diag_corr) > 0 else 0

    # --- Analysis 3: Mutual Information ---
    print("Running Analysis 3: Mutual Information...")
    Z_np = Z_groups.cpu().numpy()
    mi_matrix = np.zeros((K, K))
    for k in range(K):
        pca = PCA(n_components=3)
        z_pcs = pca.fit_transform(Z_np[:, k, :])
        for l in range(K):
            mi_vals = [
                mutual_info_regression(
                    z_pcs[:, pc].reshape(-1, 1), c_scalars[:, l], random_state=42
                )[0]
                for pc in range(3)
            ]
            mi_matrix[k, l] = max(mi_vals)
            
    mi_norm = mi_matrix / (mi_matrix.max(axis=1, keepdims=True) + 1e-10)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(mi_norm, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=group_names, yticklabels=group_names)
    plt.xlabel("Acoustic Features")
    plt.ylabel("Latent Z_k (PC1)")
    plt.title("Normalized Mutual Information")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'mutual_information_matrix.pdf'))
    plt.close()
    
    diag_mi = np.diag(mi_matrix)
    off_diag_mi = mi_matrix[~np.eye(K, dtype=bool)]
    mean_diag_mi = diag_mi.mean() if K > 0 else 0
    mean_off_diag_mi = off_diag_mi.mean() if len(off_diag_mi) > 0 else 0
    diagonality_score = mean_diag_mi / (mean_off_diag_mi + 1e-10)

    # --- Analysis 4: t-SNE ---
    print("Running Analysis 4: t-SNE...")
    fig, axes = plt.subplots(1, K, figsize=(5*K, 4))
    if K == 1: axes = [axes]
    for k in range(K):
        tsne = TSNE(n_components=2, perplexity=min(30, B-1), n_iter=1000, random_state=42)
        z_tsne = tsne.fit_transform(Z_np[:, k, :])
        sc = axes[k].scatter(z_tsne[:, 0], z_tsne[:, 1], c=c_scalars[:, k], cmap='coolwarm', s=20)
        axes[k].set_title(group_names[k])
        plt.colorbar(sc, ax=axes[k])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'tsne_per_group.pdf'))
    plt.close()

    # --- Analysis 5: Attention Entropy ---
    print("Running Analysis 5: Attention Entropy...")
    eps = 1e-10
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1).cpu().numpy()
    mean_entropy = entropy.mean(axis=0)
    std_entropy = entropy.std(axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.bar(group_names, mean_entropy, yerr=std_entropy, capsize=5)
    plt.ylabel("Entropy")
    plt.title("Mean Attention Entropy per Group")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'attention_entropy_per_group.pdf'))
    plt.close()

    # --- Final Summary ---
    print("\n" + "="*50)
    print("                SCALAR METRICS")
    print("="*50)
    print(f"Mean off-diagonal Cosine Similarity: {mean_ortho:.4f}")
    print(f"Max off-diagonal Cosine Similarity:  {max_ortho:.4f}")
    print(f"Random expected Cosine Similarity:   ~0.0000")
    print("-" * 50)
    print(f"Mean Diagonal Mutual Information:    {mean_diag_mi:.4f}")
    print(f"Mean Off-Diag Mutual Information:    {mean_off_diag_mi:.4f}")
    print(f"Diagonality Score (Signal/Leakage):  {diagonality_score:.4f}")
    print("-" * 50)
    if 'Vocal_Tract' in group_names and 'Timbre' in group_names:
        vt_idx = group_names.index('Vocal_Tract')
        timbre_idx = group_names.index('Timbre')
        print(f"Vocal_Tract/Timbre Correlation:      {c_corr[vt_idx, timbre_idx]:.4f}")
    print(f"Mean Abs Off-Diag Feature Corr:      {mean_abs_off_diag_corr:.4f}")
    print("="*50)
    print(f"All plots saved to: {args.out_dir}")
    print(f"Generated plots include: acoustic_feature_correlation.pdf")

if __name__ == '__main__':
    main()
