import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import json
import sys

# Append paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from acoustic_features import extract_acoustic_features
from sac_diagnostics import SACDebugger
from sac_model import SSAMBASACModel

def load_progress(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def analyze_logs():
    print("=== 1. Analyzing Pretraining Logs ===")
    sac_exp_dir = '/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam1.0-sig1.0-feat_sid-librispeech' # '/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-m300-lam1.0-tau0.3-sig1.0-librispeech'
    base_exp_dir = '/storage/yotam/ssamba/src/pretrain/exp/amba-base-f16-t16-b16-lr1e-4-m300-pretrain_joint-librispeech'

    try:
        sac_prog = load_progress(os.path.join(sac_exp_dir, 'progress.pkl'))
        base_prog = load_progress(os.path.join(base_exp_dir, 'progress.pkl'))
        
        # Depending on how the logs are saved (dict or list), print the final reconstruction loss
        print("Successfully loaded progress logs.")
        if isinstance(sac_prog, list) and len(sac_prog) > 0:
            print("Final SAC log entry:", sac_prog[-1])
        if isinstance(base_prog, list) and len(base_prog) > 0:
            print("Final Baseline log entry:", base_prog[-1])
            
    except Exception as e:
        print(f"Could not load logs: {e}")

def evaluate_manifold(device='cuda'):
    print("\n=== 2. Evaluating Latent Manifold & Acoustic Weights ===")
    
    # 1. Load sample dataset (LibriSpeech is best for checking pretraining)
    dataset_path = "/storage/data/LibriSpeech/train-clean-100/" 
    if not os.path.exists(dataset_path):
        print("LibriSpeech path not found. Please provide a valid directory.")
        return
        
    import glob
    import random
    wav_files = glob.glob(os.path.join(dataset_path, "**/*.flac"), recursive=True)
    if len(wav_files) == 0:
        print("No flac files found.")
        return
        
    sample_files = random.sample(wav_files, min(16, len(wav_files))) # small batch of 16
    
    audio_list = []
    for f in sample_files:
        wav, sr = torchaudio.load(f)
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            wav = resampler(wav)
        # Pad or truncate to 3 seconds (48000 samples)
        if wav.shape[1] > 48000:
            wav = wav[:, :48000]
        else:
            wav = torch.nn.functional.pad(wav, (0, 48000 - wav.shape[1]))
        audio_list.append(wav.squeeze(0))
        
    audio_batch = torch.stack(audio_list).to(device)
    
    # 2. Compute Acoustic Features & weights (w_ij)
    print("Computing acoustic features...")
    features = extract_acoustic_features(audio_batch, sample_rate=16000)
    
    sigma = 1.0
    dist_sq = torch.cdist(features, features)**2
    w_ij = torch.exp(-dist_sq / sigma)
    
    print("\nw_ij Weight Distribution (sigma=1.0):")
    print(f"Mean weight: {w_ij.mean().item():.4f}")
    print(f"Min weight: {w_ij.min().item():.4f}")
    print(f"Max weight: {w_ij.max().item():.4f}")
    
    if w_ij.mean() > 0.8:
        print(">>> WARNING: w_ij mean is very high. Sigma is likely too large! All pairs are considered positive.")
    elif w_ij.mean() < 0.1:
        print(">>> WARNING: w_ij mean is very low. Sigma is likely too small! No positive pairs found.")

    # 3. Load SAC Model and evaluate Uniformity
    try:
        print("\nLoading SAC model to extract latent representations...")
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
            vision_mamba_config=vision_mamba_config,
        )
        
        ckpt_path = '/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-m300-lam1.0-tau0.3-sig1.0-librispeech/models/best_audio_model.pth'
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        # Handle DataParallel / nested state dicts
        if not isinstance(state_dict, dict):
            state_dict = state_dict.state_dict()
        if 'module' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # Extract log-Mel spectrograms (fbank)
        fbank_list = []
        dataset_mean = -4.2677393 # LibriSpeech mean from run_sac.sh
        dataset_std = 4.5689974 # LibriSpeech std from run_sac.sh
        
        for wav in audio_list:
            wav = wav - wav.mean()
            fbank = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0), htk_compat=True, sample_frequency=16000, 
                use_energy=False, window_type='hanning', num_mel_bins=128, 
                dither=0.0, frame_shift=10
            )
            p = 1024 - fbank.shape[0]
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:1024, :]
                
            fbank = (fbank - dataset_mean) / (dataset_std * 2)
            fbank_list.append(fbank)
            
        fbank_batch = torch.stack(fbank_list).to(device)
        
        with torch.no_grad():
            output = model(fbank_batch, acoustic_features=features, return_diagnostics=True)
            z_norm = output['z_norm']
            
        diag = SACDebugger()
        uniformity, alignment = diag.compute_alignment_uniformity(z_norm, w_ij)
        
        print(f"\nSAC Model Uniformity: {uniformity:.4f}")
        print(f"SAC Model Alignment: {alignment:.4f}")
        
        if uniformity < -2.0:
            print(">>> WARNING: Very low uniformity detected. The latent space has likely collapsed into a single cluster (Mode Collapse).")
            
    except Exception as e:
        print(f"Failed to load model or compute representations: {e}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running diagnostics on {device}...")
    analyze_logs()
    evaluate_manifold(device)
