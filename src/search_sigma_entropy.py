import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Setup paths to ensure imports work correctly
sys.path.insert(0, '/storage/yotam/ssamba/src')
sys.path.insert(0, '/storage/yotam/ssamba')
sys.path.insert(0, '/storage/yotam/ssamba/Vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/mamba-1p1p1')

from sac.run_pretrain_sac import AudioDatasetWithWaveform
from sac.acoustic_features import extract_acoustic_features, get_feature_groups

def load_random_features(num_samples=5000, batch_size=32, device='cuda'):
    print(f"Loading up to {num_samples} samples from LibriSpeech train set...")
    json_path = '/storage/yotam/ssamba/librispeech_train.json'
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
        
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'dataset': 'librispeech',
        'mode': 'train',
        'mean': -4.2677393,
        'std': 4.5689974,
        'noise': False,
    }
    
    label_csv = '/storage/yotam/ssamba/src/finetune/audioset/data/class_labels_indices.csv'
    dataset = AudioDatasetWithWaveform(json_path, audio_conf=audio_conf, label_csv=label_csv, target_sr=16000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    features_list = "f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm"
    all_features = []
    
    loaded_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) >= 3:
                _, waveform, _ = batch
            else:
                waveform = batch[0]
                
            waveform = waveform.to(device)
            feats = extract_acoustic_features(
                waveform, 
                sample_rate=16000, 
                feature_stats=None, 
                features_list=features_list, 
                normalize=False
            )
            all_features.append(feats)
            loaded_count += feats.size(0)
            
            # Simple progress print
            if loaded_count % (batch_size * 10) < batch_size:
                print(f"Extracted {loaded_count} / {num_samples}...")
                
            if loaded_count >= num_samples:
                break
                
    all_features = torch.cat(all_features, dim=0)[:num_samples]
    
    # Global normalization across all samples
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True)
    all_features = (all_features - mean) / (3.0 * std + 1e-6)
    all_features = all_features.clamp(-1.0, 1.0)
    
    print(f"Successfully extracted globally normalized features of shape: {all_features.shape}")
    return all_features, features_list

def search_sigma_entropy():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    # Load 5000 samples
    N = 5000
    try:
        features, features_list = load_random_features(num_samples=N, device=device)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
        
    # Slice features
    groups, num_groups = get_feature_groups(features_list)
    print(f"Found {num_groups} active feature groups: {list(groups.keys())}")
    
    # Grid search setup
    sigmas = np.logspace(-2, 1, 50)
    batch_sizes = [16, 32, 64]
    
    results = {}
    
    out_dir = '/storage/yotam/ssamba/src/metrics/sigma_search'
    os.makedirs(out_dir, exist_ok=True)
    
    # Plotting setup
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(18, 6))
    if len(batch_sizes) == 1:
        axes = [axes]
        
    for idx, bs in enumerate(batch_sizes):
        print(f"\n{'='*50}\nEvaluating for Batch Size = {bs}\n{'='*50}")
        max_entropy = np.log(bs - 1)
        target_entropy = 0.625 * max_entropy
        print(f"Theoretical max entropy for N={bs}: {max_entropy:.4f}")
        
        bs_results = {g: [] for g in groups.keys()}
        recommendations = {}
        
        for g_name, indices in groups.items():
            g_feats = features[:, indices]
            
            entropies = []
            best_sigma = None
            min_diff_to_target = float('inf')
            
            for sigma in sigmas:
                batch_entropies = []
                # Compute over mini-batches
                for i in range(0, N - bs + 1, bs):
                    batch_feats = g_feats[i:i+bs]
                    D = torch.cdist(batch_feats, batch_feats, p=2)
                    off_diag = ~torch.eye(bs, dtype=torch.bool, device=device)
                    
                    W = torch.exp(- (D / sigma)**2)
                    W_masked = W * off_diag.float()
                    
                    W_sum = W_masked.sum(dim=1, keepdim=True) + 1e-8
                    W_norm = W_masked / W_sum
                    
                    H = - (W_norm * torch.log(W_norm + 1e-8)).sum(dim=1).mean().item()
                    batch_entropies.append(H)
                    
                mean_H = np.mean(batch_entropies)
                entropies.append(mean_H)
                
                diff = abs(mean_H - target_entropy)
                if diff < min_diff_to_target and 0.5 * max_entropy <= mean_H <= 0.75 * max_entropy:
                    min_diff_to_target = diff
                    best_sigma = sigma
                    
            if best_sigma is None:
                best_sigma = sigmas[np.argmin([abs(h - target_entropy) for h in entropies])]
                
            bs_results[g_name] = entropies
            recommendations[g_name] = best_sigma
            print(f"{g_name:<15} | Recommended sigma: {best_sigma:.4f}")
            
        results[bs] = {
            'entropies': bs_results,
            'recommendations': recommendations
        }
        
        # Plot for this batch size
        ax = axes[idx]
        for g_name, ent in bs_results.items():
            ax.plot(sigmas, ent, label=f"{g_name} ({recommendations[g_name]:.3f})", marker='.', markersize=4)
        ax.axhline(y=max_entropy, color='k', linestyle='--', label=f"Max H ({max_entropy:.2f})")
        ax.axhline(y=target_entropy, color='gray', linestyle=':', label="Target 62.5%")
        ax.set_xscale('log')
        ax.set_xlabel('Sigma (Log Scale)')
        ax.set_ylabel('Mean Shannon Entropy')
        ax.set_title(f'Batch Size = {bs}')
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'sigma_entropy_search_multibatch.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    
    # Save results
    np.save(os.path.join(out_dir, 'search_results_multibatch.npy'), results)
    
    # Print dictionary
    print("\n\n=== Final Optimal Sigma Dictionary ===")
    final_dict = {bs: results[bs]['recommendations'] for bs in batch_sizes}
    import pprint
    pprint.pprint(final_dict)

if __name__ == '__main__':
    search_sigma_entropy()
