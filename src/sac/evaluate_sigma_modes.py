import os
import sys
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

# Add parent paths so we can import SSAMBA modules
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from acoustic_features import extract_acoustic_features, get_feature_families, calculate_acoustic_feature_stats
from run_pretrain_sac import AudioDatasetWithWaveform

def evaluate_sigma_modes():
    print("Initializing...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    batch_size = 256 # Large batch to get meaningful histograms
    dataset_json = "/storage/yotam/ssamba/librispeech_train.json"
    sample_rate = 16000
    sac_features = "f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm"
    sac_sigma = 1.0

    if not os.path.exists(dataset_json):
        print(f"Error: Dataset not found at {dataset_json}. Please update the path in the script.")
        return

    audio_conf = {
        'num_mel_bins': 128, 'target_length': 1024,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'dataset': 'librispeech', 'mode': 'evaluation',
        'mean': -4.2677393, 'std': 4.5689974, 'noise': False
    }

    dataset = AudioDatasetWithWaveform(dataset_json, audio_conf=audio_conf, target_sr=sample_rate, label_csv="/storage/yotam/ssamba/src/finetune/audioset/data/class_labels_indices.csv")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 1. Pre-calculate global stats on a few batches (as done in pretraining)
    print("Calculating global feature stats for offline modes...")
    feature_stats = calculate_acoustic_feature_stats(
        dataloader, sample_rate=sample_rate, num_batches=10, 
        device=device, features_list=sac_features
    )
    if feature_stats:
        feature_stats['mean'] = feature_stats['mean'].to(device)
        feature_stats['std'] = feature_stats['std'].to(device)
    else:
        print("Failed to compute stats.")
        return

    offline_medians = feature_stats['family_medians']

    # 2. Get one large batch to evaluate
    print(f"\nLoading a single batch of {batch_size} samples...")
    batch = next(iter(dataloader))
    waveform = batch[1].to(device)

    # 3. Extract features
    print("Extracting acoustic features...")
    with torch.no_grad():
        c = extract_acoustic_features(
            waveform, sample_rate=sample_rate, 
            feature_stats=feature_stats, features_list=sac_features
        )
    
    families, _ = get_feature_families(sac_features)
    
    modes = ['dynamic_batch_median', 'offline_global_median', 'chi2_median', 'sqrt_dim']
    chi2_medians = {1: 0.455, 2: 1.386, 3: 2.366, 4: 3.357, 5: 4.351, 6: 5.348, 7: 6.346}

    # Prepare table
    print("\n--- Sigma (Bandwidth) Table ---")
    header = f"{'Family':<15} | {'D':<3} | " + " | ".join([f"{m:<22}" for m in modes])
    print(header)
    print("-" * len(header))

    # Prepare plots
    fig, axes = plt.subplots(len(families), len(modes), figsize=(5 * len(modes), 4 * len(families)))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, (fam_name, indices) in enumerate(families.items()):
        D = len(indices)
        c_fam = c[:, indices]
        
        # Calculate pairwise distances for this batch
        dist = torch.cdist(c_fam, c_fam, p=2)
        diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        off_diag_dist = dist[~diag_mask]
        
        batch_median = off_diag_dist.median().item()
        global_median = offline_medians.get(fam_name, batch_median)

        sigma_vals = []

        for j, mode in enumerate(modes):
            # Calculate local_sigma according to the formula
            if mode == 'dynamic_batch_median':
                local_sigma = batch_median / math.sqrt(math.log(2.0))
            elif mode == 'offline_global_median':
                local_sigma = global_median / math.sqrt(math.log(2.0))
            elif mode == 'chi2_median':
                local_sigma = sac_sigma * math.sqrt(chi2_medians.get(D, D - 2/3))
            elif mode == 'sqrt_dim':
                local_sigma = sac_sigma * math.sqrt(D)

            sigma_vals.append(local_sigma)

            # Calculate weights
            w = torch.exp(-(off_diag_dist / local_sigma) ** 2)
            w_np = w.cpu().numpy()

            # Plot histogram
            ax = axes[i, j]
            ax.hist(w_np, bins=50, color='royalblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xlim(0, 1)
            if i == 0:
                ax.set_title(f"{mode}")
            if j == 0:
                ax.set_ylabel(f"{fam_name} (D={D})\nCounts", fontsize=12, fontweight='bold')
            
            # Add text with stats
            mean_w = w_np.mean()
            med_w = np.median(w_np)
            std_w = np.std(w_np)
            ax.text(0.05, 0.95, f"$\sigma={local_sigma:.3f}$\nMean w={mean_w:.3f}\nMed w={med_w:.3f}\nStd={std_w:.3f}", 
                    transform=ax.transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Print table row
        row = f"{fam_name:<15} | {D:<3} | " + " | ".join([f"{val:<22.4f}" for val in sigma_vals])
        print(row)

    plt.suptitle("Histogram of Target Weights ($w_{ij}$) Across Different Modes", fontsize=16, fontweight='bold')
    
    # Create figures directory
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    out_path = os.path.join(figures_dir, "sigma_mode_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved histograms to {out_path}")

if __name__ == "__main__":
    evaluate_sigma_modes()
