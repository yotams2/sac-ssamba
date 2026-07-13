import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import builtins
import datetime
import json
from unittest.mock import patch

sys.path.insert(0, '/storage/yotam/ssamba/src')
sys.path.insert(0, '/storage/yotam/ssamba')
sys.path.insert(0, '/storage/yotam/ssamba/Vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/mamba-1p1p1')

from sac.run_pretrain_sac import main as run_pretrain
from sac.sigma_configs import OPTIMAL_SIGMAS

class StopSweepException(Exception):
    pass

def build_sys_argv(tau, out_dir):
    return [
        "run_pretrain_sac.py",
        "--sac-loss",
        "--dataset", "librispeech",
        "--data-train", "/storage/yotam/ssamba/librispeech_train.json",
        "--data-val", "/storage/yotam/ssamba/librispeech_eval.json",
        "--label-csv", "/storage/yotam/ssamba/src/finetune/audioset/data/class_labels_indices.csv",
        "--exp-dir", f"{out_dir}/exp_tau_{tau}",
        "--exp_name", f"tau_sweep_{tau}",
        "--lr", "1e-4",
        "--lr_patience", "2",
        "--n-epochs", "10",
        "--batch-size", "16",
        "--num-workers", "16",
        "--n-print-steps", "50",
        "--epoch_iter", "10000", # Set extremely high to avoid saving checkpoints
        "--diagnostic_steps", "100", # We want diagnostics frequently
        "--task", "pretrain_joint",
        "--mask_patch", "300",
        "--dataset_mean", "-4.2677393",
        "--dataset_std", "4.5689974",
        "--target_length", "1024",
        "--num_mel_bins", "128",
        "--sample_rate", "16000",
        "--model_size", "base",
        "--fshape", "16",
        "--tshape", "16",
        "--sac-lambda", "0.02",
        "--sac-temperature", str(tau),
        "--sac-sigma", "1.0",
        "--sac_features", "f0_mean,f0_var,formants,mfcc,hnr,centroid,flux,zcr_mean,rhythm",
        "--local_sigma_mode", "static_entropy_optimal",
        "--use_cross_attention", "true",
        "--num_queries_per_group", "1",
        "--proj-dim", "128"
    ]

def run_sweep():
    tau_values = [0.15, 0.2, 0.3]
    max_steps = 4000
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/storage/yotam/ssamba/src/metrics/tau_search/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save metadata so we can retrace what was used
    metadata = {
        "timestamp": timestamp,
        "tau_values_swept": tau_values,
        "max_steps_per_tau": max_steps,
        "local_sigma_mode": "static_entropy_optimal",
        "batch_size": 16,
        "sigma_values_used": OPTIMAL_SIGMAS
    }
    
    with open(os.path.join(out_dir, "sweep_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    results = {}
    
    # Store original functions to restore them between runs
    original_print = builtins.print
    original_torch_save = torch.save
    
    for tau in tau_values:
        print(f"\n{'='*50}\nStarting Sweep for tau = {tau}\n{'='*50}")
        
        # Initialize results dictionary for this tau
        tau_results = {
            'steps': [],
            'uniformity': [],  # global uniformity
            'alignment': [],   # global alignment
            'groups': {}       # per-group tracking
        }
        
        # Mock torch.save to completely suppress checkpointing
        def mock_torch_save(*args, **kwargs):
            pass
        
        # Mock print to intercept diagnostics and enforce max_steps
        def mock_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            original_print(*args, **kwargs)
            
            if "  [Diagnostics] Uniformity:" in text:
                try:
                    parts = text.split(',')
                    uni_str = parts[0].split('Uniformity:')[1].strip()
                    ali_str = parts[1].split('Alignment_Top10:')[1].strip()
                    tau_results['uniformity'].append(float(uni_str))
                    tau_results['alignment'].append(float(ali_str))
                    
                    # Compute step assuming diagnostic_steps=100
                    step = len(tau_results['uniformity']) * 100
                    tau_results['steps'].append(step)
                    
                    if step >= max_steps:
                        original_print(f"\n[Tau Sweep] Reached max_steps={max_steps}. Terminating run for tau={tau}.\n")
                        raise StopSweepException()
                except StopSweepException:
                    raise
                except Exception as e:
                    original_print(f"Error parsing diagnostics: {e}")
            
            import re
            match = re.match(r"\s+\[(.*?)\] Uniformity:\s*([-.\d]+),\s*Alignment:\s*([-.\d]+)", text)
            if match and "Diagnostics" not in text:
                group_name = match.group(1)
                uni_group = float(match.group(2))
                ali_group = float(match.group(3))
                
                if group_name not in tau_results['groups']:
                    tau_results['groups'][group_name] = {'uniformity': [], 'alignment': []}
                tau_results['groups'][group_name]['uniformity'].append(uni_group)
                tau_results['groups'][group_name]['alignment'].append(ali_group)
                    
        # Apply mocks
        builtins.print = mock_print
        torch.save = mock_torch_save
        
        sys.argv = build_sys_argv(tau, out_dir)
        
        try:
            run_pretrain()
        except StopSweepException:
            # Expected termination
            pass
        except Exception as e:
            original_print(f"Error during run for tau={tau}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            builtins.print = original_print
            
        # Store results
        results[tau] = tau_results
        
        # Save the tracking matrix progressively
        results_path = os.path.join(out_dir, "tau_sweep_results.npy")
        np.save(results_path, results)
        print(f"\nProgressively saved tracking matrix to {results_path}")
        
        # Plotting progressively
        plot_path = os.path.join(out_dir, "tau_trajectory_plot.png")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, t in enumerate(tau_values):
            if t not in results:
                continue
            steps = results[t]['steps']
            uni = results[t]['uniformity']
            align = results[t]['alignment']
            
            # Ensure array lengths match
            min_len = min(len(steps), len(uni), len(align))
            if min_len == 0:
                continue
            steps = steps[:min_len]
            uni = uni[:min_len]
            align = align[:min_len]
            
            color = colors[i % len(colors)]
            ax1.plot(steps, uni, label=f'$\\tau$ = {t}', color=color, linewidth=2)
            ax2.plot(steps, align, label=f'$\\tau$ = {t}', color=color, linewidth=2)

        ax1.set_title('Global Uniformity Trajectory', fontsize=14)
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Uniformity (Lower is Better)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        if ax1.get_legend_handles_labels()[1]:
            ax1.legend(fontsize=11)

        ax2.set_title('Global Alignment (Top 10%) Trajectory', fontsize=14)
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Alignment (Higher is Better)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        if ax2.get_legend_handles_labels()[1]:
            ax2.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Progressively saved trajectory plot to {plot_path}")
    
    # Restore originals
    torch.save = original_torch_save

if __name__ == "__main__":
    run_sweep()
