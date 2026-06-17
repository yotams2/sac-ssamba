import os
import yaml

wandb_dir = "/storage/yotam/ssamba/src/sac/wandb"
new_run_id = "8usxac1h"
old_run_id = None

# Find the old run's wandb ID
for run_folder in os.listdir(wandb_dir):
    if not run_folder.startswith("run-"): continue
    config_path = os.path.join(wandb_dir, run_folder, "files", "config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                conf = yaml.safe_load(f)
            exp_dir = conf.get("exp_dir", {}).get("value", "")
            if "feat_universal_fixed_norm_layer-med0-librispeech" in exp_dir:
                old_run_id = run_folder.split("-")[-1]
                break
        except Exception:
            pass

if not old_run_id:
    print("Could not find the older run in wandb logs.")
else:
    print(f"Found older run ID: {old_run_id}")
    
    def extract_losses(run_id):
        losses = []
        run_folder = [f for f in os.listdir(wandb_dir) if f.endswith(run_id)][0]
        log_file = os.path.join(wandb_dir, run_folder, "files", "output.log")
        
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f:
                    if "Loss:" in line and "SAC:" in line:
                        # Extract the SAC and Recon parts
                        parts = line.strip().split()
                        try:
                            sac_idx = parts.index("SAC:")
                            recon_idx = parts.index("Recon:")
                            losses.append({
                                'step': len(losses) + 1,
                                'sac': float(parts[sac_idx+1]),
                                'recon': float(parts[recon_idx+1])
                            })
                            if len(losses) >= 5: break
                        except ValueError:
                            pass
        return losses

    new_losses = extract_losses(new_run_id)
    old_losses = extract_losses(old_run_id)

    print("\n--- Newer Run (mode=sqrt_dim) ---")
    for l in new_losses: print(f"Log Step {l['step']}: SAC Loss = {l['sac']:.4f}, Recon Loss = {l['recon']:.4f}")
    
    print("\n--- Older Run (med0) ---")
    for l in old_losses: print(f"Log Step {l['step']}: SAC Loss = {l['sac']:.4f}, Recon Loss = {l['recon']:.4f}")
    
    if new_losses and old_losses:
        if [l['sac'] for l in new_losses] == [l['sac'] for l in old_losses]:
            print("\n✅ MATCH! The losses are identical. The local_sigma calculations are exactly the same.")
        else:
            print("\n❌ MISMATCH! The losses differ. The local_sigma calculation or other logic changed.")
