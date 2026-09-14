#!/usr/bin/env python3
import json
import argparse
import os
import sys
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Ensure package imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sac.acoustic_features import extract_acoustic_features
from binaural_dual_stream_sac.spatial_features import extract_spatial_features

def process_single_item(datum, features_list="f0,hnr,centroid,flux,zcr"):
    wav_path = datum['wav']
    try:
        waveform, sr = torchaudio.load(wav_path)
        # Ensure 2 channels
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]
            
        # Average to mono for acoustic feature extraction
        mono_waveform = waveform.mean(dim=0, keepdim=True)
        
        # Extract features (unnormalized)
        mono_feats_tensor = extract_acoustic_features(
            mono_waveform, 
            sample_rate=sr, 
            normalize=False, 
            features_list=features_list
        )
        
        # Squeeze to 1D and convert to list of floats for JSON
        mono_feats = [float(x) for x in mono_feats_tensor.squeeze(0).numpy()]
        
        # Extract spatial features
        spatial_feats_array = extract_spatial_features(waveform, sr=sr)
        spatial_feats = [float(x) for x in spatial_feats_array]

        # Add to datum
        datum['acoustic_features'] = mono_feats
        datum['spatial_features'] = spatial_feats
        return {'status': 'ok', 'datum': datum}
    except Exception as e:
        # Mark clearly instead of silent fallback to zeros
        datum['acoustic_features'] = None
        datum['spatial_features'] = None
        return {'status': 'error', 'wav': wav_path, 'error': str(e), 'datum': datum}

def main():
    parser = argparse.ArgumentParser(description="Precompute monaural and spatial acoustic features for SSAMBA datasets.")
    parser.add_argument('--input_json', type=str, required=True, help="Path to input dataset JSON file.")
    parser.add_argument('--output_json', type=str, required=True, help="Path to output dataset JSON file.")
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help="Number of parallel workers.")
    parser.add_argument('--features_list', type=str, default="f0,hnr,centroid,flux,zcr", help="Comma-separated feature list.")
    args = parser.parse_args()

    print(f"Loading {args.input_json}...")
    with open(args.input_json, 'r') as f:
        data_json = json.load(f)
        
    data = data_json.get('data', [])
    if not data:
        print("No 'data' field found in JSON or data is empty.")
        return

    print(f"Processing {len(data)} items with {args.num_workers} workers...")
    
    # Process items in parallel
    process_func = partial(process_single_item, features_list=args.features_list)
    
    # Using Pool for multiprocessing
    processed_data = []
    failures = []
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap(process_func, data), total=len(data)):
            processed_data.append(result['datum'])
            if result['status'] == 'error':
                failures.append(result)
            
    # Update json and save
    data_json['data'] = processed_data
    
    # Report failures
    if failures:
        fail_pct = len(failures) / len(data) * 100
        print(f"\nWARNING: {len(failures)} files failed feature extraction ({fail_pct:.2f}%).")
        print("First 20 failures:")
        for fail in failures[:20]:
            print(f"  {fail['wav']} -> {fail['error']}")
        
        if fail_pct > 0.5:
            print(f"\nLOUD WARNING: More than 0.5% ({fail_pct:.2f}%) of the dataset failed extraction!")
            print("Please investigate these files before training.")
    else:
        print("\nAll files processed successfully (0 failures).")
    
    print(f"\nSaving to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(data_json, f, indent=2)
        
    print("Done!")

if __name__ == '__main__':
    # Disable gradient tracking globally for this script
    torch.set_grad_enabled(False)
    main()

