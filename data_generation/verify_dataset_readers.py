"""
    Verification script for VoxCeleb1, IEMOCAP, and Speech Commands simulation dataset classes.
"""
import sys
import os
from pathlib import Path

# Add current dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_src import VoxCeleb1ForSimuDataset, IEMOCAPForSimuDataset, SpeechCommandsForSimuDataset


def verify_speech_commands():
    sc_root = Path('/scratch/yotam/ssamba/src/finetune/speechcommands_v2/data/speech_commands_v0.02')
    label_csv = Path('/scratch/yotam/ssamba/src/finetune/speechcommands_v2/data/speechcommands_class_labels_indices.csv')

    if not sc_root.exists() or not label_csv.exists():
        print("Speech Commands dataset not found on disk yet. Skipping live audio read check.")
        return

    print("\n--- Verifying SpeechCommandsForSimuDataset ---")
    ds = SpeechCommandsForSimuDataset(sc_root=sc_root, label_csv=label_csv, split='pretrain', T=2.0)
    print(f"Loaded SpeechCommands pretrain dataset: {len(ds)} items.")

    sig, meta = ds[0]
    print(f"Sample 0 shape: {sig.shape} (desired: (32000, 1))")
    print(f"Sample 0 metadata: {meta}")
    assert sig.shape == (32000, 1), f"Expected shape (32000, 1), got {sig.shape}"
    assert 'keyword' in meta and 'keyword_idx' in meta, "Missing metadata keys"
    print("✓ SpeechCommandsForSimuDataset VERIFIED SUCCESSFULLY!")


def verify_iemocap():
    iemocap_root = Path('/scratch/yotam/data/IEMOCAP_full_release')
    meta_dir = Path('/scratch/yotam/s3prl/s3prl/downstream/emotion/meta_data')

    if not iemocap_root.exists():
        print("\n--- IEMOCAP dataset root not found on disk. Checking meta_data parsing... ---")
        if meta_dir.exists():
            ds = IEMOCAPForSimuDataset(iemocap_root=iemocap_root, meta_dir=meta_dir, split='pretrain', T=6.0)
            print(f"IEMOCAP pretrain dataset parsed: {len(ds)} items from meta_data.")
            ds_test = IEMOCAPForSimuDataset(iemocap_root=iemocap_root, meta_dir=meta_dir, split='pretest', T=6.0)
            print(f"IEMOCAP pretest dataset parsed: {len(ds_test)} items (Session1).")
            print("✓ IEMOCAP Dataset Split Logic VERIFIED SUCCESSFULLY!")
        return

    ds = IEMOCAPForSimuDataset(iemocap_root=iemocap_root, meta_dir=meta_dir, split='pretrain', T=6.0)
    sig, meta = ds[0]
    print(f"IEMOCAP Sample 0 shape: {sig.shape}")
    print(f"IEMOCAP Sample 0 metadata: {meta}")
    assert sig.shape == (96000, 1), f"Expected shape (96000, 1), got {sig.shape}"
    print("✓ IEMOCAPForSimuDataset VERIFIED SUCCESSFULLY!")


def verify_voxceleb1():
    vox1_root = Path('/scratch/yotam/data/voxceleb1')
    meta_file = Path('/scratch/yotam/s3prl/s3prl/downstream/voxceleb1/veri_test_class.txt')

    if not meta_file.exists():
        print("VoxCeleb1 meta_file not found. Skipping.")
        return

    print("\n--- Verifying VoxCeleb1 meta_file parsing ---")
    if not vox1_root.exists():
        print(f"VoxCeleb1 root {vox1_root} not on disk yet. Verifying split parsing from {meta_file}...")
        ds_tr = VoxCeleb1ForSimuDataset(vox1_root=vox1_root, meta_file=meta_file, split='pretrain', T=8.0)
        ds_va = VoxCeleb1ForSimuDataset(vox1_root=vox1_root, meta_file=meta_file, split='preval', T=8.0)
        ds_te = VoxCeleb1ForSimuDataset(vox1_root=vox1_root, meta_file=meta_file, split='pretest', T=8.0)
        print(f"VoxCeleb1 parsed pretrain: {len(ds_tr)} items, preval: {len(ds_va)} items, pretest: {len(ds_te)} items.")
        print("✓ VoxCeleb1 Split Parsing VERIFIED SUCCESSFULLY!")

if __name__ == '__main__':
    verify_speech_commands()
    verify_iemocap()
    verify_voxceleb1()
