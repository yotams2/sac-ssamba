"""
    Generate spatial dataset JSON manifests for downstream fine-tuning.

    Usage:
        python generate_downstream_manifests.py --dataset voxceleb1 --simu_dir /scratch/yotam/ssamba/data/VoxCeleb1/simu --out_dir /scratch/yotam/ssamba/data
        python generate_downstream_manifests.py --dataset iemocap --simu_dir /scratch/yotam/ssamba/data/IEMOCAP/simu --out_dir /scratch/yotam/ssamba/data
        python generate_downstream_manifests.py --dataset speechcommands --simu_dir /scratch/yotam/ssamba/data/SpeechCommands/simu --out_dir /scratch/yotam/ssamba/data
"""
import os
import json
import glob
import numpy as np
from pathlib import Path
from jsonargparse import ArgumentParser


def generate_manifests(dataset: str, simu_dir: str, out_dir: str):
    simu_path = Path(simu_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = ['pretrain', 'preval', 'pretest']

    for split in splits:
        split_dir = simu_path / split
        if not split_dir.exists():
            print(f"Skipping split '{split}' - directory not found: {split_dir}")
            continue

        info_files = sorted(glob.glob(str(split_dir / '*_info.npz')))
        if not info_files:
            print(f"No info files found in {split_dir}")
            continue

        print(f"Processing {len(info_files)} files for {dataset} ({split})...")

        items = []
        for info_file in info_files:
            idx_str = Path(info_file).name.replace('_info.npz', '')
            wav_path = str(split_dir / f"{idx_str}.wav")

            if not os.path.exists(wav_path):
                continue

            data = dict(np.load(info_file, allow_pickle=True))

            item = {'wav': wav_path}

            if dataset == 'voxceleb1':
                item['speaker'] = str(data.get('speaker_id', ''))
                item['label'] = int(data.get('label_idx', -1))
                item['orig_rel_path'] = str(data.get('orig_rel_path', ''))
            elif dataset == 'iemocap':
                item['emotion'] = str(data.get('emotion_label', ''))
                item['label'] = int(data.get('emotion_idx', -1))
                item['speaker'] = str(data.get('speaker_id', ''))
                item['session'] = str(data.get('session', ''))
                item['orig_rel_path'] = str(data.get('orig_rel_path', ''))
            elif dataset == 'speechcommands':
                keyword = str(data.get('keyword', ''))
                label_idx = int(data.get('keyword_idx', -1))
                item['keyword'] = keyword
                item['label'] = label_idx
                item['labels'] = f"/m/spcmd{str(label_idx).zfill(2)}"
                item['file_id'] = str(data.get('file_id', ''))
                item['orig_rel_path'] = str(data.get('orig_rel_path', ''))
            else:
                # Generic fallback
                for k, v in data.items():
                    if isinstance(v, (str, int, float, np.integer, np.floating)):
                        item[k] = v.item() if hasattr(v, 'item') else v

            items.append(item)

        out_json = out_path / f"{dataset}_spatial_{split}.json"
        with open(out_json, 'w') as f:
            json.dump({'data': items}, f, indent=2)
        print(f"Saved {len(items)} entries to {out_json}")


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate spatial dataset JSON manifests')
    parser.add_argument('--dataset', type=str, required=True, choices=['voxceleb1', 'iemocap', 'speechcommands'])
    parser.add_argument('--simu_dir', type=str, required=True, help="Root directory containing pretrain/preval/pretest")
    parser.add_argument('--out_dir', type=str, default='/scratch/yotam/ssamba/data', help="Output directory for manifests")
    args = parser.parse_args()
    generate_manifests(args.dataset, args.simu_dir, args.out_dir)
