"""
    Checks the binaural IEMOCAP manifests against s3prl's emotion meta_data before any evaluation uses them:
      - every utterance of every fold (Session1-5 train/test) has exactly one simulation
      - the simulated label matches the meta_data label
      - each simulated wav is 2-channel, 16 kHz, and as long as the original utterance (no truncation/padding)
    Exits non-zero on any failure. Reads wav headers only (no torch).

    Usage:
        python verify_binaural_iemocap.py --data_root <dir with iemocap_spatial_*.json> \
            --meta_dir <s3prl>/s3prl/downstream/emotion/meta_data --iemocap_root <IEMOCAP_full_release>
"""
import glob
import json
import os
import sys
from argparse import ArgumentParser

import soundfile

FS = 16000
JSON_DATA_ROOT = "/storage/yotam/ssamba/data"


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--meta_dir', required=True)
    parser.add_argument('--iemocap_root', required=True)
    args = parser.parse_args()
    errors = []

    sims = {}
    for split in ['pretrain', 'preval', 'pretest']:
        with open(os.path.join(args.data_root, f'iemocap_spatial_{split}.json')) as f:
            for item in json.load(f)['data']:
                if item['orig_rel_path'] in sims:
                    errors.append(f"duplicate simulation: {item['orig_rel_path']}")
                if not os.path.exists(item['wav']):  # json written on another machine
                    item['wav'] = item['wav'].replace(JSON_DATA_ROOT, args.data_root.rstrip('/'), 1)
                sims[item['orig_rel_path']] = item
    print(f'{len(sims)} simulated utterances')

    meta = {}
    for path in sorted(glob.glob(os.path.join(args.meta_dir, 'Session*', '*_meta_data.json'))):
        with open(path) as f:
            for info in json.load(f)['meta_data']:
                meta[info['path']] = info['label']
    print(f'{len(meta)} utterances in meta_data')

    missing = sorted(set(meta) - set(sims))
    extra = sorted(set(sims) - set(meta))
    if missing:
        errors.append(f'{len(missing)} meta_data utterances have no simulation, e.g. {missing[0]}')
    if extra:
        errors.append(f'{len(extra)} simulations are not in meta_data, e.g. {extra[0]}')

    for rel_path, item in sims.items():
        if rel_path in meta and item['emotion'] != meta[rel_path]:
            errors.append(f"label mismatch for {rel_path}: {item['emotion']} vs {meta[rel_path]}")
        sim = soundfile.info(item['wav'])
        orig = soundfile.info(os.path.join(args.iemocap_root, rel_path))
        orig_frames = round(orig.frames * FS / orig.samplerate)
        if sim.channels != 2 or sim.samplerate != FS:
            errors.append(f'{rel_path}: {sim.channels} channels at {sim.samplerate} Hz')
        if sim.frames != orig_frames:
            errors.append(f'{rel_path}: {sim.frames} samples, original has {orig_frames}')

    if errors:
        print(f'FAILED with {len(errors)} errors, first ones:')
        print('\n'.join(errors[:20]))
        sys.exit(1)
    print('OK: every fold utterance has exactly one full-length binaural simulation with the right label')


if __name__ == '__main__':
    main()
