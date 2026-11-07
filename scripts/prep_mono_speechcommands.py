"""Speech Commands v2 manifests for run_sc_amba.sh, same split and format as prep_sc.py, without its download step.
Usage: prep_mono_speechcommands.py <dataset dir> <class labels csv> <output dir>"""
import csv
import json
import os
import sys

root, labels_csv, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
with open(labels_csv) as f:
    label_map = {row['display_name']: row['index'] for row in csv.DictReader(f)}


def read_list(name):
    with open(os.path.join(root, name)) as f:
        return [line.strip() for line in f if line.strip()]


val, test = read_list('validation_list.txt'), read_list('testing_list.txt')
held_out = set(val) | set(test)
train = [f'{c}/{w}' for c in sorted(os.listdir(root)) if c != '_background_noise_' and os.path.isdir(os.path.join(root, c))
         for w in sorted(os.listdir(os.path.join(root, c))) if f'{c}/{w}' not in held_out]
for name, files in (('train', train), ('valid', val), ('eval', test)):
    data = [{'wav': os.path.join(root, f), 'labels': '/m/spcmd' + label_map[f.split('/')[0]].zfill(2)} for f in files]
    with open(os.path.join(out, f'speechcommand_{name}_data.json'), 'w') as f:
        json.dump({'data': data}, f, indent=1)
    print(name, len(data), 'samples')
