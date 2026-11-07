#!/usr/bin/env python3
"""Rewrite a mono LibriSpeech manifest to this cluster's paths and check that every file exists.

usage: localize_librispeech_manifest.py <input.json> <output.json> <LibriSpeech dir>
The mono loaders use `wav` as it stands and fall back to silence on a failed load, so a wrong path must stop the job here.
"""
import json, os, sys

src, dst, root = sys.argv[1:4]
manifest = json.load(open(src))
missing = []
for d in manifest['data']:
    marker = 'LibriSpeech/'
    if marker not in d['wav']:
        sys.exit(f"no {marker} in path: {d['wav']}")
    d['wav'] = os.path.join(root, d['wav'].split(marker, 1)[1])
    if not os.path.isfile(d['wav']):
        missing.append(d['wav'])
print(f"{src}: {len(manifest['data'])} clips, {len(missing)} missing, e.g. {manifest['data'][0]['wav']}")
if missing:
    sys.exit(f'missing files, first: {missing[:5]}')
with open(dst + '.tmp', 'w') as f:
    json.dump(manifest, f, indent=1)
os.replace(dst + '.tmp', dst)
