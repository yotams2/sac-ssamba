#!/usr/bin/env python3
"""Back up experiment outputs from /scratch200 (scratch: wiped when unused) to /moshe-data (long term) and keep a
compact copy of the lightweight records in home.

Run it through backup.sbatch (needs a node that mounts /moshe-data). Stdlib only: no torch, no venv.

Every file under the roots in ROOTS is classified once per run; the decisions are written to a manifest
(<records>/manifest.tsv, also copied to moshe), so what is and isn't protected can be reviewed.

  light   records: json, jsonl, txt, log, out, csv, yaml, md, pkl, npz/npy < 5 MB, tensorboard events, configs.
          -> moshe mirror (all of them) and home records (logs xz-compressed). Cheap, so all runs, smoke runs too.
  weights model checkpoints (.pth .pt .tar .ckpt .bin .safetensors). Kept when they cannot be regenerated cheaply:
          - pretraining runs (a path segment containing 'pretrain'): the run's canonical weights, which is
            audio_model.final.pth, or best_audio_model.pth if there is no final yet. Days of GPU, and every downstream
            number depends on them.
          - a pretraining run with no final weights yet (still training, preempted, or died): its resume state
            (latest.pth / resume.pth), so a scratch purge mid-run costs at most one backup interval.
          - everything else is skipped: downstream/fine-tuned weights (hours to regenerate from the backed-up
            upstream + the recipe in the light files), per-epoch snapshots, optimizer states, nan dumps,
            the resume state of a finished run, smoke/test runs (a path segment starting with '_' or 'exp_smoke').
  manifest the top-level *.json* of sac-ssamba/data (feature manifests computed here): always, moshe only.
  heavy   other files >= 5 MB (embeddings, features dumps, plots in bulk): skipped, listed in the manifest.

Overrides, one glob per line relative to /scratch200/yotams2 (fnmatch, '*' crosses '/'), '#' comments for the reason:
  keep.txt  back up matching files whatever the class (e.g. a fine-tuned model behind a paper figure).
  skip.txt  never back up matching files (e.g. a superseded run).
keep wins over skip.

Nothing is ever deleted, on scratch or on moshe. Files already on moshe with the same size and mtime are not copied.
"""
import argparse
import fnmatch
import lzma
import os
import shutil
import subprocess
import sys
import time

SCRATCH = '/scratch200/yotams2'
MOSHE = '/moshe-data/blaufer-data/yotams2/scratch200-backup'
RECORDS = os.path.expanduser('~/exp-records')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = [  # (absolute path, name in the backup); data dirs are not here: sources are on moshe already
    (f'{SCRATCH}/sac-ssamba/exp', 'sac-ssamba/exp'),
    (f'{SCRATCH}/SAC-SAR-SSL/exp', 'SAC-SAR-SSL/exp'),
    (f'{SCRATCH}/SAC-SAR-SSL/exp_smoke', 'SAC-SAR-SSL/exp_smoke'),
    (f'{SCRATCH}/logs', 'logs'),
    (os.path.expanduser('~/sac-ssamba/logs'), 'home/sac-ssamba/logs'),
]
LIGHT_EXT = {'.json', '.jsonl', '.txt', '.log', '.out', '.err', '.csv', '.tsv', '.yaml', '.yml', '.md', '.pkl',
             '.npz', '.npy', '.mat', '.png', '.pdf', '.svg'}
WEIGHT_EXT = {'.pth', '.pt', '.tar', '.ckpt', '.bin', '.safetensors'}
LIGHT_MAX = 5 * 2**20       # larger "light" files count as heavy
HOME_MAX = 2 * 2**20        # largest record copied to home (after compression for logs)
HOME_EXT = {'.json', '.jsonl', '.txt', '.log', '.out', '.err', '.csv', '.tsv', '.yaml', '.yml', '.md', '.pkl'}
COMPRESS_EXT = {'.log', '.out', '.err'}
RESUME_NAMES = {'latest.pth', 'resume.pth', 'latest_model.tar'}
# canonical weights, in order of preference (SSAMBA final, SSAMBA best, SAR-SSL best). Only audio_model.final.pth
# marks a finished run; SAR-SSL writes no such marker, so its resume state is always kept (one small file per run).
FINAL_NAMES = ['audio_model.final.pth', 'best_audio_model.pth', 'best_model.tar']
# Data: only the top-level manifests of sac-ssamba/data (features computed on this cluster, hours of CPU each,
# ~200 MB in all). The audio itself came from /moshe-data (data/, sac-ssamba/) and is restored from there.
DATA_MANIFESTS = (f'{SCRATCH}/sac-ssamba/data', 'sac-ssamba/data')


def read_globs(path):
    if not os.path.isfile(path):
        return []
    return [l.split('#')[0].strip() for l in open(path) if l.split('#')[0].strip()]


def is_smoke(rel):
    return any(p.startswith('_') or p == 'exp_smoke' for p in rel.split('/')[:-1])


def classify(rel, name, size, siblings, pretrain):
    """ -> (class, keep, reason) """
    ext = os.path.splitext(name)[1].lower()
    if 'events.out.tfevents' in name:
        return 'light', True, 'tensorboard'
    if ext in WEIGHT_EXT:
        if is_smoke(rel):
            return 'weights', False, 'smoke/test run'
        if not pretrain:
            return 'weights', False, 'downstream weights: cheap to regenerate'
        canonical = next((f for f in FINAL_NAMES if f in siblings), None)
        if name == canonical:
            return 'weights', True, 'pretrain canonical weights'
        if name in RESUME_NAMES:
            if FINAL_NAMES[0] not in siblings:  # best_audio_model.pth alone doesn't mean the run is over
                return 'weights', True, 'pretrain resume state, run not finished'
            return 'weights', False, 'resume state of a finished run'
        return 'weights', False, 'snapshot/optimizer/other'
    if ext in LIGHT_EXT and size <= LIGHT_MAX:
        return 'light', True, 'record'
    if size <= LIGHT_MAX and ext not in WEIGHT_EXT:
        return 'light', True, 'small file'
    return 'heavy', False, 'large non-weight file'


def scan(keep_globs, skip_globs):
    rows = []
    for root, alias in ROOTS:
        if not os.path.isdir(root):
            continue
        for d, dirs, files in os.walk(root):
            dirs.sort()
            reld = os.path.relpath(d, root)
            pretrain = any('pretrain' in p for p in [alias] + reld.split('/'))
            sib = set(files)
            for f in sorted(files):
                p = os.path.join(d, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                rel = os.path.normpath(os.path.join(alias, reld, f))
                cls, keep, why = classify(rel, f, st.st_size, sib, pretrain)
                if any(fnmatch.fnmatch(rel, g) for g in skip_globs):
                    keep, why = False, 'skip.txt'
                if any(fnmatch.fnmatch(rel, g) for g in keep_globs):
                    keep, why = True, 'keep.txt'
                rows.append(dict(src=p, rel=rel, size=st.st_size, mtime=st.st_mtime, atime=st.st_atime,
                                 cls=cls, keep=keep, why=why))
    root, alias = DATA_MANIFESTS
    for f in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, f)
        if '.json' in f and os.path.isfile(p):
            st = os.stat(p)
            rel = f'{alias}/{f}'
            keep, why = True, 'data manifest (features computed here)'
            if any(fnmatch.fnmatch(rel, g) for g in skip_globs):
                keep, why = False, 'skip.txt'
            rows.append(dict(src=p, rel=rel, size=st.st_size, mtime=st.st_mtime, atime=st.st_atime,
                             cls='manifest', keep=keep, why=why))
    return rows


def needs_copy(r, dst):
    try:
        st = os.stat(dst)
    except OSError:
        return True
    return st.st_size != r['size'] or int(st.st_mtime) != int(r['mtime'])


def copy_moshe(rows, dry):
    todo = [r for r in rows if r['keep'] and needs_copy(r, os.path.join(MOSHE, r['rel']))]
    nbytes = sum(r['size'] for r in todo)
    print(f'[moshe] {len(todo)} files, {nbytes / 2**30:.2f} GB to copy', flush=True)
    if dry or not todo:
        return todo
    for r in todo:
        dst = os.path.join(MOSHE, r['rel'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + '.part'
        t0 = time.time()
        shutil.copyfile(r['src'], tmp)
        shutil.copystat(r['src'], tmp)
        os.replace(tmp, dst)
        if r['size'] > 100 * 2**20:
            print(f'  {r["rel"]}  {r["size"] / 2**30:.2f} GB  {time.time() - t0:.0f} s', flush=True)
    return todo


def copy_home(rows, dry):
    n = nb = 0
    for r in rows:
        if r['cls'] != 'light' or not r['keep']:  # manifests and weights: moshe only
            continue
        ext = os.path.splitext(r['rel'])[1].lower()
        if ext not in HOME_EXT or r['rel'].startswith('home/'):
            continue  # home/... is home already
        xz = ext in COMPRESS_EXT and r['size'] > 64 * 2**10
        dst = os.path.join(RECORDS, r['rel'] + ('.xz' if xz else ''))
        if os.path.exists(dst) and os.stat(dst).st_mtime >= r['mtime']:
            continue
        if not xz and r['size'] > HOME_MAX:
            continue
        n += 1
        if dry:
            nb += r['size'] if not xz else r['size'] // 20
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if xz:
            with open(r['src'], 'rb') as fi:
                data = fi.read().replace(b'\r', b'\n')  # tqdm redraws -> lines, they compress to almost nothing
            with lzma.open(dst + '.part', 'wb', preset=6) as fo:
                fo.write(data)
            if os.path.getsize(dst + '.part') > HOME_MAX:
                os.remove(dst + '.part')
                n -= 1
                continue
            os.replace(dst + '.part', dst)
        else:
            shutil.copyfile(r['src'], dst)
        os.utime(dst, (r['mtime'], r['mtime']))
        nb += os.path.getsize(dst)
    print(f'[home] {n} records {"would be " if dry else ""}written to {RECORDS} ({nb / 2**20:.1f} MB)', flush=True)


def write_manifest(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.part', 'w') as f:
        f.write('# generated by sac-ssamba/scripts/backup/backup.py ' + time.strftime('%Y-%m-%d %H:%M') + '\n')
        f.write('keep\tclass\tsize_mb\tmtime\tatime\treason\tpath\n')
        for r in rows:
            f.write(f"{int(r['keep'])}\t{r['cls']}\t{r['size'] / 2**20:.1f}\t"
                    f"{time.strftime('%Y-%m-%d', time.localtime(r['mtime']))}\t"
                    f"{time.strftime('%Y-%m-%d', time.localtime(r['atime']))}\t{r['why']}\t{r['rel']}\n")
    os.replace(path + '.part', path)


def summary(rows):
    by = {}
    for r in rows:
        k = (r['cls'], r['keep'], r['why'])
        n, s = by.get(k, (0, 0))
        by[k] = (n + 1, s + r['size'])
    print(f'{"class":8} {"keep":5} {"files":>7} {"GB":>8}  reason')
    for (c, k, w), (n, s) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f'{c:8} {str(k):5} {n:7d} {s / 2**30:8.2f}  {w}')
    print('weights kept:')
    for r in rows:
        if r['cls'] == 'weights' and r['keep']:
            print(f'  {r["size"] / 2**30:6.2f} GB  {r["why"]:45}  {r["rel"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true', help='classify and report, copy nothing')
    ap.add_argument('--no-home', action='store_true', help='do not write home records')
    args = ap.parse_args()
    if not args.dry_run and not os.path.isdir(os.path.dirname(MOSHE)):
        sys.exit(f'{os.path.dirname(MOSHE)} is not mounted on this node')
    rows = scan(read_globs(os.path.join(HERE, 'keep.txt')), read_globs(os.path.join(HERE, 'skip.txt')))
    summary(rows)
    copy_moshe(rows, args.dry_run)
    if not args.no_home:
        copy_home(rows, args.dry_run)
    if not args.dry_run:
        write_manifest(rows, os.path.join(RECORDS, 'manifest.tsv'))
        os.makedirs(MOSHE, exist_ok=True)
        shutil.copyfile(os.path.join(RECORDS, 'manifest.tsv'), os.path.join(MOSHE, 'manifest.tsv'))
        with open(os.path.join(RECORDS, 'LAST_BACKUP'), 'w') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M') + f' job {os.environ.get("SLURM_JOB_ID", "local")}\n')
    print('done', flush=True)


if __name__ == '__main__':
    main()
