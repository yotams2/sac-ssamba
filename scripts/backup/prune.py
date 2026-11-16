#!/usr/bin/env python3
"""Delete regenerable weight files of FINISHED runs on /scratch200, and stale resume states on /moshe-data.

Runs from backup.sbatch right after a successful backup.py. Stdlib only. --dry-run lists, deletes nothing.

Only weight files (.pth .pt .tar .ckpt) are ever deleted, never records, never a run's canonical/best weights, never a
file matching keep.txt, never a file modified in the last MIN_AGE_H hours. A run counts as finished only by its own
end-of-run marker:

  S1 pretraining (path has 'pretrain'), marker models/audio_model.final.pth, which must also be on moshe with the same
     size: delete per-eval snapshots audio_model.<n>.pth and the resume state latest.pth / resume.pth.
     Kept: audio_model.final.pth, best_audio_model.pth.
  S2 s3prl downstream (IEMOCAP, SID), marker states-<total_steps>.ckpt (total_steps from the newest config_*.yaml):
     delete states-*.ckpt (resume states). Kept: dev-best.ckpt.
  S3 SC2 fine-tuning (run_amba.py), marker eval_result.csv (written after the last epoch): delete
     models/best_optim_state.pth and models/resume.pth. Kept: models/best_audio_model.pth.
  S4 SAR-SSL downstream <task>/<label>/<mode>-...-R<n>/, marker <task>/<label>/.train_done_<mode>_R<n> plus
     downstream_test_res_<mode>.json: delete model<epoch>.tar and latest_model.tar. Kept: best_model.tar, ensemble_model.tar.
  S5 smoke/test runs (a path segment starting with '_' or 'exp_smoke'): weights older than SMOKE_AGE_D days.
     The gate only needs _smoke/mono_ds/.passed, which is not a weight file.

  M1 moshe: the copy of a pretraining run's resume state (latest.pth / resume.pth), once the run's
     audio_model.final.pth is on moshe with the same size as on scratch. Nothing else on moshe is touched.

Every deletion is appended to ~/exp-records/prune_log.tsv.
"""
import argparse
import fnmatch
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backup import SCRATCH, MOSHE, RECORDS, HERE, ROOTS, WEIGHT_EXT, read_globs, is_smoke  # noqa: E402

MIN_AGE_H = 24
SMOKE_AGE_D = 7
RESUME = {'latest.pth', 'resume.pth'}
SNAPSHOT = re.compile(r'^audio_model\.\d+\.pth$')
LOG = os.path.join(RECORDS, 'prune_log.tsv')


def old(p, hours=MIN_AGE_H):
    return time.time() - os.stat(p).st_mtime > hours * 3600


def total_steps(d):
    cfgs = sorted(glob.glob(os.path.join(d, 'config_*.yaml')))
    if not cfgs:
        return None
    m = re.search(r'^\s*total_steps:\s*(\d+)', open(cfgs[-1]).read(), re.M)
    return int(m.group(1)) if m else None


def same_on_moshe(src, rel):
    try:
        return os.stat(os.path.join(MOSHE, rel)).st_size == os.stat(src).st_size
    except OSError:
        return False


def candidates(keep_globs):
    """-> list of (path on scratch, rel, rule)"""
    out = []
    for root, alias in ROOTS:
        if not root.startswith(SCRATCH) or not os.path.isdir(root):
            continue  # home logs root: nothing to prune
        for d, dirs, files in os.walk(root):
            dirs.sort()
            reld = os.path.normpath(os.path.join(alias, os.path.relpath(d, root)))
            fs = set(files)
            weights = [f for f in sorted(files) if os.path.splitext(f)[1].lower() in WEIGHT_EXT]
            if not weights:
                continue
            pick = []
            if is_smoke(reld + '/x'):
                pick = [(f, 'S5 smoke/test weights') for f in weights
                        if old(os.path.join(d, f), SMOKE_AGE_D * 24)]
            elif 'pretrain' in reld and 'audio_model.final.pth' in fs:
                fin = os.path.join(d, 'audio_model.final.pth')
                if old(fin) and same_on_moshe(fin, f'{reld}/audio_model.final.pth'):
                    pick = [(f, 'S1 pretrain snapshot/resume, final backed up') for f in weights
                            if SNAPSHOT.match(f) or f in RESUME]
            elif any(f.startswith('states-') for f in weights):
                t = total_steps(d)
                if t and f'states-{t}.ckpt' in fs and old(os.path.join(d, f'states-{t}.ckpt')):
                    pick = [(f, f's3prl resume states, finished at {t}') for f in weights
                            if re.match(r'^states-\d+\.ckpt$', f)]
                    pick = [(f, 'S2 ' + w) for f, w in pick]
            elif os.path.basename(d) == 'models' and os.path.isfile(os.path.join(d, '..', 'eval_result.csv')):
                if old(os.path.join(d, '..', 'eval_result.csv')):
                    pick = [(f, 'S3 SC2 optimizer/resume, run finished') for f in weights
                            if f in ('best_optim_state.pth', 'resume.pth')]
            else:
                m = re.match(r'^([a-zA-Z]+)-.*[-_]R(\d+)$', os.path.basename(d))
                if m:
                    up = os.path.dirname(d)
                    marks = {x.lower(): x for x in os.listdir(up)}  # dir 'scratchlow-...', marker '..._scratchLOW_R8'
                    mode = m.group(1).lower()
                    done = f'.train_done_{mode}_r{m.group(2)}' in marks
                    res = marks.get(f'downstream_test_res_{mode}.json')
                    if done and res and old(os.path.join(up, res), 1):
                        pick = [(f, 'S4 SAR-SSL epoch snapshots/latest, trained+tested') for f in weights
                                if re.match(r'^model\d+\.tar$', f) or f == 'latest_model.tar']
            for f, why in pick:
                rel = f'{reld}/{f}'
                if any(fnmatch.fnmatch(rel, g) for g in keep_globs):
                    continue
                out.append((os.path.join(d, f), rel, why))
    return out


def moshe_candidates():
    """M1: moshe copies of resume states of pretraining runs whose final weights are on moshe."""
    out = []
    for root, alias in ROOTS:
        if not root.startswith(SCRATCH) or not os.path.isdir(root):
            continue
        for d, dirs, files in os.walk(root):
            reld = os.path.normpath(os.path.join(alias, os.path.relpath(d, root)))
            if 'pretrain' not in reld or 'audio_model.final.pth' not in files:
                continue
            if not same_on_moshe(os.path.join(d, 'audio_model.final.pth'), f'{reld}/audio_model.final.pth'):
                continue
            for f in sorted(RESUME):
                mp = os.path.join(MOSHE, reld, f)
                if os.path.isfile(mp):
                    out.append((mp, f'{reld}/{f}', 'M1 moshe resume state, final backed up'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    moshe_ok = os.path.isdir(os.path.dirname(MOSHE))
    if not moshe_ok and not args.dry_run:
        sys.exit(f'{os.path.dirname(MOSHE)} not mounted: S1 and M1 need it, nothing deleted')
    todo = candidates(read_globs(os.path.join(HERE, 'keep.txt')))
    todo += moshe_candidates() if moshe_ok else []
    total = 0
    for p, rel, why in todo:
        sz = os.path.getsize(p)
        total += sz
        print(f'{"would delete" if args.dry_run else "delete"}  {sz / 2**30:6.2f} GB  {why:50}  {p}', flush=True)
    print(f'[prune] {len(todo)} files, {total / 2**30:.2f} GB{" (dry run)" if args.dry_run else ""}'
          f'{"" if moshe_ok else "; moshe not mounted, S1/M1 skipped"}', flush=True)
    if args.dry_run or not todo:
        return
    with open(LOG, 'a') as log:
        for p, rel, why in todo:
            sz = os.path.getsize(p)
            os.remove(p)
            log.write(f'{time.strftime("%Y-%m-%d %H:%M")}\t{os.environ.get("SLURM_JOB_ID", "local")}\t'
                      f'{sz / 2**20:.1f}\t{why}\t{p}\n')


if __name__ == '__main__':
    main()
