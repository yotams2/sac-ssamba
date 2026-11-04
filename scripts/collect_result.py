#!/usr/bin/env python3
"""Write one run's numbers to results/<task>/<experiment>__<job_id>/ in the `results` worktree.

Stdlib only and Python 3.9 compatible: it runs on the Jupyter node, where importing torch gets OOM-killed, and under
the system Python on compute nodes. Called by scripts/results_hook.sh at job start and exit, and by hand for backfills:

    python3 scripts/collect_result.py --task voxceleb1 --exp_dir <exp dir> --job_id 23305339 --log logs/<log>.out

Understands run_amba.py output (eval_result.csv, result.csv, args.pkl), s3prl expdirs (log.log, also one level down
for fold folders), binaural pretraining (config.json, probe_metrics*.jsonl), pretrain_probe.py files and
probe_diagnostic.py logs. Never raises: a run that can't be parsed still gets a summary.json, with a parse_error field,
and the exit code is always 0 so that a job can't fail because of this script.
"""
import argparse
import datetime
import fcntl
import glob
import json
import os
import pickle
import re
import shutil
import socket
import sys
import tempfile

MAX_FILE_BYTES = 1000 * 1000
LOG_TAIL_BYTES = 256 * 1024
LOG_TAIL_LINES = 60
MAX_PROBE_ROWS = 40
PROBE_KEYS = ('sc2/linear_acc', 'sid/linear_acc', 'sid/eer', 'emo/linear_ua')


class _ArgsUnpickler(pickle.Unpickler):
    """args.pkl is an argparse.Namespace of plain values: allow that and nothing else."""

    def find_class(self, module, name):
        if (module, name) == ('argparse', 'Namespace'):
            return argparse.Namespace
        raise pickle.UnpicklingError(f'{module}.{name} not allowed')


def load_pickle(path):
    with open(path, 'rb') as f:
        return _ArgsUnpickler(f).load()


def load_args_pkl(path):
    try:
        return {k: v for k, v in vars(load_pickle(path)).items() if isinstance(v, (str, int, float, bool, type(None)))}
    except Exception:
        return {}


def yaml_value(text, key):
    """Value of the first `key: value` line in a YAML file, at any indent. Enough for s3prl's flat keys."""
    m = re.search(rf'^\s*{re.escape(key)}:\s*(.+?)\s*$', text, re.M)
    if not m or m.group(1) in ('null', '~'):
        return None
    value = m.group(1)
    if value in ('true', 'false'):
        return value == 'true'
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def latest(pattern):
    """Last match of a timestamped pattern such as args_2026-11-02-05-40-34.yaml."""
    paths = sorted(glob.glob(pattern))
    return paths[-1] if paths else None


def read_text(path):
    with open(path, errors='replace') as f:
        return f.read()


def is_s3prl(d):
    return os.path.isfile(os.path.join(d, 'log.log'))


def is_amba(d):
    return os.path.isfile(os.path.join(d, 'eval_result.csv')) or os.path.isfile(os.path.join(d, 'result.csv'))


def parse_s3prl(run_dir):
    """Metrics from log.log. It is appended to across resumes, so the last value for a step wins."""
    scores = {'train': {}, 'dev': {}, 'test': {}}
    new_best = {}
    for line in read_text(os.path.join(run_dir, 'log.log')).splitlines():
        m = re.match(r'(train|dev|test) at step (\d+): (\S+)$', line)
        if m:
            scores[m.group(1)][int(m.group(2))] = float(m.group(3))
        m = re.match(r'New best on dev at step (\d+): (\S+)$', line)
        if m:
            new_best[int(m.group(1))] = float(m.group(2))

    out = {'pipeline': 's3prl'}
    best = new_best or scores['dev']
    if best:
        step = max(best, key=lambda s: (best[s], s))
        out['best_dev'] = {'step': step, 'acc': best[step]}
        if step in scores['test']:
            out['test_at_best_dev'] = {'step': step, 'acc': scores['test'][step]}
    if scores['test']:
        step = max(scores['test'])
        out['test_final'] = {'step': step, 'acc': scores['test'][step]}

    all_steps = [s for mode in scores.values() for s in mode]
    out['last_step'] = max(all_steps) if all_steps else 0
    config_path = latest(os.path.join(run_dir, 'config_*.yaml'))
    args_path = latest(os.path.join(run_dir, 'args_*.yaml'))
    config = read_text(config_path) if config_path else ''
    args = read_text(args_path) if args_path else ''
    out['total_steps'] = yaml_value(config, 'total_steps')
    out['complete'] = bool(out['total_steps']) and out['last_step'] >= out['total_steps']
    trainable = yaml_value(args, 'upstream_trainable')
    info = {
        'model': yaml_value(args, 'upstream'),
        'pretrained': yaml_value(args, 'upstream_ckpt'),
        'seed': yaml_value(args, 'seed'),
        'lr': yaml_value(config, 'lr'),
        'frozen': None if trainable is None else not trainable,
    }
    return out, info


def parse_amba(run_dir):
    """run_amba.py. result.csv has one row per epoch; in fine-tuning, zeros for epochs not reached yet."""
    args = load_args_pkl(os.path.join(run_dir, 'args.pkl'))
    out = {'pipeline': 'run_amba'}
    rows = []
    result_path = os.path.join(run_dir, 'result.csv')
    if os.path.isfile(result_path):
        for line in read_text(result_path).split('\n'):
            values = [float(v) for v in line.replace(',', ' ').split()]
            if any(values):
                rows.append(values)
    pretrain = 'pretrain' in str(args.get('task', ''))
    if pretrain:
        out['pipeline'] = 'run_amba_pretrain'
        out['epochs_done'] = len(rows)
        if rows:
            # traintest_mask.py: masked acc and InfoNCE on train and eval, lr
            out['last_epoch'] = dict(zip(('train_acc', 'train_nce', 'eval_acc', 'eval_nce', 'lr'), rows[-1]))
    elif rows:
        # traintest.py: first column is the main metric on the validation set, second its AUC
        best = max(range(len(rows)), key=lambda i: rows[i][0])
        out['best_dev'] = {'epoch': best + 1, str(args.get('metrics', 'acc')): rows[best][0], 'auc': rows[best][1]}
        out['epochs_done'] = len(rows)

    eval_path = os.path.join(run_dir, 'eval_result.csv')
    if os.path.isfile(eval_path):
        val_acc, val_auc, test_acc, test_auc = [float(v) for v in read_text(eval_path).split()[:4]]
        # run_amba.py reloads best_audio_model.pth for these, so test is at the best validation epoch
        out['test_at_best_dev'] = {'acc': test_acc, 'auc': test_auc}
        out['dev_of_best_model'] = {'acc': val_acc, 'auc': val_auc}
    out['complete'] = None if pretrain else os.path.isfile(eval_path)
    info = {
        'model': args.get('gramt_model_id') or args.get('model_size'),
        'pretrained': args.get('pretrained_mdl_path'),
        'seed': args.get('seed'),
        'lr': args.get('lr'),
        'frozen': None if pretrain else bool(args.get('freeze_base')),
    }
    return out, info, args


def probe_rows(path):
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def summarise_probes(rows):
    """Headline columns per probe, keyed by training step or by model name. The full rows are in the copied jsonl."""
    out = []
    for row in rows[-MAX_PROBE_ROWS:]:
        short = {k: row[k] for k in PROBE_KEYS if k in row}
        if 'step' in row:
            short['step'] = row['step']
        if 'model' in row:
            short['model'] = os.path.basename(str(row['model']))
        out.append(short)
    return out


def parse_pretrain(exp_dir):
    """Binaural pretraining: config.json, plus the last saved progress entry when there is one."""
    path = os.path.join(exp_dir, 'config.json')
    config = json.loads(read_text(path)) if os.path.isfile(path) else {}
    out = {'pipeline': 'pretrain'}
    progress = os.path.join(exp_dir, 'progress.pkl')
    if os.path.isfile(progress):
        try:
            entries = load_pickle(progress)
            if entries:
                out['last_progress'] = [v for v in entries[-1] if isinstance(v, (int, float))]
        except Exception:
            pass
    model = 'binaural_sac_spectral_only' if config.get('spectral_only') else 'binaural_sac' if config else None
    return out, {'model': model, 'seed': config.get('seed'), 'lr': config.get('lr')}


def parse_probe_diagnostic(log_text):
    """The results table at the end of a probe_diagnostic.py log."""
    rows = re.findall(r'^(\S+)\s+(\S+)\s+([\d.]+)%\s*$', log_text, re.M)
    return {'pipeline': 'probe_diagnostic',
            'test_acc_percent': {f'{encoder}/{variant}': float(acc) for encoder, variant, acc in rows}}


def log_tail(path):
    """Last lines of a Slurm log, without the tqdm bars that make up most of an s3prl log."""
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - LOG_TAIL_BYTES))
        text = f.read().decode('utf-8', 'replace')
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    lines = []
    for line in re.split(r'[\r\n]+', text):
        line = line.strip()
        if line and not re.search(r'\d+%\|| \d+/\d+ \[\d', line):
            lines.append(line[:400])
    return '\n'.join(lines[-LOG_TAIL_LINES:]) + '\n'


def fold_dirs(exp_dir):
    """Subfolders holding a run of their own (iemocap, esc50: one per fold)."""
    return [d for d in sorted(glob.glob(os.path.join(exp_dir, '*'))) if os.path.isdir(d) and (is_s3prl(d) or is_amba(d))]


def collect(args, stage):
    """Fills `stage` with the raw files and returns (metrics, info)."""
    exp_dir = args.exp_dir
    copy = []          # (source, name in the results folder)
    args_json = {}
    info = {}

    def add_run(run_dir, prefix=''):
        nonlocal args_json
        if is_s3prl(run_dir):
            metrics, run_info = parse_s3prl(run_dir)
            copy.append((os.path.join(run_dir, 'log.log'), prefix + 'log.log'))
            for pattern in ('args_*.yaml', 'config_*.yaml'):
                path = latest(os.path.join(run_dir, pattern))
                if path:
                    copy.append((path, prefix + os.path.basename(path)))
        else:
            metrics, run_info, args_json = parse_amba(run_dir)
            for name in ('eval_result.csv', 'result.csv'):
                copy.append((os.path.join(run_dir, name), prefix + name))
        info.update({k: v for k, v in run_info.items() if v is not None})
        return metrics

    if args.task == 'probe_diagnostic':
        metrics = parse_probe_diagnostic(read_text(args.log)) if args.log and os.path.isfile(args.log) else {}
    elif args.probe_file:
        metrics = {'pipeline': 'pretrain_probe'}
        if os.path.isfile(args.probe_file):
            metrics['probes'] = summarise_probes(probe_rows(args.probe_file))
            copy.append((args.probe_file, os.path.basename(args.probe_file)))
    elif is_s3prl(exp_dir) or is_amba(exp_dir):
        metrics = add_run(exp_dir)
    elif fold_dirs(exp_dir):
        folds = {os.path.basename(d): add_run(d, os.path.basename(d) + '__') for d in fold_dirs(exp_dir)}
        scores = [m['test_at_best_dev']['acc'] for m in folds.values() if 'test_at_best_dev' in m]
        metrics = {'folds': folds, 'complete': all(m.get('complete') for m in folds.values())}
        if scores:
            metrics['test_at_best_dev'] = {'acc': sum(scores) / len(scores), 'n_folds': len(scores)}
    else:
        metrics, run_info = parse_pretrain(exp_dir)
        info.update({k: v for k, v in run_info.items() if v is not None})

    if not args.probe_file and args.task != 'probe_diagnostic':
        probes = sorted(glob.glob(os.path.join(exp_dir, 'probe_metrics*.jsonl')))
        for path in probes:
            copy.append((path, os.path.basename(path)))
        if probes:
            metrics['probes'] = summarise_probes(probe_rows(probes[-1]))
        for name in ('config.json', 'description.log'):
            copy.append((os.path.join(exp_dir, name), name))

    skipped = []
    for source, name in copy:
        if not os.path.isfile(source):
            continue
        if os.path.getsize(source) > MAX_FILE_BYTES:
            skipped.append(name)
        else:
            shutil.copyfile(source, os.path.join(stage, name))
    if skipped:
        metrics['skipped_files_over_1MB'] = skipped
    if args_json:
        with open(os.path.join(stage, 'args.json'), 'w') as f:
            json.dump(args_json, f, indent=2, sort_keys=True)
    return metrics, info


def has_numbers(metrics):
    return any(k in metrics for k in ('test_at_best_dev', 'test_final', 'best_dev', 'probes', 'folds', 'last_epoch',
                                      'last_progress')) or bool(metrics.get('test_acc_percent'))


def newest_mtime(exp_dir):
    """When the run last produced something: the date to record, also for runs backfilled long after they finished."""
    times = [os.path.getmtime(p) for p in glob.glob(os.path.join(exp_dir, '*')) if os.path.isfile(p)]
    times += [os.path.getmtime(p) for p in glob.glob(os.path.join(exp_dir, '*', 'log.log'))]
    return max(times) if times else None


def infer_status(metrics):
    """Status of a backfilled run, when no job told us how it ended."""
    complete = metrics.get('complete')
    if complete is None:
        return 'unknown'
    return 'done' if complete else 'incomplete'


def write_summary(args, stage):
    now = datetime.datetime.now().astimezone()
    summary = {
        'task': args.task,
        'experiment': os.path.basename(args.exp_dir.rstrip('/')),
        'job_id': args.job_id,
        'status': args.status,
        'date': now.isoformat(timespec='seconds'),
        'collected_at': now.isoformat(timespec='seconds'),
        'host': socket.gethostname(),
        'exp_dir': args.exp_dir,
        'ckpt_dir': None,
        'model': None, 'pretrained': None, 'seed': None, 'lr': None, 'frozen': None,
        'git_commit': None, 'git_dirty': None, 's3prl_commit': args.s3prl_commit or None,
        'metrics': {},
    }
    if args.exit_code is not None:
        summary['exit_code'] = args.exit_code
    if os.environ.get('SLURM_RESTART_COUNT'):
        summary['slurm_restart_count'] = int(os.environ['SLURM_RESTART_COUNT'])
    if args.git_commit:
        summary['git_commit'] = args.git_commit.replace('-dirty', '')
        summary['git_dirty'] = args.git_commit.endswith('-dirty')
    models = os.path.join(args.exp_dir, 'models')
    summary['ckpt_dir'] = models if os.path.isdir(models) else args.exp_dir

    errors = []
    try:
        metrics, info = collect(args, stage)
        summary['metrics'] = metrics
        summary.update(info)
        if args.status is None:
            summary['status'] = infer_status(metrics)
        if args.status in ('done', 'failed') and not has_numbers(metrics):
            errors.append(f'no metrics found in {args.exp_dir}')
    except Exception as e:
        errors.append(f'{type(e).__name__}: {e}')
        summary['status'] = args.status or 'unknown'
    try:
        mtime = newest_mtime(args.exp_dir)
        if mtime:
            summary['date'] = datetime.datetime.fromtimestamp(mtime).astimezone().isoformat(timespec='seconds')
    except OSError:
        pass
    try:
        if args.log and os.path.isfile(args.log):
            summary['log'] = args.log
            with open(os.path.join(stage, 'log_tail.txt'), 'w') as f:
                f.write(log_tail(args.log))
    except Exception as e:
        errors.append(f'log tail: {type(e).__name__}: {e}')
    if errors:
        summary['parse_error'] = '; '.join(errors)
    with open(os.path.join(stage, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, help='speechcommands_v2, iemocap, voxceleb1, binaural_sac_pretrain, ...')
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--job_id', required=True, help='Slurm job id; it survives requeues, so a requeued job overwrites its folder')
    parser.add_argument('--status', choices=['running', 'done', 'failed', 'interrupted', 'incomplete', 'unknown'],
                        help='omit to infer it from the run (backfills)')
    parser.add_argument('--log', help='Slurm log, for log_tail.txt')
    parser.add_argument('--probe_file', help='pretrain_probe.py output: report this file only')
    parser.add_argument('--git_commit', help='sac-ssamba commit the job ran, with a -dirty suffix if it had local changes')
    parser.add_argument('--s3prl_commit')
    parser.add_argument('--exit_code', type=int)
    parser.add_argument('--results_root', default=os.environ.get('SAC_RESULTS_ROOT', os.path.expanduser('~/sac-ssamba-results')))
    args = parser.parse_args()
    args.exp_dir = os.path.abspath(args.exp_dir)
    if args.probe_file:
        args.probe_file = os.path.abspath(args.probe_file)

    if not os.path.isdir(args.results_root):
        print(f'[collect_result] no results worktree at {args.results_root}, nothing written')
        return
    experiment = os.path.basename(args.exp_dir.rstrip('/'))
    target = os.path.join(args.results_root, 'results', args.task, f'{experiment}__{args.job_id}')
    with tempfile.TemporaryDirectory() as stage:
        summary = write_summary(args, stage)
        # Same lock as sync_results.sh: its rebase must not see a half-written folder
        with open(os.path.join(args.results_root, '.sync.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if os.path.isdir(target):
                shutil.rmtree(target)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copytree(stage, target)
    print(f"[collect_result] {target}: status {summary['status']}"
          + (f", parse_error: {summary['parse_error']}" if 'parse_error' in summary else ''))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # a job must never fail because of its results bookkeeping
        print(f'[collect_result] failed: {type(e).__name__}: {e}', file=sys.stderr)
    sys.exit(0)
