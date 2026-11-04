# Experiment results (branch `results`)

Written by jobs on the TAU cluster, not by hand. Code lives on `main`. One folder per run:

    results/<task>/<experiment>__<job_id>/
      summary.json      task, model, job_id, date, git_commit, git_dirty, seed, lr, frozen, status, metrics, ckpt_dir, ...
      log_tail.txt      last lines of the Slurm log, progress bars removed
      args.json / args_*.yaml / config_*.yaml / config.json   resolved configuration
      log.log, result.csv, eval_result.csv, probe_metrics*.jsonl   raw metrics, when they exist

`status`: `running` (job started, or was killed so hard that nothing else could be written), `done`, `failed`,
`interrupted` (preempted, time limit or scancel; a requeued job overwrites its folder), `incomplete` (backfilled run that
stopped early), `unknown` (backfilled pretraining run that has no completion marker).

`job_id` is the Slurm job id. `runai` marks runs from the old RunAI server, copied to this cluster and backfilled here:
they have no job id and no git commit. They also predate fixes described in `TODO.md` on `main`, so read those notes
before comparing them with newer runs.

For downstream runs, the number to report is `metrics.test_at_best_dev`, not `metrics.test_final` or the best test score.
Probe values in `summary.json` and `probe_metrics*.jsonl` are fractions (0.71 = 71%).

Files over 1 MB, checkpoints and audio are never stored here. Checkpoints stay at `ckpt_dir` on the cluster.
