"""
Run training and test processes for self-supervised SSAMBA model on spatial acoustic tasks.
Tasks: TDOA, DRR, T60, C50, ABS
Training Modes: finetune, lineareval, scratchLOW

Examples:
    python3 src/spatial_downstream/run_downstream.py --ds-train --ds-trainmode finetune --simu-exp --ds-nsimroom 8 --ds-task TDOA --gpu-id 0
    python3 src/spatial_downstream/run_downstream.py --ds-test --ds-trainmode finetune --simu-exp --ds-nsimroom 8 --ds-task TDOA --gpu-id 0
"""

import os
import sys
import copy
import json
import shutil
import scipy.io
import numpy as np
import torch

# Ensure spatial_downstream imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from opt import opt_downstream
from common_utils import set_seed, set_random_seed, get_nparams
import dataset as at_dataset
import learner as at_learner
import model as at_model

opts = opt_downstream()
args = opts.parse()
dirs = opts.dir(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

set_seed(args.seed)

sound_speed = args.acoustic_setting['sound_speed']
fs = args.acoustic_setting['fs']
seeds = {'train': int(args.seed + 2e8), 'val': int(args.seed + 1e8), 'test': int(args.seed + 1)}

# Determine audio clip duration T per task
if args.ds_task == ['TDOA']:
    T = 1.04
else:
    T = 4.112

print(f"Downstream task duration: {T}s")
selecting = at_dataset.Selecting(select_range=[0, int(T * fs)])

# STFT specifications
win_len = 512
nfft = 512
win_shift_ratio = 0.5
fre_used_ratio = 1.0
nf = nfft // 2
nt = int((T * fs - win_len * (1 - win_shift_ratio)) / (win_len * win_shift_ratio))

print(f"STFT frames: nf={nf}, nt={nt}")

# Network initialization
param_frozen = (args.ds_trainmode == 'lineareval')
net = at_model.SSAMBASpatialDownstreamModel(
    fshape=args.fshape,
    tshape=args.tshape,
    input_fdim=nf,
    input_tdim=nt,
    model_size=args.model_size,
    embed_dim=args.embed_dim,
    downstream_token=args.ds_token,
    downstream_head=args.ds_head,
    downstream_embed=args.ds_embed,
    downstream_dlabel=1,
    load_pretrained_mdl_path=args.pretrained_mdl_path if args.ds_trainmode != 'scratchLOW' else None,
    param_frozen=param_frozen,
    device=device
)

nparam, nparam_sum = get_nparams(net, param_key_list=['ssamba_model', 'mlp_head'])
print(f"# Parameters (M): Total={round(nparam_sum, 2)}, breakdown={[f'{k}: {round(v, 2)}' for k, v in nparam.items()]}")

# Training stage
if args.ds_train:
    print(f"Starting downstream training stage: {args.ds_trainmode}")
    print(f"Simulated rooms: {args.ds_nsimroom}")

    log_dir = f"log_task_{args.ds_trainmode}"
    init_state_dict = copy.deepcopy(net.state_dict())

    for task in args.ds_task:
        net.task = task
        set_seed(args.seed)
        task_time_dir = dirs['log_task'].replace('TASK', task)
        os.makedirs(task_time_dir, exist_ok=True)

        nepoch = args.ds_setting[task]['nepoch']
        num = args.ds_setting[task]['num']
        bs_set = args.ds_setting[task]['bs_set']
        lr_set = args.ds_setting[task]['lr_set']
        stages = ['train', 'val', 'test', 'test_large']
        data_num = {'train': num, 'val': 1000, 'test': 1000, 'test_large': 4000}
        test_bs = 16
        early_stop_patience = 10
        smooth_alpha = 0.6
        nepoch_ensemble = 5
        ntrials = args.ds_setting[task]['ntrial']

        nlrs = len(lr_set)
        nbss = len(bs_set)

        result_name = f"result-{task}-{args.ds_trainmode}.mat"
        result_name_temporal = f"result-{task}-{args.ds_trainmode}_temporal.mat"

        if os.path.exists(os.path.join(task_time_dir, result_name_temporal)):
            print(f"Temporal result {result_name_temporal} exists. Resuming...")
            mat_data = scipy.io.loadmat(os.path.join(task_time_dir, result_name_temporal))
            val_losses = mat_data['val_losses']
            test_losses = mat_data['test_losses']
            val_metrics = mat_data['val_metrics']
            test_metrics = mat_data['test_metrics']
            ensemble_epochs = mat_data['ensemble_epoch']
        else:
            val_losses = np.zeros((nlrs, nbss, ntrials))
            test_losses = np.zeros((nlrs, nbss, ntrials))
            val_metrics = np.zeros((nlrs, nbss, ntrials))
            test_metrics = np.zeros((nlrs, nbss, ntrials))
            ensemble_epochs = np.zeros((nlrs, nbss, ntrials, 2))

        for trial_idx in range(ntrials):
            for bs_idx in range(nbss):
                for lr_idx in range(nlrs):
                    set_seed(args.seed)
                    lr_init = lr_set[lr_idx]
                    bs = bs_set[bs_idx]

                    print(f"\nTask: {task} | trial: {trial_idx}/{ntrials} | bs: {bs} | lr: {lr_init}")
                    task_dir = dirs[log_dir].replace('TASK', task).replace('NUM', str(num)).replace('LR', str(lr_init)).replace('BAS', str(bs)).replace('TRI', str(trial_idx))
                    os.makedirs(task_dir, exist_ok=True)

                    if val_losses[lr_idx, bs_idx, trial_idx] == 0:
                        datasets = {}
                        for stage in stages:
                            if stage == 'train':
                                data_dir = dirs['micsig_train_simu'][trial_idx]
                            else:
                                data_dir = dirs['micsig_val_simu'] if 'val' in stage else dirs['micsig_test_simu']

                            datasets[stage] = at_dataset.FixMicSigDataset(
                                data_dir=data_dir,
                                load_anno=True,
                                load_dp=False,
                                fs=fs,
                                dataset_sz=data_num[stage],
                                transforms=[selecting]
                            )

                        kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
                        dataloader_train = torch.utils.data.DataLoader(dataset=datasets['train'], batch_size=bs, shuffle=True, **kwargs)
                        dataloader_val = torch.utils.data.DataLoader(dataset=datasets['val'], batch_size=test_bs, shuffle=False, **kwargs)
                        dataloader_test = torch.utils.data.DataLoader(dataset=datasets['test'], batch_size=test_bs, shuffle=False, **kwargs)
                        dataloader_test_large = torch.utils.data.DataLoader(dataset=datasets['test_large'], batch_size=test_bs, shuffle=False, **kwargs)

                        net.load_state_dict(init_state_dict)
                        learner = at_learner.STFTLearner(
                            net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft,
                            fre_used_ratio=fre_used_ratio, fs=fs, task=task, ch_mode='M'
                        )

                        if len(args.gpu_id.split(',')) > 1 and torch.cuda.device_count() > 1:
                            learner.mul_gpu()
                        if use_cuda:
                            learner.cuda()
                        else:
                            learner.cpu()
                        if args.use_amp:
                            learner.amp()

                        if args.checkpoint_start:
                            learner.resume_checkpoint(checkpoints_dir=task_dir, from_latest=True)
                        else:
                            if args.ds_trainmode == 'finetune':
                                if args.pretrained_mdl_path:
                                    learner.model.load_pretrained_weights(args.pretrained_mdl_path)
                            elif args.ds_trainmode == 'lineareval':
                                if args.pretrained_mdl_path:
                                    learner.load_checkpoint_best(checkpoints_dir=os.path.dirname(args.pretrained_mdl_path), param_frozen=True)

                        loss_val_list = []
                        lr = lr_init
                        cnt_stop = 0
                        best_epoch = 1

                        for epoch in range(learner.start_epoch, nepoch + 1):
                            set_random_seed(seeds['train'])
                            loss_train, metric_train = learner.train_epoch(dataloader_train, lr=lr, epoch=epoch, return_metric=True)

                            set_random_seed(seeds['val'])
                            loss_val, metric_val = learner.test_epoch(dataloader_val, return_metric=True)

                            set_random_seed(seeds['test'])
                            loss_test, metric_test = learner.test_epoch(dataloader_test, return_metric=True)

                            print(f"Epoch {epoch}/{nepoch} | Val MAE: {metric_val:.4f} | Test MAE: {metric_test:.4f}")

                            loss_val_list.append(loss_val)
                            loss_val_list_smooth = learner.smooth_data(data_list=loss_val_list, alpha=smooth_alpha)

                            stop_flag, is_best_epoch = learner.early_stopping(current_score=-loss_val_list_smooth[-1], patience=early_stop_patience)
                            learner.save_checkpoint(epoch=epoch, checkpoints_dir=task_dir, is_best_epoch=is_best_epoch, save_extra_hist=True)

                            if is_best_epoch:
                                best_epoch = epoch

                            if stop_flag:
                                cnt_stop += 1
                                if cnt_stop <= 1:
                                    lr /= 10.0
                                    print("Learning rate decayed by 10x")
                                    learner.early_stop_counter = 0
                                else:
                                    break

                        print("\nTraining completed. Ensembling top models...")
                        st_epoch = max(1, best_epoch - nepoch_ensemble + 1)
                        ed_epoch = best_epoch
                        epochs_to_ensemble = list(range(st_epoch, ed_epoch + 1))
                        learner.ensembling(checkpoints_dir=task_dir, epochs=epochs_to_ensemble)

                        set_random_seed(seeds['test'])
                        best_loss_test, best_metric_test = learner.test_epoch(dataloader_test_large, return_metric=True)
                        set_random_seed(seeds['val'])
                        best_loss_val, best_metric_val = learner.test_epoch(dataloader_val, return_metric=True)

                        print(f"Ensemble Results | Test MAE: {best_metric_test:.4f} | Val MAE: {best_metric_val:.4f}")

                        epochs_remove = [i for i in range(1, st_epoch)] + [i for i in range(best_epoch + 1, epoch + 1)]
                        learner.remove_checkpoint_epochs(checkpoints_dir=task_dir, epochs=epochs_remove)

                        val_losses[lr_idx, bs_idx, trial_idx] = best_loss_val
                        val_metrics[lr_idx, bs_idx, trial_idx] = best_metric_val
                        test_losses[lr_idx, bs_idx, trial_idx] = best_loss_test
                        test_metrics[lr_idx, bs_idx, trial_idx] = best_metric_test
                        ensemble_epochs[lr_idx, bs_idx, trial_idx, :] = [st_epoch, ed_epoch]

                        scipy.io.savemat(os.path.join(task_time_dir, result_name_temporal), {
                            'val_losses': val_losses, 'val_metrics': val_metrics,
                            'test_losses': test_losses, 'test_metrics': test_metrics,
                            'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrials,
                            'ensemble_epoch': ensemble_epochs
                        })

        metric_avg = np.mean(val_metrics, axis=-1)
        best_flat_idx = metric_avg.argmin()
        ncol = metric_avg.shape[1]
        best_lr_idx = best_flat_idx // ncol
        best_bs_idx = best_flat_idx % ncol
        best_lr = lr_set[best_lr_idx]
        best_bs = bs_set[best_bs_idx]

        best_val_metric = np.mean(val_metrics, axis=-1)[best_lr_idx, best_bs_idx]
        best_test_metric = np.mean(test_metrics, axis=-1)[best_lr_idx, best_bs_idx]

        print(f"\nFinal Task Summary [{task}] | Best BS: {best_bs} | Best LR: {best_lr} | Val MAE: {best_val_metric:.4f} | Test MAE: {best_test_metric:.4f}")

        scipy.io.savemat(os.path.join(task_time_dir, result_name), {
            'val_losses': val_losses, 'val_metrics': val_metrics,
            'test_losses': test_losses, 'test_metrics': test_metrics,
            'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrials,
            'best_lr_idx': best_lr_idx, 'best_bs_idx': best_bs_idx,
            'ensemble_epoch': ensemble_epochs
        })

        if os.path.exists(os.path.join(task_time_dir, result_name_temporal)):
            os.remove(os.path.join(task_time_dir, result_name_temporal))

        res_file = os.path.join(task_time_dir, f"downstream_test_res_{args.ds_trainmode}.json")
        res_list = [{
            'task': task,
            'best_lr': float(best_lr),
            'best_bs': int(best_bs),
            'val_mae': float(best_val_metric),
            'test_mae': float(best_test_metric)
        }]
        with open(res_file, 'w') as f:
            json.dump(res_list, f, indent=4)
        print(f"Saved test summary results to {res_file}")

# Testing stage
if args.ds_test:
    print(f"Downstream evaluation stage: {args.ds_trainmode}")
    for task in args.ds_task:
        task_time_dir = dirs['log_task'].replace('TASK', task)
        res_file = os.path.join(task_time_dir, f"downstream_test_res_{args.ds_trainmode}.json")
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                res_data = json.load(f)
            print(f"Loaded existing evaluation summary for {task}: {res_data}")
        else:
            print(f"No previous results found in {task_time_dir}")
