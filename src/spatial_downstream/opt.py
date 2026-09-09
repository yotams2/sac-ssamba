import argparse
import time
import os
import numpy as np

class opt_downstream:
    """
    Downstream options parser for SSAMBA spatial acoustic tasks (TDOA, DRR, T60, C50, ABS).
    Adapted from SAR-SSL framework.
    """
    def __init__(self):
        time_stamp = time.time()
        local_time = time.localtime(time_stamp)
        self.time = time.strftime('%m%d%H%M', local_time)
        code_dir = os.path.dirname(os.path.abspath(__file__))
        self.work_dir = os.path.abspath(os.path.join(code_dir, '../..'))

        self.acoustic_setting = {
            'sound_speed': 343.0,
            'fs': 16000,
            'snr_range': [15, 30],
            'nmic': 2,
            'mic_dist_range': [0.03, 0.20]
        }

    def parse(self):
        parser = argparse.ArgumentParser(description='Downstream spatial tasks for SSAMBA models')

        # Execution flags
        parser.add_argument('--gpu-id', type=str, default='0', help='GPU ID(s) (default: 0)')
        parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers (default: 4)')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
        parser.add_argument('--use-amp', action='store_true', default=False, help='Use automatic mixed precision')
        parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
        parser.add_argument('--checkpoint-start', action='store_true', default=False, help='Resume from saved checkpoints')

        # SSAMBA Model parameters
        parser.add_argument('--pretrained_mdl_path', type=str, default=None, help='Path to pre-trained model checkpoint (.pth or .tar)')
        parser.add_argument('--model_size', type=str, default='base', choices=['base', 'small', 'tiny', 'binaural_sac'], help='Model size (default: base)')
        parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension of spatial encoder (default: 768)')
        parser.add_argument('--fshape', type=int, default=16, help='Frequency patch shape (default: 16)')
        parser.add_argument('--tshape', type=int, default=16, help='Time patch shape (default: 16)')
        parser.add_argument('--input_fdim', type=int, default=256, help='Number of frequency bins (default: 256)')

        # Downstream task arguments
        parser.add_argument('--simu-exp', action='store_true', default=True, help='Experiments on simulated data (default: True)')
        parser.add_argument('--ds-train', action='store_true', default=False, help='Train downstream model')
        parser.add_argument('--ds-test', action='store_true', default=False, help='Test downstream model')
        parser.add_argument('--ds-trainmode', type=str, default='finetune', choices=['finetune', 'lineareval', 'scratchLOW'], help='Training mode (default: finetune)')
        parser.add_argument('--ds-task', type=str, nargs='+', default=['TDOA'], choices=['TDOA', 'DRR', 'C50', 'T60', 'ABS'], help='Downstream task(s) to run')
        parser.add_argument('--ds-token', type=str, default='all', choices=['all', 'cls'], help='Downstream token pooling mode (default: all)')
        parser.add_argument('--ds-head', type=str, default='mlp', choices=['mlp'], help='Downstream head type (default: mlp)')
        parser.add_argument('--ds-embed', type=str, default='spat', choices=['spat', 'spec', 'spec_spat'], help='Downstream stream embedding to evaluate (default: spat)')
        parser.add_argument('--ds-nsimroom', type=int, default=8, help='Number of simulated room conditions for training (default: 8)')
        parser.add_argument('--exp-dir', type=str, default='exp_spatial_downstream', help='Experiment output directory')

        args = parser.parse_args()

        assert (args.ds_train or args.ds_test), 'Must specify either --ds-train or --ds-test'
        
        self.simu_exp = args.simu_exp
        self.time = args.time if hasattr(args, 'time') else self.time
        args.time = self.time
        args.work_dir = self.work_dir
        args.acoustic_setting = self.acoustic_setting

        # Setup dataset trial settings
        bs_set = [8]
        lr_set = [0.001, 0.0005, 0.0001, 0.00005]
        nepoch = 200
        num = args.ds_nsimroom * 100
        ntrial = int(max(1, round(32 / (args.ds_nsimroom + 1e-4))))

        args.ds_setting = {}
        for task in ['TDOA', 'DRR', 'C50', 'T60', 'ABS']:
            args.ds_setting[task] = {
                'nepoch': nepoch,
                'num': num,
                'lr_set': lr_set,
                'bs_set': bs_set,
                'ntrial': ntrial
            }

        return args

    def dir(self, args):
        dirs = {}
        dirs['code'] = os.path.join(self.work_dir, 'src/spatial_downstream')
        dirs['exp'] = os.path.join(self.work_dir, args.exp_dir)

        # Primary data directory in SAR-SSL, fallback to ssamba data directory
        sar_ssl_data = '/storage/yotam/SAR-SSL/SAR-SSL/data'
        local_data = os.path.join(self.work_dir, 'data')
        
        if os.path.exists(sar_ssl_data):
            dirs['data'] = sar_ssl_data
            dirs['gerdata'] = sar_ssl_data
        else:
            dirs['data'] = local_data
            dirs['gerdata'] = local_data

        # Simulated downstream dataset directories
        micsig_train_simu_dir = os.path.join(dirs['gerdata'], 'MicSig/simu_ds/train')
        dirs['micsig_train_simu'] = []
        
        ntrial = args.ds_setting[args.ds_task[0]]['ntrial'] if len(args.ds_task) > 0 else 1
        for trial_idx in range(ntrial):
            room_dirs = []
            for room_idx in range(args.ds_nsimroom):
                idx = trial_idx * args.ds_nsimroom + room_idx + 1
                room_dirs.append(os.path.join(micsig_train_simu_dir, f'R{idx}'))
            dirs['micsig_train_simu'].append(room_dirs)

        dirs['micsig_val_simu'] = os.path.join(dirs['gerdata'], 'MicSig/simu_ds/val')
        dirs['micsig_test_simu'] = os.path.join(dirs['gerdata'], 'MicSig/simu_ds/test')

        dirs['log_pretrain'] = os.path.join(dirs['exp'], 'pretrain', self.time)
        dirs['log_task'] = os.path.join(dirs['exp'], 'TASK', self.time)

        data_model_flag = 'sim_' if args.simu_exp else 'real_'
        extra_info = f'R{args.ds_nsimroom}'

        dirs['log_task_scratchLOW'] = os.path.join(
            dirs['log_task'], f'scratchlow-{args.ds_token}-{args.ds_head}-NUM-LR-BAS-TRI-{args.ds_embed}-{data_model_flag}{extra_info}'
        )
        dirs['log_task_finetune'] = os.path.join(
            dirs['log_task'], f'finetune-{args.ds_token}-{args.ds_head}-NUM-LR-BAS-TRI-{args.ds_embed}-{data_model_flag}{extra_info}'
        )
        dirs['log_task_lineareval'] = os.path.join(
            dirs['log_task'], f'lineareval-{args.ds_token}-{args.ds_head}-NUM-LR-BAS-TRI-{args.ds_embed}-{data_model_flag}{extra_info}'
        )

        return dirs
