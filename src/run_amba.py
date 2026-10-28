#Adopted from run.py by yuan gong, made for running experiments with ssamba model

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
from models import AMBAModel
import numpy as np
from traintest import train, validate
from traintest_mask import trainmask
try:
    import wandb
except ImportError:
    wandb = None
sys.path.append('/share/naplab/users/ss6928/ssast/modules')

class BinauralSACClassifier(torch.nn.Module):
    def __init__(self, ckpt, label_dim, pool_real_time_only=False):
        super().__init__()
        import sys
        sys.path.insert(0, '/storage/yotam/ssamba/src')
        from binaural_dual_stream_sac.downstream_experts import MonauralSpectralUpstreamExpert
        
        self.encoder = MonauralSpectralUpstreamExpert(ckpt)
        spectral_enc = getattr(self.encoder, 'spectral_encoder', getattr(self.encoder, 'model', None))
        if hasattr(spectral_enc, 'spectral_encoder'):
            spectral_enc = spectral_enc.spectral_encoder
        self.spectral_encoder = spectral_enc
        if hasattr(self.spectral_encoder, 'embed_dim'):
            self.encoder_dim = self.spectral_encoder.embed_dim
        elif hasattr(self.spectral_encoder, 'original_embedding_dim'):
            self.encoder_dim = self.spectral_encoder.original_embedding_dim
        elif hasattr(self.spectral_encoder, 'v') and hasattr(self.spectral_encoder.v, 'pos_embed'):
            self.encoder_dim = self.spectral_encoder.v.pos_embed.shape[-1]
        else:
            self.encoder_dim = 768
        
        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.encoder_dim),
            torch.nn.Linear(self.encoder_dim, label_dim)
        )
        self.pool_real_time_only = pool_real_time_only
        
    def forward(self, x, task=None, cluster=None, mask_patch=None):
        # x is [B, 4, 256, 1024] 4-channel complex STFT directly
        h = self.spectral_encoder._encode_with_mamba(x)
        
        if task == 'ft_avgtok':
            h_tokens = h[:, 1:, :] # Skip the CLS token -> [B, 1024, D]
            if self.pool_real_time_only:
                # Patch grid: 16 frequency x 64 time patches.
                # Average only over time frames that hold real audio.
                x_time_energy = x.abs().sum(dim=(1, 2)) # [B, 1024]
                patch_energy = x_time_energy.view(x.size(0), 64, 16).sum(dim=-1) # [B, 64]
                time_mask = (patch_energy > 0).float() # [B, 64]
                zero_mask = (time_mask.sum(dim=-1, keepdim=True) == 0)
                time_mask = torch.where(zero_mask, torch.ones_like(time_mask), time_mask)
                h_grid = h_tokens.reshape(x.size(0), 16, 64, -1) # [B, 16, 64, D]
                h_masked = h_grid * time_mask.unsqueeze(1).unsqueeze(-1)
                h_pool = h_masked.sum(dim=(1, 2)) / (16.0 * time_mask.sum(dim=-1, keepdim=True))
            else:
                h_pool = h_tokens.mean(dim=1)
        elif task == 'ft_cls':
            h_pool = h[:, 0, :]
        else:
            raise ValueError(f"Task {task} not supported for BinauralSACClassifier")
            
        return self.mlp_head(h_pool)




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")

parser.add_argument("--dataset", type=str, help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--accum_grad', default=1, type=int, help='gradient accumulation steps')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# only used in pretraining stage or from-scratch fine-tuning experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

parser.add_argument("--resume_model", type=str, default=None, help="path to model to resume")
parser.add_argument("--resume_optimizer", type=str, default=None, help="path to optimizer to resume")
parser.add_argument("--wandb_id", type=str, default=None, help="wandb run id to resume")

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
# during self-supervised pretraining stage, no patch split overlapping is used (to aviod shortcuts), i.e., fstride=fshape and tstride=tshape
# during fine-tuning, using patch split overlapping (i.e., smaller {f,t}stride than {f,t}shape) improves the performance.
# it is OK to use different {f,t} stride in pretraining and finetuning stages (though fstride is better to keep the same)
# but {f,t}stride in pretraining and finetuning stages must be consistent.
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument('--model_size', help='the size of AST models', type=str, default='base384')

parser.add_argument("--task", type=str, default='ft_cls', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])

# pretraining augments
#parser.add_argument('--pretrain_stage', help='True for self-supervised pretraining stage, False for fine-tuning stage', type=ast.literal_eval, default='False')
parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")

# fine-tuning arguments
parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="the ssl pretrained models path")
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
parser.add_argument("--noise", help='if augment noise in finetuning', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument("--freeze_base", action="store_true", default=False, help="freeze base encoder weights for linear probing")
parser.add_argument('--use_wandb', action='store_true', help='Enable logging to Weights & Biases')

# vision mamba arguments
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--embed_dim', type=int, default=768)
parser.add_argument('--depth', type=int, default=12)
parser.add_argument('--rms_norm', type=str, choices=['true', 'false'], default='true')
parser.add_argument('--residual_in_fp32', type=str, choices=['true', 'false'], default='true')
parser.add_argument('--fused_add_norm', type=str, choices=['true', 'false'], default='true')
parser.add_argument('--if_rope', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--if_rope_residual', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--bimamba_type', type=str, default='v2')
parser.add_argument('--drop_path_rate', type=float, default=0.1)
parser.add_argument('--stride', type=int, default=16)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--drop_rate', type=float, default=0.)
parser.add_argument('--norm_epsilon', type=float, default=1e-5)
parser.add_argument('--if_bidirectional', type=str, choices=['true', 'false'], default='true')
parser.add_argument('--final_pool_type', type=str, default='none')
parser.add_argument('--if_abs_pos_embed', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--if_bimamba', type=str, choices=['true', 'false'], default='true')
parser.add_argument('--if_cls_token', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--if_devide_out', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--use_double_cls_token', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--use_middle_cls_token', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--pool_real_time_only', help='if average only over real audio time frames for binaural_sac', type=ast.literal_eval, default='False')



def main():
    print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
    args = parser.parse_args()
    if args.use_wandb and wandb is not None:
        project_name = "SSAMBA"  # default project name
        if args.dataset == "esc50":
            project_name = "amba_esc"
        elif args.dataset == "audioset":
            project_name = "amba_as"
        elif args.dataset == "speechcommands":
            project_name = "amba_sc"
        
        if args.wandb_id:
            wandb.init(project=project_name, config=args, id=args.wandb_id, resume="must")
        else:
            wandb.init(project=project_name, config=args)

    args.rms_norm = args.rms_norm == 'true'
    args.residual_in_fp32 = args.residual_in_fp32 == 'true'
    args.fused_add_norm = args.fused_add_norm == 'true'
    args.if_rope = args.if_rope == 'true'
    args.if_rope_residual = args.if_rope_residual == 'true'
    args.if_bidirectional = args.if_bidirectional == 'true'
    args.if_abs_pos_embed = args.if_abs_pos_embed == 'true'
    args.if_bimamba = args.if_bimamba == 'true'
    args.if_cls_token = args.if_cls_token == 'true'
    args.if_devide_out = args.if_devide_out == 'true'
    args.use_double_cls_token = args.use_double_cls_token == 'true'
    args.use_middle_cls_token = args.use_middle_cls_token == 'true'

    audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
                  'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}

    val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                      'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

    dataset_class = dataloader.BinauralSTFTClassificationDataset if args.model_size == 'binaural_sac' else dataloader.AudioDataset

    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataset_class(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataset_class(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
        
    dataset = dataset_class(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(train_loader)}")

    val_loader = torch.utils.data.DataLoader(
        dataset_class(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))

    vision_mamba_config = {
        'img_size': (args.num_mel_bins, args.target_length),
        'patch_size': args.patch_size,
        'stride': args.stride,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'channels': args.channels,
        'num_classes': args.num_classes,
        'drop_rate': args.drop_rate,
        'drop_path_rate': args.drop_path_rate,
        'norm_epsilon': args.norm_epsilon,
        'rms_norm': args.rms_norm,
        'residual_in_fp32': args.residual_in_fp32,
        'fused_add_norm': args.fused_add_norm,
        'if_rope': args.if_rope,
        'if_rope_residual': args.if_rope_residual,
        'bimamba_type': args.bimamba_type,
        'if_bidirectional': args.if_bidirectional,
        'final_pool_type': args.final_pool_type,
        'if_abs_pos_embed': args.if_abs_pos_embed,
        'if_bimamba': args.if_bimamba,
        'if_cls_token': args.if_cls_token,
        'if_devide_out': args.if_devide_out,
        'use_double_cls_token': args.use_double_cls_token,
        'use_middle_cls_token': args.use_middle_cls_token,
    }

    if 'pretrain' in args.task:
        cluster = (args.num_mel_bins != args.fshape)
        if cluster == True:
            print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(args.num_mel_bins, args.fshape))
        else:
            print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(args.num_mel_bins, args.fshape))
        audio_model = AMBAModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fshape, tstride=args.tshape,
                           input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=True, vision_mamba_config=vision_mamba_config)
    else:
        if args.model_size == 'binaural_sac':
            pool_real_time_only = getattr(args, 'pool_real_time_only', False)
            audio_model = BinauralSACClassifier(args.pretrained_mdl_path, args.n_class, pool_real_time_only=pool_real_time_only)
        else:
            audio_model = AMBAModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                               input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                               load_pretrained_mdl_path=args.pretrained_mdl_path, vision_mamba_config=vision_mamba_config)

    if args.freeze_base:
        print('Freezing base encoder parameters for linear probing...')
        for name, p in audio_model.named_parameters():
            if 'mlp_head' not in name:
                p.requires_grad = False

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)

    print("\nCreating experiment directory: %s" % args.exp_dir)
    if os.path.exists("%s/models" % args.exp_dir) == False:
        os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

    if 'pretrain' not in args.task:
        print('Now starting fine-tuning for {:d} epochs'.format(args.n_epochs))
        train(audio_model, train_loader, val_loader, args)
    else:
        print('Now starting self-supervised pretraining for {:d} epochs'.format(args.n_epochs))
        trainmask(audio_model, train_loader, val_loader, args)

    if args.data_eval != None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
        if not isinstance(audio_model, torch.nn.DataParallel):
            audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd, strict=False)

        args.loss_fn = torch.nn.BCEWithLogitsLoss()
        stats, _ = validate(audio_model, val_loader, args, 'valid_set')
        val_acc = stats[0]['acc']
        val_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the validation set---------------')
        print("Accuracy: {:.6f}".format(val_acc))
        print("AUC: {:.6f}".format(val_mAUC))

        eval_loader = torch.utils.data.DataLoader(
            dataset_class(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
        eval_acc = stats[0]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the test set---------------')
        print("Accuracy: {:.6f}".format(eval_acc))
        print("AUC: {:.6f}".format(eval_mAUC))
        if args.use_wandb and wandb is not None:
            wandb.log({"val_accuracy": val_acc, "val_mAUC": val_mAUC})
        np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])


if __name__ == '__main__':
    main()

