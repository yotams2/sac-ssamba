"""Both finetune paths load weights with strict=False, so a key mismatch would silently leave random weights.
Compare every encoder block in the built models with the checkpoint. Usage: check_mono_upstream_load.py <ckpt>"""
import os
import sys

import torch

ckpt = sys.argv[1]
sd = torch.load(ckpt, map_location='cpu')
prefix = next(p for p in ('module.encoder.', 'encoder.', 'module.') if any(k.startswith(p + 'v.layers.') for k in sd))
print('checkpoint prefix:', prefix)


def compare(name, v):
    same = diff = 0
    for k, p in v.named_parameters():
        if '.layers.' not in '.' + k:
            continue
        if torch.equal(p.detach().cpu(), sd[f'{prefix}v.{k}']):
            same += 1
        else:
            diff += 1
    print(f'{name}: {same} encoder tensors match the checkpoint, {diff} differ')
    assert same > 0 and diff == 0, name


sys.path.insert(0, os.path.join(os.environ['SAC_SSAMBA_SRC']))
from models import AMBAModel  # noqa: E402

config = dict(img_size=(128, 128), patch_size=16, stride=10, embed_dim=768, depth=24, rms_norm=False, residual_in_fp32=False,
              fused_add_norm=False, drop_path_rate=0.1, channels=1, num_classes=1000, drop_rate=0.0, norm_epsilon=1e-5,
              if_rope=False, if_rope_residual=False, bimamba_type='v2', if_bidirectional=True, final_pool_type='none',
              if_abs_pos_embed=True, if_bimamba=False, if_cls_token=True, if_devide_out=True, use_double_cls_token=False,
              use_middle_cls_token=False)
sc = AMBAModel(label_dim=35, fshape=16, tshape=16, fstride=10, tstride=10, input_fdim=128, input_tdim=128, model_size='base',
               pretrain_stage=False, load_pretrained_mdl_path=ckpt, vision_mamba_config=config)
compare('run_amba path (SC2)', sc.v)
del sc

from s3prl.upstream.ssamba.expert import UpstreamExpert  # noqa: E402

compare('s3prl path (IEMOCAP, VoxCeleb1)', UpstreamExpert(ckpt, 'base_a', window_secs=10.0).model.v)
print('load check passed')
