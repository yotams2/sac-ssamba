# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : hubconf.py

# Authors
# - Leo

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


# Frame-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssamba_baseline(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/storage/yotam/ssamba/src/pretrain/exp/amba-base-f16-t16-b16-lr1e-4-m300-pretrain_joint-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-m300-lam1.0-tau0.3-sig1.0-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)



