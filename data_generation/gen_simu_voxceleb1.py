"""
    Generate simulated room impulse responses and binaural microphone signals for VoxCeleb1.

    Examples:
        python gen_simu_voxceleb1.py --stage pretrain --data_num 50000 --vox1_root /scratch/yotam/data/voxceleb1 --meta_file /scratch/yotam/s3prl/s3prl/downstream/voxceleb1/veri_test_class.txt --save_to ../../data/VoxCeleb1/simu --gpus [0,1]
        python gen_simu_voxceleb1.py --stage preval --data_num 4000 --vox1_root /scratch/yotam/data/voxceleb1 --meta_file /scratch/yotam/s3prl/s3prl/downstream/voxceleb1/veri_test_class.txt --save_to ../../data/VoxCeleb1/simu --gpus [0]
        python gen_simu_voxceleb1.py --stage pretest --data_num 4874 --vox1_root /scratch/yotam/data/voxceleb1 --meta_file /scratch/yotam/s3prl/s3prl/downstream/voxceleb1/veri_test_class.txt --save_to ../../data/VoxCeleb1/simu --gpus [0]
"""
import os
cpu_num = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

import numpy as np
import tqdm
import inspect
import importlib
import multiprocessing as mp
from functools import partial
from jsonargparse import ArgumentParser
from typing import *
from pathlib import Path
from utils_simu_rir_sig import *
from utils_src import *
from utils_noise import *
from utils_array import *


def GenerateRandomMicSigVoxCeleb1(
    room_sz_range: Union[List[Tuple[float, float]], np.ndarray]=[(3,15), (3,10), (2.5,6)],
    T60_range: Tuple[float,float]=(0.2, 1.3),
    abs_weights_range: List[Tuple[float, float]]=[(0.5,1)]*6,
    mic_array_cfg: Dict[str, Any]=mic_array_cfg_2ch,
    array_pos_ratio_range: Union[List[Tuple[float, float]], np.ndarray]=[(0.2,0.8), (0.2,0.8), (0.1,0.5)],
    num_source_range: Tuple[int,int]=(1,1),
    source_state: str='static',
    min_src_array_dist: float=0.3,
    min_src_boundary_dist: float=0.3,
    traj_pt_mode: str='time',
    snr_range: Tuple[float,float]=(15,30),
    noise_type: str='diffuse_white',
    fs: int=16000,
    c: float=343.0,
    ism_db: float=12,
    T: float=8.0, # 8.0s duration for VoxCeleb1
    vox1_root: str='',
    meta_file: str='',
    noi_dir: str='',
    stage: str='pretrain',
    data_num: int=1,
    save_to: str='',
    save_dp: bool=False,
    gpu_conv: bool=False,
    gpus: List[int]=[0,0,1,1],
    use_gpu: bool=True,
):
    if traj_pt_mode == 'time':
        if 'static' in source_state:
            nb_points = 1
        elif 'moving' in source_state:
            nb_points = int(T/0.1)
        else:
            raise Exception('Unknown source state: {}'.format(source_state))
    else:
        nb_points = None

    assert stage in ['pretrain', 'preval', 'pretest'], stage
    if stage == 'pretrain':
        seed = 1
    elif stage == 'preval':
        seed = 2e6
    elif stage == 'pretest':
        seed = 3e6
    seed = int(seed)
    args = locals().copy()

    Path(save_to + '/' + stage).mkdir(parents=True, exist_ok=True)
    save_to_file = os.path.join(save_to, stage, f'all_info.npz')

    print('Args:')
    print(dict(args), '\n')

    spatialacoustics = SpatialAcoustics()
    roomir = RoomImpulseResponse(fs=fs, c=c, ism_db=ism_db)
    sa_cfgs = []

    for idx in range(data_num):
        if (idx+1)%10000 == 0:
            print('Generating MicSig config: {}/{} ({}%)'.format(idx+1, data_num, round((idx+1)/data_num*100,2)))
        sa_cfg = spatialacoustics.generate_random_spatial_acoustics(
            room_sz_range=room_sz_range,
            T60_range=T60_range,
            abs_weights_range=abs_weights_range,
            c=c,
            ism_db=ism_db,
            mic_array_cfg=mic_array_cfg,
            array_pos_ratio_range=array_pos_ratio_range,
            num_source_range=num_source_range,
            source_state=source_state,
            min_src_array_dist=min_src_array_dist,
            min_src_boundary_dist=min_src_boundary_dist,
            traj_pt_mode=traj_pt_mode,
            nb_points=nb_points,
            room_cfg=None,
            seed=seed,
            idx=idx
        )
        sa_cfgs.append(sa_cfg)

    np.savez_compressed(save_to_file, args=args, cfgs=sa_cfgs)

    mic_sig_or_rir = MicrophoneSignalOrRIR()

    srcdataset = VoxCeleb1ForSimuDataset(
        vox1_root=vox1_root,
        meta_file=meta_file,
        split=stage,
        T=T,
        fs=fs,
        num_source=max(num_source_range),
        size=data_num,
    )
    print(f'VoxCeleb1 source dataset: {len(srcdataset)} items, T={T}s, split={stage}')

    noidataset = NoiseSignal(
        T=T,
        fs=fs,
        nmic=mic_array_cfg['mic_pos_relative'].shape[0],
        noise_type=noise_type,
        noise_path=noi_dir,
        c=c
    )

    pbar = tqdm.tqdm(total=data_num)
    pbar.set_description(f'VoxCeleb1 ({stage}) micro-signals')

    gen_range = range(min(data_num, len(srcdataset)))

    if use_gpu:
        def init_env_var(gpus: List[int]):
            i = queue.get()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
            import gpuRIR
            importlib.reload(gpuRIR)

        queue = mp.Queue()
        for gid in gpus:
            queue.put(gid)

        p = mp.Pool(processes=len(gpus), initializer=init_env_var, initargs=(queue,))

        for _ in p.imap_unordered(
            partial(
                mic_sig_or_rir.generate_microphone_signal,
                sa_cfgs=sa_cfgs,
                fs=fs,
                c=c,
                roomir=roomir,
                srcdataset=srcdataset,
                noidataset=noidataset,
                snr_range=snr_range,
                save_to=os.path.join(save_to, stage),
                save_dp=save_dp,
                gpu_conv=gpu_conv,
                seed=seed,
            ),
            gen_range,
            chunksize=100,
        ):
            pbar.update()
        p.close()
        p.join()
    else:
        for idx in gen_range:
            mic_sig_or_rir.generate_microphone_signal(
                idx=idx,
                sa_cfgs=sa_cfgs,
                fs=fs,
                c=c,
                roomir=roomir,
                srcdataset=srcdataset,
                noidataset=noidataset,
                snr_range=snr_range,
                save_to=os.path.join(save_to, stage),
                save_dp=save_dp,
                gpu_conv=gpu_conv,
                seed=seed,
            )
            pbar.update()
    pbar.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate spatialized VoxCeleb1 data')
    parser.add_function_arguments(GenerateRandomMicSigVoxCeleb1)
    args = parser.parse_args()
    GenerateRandomMicSigVoxCeleb1(**vars(args))
