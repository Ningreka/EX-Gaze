import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

from argparse import ArgumentParser


def set_args(argparser: ArgumentParser):
    argparser.add_argument("--config_file", type=str, default="configs/train_config/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre0.5.py")
    argparser.add_argument("--resume", action="store_true",default=False)

if __name__ == '__main__':
    p = ArgumentParser()
    set_args(p)
    args = p.parse_args()

    # config_file = Path(args.config_file)
    cfg = Config.fromfile(args.config_file)
    
    cfg["resume"] = args.resume
    torch.multiprocessing.set_sharing_strategy('file_system')

    runner = Runner.from_cfg(cfg)

    runner.train()
    runner.test()
