import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

from argparse import ArgumentParser

from configs._base_.data_split import train_user_list,val_user_list,test_user_list


def set_args(argparser: ArgumentParser):
    argparser.add_argument("--config_file", type=str, default="configs/train_config/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre0.5.py")


if __name__ == '__main__':
    p = ArgumentParser()
    set_args(p)
    args = p.parse_args()

    # config_file = Path(args.config_file)
    cfg = Config.fromfile(args.config_file)
    # cfg["train_cfg"].update(
    #     max_epochs=1
    # )
    # cfg.update(test_dataloader=cfg["train_dataloader"])
    cfg.test_dataloader.dataset.update(session_list=['102','201','202'],user_list =range(1,49),ann_file_name="origin_labelled_pupil_dataset.json")
    cfg.update(resume=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    runner = Runner.from_cfg(cfg)

    # runner.val()
    runner.test()
