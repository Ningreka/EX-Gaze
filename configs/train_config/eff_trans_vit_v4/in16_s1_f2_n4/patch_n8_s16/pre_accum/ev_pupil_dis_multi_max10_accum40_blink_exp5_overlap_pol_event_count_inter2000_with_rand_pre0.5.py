from mmengine.config import read_base

with read_base():
    from configs.train_config.eff_trans_vit_v4.in16_s1_f2_n4.patch_n8_s16.pre_accum.ev_pupil_dis_multi_max10_accum40_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre import *

rand_dis = 0.5
train_dataloader["dataset"]["dataset_pipeline"][1].update(random_dis_range=rand_dis,
                                                          random_rot_range=0.1,
                                                          random_scale_range=0)

test_dataloader["dataset"]["dataset_pipeline"][1].update(random_dis_range=0,
                                                         random_rot_range=0,
                                                         random_scale_range=0)

val_dataloader["dataset"]["dataset_pipeline"][1].update(random_dis_range=0,
                                                        random_rot_range=0,
                                                        random_scale_range=0)

# event only pupil disp
work_dir = f"misc/result/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/multi_max10_accum40_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre_dis{rand_dis}/"
