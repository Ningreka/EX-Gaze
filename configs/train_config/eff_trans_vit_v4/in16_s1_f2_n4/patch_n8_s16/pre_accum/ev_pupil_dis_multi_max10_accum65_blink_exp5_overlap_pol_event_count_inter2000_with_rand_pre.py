from mmengine.config import read_base

with read_base():
    from configs.train_config.eff_trans_vit_v4.in16_s1_f2_n4.patch_n8_s16.pre_accum.ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre import *
    from configs.dataset_config.pre_accum_pol_even_count_inter2000.patch_n8_s16.multi_max10_accum65_blink_exp5_ev_overlap_pupil_disp_with_rand_pre import \
        train_dataloader, test_dataloader, val_dataloader

rand_dis = 1.5
train_dataloader["dataset"]["dataset_pipeline"][1].update(random_dis_range=rand_dis,
                                                          random_rot_range=0.3,
                                                          random_scale_range=0.1)

model.data_preprocessor.update(
    mean=[0.00148052, 0.00124122], std=[0.03893584, 0.0356804]
)

# event only pupil disp
work_dir = f"misc/result/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/multi_max10_accum65_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre_dis{rand_dis}/"
