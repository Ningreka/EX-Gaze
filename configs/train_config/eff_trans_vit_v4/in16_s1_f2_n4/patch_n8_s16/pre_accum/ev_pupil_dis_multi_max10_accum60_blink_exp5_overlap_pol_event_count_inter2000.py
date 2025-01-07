from mmengine.config import read_base

with read_base():
    from configs.train_config.eff_trans_vit_v4.in16_s1_f2_n4.patch_n8_s16.pre_accum.ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre import *
    from configs.dataset_config.pre_accum_pol_even_count_inter2000.patch_n8_s16.multi_max10_accum60_blink_exp5_ev_overlap_pupil_disp_with_rand_pre import \
        train_dataloader, test_dataloader, val_dataloader

train_dataloader["dataset"]["dataset_pipeline"].pop(1)

val_dataloader["dataset"]["dataset_pipeline"].pop(1)

test_dataloader["dataset"]["dataset_pipeline"].pop(1)

model.data_preprocessor.update(
    mean=[0.00138146, 0.00116608],std= [0.03758078, 0.03458747]
)

# event only pupil disp
work_dir = f"misc/result/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/multi_max10_accum60_blink_exp5_overlap_pol_event_count_inter2000/"
