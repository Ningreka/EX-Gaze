from mmengine.config import read_base

with read_base():
    from configs._base_.default_runtime import *
    from configs._base_.defualt_schedule import *
    from configs.model_config.detectors.efficient_trans_v4.cnn_16_s1_transformer_f2_n4 import \
        efficient_trans_vit as model
    from configs.dataset_config.pre_accum_pol_even_count_inter2000.patch_n8_s16.multi_max10_accum50_blink_exp5_ev_overlap_pupil_disp_with_rand_pre import \
        train_dataloader, val_dataloader, test_dataloader

rand_dis = 1.5  # perturbation
train_dataloader["dataset"]["dataset_pipeline"][1].update(random_dis_range=rand_dis,
                                                          random_rot_range=0.3,
                                                          random_scale_range=0.1)

model.data_preprocessor.update(
    mean=[0.00118537, 0.00101531],std= [0.03475869, 0.03229738]
)

model.update(
    init_cfg=[
        dict(type="Xavier", layer="Linear"),
        dict(type='Kaiming', layer='Conv2d')
    ]
)

model.bbox_head.ref_bbox_shape = [50, 50]

max_epochs = 80

train_cfg.update(
    max_epochs=max_epochs
)
# learning rate

base_lr = 0.004 / 16

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=5.0e-6,
        by_epoch=False,
        begin=0,
        end=4000),
    dict(
        type='CosineAnnealingLR',
        eta_min=2.0e-5,
        begin=40,
        end=80,
        T_max=40,
        by_epoch=True,
        convert_to_iter_based=True),
]

# optimizer
optim_wrapper["optimizer"].update(
    type="torch.optim.AdamW",
    lr=base_lr,
    eps=1e-8,
    weight_decay=0.05
)

# event only pupil disp
work_dir = f"misc/result/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre_dis{rand_dis}/"
experiment_name = "trans vit"
