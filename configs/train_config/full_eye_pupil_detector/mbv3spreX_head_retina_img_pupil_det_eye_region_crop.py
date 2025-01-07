from mmengine.config import read_base
from metrics.EllipseMetric import BestPredictEllipseMetric

with read_base():
    from configs._base_.defualt_schedule import *
    from configs._base_.default_runtime import *
    from configs.model_config.detectors.full_eye_pupil_detector.mbv3spreX_head_retina_img_pupil_detector import \
        img_pupil_detector as model
    from configs.dataset_config.pupil_detect.pupil_img_eye_region_crop import train_dataloader, \
        val_dataloader, test_dataloader

val_evaluator = dict(type=BestPredictEllipseMetric, mask_size=(260, 346), prefix="ev_eye pupil")
test_evaluator = val_evaluator.copy()

model.update(init_cfg=dict(
    type="Kaiming",
    layer="Conv2d"))

max_epochs = 100

train_cfg.update(
    max_epochs=max_epochs
)
# learning rate

base_lr = 2.5e-4

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=8e-5,
        begin=40,
        end=70,
        T_max=30,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=1.5e-5,
        begin=80,
        end=100,
        T_max=20,
        by_epoch=True,
        convert_to_iter_based=True),
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
)

work_dir = f"misc/result/mbv3spreX_head_retina_img_pupil_det/eye_region_crop"
experiment_name = "img pupil detection"