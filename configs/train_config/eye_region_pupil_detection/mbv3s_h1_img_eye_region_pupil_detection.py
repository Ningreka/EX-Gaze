from mmengine.config import read_base
from metrics.EyePupilMetric import EyePupilMetric

with read_base():
    from configs._base_.default_runtime import *
    from configs._base_.defualt_schedule import *
    from configs.dataset_config.eye_pupil_detection.eye_region_pupil_detection import train_dataloader, val_dataloader, \
        test_dataloader
    from configs.model_config.detectors.full_eye_pupil_detector.mbv3s_head1_img_eye_region_pupil_detector import \
        img_eye_region_pupil_detector as model

val_evaluator = dict(type=EyePupilMetric, mask_size=(260, 346), prefix="ev_eye_region&pupil")
test_evaluator = val_evaluator.copy()


model.update(init_cfg=dict(
    type="Kaiming",
    layer="Conv2d"))

max_epochs = 80

train_cfg.update(
    max_epochs=max_epochs
)
# learning rate

base_lr = 3e-4

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=8000),
    dict(
        type='CosineAnnealingLR',
        eta_min=1.5e-5,
        begin=40,
        end=80,
        T_max=40,
        by_epoch=True,
        convert_to_iter_based=True),
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
)

work_dir = f"misc/result/img_eye_region_pupil_detection/mbv3s_h1"
experiment_name = "img pupil detection"
