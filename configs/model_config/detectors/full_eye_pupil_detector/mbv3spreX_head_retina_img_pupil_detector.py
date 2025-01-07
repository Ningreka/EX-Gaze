from model.backbones.mobilenet import MobileNetBackbonePreX
from mmengine.config import read_base

with read_base():
    from configs.model_config.detectors.full_eye_pupil_detector.mbv3s_head_retina_img_pupil_detector import img_pupil_detector

img_pupil_detector.backbone=dict(
    type=MobileNetBackbonePreX,
    net_version="v3_small",
    with_stem=False,
    reduced_tail=False,
    fake_multi_scale=True,
    pre_x_layers=8
)

img_pupil_detector.bbox_head.update(
    stacked_convs=2,
    in_channels=48,
)

img_pupil_detector.bbox_head.anchor_generator.update(
    scales=[1, 1.5,2, 3],
    base_sizes=[16],
    ratios=[1.0],
    strides=[16],
)
img_pupil_detector.bbox_head.train_cfg.assigner.update(
    pos_iou_thr=0.3,
    neg_iou_thr=0.1)
