from mmcv.ops import nms_rotated
from mmdet.models import RetinaHead, FocalLoss
from mmdet.models.task_modules import MaxIoUAssigner, PseudoSampler
from mmengine.config import read_base
from mmrotate.models import FakeRotatedAnchorGenerator, DeltaXYWHTRBBoxCoder, GDLoss, RBboxOverlaps2D

from model.detectors.base_pupil_detector import BasePupilDetector
from model.backbones.mobilenet import MobileNetBackbone

with read_base():
    from configs.model_config.stems.img_stem import img_stem
    from configs.model_config.data_preprocessors.pupil_detect_data_preprocessor import pupil_detect_data_preprocessor

img_stem.update(
    pool=False,
    active_layer="Hardswish"
)

img_pupil_detector = dict(
    type=BasePupilDetector,
    stem=img_stem,
    backbone=dict(
        type=MobileNetBackbone,
        net_version="v3_small",
        with_stem=False,
        reduced_tail=False,
        fake_multi_scale=True
    ),
    bbox_head=dict(
        type=RetinaHead,
        num_classes=1,
        in_channels=96,
        stacked_convs=4,
        feat_channels=128,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type=FakeRotatedAnchorGenerator,
            angle_version="le90",
            # scales=[0.75, 1.5],
            base_sizes=[50],
            ratios=[1.0],
            scales=[1],
            strides=[32],
        ),
        bbox_coder=dict(
            type=DeltaXYWHTRBBoxCoder,
            angle_version="le90",
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type=GDLoss,
            loss_type='kld',
            # fun='log1p',
            tau=1,
            # sqrt=False,
            loss_weight=5.5),
        train_cfg=dict(
            assigner=dict(
                type=MaxIoUAssigner,
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type=RBboxOverlaps2D)),
            sampler=dict(
                type=PseudoSampler),  # Focal loss should use PseudoSampler
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.1,
            nms=dict(type=nms_rotated, iou_threshold=0.1),
            max_per_img=2000)),
    data_preprocessor=pupil_detect_data_preprocessor
)
