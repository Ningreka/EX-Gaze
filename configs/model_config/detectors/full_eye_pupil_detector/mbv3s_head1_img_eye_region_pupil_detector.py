#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 Chen Ning. All Rights Reserved 
#
# @Time    : 2024/3/30 22:09
# @Author  : Chen Ning
# @Email   : cn13328822083@163.com
# @File    : mbv3s_head1_img_eye_region_pupil_detector.py
# @Software: PyCharm

from mmengine.config import read_base

from ev_eye_model.detectors.base_pupil_detector import BasePupilDetector

with read_base():
    from configs.model_config.stems.img_stem import img_stem
    from configs.model_config.backbones.mobilenet_v3_small import mobilenet_v3_small
    from configs.model_config.heads.detection_heads.conv1_hardswish_eye_region_pupil_detection_head import eye_pupil_detection_head
    from configs.model_config.data_preprocessors.img_eye_region_pupil_data_preprocessor import img_eye_region_pupil_data_preprocessor

img_stem.update(
    pool=False,
    active_layer="Hardswish"
)

img_eye_region_pupil_detector = dict(
    type=BasePupilDetector,
    stem=img_stem,
    backbone=mobilenet_v3_small,
    bbox_head=eye_pupil_detection_head,
    data_preprocessor=img_eye_region_pupil_data_preprocessor
)
