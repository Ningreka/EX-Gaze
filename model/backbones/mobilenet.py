from functools import partial
from typing import Union, List

import mmengine
import mmdet
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType
import logging
import torch
import torchvision
from torchvision.models.mobilenetv3 import InvertedResidualConfig
from torchvision.ops import Conv2dNormActivation

from registry import EV_MODELS


@EV_MODELS.register_module()
class MobileNetBackbone(BaseModule):
    # down sample scale x16
    arch_settings = {
        "v2": (32, 320),
        "v3_small": (16, 96),
        "v3_large": (16, 160)
    }  # (in chanel, out channel)

    def __init__(self,
                 net_version: str,
                 with_stem: bool = False,
                 stem_input_channels: int = 1,
                 reduced_tail: bool = False,
                 fake_multi_scale: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        if init_cfg is None:
            init_cfg = dict(
                type='Xavier'
            )
        if init_cfg == "pretrained":
            super().__init__(None)
            with_pretrain = True
            logging.log(logging.WARNING, "backbone use pre trained weights")
        else:
            super().__init__(init_cfg)
            with_pretrain = False
        assert (net_version in self.arch_settings.keys())
        self.net_version = net_version
        self.stem_channels = self.arch_settings[self.net_version][0]
        self.output_channels = self.arch_settings[self.net_version][1]
        if self.net_version != "v2" and reduced_tail:
            self.output_channels = self.output_channels // 2

        if self.net_version == "v2":
            self.features = torchvision.models.mobilenet_v2(
                weights=torchvision.models.MobileNet_V2_Weights if with_pretrain else None).features
            # self.features.pop(18)
        elif self.net_version == "v3_small":
            self.features = torchvision.models.mobilenet_v3_small(
                weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if with_pretrain else None,
                reduced_tail=reduced_tail).features
            # self.features.pop(12)
        else:
            self.features = torchvision.models.mobilenet_v3_large(
                weights=torchvision.models.MobileNet_V3_Large_Weights if with_pretrain else None,
                reduced_tail=reduced_tail).features
            # self.features.pop(16)

        if not with_stem:
            self.features.pop(0)
        else:
            stem_conv: Conv2dNormActivation = self.features[0]
            stem_conv[0] = torch.nn.Conv2d(
                stem_input_channels,
                stem_conv.out_channels,
                3,
                2,
                1,
            )
        self.features.pop(-1)

        self.fake_multi_scale = fake_multi_scale

    def forward(self, stem_feature):
        feat = self.features(stem_feature)
        if self.fake_multi_scale:
            return tuple([feat])
        else:
            return feat


@EV_MODELS.register_module()
class MobileNetBackbonePreX(MobileNetBackbone):
    """use pre x layers"""

    def __init__(self,
                 net_version: str,
                 with_stem: bool = False,
                 stem_input_channels: int = 1,
                 reduced_tail: bool = False,
                 fake_multi_scale: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None,
                 pre_x_layers=9):
        super().__init__(net_version, with_stem, stem_input_channels,
                         reduced_tail, fake_multi_scale, init_cfg)

        assert pre_x_layers < len(self.features)
        self.features = self.features[:pre_x_layers]
        self.output_channels = self.features[-1].out_channels
