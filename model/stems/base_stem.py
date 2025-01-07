from typing import Union, List

import mmengine
import mmdet
from mmengine.model import BaseModule
import torch
from torchvision.ops.misc import Conv2dNormActivation
from registry import EV_MODELS

# CHECKED
@EV_MODELS.register_module()
class BaseStem(BaseModule):
    num_input = 1

    def __init__(self,
                 in_channels: int,
                 stem_channels: int,
                 active_layer: str = "Relu",
                 kernel_size: int = 3,
                 pool: bool = True,
                 init_cfg: Union[dict, List[dict], None] = None):
        if init_cfg is None:
            init_cfg = dict(
                type='Normal',
                std=0.01
            )
        super().__init__(init_cfg)
        self.conv = Conv2dNormActivation(in_channels=in_channels,
                                         out_channels=stem_channels,
                                         kernel_size=kernel_size, stride=2, norm_layer=torch.nn.BatchNorm2d,
                                         activation_layer=EV_MODELS.get(active_layer))
        if pool:
            self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool = None

    def forward(self, x):
        stem_feature = self.conv(x)
        if self.max_pool:
            stem_feature = self.max_pool(stem_feature)
        return stem_feature
