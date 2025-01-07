from typing import Union, List, Callable

import math

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torchvision.ops import Conv2dNormActivation
from registry import EV_MODELS, OptionLayerType

@EV_MODELS.register_module()
class TransformerStem(BaseModule):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int,
                 last_kernel: int,
                 last_stride: int,
                 out_channels: List[int] = [],
                 kernels: List[int] = [],
                 strides: List[int] = [],
                 norm_layers: Union[List[OptionLayerType], OptionLayerType] = [],
                 active_layers: Union[List[OptionLayerType], OptionLayerType] = [],
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)

        assert (len(out_channels) == len(kernels) and len(kernels) == len(strides))
        conv_num = len(out_channels)
        if not isinstance(norm_layers, List):
            norm_layers = [norm_layers] * conv_num
        else:
            assert len(norm_layers) == conv_num
        if not isinstance(active_layers, List):
            active_layers = [active_layers] * conv_num
        else:
            assert len(active_layers) == conv_num

        if conv_num > 0:
            seq_proj = nn.Sequential()
            for i, (oc, k, s, n, a) in enumerate(zip(out_channels, kernels, strides, norm_layers, active_layers)):
                seq_proj.add_module(
                    f"conv{k}_{str(n).split('.')[-1]}_{str(a).split('.')[-1]}_{i}",
                    Conv2dNormActivation(
                        in_channels=in_channels,
                        out_channels=oc,
                        kernel_size=k,
                        stride=s,
                        norm_layer=EV_MODELS.get(n) if isinstance(n, str) else n,
                        activation_layer=EV_MODELS.get(a) if isinstance(a, str) else a,
                    ),
                )
                in_channels = oc
            if last_kernel is not None and last_stride is not None:
                seq_proj.add_module(
                    "conv_last",
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=last_kernel,
                              stride=last_stride)
                )
            self.conv_proj: nn.Module = seq_proj
        else:
            assert last_kernel is not None and last_stride is not None
            self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=last_kernel,
                                       stride=last_stride)

    def init_weights(self):
        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        elif isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

    def forward(self, x):
        return self.conv_proj(x)
