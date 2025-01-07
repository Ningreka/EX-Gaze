from typing import Optional, Callable, List

import torch
from mmdet.utils import ConfigType, OptConfigType
import torch.nn as nn
from mmengine.model import BaseModule
from torchvision.ops import Conv2dNormActivation

from model.blocks.mbconv_v3 import InvertedResidual, FusedMBConv, ConvConfig, MBConvConfig, FusedMBConvConfig
from model.blocks.seperable_self_attention import LinearAttnEncoder
from model.detectors.base_disp_detector import BaseDispDetector
from registry import EV_MODELS

CNNEncoderConfig_stem_only = [ConvConfig(2, 16, 3, 2, 1, nn.SiLU)]

CNNEncoderConfig_16_s0 = [
    ConvConfig(2, 16, 3, 2, 1, nn.SiLU),  # in 16 out 8
    FusedMBConvConfig(16, 16, 16, 3, 1, 1, True),  # in 8 out 8
    FusedMBConvConfig(16, 24, 64, 3, 2, 1, False),  # in 8 out 4
    FusedMBConvConfig(24, 40, 72, 3, 1, 0, False),  # in 4 out 2
    FusedMBConvConfig(40, 48, 120, 2, 1, 0, False)  # in 2 out 1
]

CNNEncoderConfig_16_s1 = [
    ConvConfig(2, 16, 3, 2, 0, nn.SiLU),  # in 16 out 7
    FusedMBConvConfig(16, 32, 48, 3, 2, 0, False),  # in 7 out 3
    FusedMBConvConfig(32, 32, 96, 3, 1, 1, False),  # in 3 out 3
    FusedMBConvConfig(32, 48, 96, 3, 1, 0, False)  # in 3 out 1
]

CNNEncoderConfig_16_b0 = [
    ConvConfig(2, 24, 3, 2, 1, nn.SiLU),  # in 16 out 8
    FusedMBConvConfig(24, 40, 72, 3, 2, 1, False),  # in 8 out 4
    FusedMBConvConfig(40, 40, 120, 3, 1, 1, True),  # in 4 out 4
    FusedMBConvConfig(40, 48, 120, 3, 1, 0, False),  # in 4 out 2
    FusedMBConvConfig(48, 64, 144, 2, 1, 0, False)  # in 2 out 1
]

CNNEncoderConfig_16_b1 = [
    ConvConfig(2, 24, 3, 2, 0, nn.SiLU),  # in 16 out 7
    FusedMBConvConfig(24, 48, 72, 3, 2, 0, False),  # in 7 out 3
    FusedMBConvConfig(48, 48, 144, 3, 1, 1, False),  # in 3 out 3
    FusedMBConvConfig(48, 64, 144, 3, 1, 0, False)  # in 3 out 1
]


cnn_config_map = {
    "CNNEncoderConfig_0": [],
    "CNNEncoderConfig_stem_only": CNNEncoderConfig_stem_only,
    "CNNEncoderConfig_16_s0": CNNEncoderConfig_16_s0,
    "CNNEncoderConfig_16_s1":CNNEncoderConfig_16_s1,
    "CNNEncoderConfig_16_b0": CNNEncoderConfig_16_b0,
    "CNNEncoderConfig_16_b1": CNNEncoderConfig_16_b1
}


class CNNEncoder(BaseModule):
    # [B*P,C,H,W] -> [B*P,D,1,1]
    # []
    def __init__(
            self,
            in_channels: int,
            input_size: int,
            conv_config_list: List[ConvConfig],
            pool_conv_feat: bool = False
    ):
        super().__init__()

        self.conv_list = nn.Sequential()
        self.out_channels = in_channels
        self.inter_out_size = input_size

        if len(conv_config_list) > 0:
            assert (conv_config_list[0].block == Conv2dNormActivation)
            if in_channels is not None:
                conv_config_list[0].in_channels = in_channels
            self.conv_list.append(
                Conv2dNormActivation(
                    conv_config_list[0].in_channels,
                    conv_config_list[0].out_channels,
                    conv_config_list[0].kernel_size,
                    conv_config_list[0].stride,
                    conv_config_list[0].padding,
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=conv_config_list[0].activation_layer
                )
            )
            self.out_channels = conv_config_list[0].out_channels
            self.inter_out_size = conv_config_list[0].cal_output_size(self.inter_out_size)
            for conv_config in conv_config_list[1:]:
                assert (self.out_channels == conv_config.in_channels)
                self.conv_list.append(conv_config.block(conv_config))
                self.out_channels = conv_config.out_channels
                self.inter_out_size = conv_config.cal_output_size(self.inter_out_size)

        if pool_conv_feat:
            self.conv_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.conv_pool = None
            self.out_channels = self.out_channels * self.inter_out_size * self.inter_out_size

    def forward(self, x):
        conv_feat: torch.Tensor = self.conv_list(x)  # [B*P,in_c,in_size,in_size] -> [B*P,D,out_s,out_s]
        if self.conv_pool:
            conv_feat = self.conv_pool(conv_feat)  # [B*P,D,1,1]
        conv_feat: torch.Tensor = torch.flatten(conv_feat, start_dim=1)  # [B*P,out_c]
        return conv_feat


@EV_MODELS.register_module()
class EfficientTransVit(BaseDispDetector):

    def __init__(
            self,
            patch_num: int,
            patch_size: int,
            in_channels: int,
            cnn_encoder_config: str,
            transformer_encoder_config: ConfigType,
            bbox_head: ConfigType,
            pool_conv_feat: bool = False,
            data_preprocessor: OptConfigType = None,
            init_cfg: Optional[dict] = None
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.cnn_stage = CNNEncoder(in_channels, patch_size, cnn_config_map[cnn_encoder_config],
                                    pool_conv_feat=pool_conv_feat)
        self.conv_dim = self.cnn_stage.out_channels

        self.patch_num = patch_num
        self.patch_size = patch_size
        if transformer_encoder_config is not None:
            assert (self.patch_num == transformer_encoder_config["patch_num"] and
                    self.conv_dim == transformer_encoder_config["attn_unit_dim"])
            self.transformer_stage = EV_MODELS.build(transformer_encoder_config)
        else:
            self.transformer_stage = None

        assert bbox_head["hidden_dim"] == self.conv_dim
        self.bbox_head = EV_MODELS.build(bbox_head)

    def extract_feat(self, batch_inputs: torch.Tensor):
        B, P, C, H, W = batch_inputs.shape
        torch._assert(H == W and H == self.patch_size, f"size must be {self.patch_size}*{self.patch_size}")
        torch._assert(self.patch_num == P, f"expected patch num is {self.patch_num}, but get {P}")
        conv_feat: torch.Tensor = self.cnn_stage(batch_inputs.view(-1, C, H, W))  # [B*P,D]

        conv_feat = conv_feat.view(B, P, self.conv_dim)  # [B,P,D]

        if self.transformer_stage:
            trans_feat = self.transformer_stage(conv_feat)  # [B, P, D]
        else:
            trans_feat = conv_feat

        return torch.mean(trans_feat, dim=1)
