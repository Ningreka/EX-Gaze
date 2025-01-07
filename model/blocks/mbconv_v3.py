from typing import List, Callable

from torch import nn, Tensor
from torchvision.models._utils import _make_divisible
from torchvision.ops import Conv2dNormActivation, SqueezeExcitation, StochasticDepth


class ConvConfig:
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            activation_layer: nn.Module = nn.ReLU
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_layer = activation_layer

        self.block = Conv2dNormActivation

    def cal_output_size(self,input_size):
        return (input_size + 2 * self.padding - self.kernel_size) // self.stride + 1

class MBConvConfig(ConvConfig):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expanded_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            use_se: bool,
            use_res: bool,
            activation_layer: nn.Module = nn.ReLU,
            se_scale_activation=nn.Hardsigmoid
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, activation_layer)
        self.expanded_channels = expanded_channels
        self.use_se = use_se
        self.use_res = use_res
        self.se_scale_activation = se_scale_activation

        self.block = InvertedResidual


class FusedMBConvConfig(ConvConfig):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expanded_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            use_res: bool,
            stochastic_depth_prob: float = 0.2,
            activation_layer=nn.SiLU,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, activation_layer)
        self.expanded_channels = expanded_channels
        self.use_res = use_res
        self.stochastic_depth_prob = stochastic_depth_prob

        self.block = FusedMBConv


class InvertedResidual(nn.Module):
    def __init__(self,
                 config: MBConvConfig
                 ) -> None:
        super().__init__()

        if config.use_res:
            assert(config.stride == 1 and config.in_channels == config.out_channels)
        self.use_res_connect = config.use_res

        layers: List[nn.Module] = []
        # activation_layer = nn.Hardswish if use_hs else nn.ReLU

        # expand
        if config.expanded_channels != config.in_channels:
            layers.append(
                Conv2dNormActivation(
                    config.in_channels,
                    config.expanded_channels,
                    kernel_size=1,
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=config.activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                config.expanded_channels,
                config.expanded_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                groups=config.expanded_channels,
                norm_layer=nn.BatchNorm2d,
                activation_layer=config.activation_layer,
            )
        )
        if config.use_se:
            squeeze_channels = _make_divisible(config.expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(config.expanded_channels, squeeze_channels, scale_activation=config.se_scale_activation))

        # project
        layers.append(
            Conv2dNormActivation(
                config.expanded_channels, config.out_channels, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=None
            )
        )

        self.blocks = nn.Sequential(*layers)
        self.out_channels = config.out_channels
        self._is_cn = config.stride > 1

    def forward(self, input):
        result = self.blocks(input)
        if self.use_res_connect:
            result += input
        return result


class FusedMBConv(nn.Module):
    def __init__(
            self,
            config:FusedMBConvConfig
    ) -> None:
        super().__init__()

        if not (1 <= config.stride <= 2):
            raise ValueError("illegal stride value")

        if config.use_res:
            assert (config.stride == 1 and config.in_channels == config.out_channels)
        self.use_res = config.use_res

        layers: List[nn.Module] = []
        # activation_layer = nn.SiLU

        if config.expanded_channels != config.in_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    config.in_channels,
                    config.expanded_channels,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=config.activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    config.expanded_channels, config.out_channels, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    config.in_channels,
                    config.out_channels,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=config.activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(config.stochastic_depth_prob, "row")
        self.out_channels = config.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res:
            result = self.stochastic_depth(result)
            result += input
        return result
