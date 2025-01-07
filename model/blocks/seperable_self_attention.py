from functools import partial
from typing import Optional, Callable
from collections import OrderedDict

import torch
from mmengine.model import BaseModule
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation
from torchvision.ops.misc import ConvNormActivation

from registry import EV_MODELS


class LinearSelfAttention(BaseModule):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
            self,
            embed_dim: int,
            attn_dropout: Optional[float] = 0.0,
            bias: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = Conv2dNormActivation(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            norm_layer=None,
            activation_layer=None
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = Conv2dNormActivation(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            norm_layer=None,
            activation_layer=None
        )
        self.embed_dim = embed_dim

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out


class LinearSelfAttentionConv1d(BaseModule):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, N)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(B, C, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    """

    def __init__(
            self,
            embed_dim: int,
            attn_dropout: Optional[float] = 0.0,
            bias: Optional[bool] = True,
    ) -> None:
        super().__init__()

        self.qkv_proj = ConvNormActivation(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            norm_layer=None,
            activation_layer=None,
            conv_layer=nn.Conv1d
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvNormActivation(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            norm_layer=None,
            activation_layer=None,
            conv_layer=nn.Conv1d
        )
        self.embed_dim = embed_dim

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # [B, C, N] --> [B, h + 2d, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, N]
        # value, key --> [B, d, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out


class LinearSelfAttention_v2(BaseModule):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(B, N, C)` where :math:`B` is the batch size, :math:`C` is the input channels,
         and :math:`N` is the number of patches
        - Output: same as the input

    """

    def __init__(
            self,
            embed_dim: int,
            attn_dropout: Optional[float] = 0.0,
            bias: Optional[bool] = True,
    ) -> None:
        super().__init__()

        self.qkv_proj = nn.Linear(
            in_features=embed_dim, out_features=1 + (2 * embed_dim), bias=bias
        )  # no norm, no act

        self.attn_dropout = nn.Dropout(p=attn_dropout)

        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=bias
        )  # no norm, no act
        self.embed_dim = embed_dim

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # [B, N, C] --> [B, N, 1 + 2d]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, N, 1]
        # value, key --> [B, N, d]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=2
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # [B, N, 1].T @ [B, N, d] --> [B, 1, d]
        context_vector = torch.bmm(torch.transpose(context_scores, 1, 2), key)

        # combine context vector with values
        # [B, N, d] * [B, 1, d] --> [B, N, d]
        out = torch.mul(F.relu(value), context_vector)
        out = self.out_proj(out)
        return out


class LinearAttnFFN(BaseModule):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
            self,
            opts,
            embed_dim: int,
            ffn_latent_dim: int,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.1,
            ffn_dropout: Optional[float] = 0.0,
            norm_layer: Optional[str] = "layer_norm_2d",
            act_layer=nn.Hardswish
    ) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(
            opts, embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Conv2dNormActivation(
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                norm_layer=None,
                activation_layer=act_layer
            ),
            nn.Dropout(p=ffn_dropout),
            Conv2dNormActivation(
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                norm_layer=None,
                activation_layer=None
            ),
            nn.Dropout(p=dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__repr__()
        self.norm_name = norm_layer

    def forward(
            self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # self-attention
        x = x + self.pre_norm_attn(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class LinearAttnFFN_v2(BaseModule):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, N, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

    Shape:
        - Input: :math:`(B, N, C_{in})` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.1,
            ffn_dropout: Optional[float] = 0.0,
            act_layer=nn.Hardswish
    ) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention_v2(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            # [B, N, ed] --> [B, N, fd]
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_layer(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__repr__()
        self.norm_name = "layer norm"

    def forward(self, x: Tensor) -> Tensor:
        # self-attention
        x = x + self.pre_norm_attn(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


@EV_MODELS.register_module()
class LinearAttnEncoder(BaseModule):
    def __init__(
            self,
            patch_num: int,
            attn_unit_dim: int,
            ffn_multiplier=2.0,
            n_attn_blocks: Optional[int] = 2,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            act_layer=nn.ReLU,
            norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            learnable_position_embeddings: bool = True
    ):
        super().__init__()
        if learnable_position_embeddings:
            self.pos_embedding = nn.Parameter(torch.empty(1, patch_num, attn_unit_dim).normal_(std=0.02))  # from BERT
        else:
            theta_0 = torch.pi * 2 / patch_num
            theta_0_0 = theta_0 / attn_unit_dim
            angels = torch.arange(0, patch_num * attn_unit_dim).reshape((1, patch_num, attn_unit_dim))  # 0 - 47,47-..
            angels = angels * theta_0_0 - theta_0 / 2
            self.pos_embedding = nn.Parameter(torch.sin(angels), requires_grad=False)
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(n_attn_blocks):
            layers[f"encoder_layer_{i}"] = LinearAttnFFN_v2(
                embed_dim=attn_unit_dim,
                ffn_latent_dim=int(attn_unit_dim * ffn_multiplier),
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                act_layer=act_layer
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(attn_unit_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))