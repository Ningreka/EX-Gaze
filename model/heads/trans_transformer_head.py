from functools import partial
from typing import List, Callable, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.utils import OptConfigType
from mmrotate.structures import RotatedBoxes
from mmrotate.registry import TASK_UTILS as MMROTATE_TASK_UTILS
from mmrotate.registry import MODELS as MMROTATE_MODELS

from model.heads.single_displacement_head import SingleDisplacementHead
from registry import EV_MODELS


@EV_MODELS.register_module()
class TransTransformerHead(SingleDisplacementHead):
    """

    """

    def __init__(self,
                 hidden_dim: int,
                 pool_dim: int,
                 is_distil: bool,
                 norm_layer: OptConfigType = None,
                 bbox_coder: OptConfigType = None,
                 is_delta_coder: bool = True,
                 ref_bbox_shape: List[int] = None,
                 use_crop: bool = False,
                 weighted_loss: bool = True,
                 bbox_cls=RotatedBoxes,
                 loss_decoded_bbox=True,
                 loss_bbox: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        if init_cfg is None:
            init_cfg = dict(
                type='Xavier',
                layer='Linear')

        super()._init_detection_cfg(
            bbox_coder,
            is_delta_coder,
            ref_bbox_shape,
            use_crop,
            weighted_loss,
            bbox_cls,
            loss_decoded_bbox,
            loss_bbox,
            init_cfg)

        if norm_layer is not None:
            self.norm = MMROTATE_MODELS.build(norm_layer)
        else:
            self.norm = None

        self.pool_dim = pool_dim

        self.reg_bbox = nn.Linear(hidden_dim, out_features=self.bbox_coder.encode_size)

        self.loss_decoded_bbox = loss_decoded_bbox

    def forward(self, x: torch.Tensor):
        if self.norm:
            x = self.norm(x)
        if self.pool_dim is not None:
            return self.reg_bbox(x.mean(self.pool_dim))
        else:
            return self.reg_bbox(x)

    def parse_ref_bboxes(self, batch_img_metas, device):
        if not self.is_delta_coder:
            return None
        ref_bboxes = []
        for img_meta in batch_img_metas:
            input_shape = img_meta["ori_shape"]
            if self.ref_bbox_shape is not None:
                ref_bboxes.append(
                    torch.Tensor(
                        [[input_shape[1] / 2, input_shape[0] / 2, self.ref_bbox_shape[1], self.ref_bbox_shape[0], 0]])
                )
            else:
                ref_bboxes.append(torch.Tensor(
                    [[input_shape[1] / 2, input_shape[0] / 2, input_shape[1], input_shape[0], 0]]))
        ref_bboxes = torch.cat(ref_bboxes).to(device=device)

        ref_bboxes = self.bbox_cls(ref_bboxes)
        return ref_bboxes