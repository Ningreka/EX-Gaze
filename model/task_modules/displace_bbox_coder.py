from typing import Union

import torch
from mmdet.models.task_modules.coders import BaseBBoxCoder
from mmdet.structures.bbox import BaseBoxes

from registry import EV_TASK_UTILS

@EV_TASK_UTILS.register_module()
class DisplaceBBoxCoder(BaseBBoxCoder):
    def __init__(self, use_box_type: bool = False, **kwargs):
        super().__init__(use_box_type, **kwargs)

    def encode(self, bboxes: Union[torch.Tensor, BaseBoxes], gt_bboxes: Union[torch.Tensor, BaseBoxes]) -> torch.Tensor:
        assert isinstance(bboxes, (torch.Tensor, BaseBoxes))
        assert isinstance(gt_bboxes, (torch.Tensor, BaseBoxes))
        bboxes = bboxes.squeeze()
        gt_bboxes = gt_bboxes.squeeze()

        assert bboxes.shape == gt_bboxes.shape

        if isinstance(bboxes, torch.Tensor):
            pre_bboxes_tensor = bboxes
        else:
            pre_bboxes_tensor = bboxes.tensor
        if isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes_tensor = gt_bboxes
        else:
            gt_bboxes_tensor = gt_bboxes.tensor

        displace = gt_bboxes_tensor - pre_bboxes_tensor
        return displace

    def decode(self, bboxes: Union[torch.Tensor, BaseBoxes], bboxes_displace_pred: torch.Tensor):
        assert isinstance(bboxes, (torch.Tensor, BaseBoxes))
        assert isinstance(bboxes_displace_pred, torch.Tensor)
        bboxes = bboxes.squeeze()
        bboxes_displace_pred = bboxes_displace_pred.squeeze()

        assert bboxes.shape == bboxes_displace_pred.shape

        if isinstance(bboxes, torch.Tensor):
            pre_bboxes_tensor = bboxes
        else:
            pre_bboxes_tensor = bboxes.tensor

        bboxes_pred = bboxes_displace_pred + pre_bboxes_tensor

        if self.use_box_type:
            return bboxes.__class__(bboxes_pred)
        else:
            return bboxes_pred
