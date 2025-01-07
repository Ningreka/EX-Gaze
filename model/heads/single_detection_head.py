from typing import Tuple, Union, List

import torch
from torchvision.ops.misc import Conv2dNormActivation
import numpy as np
from mmengine.structures import InstanceData
from mmengine.model import BaseModule
from mmrotate.registry import TASK_UTILS as MMROTATE_TASK_UTILS
from mmrotate.registry import MODELS as MMROTATE_MODELS
from mmrotate.structures import RotatedBoxes
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.utils import unpack_gt_instances

from registry import EV_MODELS


# CHECKED
@EV_MODELS.register_module()
class SingleDetectionHead(BaseModule):
    def __init__(self, in_channels: int = 256,
                 pre_conv: int = 0,
                 pre_conv_channel: int = 128,
                 pre_conv_kernel: int = 3,
                 pre_stride: int = 2,
                 enlarge_channels: int = None,
                 feat_channels: int = 64,
                 stacked_convs: int = 2,
                 active_layer: str = "Relu",
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder: OptConfigType = None,
                 is_delta_coder: bool = True,
                 ref_bbox_shape: List[int] = None,
                 use_crop: bool = False,
                 weighted_loss: bool = True,
                 bbox_cls=RotatedBoxes,
                 loss_decoded_bbox=True,
                 loss_bbox: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        """

        :param feat_channels: int
        :param in_channels: int
        :param stacked_convs: int
        :param kernel_size: int
        :param active_layer: str
        :param train_cfg
        :param test_cfg
        :param bbox_coder: OptConfigType
        :param is_delta_coder: bool 
        :param ref_bbox_shape: List[int]
        :param use_crop: bool
        :param bbox_cls
        :param loss_decoded_bbox
        :param loss_bbox: OptConfigType
        :param init_cfg: OptConfigType
        """
        self._init_detection_cfg(
            bbox_coder,
            is_delta_coder,
            ref_bbox_shape,
            use_crop,
            weighted_loss,
            bbox_cls,
            loss_decoded_bbox,
            loss_bbox,
            init_cfg)

        if pre_conv > 0:
            self.pre_conv = torch.nn.Sequential()
            for i in range(pre_conv):
                self.pre_conv.append(
                    Conv2dNormActivation(
                        in_channels=in_channels,
                        out_channels=pre_conv_channel,
                        kernel_size=pre_conv_kernel,
                        stride=pre_stride,
                        norm_layer=torch.nn.BatchNorm2d,
                        activation_layer=EV_MODELS.get(active_layer)
                    )
                )
                in_channels = pre_conv_channel
                pre_stride = 1
        else:
            self.pre_conv = None

        if enlarge_channels is not None:
            self.conv_enlarge = Conv2dNormActivation(
                in_channels, enlarge_channels, kernel_size=1,
                norm_layer=torch.nn.BatchNorm2d,
                activation_layer=EV_MODELS.get(active_layer))
            in_channels = enlarge_channels
        else:
            self.conv_enlarge = None

        self.pool = torch.nn.AdaptiveAvgPool2d(1)

        if stacked_convs > 0:
            self.convs = torch.nn.Sequential()
            for i in range(stacked_convs):
                self.convs.append(
                    Conv2dNormActivation(
                        in_channels=in_channels,
                        out_channels=feat_channels,
                        kernel_size=1,
                        stride=1,
                        activation_layer=EV_MODELS.get(active_layer)
                    )
                )
                in_channels = feat_channels
        else:
            self.convs = None
            feat_channels = in_channels

        self.reg_bbox = torch.nn.Linear(feat_channels, out_features=self.bbox_coder.encode_size)

    def _init_detection_cfg(self,
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
                type='Normal',
                layer='Conv2d',
                std=0.01)
        if loss_bbox is None:
            loss_bbox = dict(type='GDLoss', loss_type='gwd')
        if bbox_coder is None:
            bbox_coder = dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version="le90",
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))

        super().__init__(init_cfg)
        self.bbox_coder = MMROTATE_TASK_UTILS.build(bbox_coder)
        self.bbox_cls = bbox_cls
        self.is_delta_coder = is_delta_coder
        if ref_bbox_shape is not None:
            assert len(ref_bbox_shape) == 2
        self.ref_bbox_shape = ref_bbox_shape
        self.use_crop = use_crop
        self.weighted_loss = weighted_loss
        self.loss_decoded_bbox = loss_decoded_bbox
        self.loss_bbox = MMROTATE_MODELS.build(loss_bbox)

    def forward(self, x: torch.Tensor):
        if self.pre_conv:
            x = self.pre_conv(x)

        if self.conv_enlarge:
            x = self.conv_enlarge(x)

        x = self.pool(x)

        if self.convs:
            x = self.convs(x)  # stacked conv

        x = torch.flatten(x, 1)

        return self.reg_bbox(x)

    def parse_ref_bboxes(self, batch_img_metas, device):
        if not self.is_delta_coder:
            return None
        ref_bboxes = []
        for img_meta in batch_img_metas:
            input_shape = img_meta["batch_input_shape"]
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

    def parse_cropped_bboxes(self, batch_img_metas, device) -> torch.Tensor:
        if not self.use_crop:
            return None
        else:
            cropped_bboxes = []
            for img_meta in batch_img_metas:
                cropped_area = img_meta["cropped_area"]
                cropped_bboxes.append(torch.Tensor([cropped_area]))
            cropped_bboxes = torch.cat(cropped_bboxes).to(device=device)
            return cropped_bboxes

    def get_target_bboxes(self, batch_gt_instances):

        target_bboxes = []
        if self.weighted_loss:
            loss_weights = []

        for gt_instance in batch_gt_instances:
            if isinstance(gt_instance["bboxes"], torch.Tensor):
                target = gt_instance["bboxes"]
            else:
                target = gt_instance["bboxes"].tensor
            if target.shape[0] != 1:
                raise ValueError(f"expect 1 gt box, but get {target}")
            if self.weighted_loss:
                loss_weights.append(gt_instance["weights"])

            target_bboxes.append(target)

        target_bboxes = self.bbox_cls(torch.cat(target_bboxes))

        loss_weights = torch.cat(loss_weights).squeeze() if self.weighted_loss else None
        return target_bboxes, loss_weights

    def decode_predict(self, box_preds, decode_ref_bboxes, cropped_bboxes=None, predict=True):
        if predict or self.loss_decoded_bbox:
            
            box_preds = self.bbox_coder.decode(decode_ref_bboxes, box_preds)
            
            if self.use_crop:
                assert cropped_bboxes is not None
                assert cropped_bboxes.shape[0] == box_preds.shape[0]
                # recover the cropped predict
                box_preds.tensor[..., :2] += cropped_bboxes[..., :2]
            return box_preds
        else:
            return box_preds

    def loss(self, x, batch_data_samples):
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(batch_data_samples)
        # get target gt bbox and corresponding loss weight
        target_bboxes, loss_weights = self.get_target_bboxes(batch_gt_instances)
        # get ref bbox for box decoding
        ref_bboxes = self.parse_ref_bboxes(batch_img_metas, target_bboxes.device)

        cropped_bboxes = self.parse_cropped_bboxes(batch_img_metas, target_bboxes.device)

        # predict bbox
        box_preds = self.forward(x)
        box_preds = self.decode_predict(box_preds, ref_bboxes, cropped_bboxes=cropped_bboxes, predict=False)

        if not self.loss_decoded_bbox:
            target_bboxes = self.bbox_coder.encode(ref_bboxes, target_bboxes)
        if self.weighted_loss:
            loss_weights = loss_weights.to(box_preds.device)
        return {"bbox_loss": self.loss_bbox(box_preds.tensor, target_bboxes.tensor, weight=loss_weights)}

    @staticmethod
    def unpack_img_metas(batch_data_samples):
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
        return batch_img_metas

    def predict(self, x, batch_data_samples):
        batch_img_metas = self.unpack_img_metas(batch_data_samples)
        ref_bboxes = self.parse_ref_bboxes(batch_img_metas, x.device)
        cropped_bboxes = self.parse_cropped_bboxes(batch_img_metas, x.device)

        # predict bbox
        box_preds = self.forward(x)
        box_preds = self.decode_predict(box_preds, ref_bboxes, cropped_bboxes=cropped_bboxes, predict=True)

        results = [InstanceData(bboxes=pred.detach()) for pred in box_preds]
        return results
