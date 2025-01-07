from typing import List

import torch
from mmdet.models import L1Loss
from mmdet.models.task_modules import DeltaXYWHBBoxCoder
from mmdet.utils import OptConfigType
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
from mmdet.registry import MODELS as MMDET_MODELS
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmrotate.models import DeltaXYWHTRBBoxCoder
from mmrotate.structures import RotatedBoxes
from mmrotate.registry import TASK_UTILS as MMROTATE_TASK_UTILS
from mmrotate.registry import MODELS as MMROTATE_MODELS
from torchvision.ops import Conv2dNormActivation

from model.heads.single_detection_head import SingleDetectionHead
from registry import EV_MODELS


def unpack_gt_instances(batch_data_samples, only_meta=False):
    batch_eye_region_instances = []
    batch_pupil_region_instances = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        if not only_meta:
            batch_eye_region_instances.append(data_sample.eye_region_instance)
            batch_pupil_region_instances.append(data_sample.pupil_instance)
    if only_meta:
        return batch_img_metas
    else:
        return batch_eye_region_instances, batch_pupil_region_instances, batch_img_metas


def _build_stack_convs(stacked_convs, in_channels, feat_channels, active_layer):
    convs = torch.nn.Sequential()
    for i in range(stacked_convs):
        convs.append(
            Conv2dNormActivation(
                in_channels=in_channels,
                out_channels=feat_channels,
                kernel_size=1,
                stride=1,
                activation_layer=EV_MODELS.get(active_layer)
            )
        )
        in_channels = feat_channels
    return convs


@EV_MODELS.register_module()
class EyePupilDetectionHead(BaseModule):
    def __init__(
            self,
            in_channels: int = 256,
            enlarge_channels: int = None,
            feat_channels: int = 64,
            stacked_convs: int = 2,
            uniform_reg: bool = False,
            active_layer: str = "Relu",
            is_delta_coder: bool = True,
            eye_region_ref_bbox_shape: List[int] = None,
            pupil_ref_bbox_shape: List[int] = None,
            reg_decoded_eye_region_bbox=False,
            reg_decoded_pupil_bbox=True,
            loss_eye_region_bbox: OptConfigType = None,
            loss_pupil_bbox: OptConfigType = None,
            init_cfg: OptConfigType = None
    ):
        # init_detection_cfg ----------------------------------------
        if init_cfg is None:
            init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01)
        super().__init__(init_cfg)

        if loss_eye_region_bbox is None:
            loss_eye_region_bbox = dict(type=L1Loss, loss_weight=1.0)
        if loss_pupil_bbox is None:
            loss_pupil_bbox = dict(type='GDLoss', loss_type='gwd')
        eye_region_bbox_coder = dict(
            type=DeltaXYWHBBoxCoder,
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0])
        pupil_bbox_coder = dict(
            type=DeltaXYWHTRBBoxCoder,
            angle_version="le90",
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            use_box_type=False
        )

        self.eye_region_bbox_coder = MMDET_TASK_UTILS.build(eye_region_bbox_coder)
        self.pupil_bbox_coder = MMROTATE_TASK_UTILS.build(pupil_bbox_coder)

        # self.eye_region_bbox_cls = HorizontalBoxes
        # self.pupil_bbox_cls = RotatedBoxes

        self.eye_region_ref_bbox_shape = eye_region_ref_bbox_shape
        if eye_region_ref_bbox_shape is not None:
            assert len(eye_region_ref_bbox_shape) == 2

        self.pupil_ref_bbox_shape = pupil_ref_bbox_shape
        if pupil_ref_bbox_shape is not None:
            assert len(pupil_ref_bbox_shape) == 2

        self.is_delta_coder = is_delta_coder
        self.reg_decoded_eye_region_bbox = reg_decoded_eye_region_bbox
        self.reg_decoded_pupil_bbox = reg_decoded_pupil_bbox

        self.loss_eye_region_bbox = MMDET_MODELS.build(loss_eye_region_bbox)
        self.loss_pupil_bbox = MMROTATE_MODELS.build(loss_pupil_bbox)

        # init model -------------------------------------------------------------

        if enlarge_channels is not None:
            self.conv_enlarge = Conv2dNormActivation(
                in_channels, enlarge_channels, kernel_size=1,
                norm_layer=torch.nn.BatchNorm2d,
                activation_layer=EV_MODELS.get(active_layer))
            in_channels = enlarge_channels
        else:
            self.conv_enlarge = None

        self.pool = torch.nn.AdaptiveAvgPool2d(1)

        self.uniform_reg = uniform_reg

        if self.uniform_reg:
            if stacked_convs > 0:
                self.uniform_stack_convs = _build_stack_convs(stacked_convs, in_channels, feat_channels, active_layer)
            else:
                self.uniform_stack_convs = torch.nn.Sequential()
                feat_channels = in_channels
            self.uniform_reg_bbox = torch.nn.Linear(feat_channels,
                                                    out_features=self.eye_region_bbox_coder.encode_size + self.pupil_bbox_coder.encode_size)

        else:
            if stacked_convs > 0:
                self.eye_region_stack_convs = _build_stack_convs(stacked_convs, in_channels, feat_channels, active_layer)
                self.pupil_stack_convs = _build_stack_convs(stacked_convs, in_channels, feat_channels, active_layer)
            else:
                self.eye_region_stack_convs = torch.nn.Sequential()
                self.pupil_stack_convs = torch.nn.Sequential()
                feat_channels = in_channels

            self.reg_eye_region_bbox = torch.nn.Linear(feat_channels,
                                                       out_features=self.eye_region_bbox_coder.encode_size)

            self.reg_pupil_bbox = torch.nn.Linear(feat_channels, out_features=self.pupil_bbox_coder.encode_size)

    def forward(self, x: torch.Tensor):
        if self.conv_enlarge:
            x = self.conv_enlarge(x)

        x = self.pool(x)

        if self.uniform_reg:
            x = self.uniform_stack_convs(x)  # stacked conv
            x = torch.flatten(x, 1)
            return self.uniform_reg_bbox(x)
        else:
            eye_feat = self.eye_region_stack_convs(x)  # stacked conv
            pupil_feat = self.pupil_stack_convs(x)

            eye_feat = torch.flatten(eye_feat, 1)
            pupil_feat = torch.flatten(pupil_feat, 1)

            return self.reg_eye_region_bbox(eye_feat), self.reg_pupil_bbox(pupil_feat)

    def get_target_bboxes(self, batch_eye_region_instances, batch_pupil_region_instances):
        target_eye_region_bboxes = []
        target_pupil_bboxes = []
        for gt_eye_region_instance, gt_pupil_instance in zip(batch_eye_region_instances,
                                                             batch_pupil_region_instances):
            eye_region_bbox = gt_eye_region_instance["bbox"]
            pupil_bbox = gt_pupil_instance["bbox"]

            eye_region_bbox = eye_region_bbox if isinstance(eye_region_bbox, torch.Tensor) else eye_region_bbox.tensor
            pupil_bbox = pupil_bbox if isinstance(pupil_bbox, torch.Tensor) else pupil_bbox.tensor

            target_eye_region_bboxes.append(eye_region_bbox)
            target_pupil_bboxes.append(pupil_bbox)

        # target_eye_region_bboxes = HorizontalBoxes(torch.cat(target_eye_region_bboxes))
        target_eye_region_bboxes = torch.cat(target_eye_region_bboxes)
        # target_pupil_bboxes = RotatedBoxes(torch.cat(target_pupil_bboxes))
        target_pupil_bboxes = torch.cat(target_pupil_bboxes)

        return target_eye_region_bboxes, target_pupil_bboxes

    def parse_ref_bboxes(self, batch_img_metas, device):
        if not self.is_delta_coder:
            return None, None
        input_shape = batch_img_metas[0]["batch_input_shape"]  # h,w
        if self.eye_region_ref_bbox_shape is not None:
            eye_region_ref_bboxes = torch.tensor([input_shape[1] / 2, input_shape[0] / 2,
                                                  self.eye_region_ref_bbox_shape[1],
                                                  self.eye_region_ref_bbox_shape[0]],
                                                 device=device)
        else:
            eye_region_ref_bboxes = torch.tensor(
                [input_shape[1] / 2, input_shape[0] / 2, input_shape[1], input_shape[0]], device=device)
        eye_region_ref_bboxes = HorizontalBoxes.cxcywh_to_xyxy(eye_region_ref_bboxes)
        eye_region_ref_bboxes = HorizontalBoxes(eye_region_ref_bboxes.repeat((len(batch_img_metas), 1)))

        if self.pupil_ref_bbox_shape is not None:
            pupil_ref_bboxes = torch.tensor([input_shape[1] / 2, input_shape[0] / 2,
                                             self.pupil_ref_bbox_shape[1], self.pupil_ref_bbox_shape[0], 0],
                                            device=device)
        else:
            pupil_ref_bboxes = torch.tensor(
                [input_shape[1] / 2, input_shape[0] / 2, input_shape[1] / 8, input_shape[0] / 8, 0], device=device)
        pupil_ref_bboxes = RotatedBoxes(pupil_ref_bboxes.repeat((len(batch_img_metas), 1)))

        return eye_region_ref_bboxes, pupil_ref_bboxes

    def decode_predict(self, box_preds, eye_region_ref_bboxes, pupil_ref_bboxes, predict=False):
        if self.uniform_reg:
            eye_region_pred = box_preds[:, :4]
            pupil_pred = box_preds[:, 4:]
        else:
            eye_region_pred, pupil_pred = box_preds

        if predict or self.reg_decoded_eye_region_bbox:
            eye_region_pred = self.eye_region_bbox_coder.decode(eye_region_ref_bboxes, eye_region_pred)
        if predict or self.reg_decoded_pupil_bbox:
            pupil_pred = self.pupil_bbox_coder.decode(pupil_ref_bboxes, pupil_pred)

        return eye_region_pred, pupil_pred

    def loss(self, x, batch_data_samples):
        batch_eye_region_instances, batch_pupil_region_instances, batch_img_metas = unpack_gt_instances(
            batch_data_samples)

        # get target gt bbox, eye region box pupil box -----------------
        target_eye_region_bboxes, target_pupil_bboxes = self.get_target_bboxes(batch_eye_region_instances,
                                                                               batch_pupil_region_instances)

        # get ref bbox for box decoding
        eye_region_ref_bboxes, pupil_ref_bboxes = self.parse_ref_bboxes(batch_img_metas,
                                                                        target_eye_region_bboxes.device)

        # predict bbox
        box_preds = self.forward(x)
        eye_region_preds, pupil_preds = self.decode_predict(box_preds, eye_region_ref_bboxes, pupil_ref_bboxes,
                                                            predict=False)

        if not self.reg_decoded_eye_region_bbox:
            target_eye_region_bboxes = self.eye_region_bbox_coder.encode(eye_region_ref_bboxes,
                                                                         target_eye_region_bboxes)
        if not self.reg_decoded_pupil_bbox:
            target_pupil_bboxes = self.pupil_bbox_coder.encode(pupil_ref_bboxes, target_pupil_bboxes)

        return {
            "eye_region_bbox_loss": self.loss_eye_region_bbox(eye_region_preds, target_eye_region_bboxes),
            "pupil_bbox_loss": self.loss_pupil_bbox(pupil_preds, target_pupil_bboxes)}

    def predict(self, x, batch_data_samples):
        batch_img_metas = unpack_gt_instances(batch_data_samples, only_meta=True)
        # get ref bbox for box decoding
        eye_region_ref_bboxes, pupil_ref_bboxes = self.parse_ref_bboxes(batch_img_metas, x.device)

        # predict bbox
        box_preds = self.forward(x)
        eye_region_preds, pupil_preds = self.decode_predict(box_preds, eye_region_ref_bboxes, pupil_ref_bboxes,
                                                            predict=True)

        results = [dict(eye_region_pred=InstanceData(bbox=torch.unsqueeze(e_pred.detach(), dim=0)),
                        pupil_pred=InstanceData(bbox=torch.unsqueeze(p_pred.detach(), dim=0)))
                   for (e_pred, p_pred) in zip(eye_region_preds, pupil_preds)]

        return results
