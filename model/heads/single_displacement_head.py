from typing import Dict

import torch
from mmdet.models.utils import unpack_gt_instances
from mmengine.structures import InstanceData

from model.heads.single_detection_head import SingleDetectionHead
from registry import EV_MODELS


@EV_MODELS.register_module()
class SingleDisplacementHead(SingleDetectionHead):

    def predict_bbox(self, x, pre_state, decode_ref_bboxes, cropped_bboxes=None, predict=True):
        displacement_preds = self.forward(x)
        if isinstance(pre_state, self.bbox_cls):
            pre_state_boxes = pre_state
        elif len(pre_state.shape) > 2: # TODO test
            pre_state_boxes = self.bbox_cls(pre_state.squeeze(1))
        else:
            pre_state_boxes = self.bbox_cls(pre_state)

        pre_encoded_box = self.bbox_coder.encode(decode_ref_bboxes, pre_state_boxes)

        box_preds = displacement_preds + pre_encoded_box

        box_preds = self.decode_predict(box_preds, decode_ref_bboxes, cropped_bboxes, predict)

        return box_preds

    def loss(self, x, pre_state, batch_data_samples):
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(batch_data_samples)
        # get target gt bbox and corresponding loss weight
        target_bboxes, loss_weights = self.get_target_bboxes(batch_gt_instances)
        # get ref bbox for box decoding
        ref_bboxes = self.parse_ref_bboxes(batch_img_metas, target_bboxes.device)
        cropped_bboxes = self.parse_cropped_bboxes(batch_img_metas, target_bboxes.device)

        # predict bbox
        box_preds = self.predict_bbox(x, pre_state, ref_bboxes, cropped_bboxes=cropped_bboxes, predict=False)

        if not self.loss_decoded_bbox:
            target_bboxes = self.bbox_coder.encode(ref_bboxes, target_bboxes)

        loss_weights = loss_weights.to(box_preds.device)
        return {"bbox_loss": self.loss_bbox(box_preds.tensor, target_bboxes.tensor, weight=loss_weights)}

    def predict(self, x, pre_state_param, batch_data_samples):
        batch_img_metas = self.unpack_img_metas(batch_data_samples)
        ref_bboxes = self.parse_ref_bboxes(batch_img_metas, x.device)
        cropped_bboxes = self.parse_cropped_bboxes(batch_img_metas, x.device)

        # predict bbox
        box_preds = self.predict_bbox(x, pre_state_param, ref_bboxes, cropped_bboxes=cropped_bboxes, predict=True)

        results = [InstanceData(bboxes=pred.detach()) for pred in box_preds]
        return results
