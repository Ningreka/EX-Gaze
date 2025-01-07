import cv2
import numpy as np
import torch
import torchvision

from mmrotate.structures import RotatedBoxes
from misc.ev_eye_dataset_utils import ellipse_mask


def ellipse_iou(gt_ellipses:torch.Tensor, pred_ellipses:torch.Tensor, mask_size,with_f1_score=False):
    assert len(gt_ellipses) == len(pred_ellipses)
    gt_corners = RotatedBoxes.rbox2corner(gt_ellipses).cpu().numpy()
    pred_corners = RotatedBoxes.rbox2corner(pred_ellipses).cpu().numpy()
    gt_masks = np.zeros([gt_ellipses.shape[0], *mask_size])  # (batch_num, *frame_size)
    pred_masks = np.zeros([pred_ellipses.shape[0], *mask_size])  # (batch_num, *frame_size)

    for i in range(gt_ellipses.shape[0]):
        gt_masks[i, ...] = ellipse_mask(mask_size, gt_corners[i, ...])
        pred_masks[i, ...] = ellipse_mask(mask_size, pred_corners[i, ...])

    intersection = np.sum((gt_masks + pred_masks) == 2, axis=(1, 2))  # intersection
    gt_area = np.sum(gt_masks, axis=(1, 2))
    pred_area = np.sum(pred_masks, axis=(1, 2))
    union = gt_area + pred_area - intersection

    iou = intersection / union
    if with_f1_score:
        f1_score = (2 * intersection) / (gt_area + pred_area)
        return iou, f1_score
    else:
        return iou


def bbox_iou(gt_boxes, pred_boxes):
    assert len(gt_boxes) == len(pred_boxes)
    ious = torchvision.ops.box_iou(gt_boxes, pred_boxes)
    return ious.diag()
