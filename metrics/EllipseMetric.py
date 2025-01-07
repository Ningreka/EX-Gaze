from typing import Any, Sequence, Optional

import torch
import numpy as np
from mmengine.evaluator import BaseMetric
from mmrotate.structures.bbox import RotatedBoxes
import cv2

from metrics.functions.ellipse_iou import ellipse_iou
from registry import EV_METRICS


@EV_METRICS.register_module()
class EllipseMetric(BaseMetric):
    default_prefix = "single ellipse detection"

    def __init__(self, mask_size, collect_device: str = 'cpu', prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        self.mask_size = mask_size

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        gt_ellipses = []
        pred_ellipses = []
        for data_sample in data_samples:
            gt_ellipse = data_sample["gt_instances"]["bboxes"]
            pred_ellipse = data_sample["pred_instances"]["bboxes"]
            gt_ellipses.append(gt_ellipse if isinstance(gt_ellipse, torch.Tensor) else gt_ellipse.tensor)
            pred_ellipses.append(pred_ellipse if isinstance(pred_ellipse, torch.Tensor) else pred_ellipse.tensor)

        gt_ellipses = torch.cat(gt_ellipses).cpu()
        pred_ellipses = torch.cat(pred_ellipses).cpu()

        center_dist = torch.sqrt(torch.sum(torch.square(gt_ellipses[:, 0:2] - pred_ellipses[:, 0:2]), dim=1))
        iou = torch.Tensor(ellipse_iou(gt_ellipses, pred_ellipses, self.mask_size))

        self.results.append([iou, center_dist])

    def compute_metrics(self, results: list) -> dict:
        ious = []
        dists = []

        for result in results:
            ious.append(result[0])
            dists.append(result[1])

        ious = torch.cat(ious)
        dists = torch.cat(dists)

        return dict(mIoU=torch.mean(ious), maxIoU=torch.max(ious), minIoU=torch.min(ious),
                    mDist=torch.mean(dists), maxDist=torch.max(dists), minDist=torch.min(dists))


@EV_METRICS.register_module()
class BestPredictEllipseMetric(EllipseMetric):
    default_prefix = "best single ellipse detection"

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        gt_ellipses = []
        pred_ellipses = []
        for data_sample in data_samples:
            gt_ellipse = data_sample["gt_instances"]["bboxes"]
            pred_ellipse = data_sample["pred_instances"]["bboxes"]

            gt_ellipses.append(gt_ellipse if isinstance(gt_ellipse, torch.Tensor) else gt_ellipse.tensor)

            if len(pred_ellipse) == 0:
                # if no target
                pred_ellipses.append(torch.zeros_like(gt_ellipses[-1]))
            else:
                best_idx = data_sample["pred_instances"]['scores'].argmax()
                if isinstance(pred_ellipse, torch.Tensor):
                    pred_ellipse = torch.unsqueeze(pred_ellipse[best_idx], 0)
                else:
                    pred_ellipse = pred_ellipse[best_idx].tensor
                pred_ellipses.append(pred_ellipse)

        gt_ellipses = torch.cat(gt_ellipses).cpu()
        pred_ellipses = torch.cat(pred_ellipses).cpu()

        center_dist = torch.sqrt(torch.sum(torch.square(gt_ellipses[:, 0:2] - pred_ellipses[:, 0:2]), dim=1))
        iou = torch.Tensor(ellipse_iou(gt_ellipses, pred_ellipses, self.mask_size))

        self.results.append([iou, center_dist])