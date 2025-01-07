from typing import Optional, Any, Sequence

import torch
from mmengine.evaluator import BaseMetric

from metrics.functions.ellipse_iou import ellipse_iou, bbox_iou
from registry import EV_METRICS


@EV_METRICS.register_module()
class EyePupilMetric(BaseMetric):
    default_prefix = "eye box pupil ellipse detection"

    def __init__(self, mask_size, collect_device: str = 'cpu', prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        self.mask_size = mask_size

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        # eye region instance, pupil region instance, pred_instances["eye_region_pred","pupil_pred"]
        gt_eye_region_boxes = []
        gt_pupil_ellipses = []

        pred_eye_region_boxes = []
        pred_pupil_ellipses = []

        for data_sample in data_samples:
            gt_eye_region_box = data_sample["eye_region_instance"]["bbox"]
            gt_pupil_ellipse = data_sample["pupil_instance"]["bbox"]

            pred_eye_region_box = data_sample["pred_instances"]["eye_region_pred"]["bbox"]
            pred_pupil_ellipse = data_sample["pred_instances"]["pupil_pred"]["bbox"]

            gt_eye_region_boxes.append(
                gt_eye_region_box if isinstance(gt_eye_region_box, torch.Tensor) else gt_eye_region_box.tensor)
            gt_pupil_ellipses.append(
                gt_pupil_ellipse if isinstance(gt_pupil_ellipse, torch.Tensor) else gt_pupil_ellipse.tensor)

            pred_eye_region_boxes.append(
                pred_eye_region_box if isinstance(pred_eye_region_box, torch.Tensor) else pred_eye_region_box.tensor)
            pred_pupil_ellipses.append(
                pred_pupil_ellipse if isinstance(pred_pupil_ellipse, torch.Tensor) else pred_pupil_ellipse.tensor)

        gt_eye_region_boxes = torch.cat(gt_eye_region_boxes).cpu()
        gt_pupil_ellipses = torch.cat(gt_pupil_ellipses).cpu()

        pred_eye_region_boxes = torch.cat(pred_eye_region_boxes).cpu()
        pred_pupil_ellipses = torch.cat(pred_pupil_ellipses).cpu()

        eye_region_iou = bbox_iou(gt_eye_region_boxes, pred_eye_region_boxes)

        pupil_center_dist = torch.sqrt(
            torch.sum(torch.square(gt_pupil_ellipses[:, 0:2] - pred_pupil_ellipses[:, 0:2]), dim=1))
        pupil_iou = torch.Tensor(ellipse_iou(gt_pupil_ellipses, pred_pupil_ellipses, self.mask_size))

        self.results.append([eye_region_iou, pupil_iou, pupil_center_dist])

    def compute_metrics(self, results: list) -> dict:
        eye_region_ious = []
        pupil_ious = []
        pupil_center_dists = []

        for result in results:
            eye_region_ious.append(result[0])
            pupil_ious.append(result[1])
            pupil_center_dists.append(result[2])

        eye_region_ious = torch.cat(eye_region_ious)
        pupil_ious = torch.cat(pupil_ious)
        pupil_center_dists = torch.cat(pupil_center_dists)

        return dict(
            mEyeIoU=torch.mean(eye_region_ious), maxEyeIoU=torch.max(eye_region_ious),
            minEyeIoU=torch.min(eye_region_ious),
            mPupilIoU=torch.mean(pupil_ious), maxPupilIoU=torch.max(pupil_ious), minPupilIoU=torch.min(pupil_ious),
            mPupilDist=torch.mean(pupil_center_dists), maxPupilDist=torch.max(pupil_center_dists),
            minPupilDist=torch.min(pupil_center_dists))
