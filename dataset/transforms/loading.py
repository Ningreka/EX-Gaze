from typing import Dict, Optional, Union, Tuple, List, Sequence
from pathlib import Path

import torch
from math import sqrt

import mmengine
from mmcv import BaseTransform
from mmrotate.structures import RotatedBoxes
from mmdet.structures.bbox import HorizontalBoxes

import numpy as np
import cv2
import h5py

from registry import EV_TRANSFORMS
from .utils import parse_item_in_results
from misc.ev_eye_dataset_utils import ellipse_mask


@EV_TRANSFORMS.register_module()
class LoadTrackingTargetEllipse2BBox(BaseTransform):
    """
    require:
        target_gt
        target_gt["instances"][{"ellipse","ellipse_label"}]
    add:
        gt_bboxes
        gt_bboxes_labels
    
    """

    def __init__(self, target_gt="cur_gt", target_labels=None, output_prefix="gt", distance_as_weight=False):
        """
        add
            f"{self.output_prefix}_bboxes"
            f"{self.output_prefix}_bboxes_labels"
            f"{self.output_prefix}_bboxes_weights"

        :param target_gt: 
        :param target_labels:
        """
        self.target_gt = target_gt
        self.target_labels = target_labels
        self.output_prefix = output_prefix
        self.distance_as_weight = distance_as_weight

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if self.target_gt is None:
            gt_item = results
        else:
            gt_item = results[self.target_gt]

        if self.distance_as_weight:
            instance_distances = results["instance_distances"]
        bboxes = []
        labels = []
        weights = []

        for idx, ellipse in enumerate(gt_item['instances']):
            ellipse_label = ellipse['ellipse_label']
            assert ellipse_label not in labels, f"the ellipse in target should have unique label, but get {gt_item['instances']}"
            if self.target_labels is None or ellipse_label in self.target_labels:
                # bboxes.append(np.asarray(ellipse["ellipse"], dtype=np.float32))
                bboxes.append(ellipse["ellipse"])
                labels.append(ellipse_label)
                if self.distance_as_weight:
                    weights.append(instance_distances[idx])

        bboxes = RotatedBoxes(np.array(bboxes, dtype=np.float32))
        results[f"{self.output_prefix}_bboxes"] = bboxes
        results[f"{self.output_prefix}_bboxes_labels"] = torch.Tensor(labels).to(dtype=int)
        if self.distance_as_weight:
            results[f"{self.output_prefix}_bboxes_weights"] = torch.Tensor(weights)
        else:
            results[f"{self.output_prefix}_bboxes_weights"] = torch.ones_like(
                results[f"{self.output_prefix}_bboxes_labels"])

        return results


@EV_TRANSFORMS.register_module()
class LoadEyeRegionPupilBBox(BaseTransform):
    """
    require:
        eye_region
        pupil
    add:
        eye_region_bbox
        pupil_bbox
    """

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        assert ("eye_region" in results) and ("pupil" in results)

        results["eye_region_bbox"] = HorizontalBoxes(np.array([results["eye_region"]], dtype=np.float32))
        results["pupil_bbox"] = RotatedBoxes(np.array([results["pupil"]], dtype=np.float32))

        return results


@EV_TRANSFORMS.register_module()
class ParseDistance(BaseTransform):

    def __init__(self, square_dist=True):
        self.square_dist = square_dist

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        pre_item = results["pre_gt"]
        cur_item = results["cur_gt"]

        distances = []
        for pre_ellipse, cur_ellipse in zip(pre_item['instances'], cur_item["instances"]):
            assert pre_ellipse["ellipse_label"] == cur_ellipse["ellipse_label"]
            pre_xy = np.array(pre_ellipse["ellipse"][:2])
            cur_xy = np.array(cur_ellipse["ellipse"][:2])

            d2 = np.sum(np.square(pre_xy - cur_xy))
            if self.square_dist:
                distances.append(d2)
            else:
                distances.append(sqrt(d2))
        results["instance_distances"] = np.array(distances)
        return results


class RandomPreState(BaseTransform):
    """
    generate a rand pre gt according current gt
    require:
        gt_bboxes
    add:
        pre_bboxes
    """

    def __init__(self, random_dis_range=5, random_rot_range=0.5, random_scale_range=3) -> None:
        if isinstance(random_dis_range, Sequence):
            assert len(random_dis_range) >= 2
            self.random_dis_range = random_dis_range  # rand disp
        else:
            self.random_dis_range = (random_dis_range,) * 2
        self.random_rot_range = random_rot_range
        if isinstance(random_scale_range, Sequence):
            assert len(random_scale_range) >= 2
            self.random_scale_range = random_scale_range  # rand scale
        else:
            self.random_scale_range = (random_scale_range,) * 2

    def _rand_transform(self, ellipse):
        x, y, w, h, t = np.random.normal(ellipse,
                                         [self.random_dis_range[0],
                                          self.random_dis_range[1],
                                          self.random_scale_range[0],
                                          self.random_scale_range[1],
                                          self.random_rot_range])
        return [x, y, w, h, t]

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        pre_bboxes = []
        for gt_bbox in results["gt_bboxes"]:
            pre_bboxes.append(
                self._rand_transform(gt_bbox.numpy()[0])
            )

        results["pre_bboxes"] = RotatedBoxes(np.array(pre_bboxes), dtype=np.float32)
        return results


@EV_TRANSFORMS.register_module()
class LoadEventVolume(BaseTransform):
    """
    require:
        events_tore_file
        events_between
    add:
        tore_volume
    """

    def __init__(self, events_volume_file: Union[str, Path]):
        self.events_volume_file = h5py.File(events_volume_file, mode='r')

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        event_volume = np.array(self.events_volume_file[str(results["events_between"])])
        results["event_volume"] = event_volume
        return results
