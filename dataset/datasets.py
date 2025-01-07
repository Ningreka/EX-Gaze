from pathlib import Path
from typing import List, Optional, Union, Sequence, Callable
import math
import random

import numpy as np
import torch
from mmengine.dataset.base_dataset import BaseDataset

from misc.ev_eye_dataset_utils import classes as ev_eye_classes
from registry import EV_DATASETS

@EV_DATASETS.register_module()
class EyeDetectionDataset(BaseDataset):
    """
    annotation sample
    ann = {
        "metainfo":
            {
                "dataset_name": "near eye labelled dataset",
                "task_name": "near eye detection",
                "classes": ["pupil", "iris"],
                "frame_size": [260, 346],
                "frame_rate": 25,
                "user_id": 1,
                "eye": "left",
                "session": "201"
            },
        "data_list":
            [
                {
                    "img_shape": [260, 346],
                    "img_id": 1065,
                    "timestamp": 1657711738218240,
                    "img_filename": "001065_1657711738218240.png",
                    "instances": [
                      {"ellipse": [205, 151, 20.006, 18, -3.117],
                        "ellipse_label": 0},
                      {"ellipse": [ 204, 148, 44, 47, 0],
                        "ellipse_label": 1}]},
                },
            ]
    }
    """

    def __init__(self, detect_classes, **kwargs):
        for c in detect_classes:
            if c not in ev_eye_classes:
                raise ValueError(f"tracking classes should be in {ev_eye_classes},but get {detect_classes}")
        self.detect_classes = detect_classes
        super().__init__(**kwargs)

    def get_cat_ids(self, idx: int) -> List[int]:
        data = self.data_list[idx]

        classes = self.metainfo["classes"]
        cat_list = []
        for idx, cls in enumerate(classes):
            if cls in data and data[cls] is not None:
                cat_list.append(idx)
        return cat_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        This method should return dict or list of dict. Each dict or list
        contains the data information of a training sample. If the protocol of
        the sample annotations is changed, this function can be overridden to
        update the parsing logic while keeping compatibility.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list[dict]: Parsed annotation.
        """
        init_len = len(raw_data_info["instances"])
        for i in range(init_len - 1, -1, -1):
            
            label_i = raw_data_info["instances"][i]["ellipse_label"]
            if label_i in self.detect_classes:
                raw_data_info["instances"][i]["ellipse_label"] = self.detect_classes.index(label_i)
            else:
                raw_data_info["instances"].pop(i)

        if 'img_path' in self.data_prefix:
            img_path_prefix = Path(self.data_prefix['img_path'])
            raw_data_info['img_path'] = Path.joinpath(img_path_prefix,
                                                      raw_data_info['img_filename'])

        raw_data_info['img_id'] = f"{self.metainfo['user_id']}/{self.metainfo['eye']}/" \
                                  f"{self.metainfo['session']}/{raw_data_info['img_id']}"
        if self.metainfo.setdefault('eye_region', None) is not None:
            raw_data_info['eye_region'] = self.metainfo['eye_region']

        return raw_data_info

    def filter_data(self) -> List[dict]:
        filtered_data_list = []
        for data in self.data_list:
            if len(data["instances"]) > 0:
                filtered_data_list.append(data)

        return filtered_data_list


@EV_DATASETS.register_module()
class EvEyeDataset(BaseDataset):
    """

    ann = {
        "metainfo":
            {
                "dataset_name": "ev eye landmark labelled ev_eye_dataset",
                "task_name": "origin_landmark_tracking",
                "classes": ["pupil", "iris"],
                "frame_size": [260, 346],
                "frame_rate": 25,
                "user_id": 1,
                "eye": "left",
                "session": "201"
            },
        "data_list":
            [
                {
                    "pre_gt": {
                        "img_id": 1065,
                        "timestamp": 1657711738218240,
                        "img_filename": "001065_1657711738218240.png",
                        "instances": [
                          {"ellipse": [205, 151, 20.006, 18, -3.117],
                            "ellipse_label": 0},
                          {"ellipse": [ 204, 148, 44, 47, 0],
                            "ellipse_label": 1}]},
                    "cur_gt": {
                        "img_id": 1066,
                        "timestamp": 1657711738258240,
                        "img_filename": "001066_1657711738258240.png",
                        "instances": [
                          {"ellipse": [201, 151, 19, 18, 0],
                            "ellipse_label": 0},
                          {"ellipse": [204, 152, 46, 44, 0],
                            "ellipse_label": 1}]},
                    "events_between": 1657711738258240,
                    "img_shape": [260, 346]
                },
            ]
    }

    """

    def __init__(self, tracking_classes, events_file=None, filter_dist=0, filter_prob=0.1, forward=True,
                 interval_list_prefix=False, **kwargs):
        if events_file is not None:
            events = np.load(events_file)
            self.events = np.stack((events["x"], events["y"], events["p"], events["t"]), axis=1)
        else:
            self.events = None
        for c in tracking_classes:
            if c not in ev_eye_classes:
                raise ValueError(f"tracking classes should be in {ev_eye_classes},but get {tracking_classes}")
        self.tracking_classes = tracking_classes
        self.filter_dist = filter_dist
        self.filter_prob = filter_prob
        self.forward = forward
        self.interval_list_prefix = interval_list_prefix
        super().__init__(**kwargs)

    def get_cat_ids(self, idx: int) -> List[int]:
        cat_ids = []
        data = self.data_list[idx]
        for instance in data["cur_gt"]["instances"]:
            if instance["ellipse_label"] not in cat_ids:
                cat_ids.append(instance["ellipse_label"])
        return cat_ids

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        assert "cur_gt" in raw_data_info and "pre_gt" in raw_data_info, (
            f'raw_data_info: {raw_data_info} dose not contain cur_gt and pre_gt')
        assert len(raw_data_info["cur_gt"]["instances"]) == len(raw_data_info["pre_gt"]["instances"])
        init_len = len(raw_data_info["cur_gt"]["instances"])
        for i in range(init_len - 1, -1, -1):
            assert raw_data_info["cur_gt"]["instances"][i]["ellipse_label"] \
                   == raw_data_info["pre_gt"]["instances"][i]["ellipse_label"]
            label_i = raw_data_info["cur_gt"]["instances"][i]["ellipse_label"]
            if label_i in self.tracking_classes:
                raw_data_info["cur_gt"]["instances"][i]["ellipse_label"] \
                    = raw_data_info["pre_gt"]["instances"][i]["ellipse_label"] = self.tracking_classes.index(label_i)
            else:
                raw_data_info["cur_gt"]["instances"].pop(i)
                raw_data_info["pre_gt"]["instances"].pop(i)

        if 'img_path' in self.data_prefix:
            img_path_prefix = Path(self.data_prefix['img_path'])
            raw_data_info['cur_gt']['img_path'] = Path.joinpath(img_path_prefix,
                                                                raw_data_info['cur_gt']['img_filename'])
            raw_data_info['pre_gt']['img_path'] = Path.joinpath(img_path_prefix,
                                                                raw_data_info['pre_gt']['img_filename'])

        img_id_prefix = f"{self.metainfo['user_id']}/{self.metainfo['eye']}/{self.metainfo['session']}"
        raw_data_info['cur_gt']['img_id'] = f"{img_id_prefix}/{raw_data_info['cur_gt']['img_id']}"
        raw_data_info['pre_gt']['img_id'] = f"{img_id_prefix}/{raw_data_info['pre_gt']['img_id']}"
        # if 'event_between' in self.data_prefix:
        #     event_between_prefix = Path(self.data_prefix['event_between'])
        #     raw_data_info['event_between'] = Path.joinpath(event_between_prefix, raw_data_info['event_between'])
        if self.metainfo.setdefault('eye_region', None) is not None:
            raw_data_info['eye_region'] = self.metainfo['eye_region']

        if self.events is not None:
            raw_data_info["events"] = self.events

        if self.interval_list_prefix:
            prefix = f"{self.metainfo['user_id']}/{self.metainfo['eye']}/{self.metainfo['session']}/"
            for i in range(len(raw_data_info["interval_list"])):
                raw_data_info["interval_list"][i] = prefix + raw_data_info["interval_list"][i]

        return raw_data_info

    def filter_data(self) -> List[dict]:
        tracking_classes_idx = [ev_eye_classes.index(tc) for tc in self.tracking_classes]
        filtered_data_list = []
        for data in self.data_list:
            cur_gt_instances = data["cur_gt"]['instances']
            pre_gt_instances = data["pre_gt"]["instances"]

            save_idx = []
            for idx, (cur_instance, pre_instance) in enumerate(zip(cur_gt_instances, pre_gt_instances)):
                if cur_instance["ellipse_label"] != pre_instance["ellipse_label"]:
                    break
                if cur_instance["ellipse_label"] in tracking_classes_idx:
                    if self.filter_dist <= 0:
                        save_idx.append(idx)
                    else:
                        p1 = cur_instance["ellipse"][:2]
                        p2 = pre_instance["ellipse"][:2]
                        dist = math.dist(p1, p2)
                        if dist >= self.filter_dist:
                            save_idx.append(idx)
                        else:
                            if random.random() < self.filter_prob:
                                save_idx.append(idx)
            if len(save_idx) == 0:
                continue
            else:
                temp_cur_instances = []
                temp_pre_instances = []
                for idx in save_idx:
                    temp_cur_instances.append(cur_gt_instances[idx])
                    temp_pre_instances.append(pre_gt_instances[idx])
                data["cur_gt"]['instances'] = temp_cur_instances
                data["pre_gt"]['instances'] = temp_pre_instances
                filtered_data_list.append(data)

        return filtered_data_list
