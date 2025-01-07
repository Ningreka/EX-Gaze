import copy
import logging
import re
from pathlib import Path
from typing import List, Union

from mmengine.dataset import BaseDataset

from misc import ev_eye_dataset_utils
from registry import EV_DATASETS
from misc.ev_eye_dataset_utils import mini_dataset_dir, img_shape
from configs._base_.data_split import train_user_list,test_user_list


@EV_DATASETS.register_module()
class EyeRegionDataset(BaseDataset):
    """
    ann = {
        "metainfo":
            {
                "dataset_name": "near eye region dataset",
                "task_name": "near eye detection",
                "classes": ["eye_region"]
            },
        "data_list":
            [
                {
                    "img_shape": [260, 346],
                    "img_filename": "u1_e_left_s_101_000160_1657710792537482.png",
                    "instances": [{"bbox": [0, 0, 10, 20], "bbox_label": 1}], # [x1,y1,x2,y2]
                },
            ]
    }
    :parameter
    ann_file (str, optional): Annotation file path. Defaults to ''.
    metainfo (dict, optional): Meta information for ev_eye_dataset, such as class
        information. Defaults to None.
    data_root (str, optional): The root directory for ``data_prefix`` and
        ``ann_file``. Defaults to ''.
    data_prefix (dict): Prefix for training data. Defaults to dict(img_path='').
    filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few data in
        annotation file to facilitate training/testing on a smaller
    serialize_data (bool, optional): Whether to hold memory using serialized objects,
        when enabled, data loader workers can use shared RAM from master process
        instead of making a copy. Defaults to True.
    pipeline (list, optional): Processing pipeline. Defaults to [].
    test_mode (bool, optional): ``test_mode=True`` means in test phase. Defaults to False.
    lazy_init (bool, optional): Whether to load annotation during instantiation.
        In some cases, such as visualization, only the meta information of the
        ev_eye_dataset is needed, which is not necessary to load annotation file.
        ``Basedataset`` can skip load annotations to save time by set ``lazy_init=True``. Defaults to False.
    max_refetch (int, optional): If ``Basedataset.prepare_data`` get a None img.
        The maximum extra number of cycles to get a valid image. Defaults to 1000.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        # if not self.test_mode:
        assert len(raw_data_info["instances"]) == 1

        assert raw_data_info["instances"][0]["bbox_label"] == 0

        raw_data_info["instances"][0]["ignore_flag"] = [0]

        if 'img_path' in self.data_prefix:
            img_path_prefix = Path(self.data_prefix['img_path'])
            raw_data_info['img_path'] = Path.joinpath(img_path_prefix,
                                                      raw_data_info['img_filename'])
        raw_data_info["img_id"] = raw_data_info['img_path'].stem

        return raw_data_info

def parse_filename(filename: str):
    match = re.match("u(\d+)_e_(left|right)_s_(\d{3})_(\w+.png)", filename)

    return match.group(1)

def annotation_generate():
    import pandas as pd
    import json
    root_dir = mini_dataset_dir / "eye_region_dataset"
    origin_annotation_path = root_dir / "eye_region_annotation.csv"
    train_annotation_path = root_dir / "eye_region_train_dataset.json"
    test_annotation_path = root_dir / "eye_region_test_dataset.json"

    origin_annotation = pd.read_csv(origin_annotation_path)
    train_annotation = {
        "metainfo":
            {
                "dataset_name": "near eye region train dataset",
                "task_name": "near eye detection",
                "classes": ["eye_region"]
            }
    }
    test_annotation = copy.deepcopy(train_annotation)
    test_annotation["metainfo"]["dataset_name"]= "near eye region test dataset"

    train_data_list = []
    test_data_list = []

    for row, sample in origin_annotation.iterrows():
        img_filename = sample["filename"]
        if not (root_dir / img_filename).exists():
            logging.log(logging.WARN, f"{img_filename} not exists")
            continue
        user_id = int(parse_filename(img_filename))
        region_shape = eval(sample["region_shape_attributes"])
        x_min, y_min = region_shape["x"], region_shape["y"]
        x_max, y_max = x_min + region_shape["width"], y_min + region_shape["height"]
        data = dict(
            img_shape=img_shape,
            img_filename=img_filename,
            instances=[{"bbox": [x_min, y_min, x_max, y_max], "bbox_label": 0}]
        )
        if user_id in train_user_list:
            train_data_list.append(data)
        else:
            test_data_list.append(data)

    train_annotation["data_list"] = train_data_list
    test_annotation["data_list"] = test_data_list
    with open(train_annotation_path, "w") as f:
        json.dump(train_annotation, f, indent=2)

    with open(test_annotation_path, "w") as f:
        json.dump(test_annotation, f, indent=2)

if __name__ == '__main__':
    annotation_generate()
