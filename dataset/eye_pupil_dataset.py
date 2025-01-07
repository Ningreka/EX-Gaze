from pathlib import Path
from typing import Union, List

from mmengine.dataset import BaseDataset

from registry import EV_DATASETS


@EV_DATASETS.register_module()
class EyePupilDataset(BaseDataset):
    """
    ann = {
        "metainfo":
            {
                "dataset_name": "near eye pupil labelled dataset",
                "task_name": "near eye pupil detection",
                "classes": ["eye_region","pupil"],
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
                    "eye_region": [0, 0, 10, 20], # [x1,y1,x2,y2]
                    "pupil": [ 204, 148, 44, 47, 0], # [xc,yc,w,h,t]
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
        if not self.test_mode:
            assert ("eye_region" in raw_data_info and "pupil" in raw_data_info)

        if 'img_path' in self.data_prefix:
            img_path_prefix = Path(self.data_prefix['img_path'])
            raw_data_info['img_path'] = Path.joinpath(img_path_prefix,
                                                      raw_data_info['img_filename'])

        raw_data_info['img_id'] = f"{self.metainfo['user_id']}/{self.metainfo['eye']}/" \
                                  f"{self.metainfo['session']}/{raw_data_info['img_id']}"

        return raw_data_info

