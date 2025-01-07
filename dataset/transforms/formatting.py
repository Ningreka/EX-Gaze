from typing import Dict, Optional, Union, Tuple, List

import torch
import numpy as np
from mmcv import BaseTransform
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData, BaseDataElement
from mmdet.structures import DetDataSample
from mmengine.config import ConfigDict
from mmengine.utils.misc import is_seq_of
from mmcv.transforms.formatting import to_tensor
from .utils import DictPath, build_dict_path

from registry import EV_TRANSFORMS


@EV_TRANSFORMS.register_module()
class MapResultItems(BaseTransform):
    def __init__(self, source: Union[DictPath, ConfigDict], target: Union[DictPath, ConfigDict]):
        if isinstance(source, DictPath):
            self.source = source
        else:
            self.source = build_dict_path(source)
        if isinstance(target, DictPath):
            self.target = target
        else:
            self.target = build_dict_path(target)

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        item = self.source.get_dict_value(results)
        assert item is not None
        self.target.set_dict_value(results, item)

        return results

@EV_TRANSFORMS.register_module()
class PackEvPupilTrackingInputs(BaseTransform):
    """
    require
        *input_keys
        *meta_keys
        "cur_gt"."pupil_ellipse"
    return

    input_keys=("concatenated_volume")
                ("ellipse_mask","tore_volume")
                (("pre_gt","pupil_ellipse"),"tore_volume")
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_bboxes_weights': 'weights'
    }

    def __init__(self, input_keys: Union[List[str], Dict[str, str]] = ["event_volume"],
                 meta_keys=('img_id', 'img_path', 'ori_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.input_keys = input_keys
        self.meta_keys = meta_keys

    def format_img(self, img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            img = to_tensor(img)
        else:
            img = to_tensor(img).permute(2, 0, 1).contiguous()
        return img

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        data = {}

        def _pack(key, value):
            if value == "img":
                data[key] = self.format_img(results[value])
                return
            if is_seq_of(results[value], np.ndarray):
                data[key] = torch.from_numpy(np.array(results[value]))
            elif isinstance(results[value], BaseBoxes):
                data[key] = results[value]
            elif isinstance(results[value], np.ndarray):
                data[key] = torch.from_numpy(results[value])
            else:
                data[key] = torch.Tensor(results[value])

        if isinstance(self.input_keys, (List, Tuple)):
            for k in self.input_keys:
                _pack(k, k)
        elif isinstance(self.input_keys, Dict):
            for k, v in self.input_keys.items():
                _pack(k, v)
        else:
            raise TypeError(f"require list tuple or dict, but get {type(self.input_keys)}")

        # pack data_sample
        data_sample = DetDataSample()
        gt_instances = InstanceData()
        ignored_instances = InstanceData()

        for key in self.mapping_table.keys():
            if key in results:
                gt_instances[self.mapping_table[key]] = results[key]
        ignored_instances["bboxes"] = torch.zeros((0, 5))
        ignored_instances["labels"] = torch.Tensor()
        ignored_instances["weights"] = torch.Tensor()

        data_sample.gt_instances = gt_instances
        data_sample.ignored_instances = ignored_instances

        # pack img meta info
        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                                   f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)

        data["data_samples"] = data_sample
        return data


@EV_TRANSFORMS.register_module()
class PackEyeRegionPupilDetInputs(PackEvPupilTrackingInputs):

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        data = {}
        assert isinstance(self.input_keys, Dict) and len(self.input_keys) == 1

        k = list(self.input_keys.keys())[0]
        data[k] = self.format_img(results[self.input_keys[k]])

        # pack data_sample
        data_sample = BaseDataElement()
        eye_region_instance = InstanceData(bbox=results["eye_region_bbox"])
        pupil_instance = InstanceData(bbox=results["pupil_bbox"])

        data_sample.eye_region_instance = eye_region_instance
        data_sample.pupil_instance = pupil_instance

        # pack img meta info
        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                                   f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)

        data["data_samples"] = data_sample
        return data
