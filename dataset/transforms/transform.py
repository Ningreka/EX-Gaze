from typing import Dict, Optional, Union, Tuple, List, Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import BaseBoxes
from mmrotate.structures import RotatedBoxes

from dataset.transforms.utils import DictPath, build_dict_path
from registry import EV_TRANSFORMS


class RandomEllipseGtTransform(BaseTransform):
    """
    require:
        transform_target

    :param transform_target: 
    :param sync_transform:
    """

    def __init__(self, transform_target='pre_gt', sync_transform=True, random_dis_range=5, random_rot_range=0.5,
                 random_scale_range=3) -> None:
        self.transform_target = transform_target
        self.sync_transform = sync_transform

        if isinstance(random_dis_range, Sequence):
            assert len(random_dis_range) == 2
            self.random_dis_range = random_dis_range  # range for random displace
        else:
            self.random_dis_range = (random_dis_range,) * 2
        self.random_rot_range = random_rot_range
        if isinstance(random_scale_range, Sequence):
            assert len(random_scale_range) == 2
            self.random_scale_range = random_scale_range  # range for random rescale
        else:
            self.random_scale_range = (random_scale_range,) * 2

    def _rand_transform(self):
        dis_transform = np.random.normal(0,
                                         [self.random_dis_range[0],
                                          self.random_dis_range[1],
                                          0,
                                          0,
                                          self.random_rot_range])
        scale_transform = np.exp(np.random.normal(0, [0, 0, self.random_scale_range[0], self.random_scale_range[1], 0]))
        return dis_transform, scale_transform

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if self.transform_target is None:
            gt_item = results
        else:
            assert self.transform_target in results, f"results does not contains key: {self.transform_target}"
            gt_item = results[self.transform_target]
        if self.sync_transform:
            dis_transform, scale_transform = self._rand_transform()
        for instance in gt_item['instances']:
            if not self.sync_transform:
                dis_transform, scale_transform = self._rand_transform()
            ellipse = instance['ellipse']
            ellipse = (ellipse + dis_transform) * scale_transform
            instance['ellipse'] = ellipse

        return results


@EV_TRANSFORMS.register_module()
class FakeEvResize(BaseTransform):
    """
    Added
    Keys:

    - scale
    - scale_factor
    - keep_ratio
    """

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        results['scale'] = results['img_shape']
        results['scale_factor'] = (1.0, 1.0)
        results['keep_ratio'] = True
        return results


@EV_TRANSFORMS.register_module()
class TrackingObjCrop(BaseTransform):
    """
    required

    added
        cropped size
    """

    def __init__(self, img_like_targets: Union[List[DictPath], List[Dict]] = [],
                 label_like_targets: Union[List[DictPath], List[Dict]] = [],
                 box_cls=RotatedBoxes):
        self.img_like_targets = []
        self.label_like_targets = []
        self.box_cls = box_cls
        for target in img_like_targets:
            if isinstance(target, DictPath):
                self.img_like_targets.append(target)
            elif isinstance(target, Dict):
                self.img_like_targets.append(build_dict_path(target))
            else:
                raise TypeError

        for target in label_like_targets:
            if isinstance(target, DictPath):
                self.label_like_targets.append(target)
            elif isinstance(target, Dict):
                self.label_like_targets.append(build_dict_path(target))
            else:
                raise TypeError

    def _crop_img_data(self, data: np.ndarray, x, y, w, h):
        if data.ndim >= 3:
            data = data[..., y:y + h, x:x + w]
        else:
            data = data[y:y + h, x:x + w]
        return data

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        eye_region = results.setdefault("eye_region", None)
        assert eye_region is not None
        # eye region use opencv style (x,y,width,height)
        x, y, width, height = eye_region
        for target in self.img_like_targets:
            data = target.get_dict_value(results)
            if isinstance(data, np.ndarray):
                data = self._crop_img_data(data, x, y, width, height)
            elif isinstance(data, List):
                for i, item in enumerate(data):
                    data[i] = self._crop_img_data(item, x, y, width, height)

            target.set_dict_value(results, data)

        for label_target in self.label_like_targets:
            labels = label_target.get_dict_value(results)
            if isinstance(labels, List):
                labels = self.box_cls(labels)
            labels.translate_([-x, -y])
            label_target.set_dict_value(results, labels)

        results["cropped_area"] = eye_region
        return results


from abc import ABCMeta, abstractmethod


class EyeCrop(BaseTransform, metaclass=ABCMeta):
    def __init__(self, min_scale):
        self.min_scale = min_scale

    def boxes_bound(self, img_size, target_bboxes: RotatedBoxes):
        """

        :param img_size:
        :param target_bboxes:
        :return: [center_x, center_y, box_w, box_h]
        """
        img_h, img_w = img_size
        corners = RotatedBoxes.rbox2corner(target_bboxes.tensor)
        x_left, y_top = corners.min(dim=1).values.min(dim=0).values
        x_right, y_bottom = corners.max(dim=1).values.max(dim=0).values
        center_x, center_y = (x_left + x_right) / 2, (y_top + y_bottom) / 2
        box_w = min((x_right - x_left) * self.min_scale, (img_w - center_x) * 2)
        box_h = min((y_bottom - y_top) * self.min_scale, (img_h - center_y) * 2)

        return np.array([center_x, center_y, box_w, box_h])

    @staticmethod
    def crop_pre_mask(results, crop_box):
        pre_mask = results["pre_mask"]
        cropped_mask = []
        for mask in pre_mask:
            cropped_mask.append(mask[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]])
        results["pre_mask"] = np.array(cropped_mask)

    @staticmethod
    def translate_box(results, crop_box):
        boxes = results["gt_bboxes"]
        boxes.translate_(-crop_box[:2])
        results["gt_bboxes"] = boxes
        results["cropped_area"] = [*crop_box[:2], crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]]

    @staticmethod
    def crop_data(results, crop_box):
        """

        :param results:
        :param crop_box: [x_left,y_top,x_right,y_bottom]
        :return:
        """
        img = results["img"]
        # crop the image
        img = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], ...]
        if "pre_mask" in results:
            EyeCrop.crop_pre_mask(results, crop_box)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]

        # translate box
        EyeCrop.translate_box(results, crop_box)

        return results

    @abstractmethod
    def get_crop_box(self):
        pass


@EV_TRANSFORMS.register_module()
class EyeRandomCenterCrop(EyeCrop):
    """
    rand crop around pupil

    """

    def __init__(self, min_scale, max_width, max_height) -> None:
        assert min_scale >= 1
        super().__init__(min_scale)
        self.max_width = max_width
        self.max_height = max_height

    def get_crop_box(self, img_size, boxes_bound):
        """

        :param img_size:
        :param boxes_bound:
        :return: [x_left,y_top,x_right,y_bottom]
        """
        img_h, img_w = img_size
        max_margin_w = img_w - boxes_bound[2]
        max_margin_h = img_h - boxes_bound[3]
        max_margin_left, max_margin_top = np.floor(boxes_bound[:2] - boxes_bound[-2:] / 2)
        max_margin_right = np.floor(img_w - max_margin_left - boxes_bound[2])
        max_margin_bottom = np.floor(img_h - max_margin_top - boxes_bound[3])

        min_margin_w = img_w - max(self.max_width, boxes_bound[2])
        min_margin_left = np.ceil(min_margin_w * max_margin_left / max_margin_w)
        min_margin_right = np.ceil(min_margin_w * max_margin_right / max_margin_w)
        min_margin_h = img_h - max(self.max_height, boxes_bound[3])
        min_margin_top = np.ceil(min_margin_h * max_margin_top / max_margin_h)
        min_margin_bottom = np.ceil(min_margin_h * max_margin_bottom / max_margin_h)

        margin_left = int(max_margin_left) if min_margin_left >= max_margin_left else \
            np.random.randint(min_margin_left, max_margin_left)
        margin_right = int(max_margin_right) if min_margin_right >= max_margin_right else \
            np.random.randint(min_margin_right, max_margin_right)
        margin_top = int(max_margin_top) if min_margin_top >= max_margin_top else \
            np.random.randint(min_margin_top, max_margin_top)
        margin_bottom = int(max_margin_bottom) if min_margin_bottom >= max_margin_bottom else \
            np.random.randint(min_margin_bottom, max_margin_bottom)

        return np.array([margin_left, margin_top, img_w - margin_right, img_h - margin_bottom])

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        image_size = results['img'].shape[:2]
        boxes_bound = self.boxes_bound(image_size, results["gt_bboxes"])
        crop_box = self.get_crop_box(image_size, boxes_bound)

        results = self.crop_data(results, crop_box)
        return results


class EyeStaticCenterCrop(EyeCrop, metaclass=ABCMeta):
    """
    crop in static size
    """

    def __init__(self, min_scale, crop_width, crop_height, strict_crop=False):
        super().__init__(min_scale)
        self.crop_width = crop_width
        self.crop_height = crop_height
        
        self.strict_crop = strict_crop

    def get_crop_box(self, img_size, boxes_bound):
        """

        :param img_size:
        :param boxes_bound:
        :return: [x_left,y_top,x_right,y_bottom]
        """
        img_h, img_w = img_size
        if self.strict_crop:
            assert img_h > self.crop_height and img_w > self.crop_width

        center_x, center_y, box_w, box_h = boxes_bound
        crop_width, crop_height = self.crop_width, self.crop_height
        if not self.strict_crop:
            if box_w > self.crop_width:
                crop_width = box_w
            if box_h > self.crop_height:
                crop_height = box_h

        x_right = min(int(center_x + crop_width / 2), img_w - 1)
        x_left = max(int(x_right - crop_width), 0)
        y_bottom = min(int(center_y + crop_height / 2), img_h - 1)
        y_top = max(int(y_bottom - crop_height), 0)

        return np.array([x_left, y_top, x_right, y_bottom])


@EV_TRANSFORMS.register_module()
class PrestateCenterCrop(EyeStaticCenterCrop):
    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        image_size = results['img'].shape[:2]
        
        pre_state_bound = self.boxes_bound(img_size=image_size, target_bboxes=results["pre_bboxes"])
        crop_box = self.get_crop_box(img_size=image_size, boxes_bound=pre_state_bound)
        results = self.crop_data(results, crop_box)
        return results


@EV_TRANSFORMS.register_module()
class EvPrestateCenterCrop(EyeStaticCenterCrop):
    @staticmethod
    def crop_data(results, crop_box):
        """

        :param results:
        :param crop_box: [x_left,y_top,x_right,y_bottom]
        :return:
        """
        event_volume = results["event_volume"]
        # crop the image
        event_volume = event_volume[..., crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        if "pre_mask" in results:
            EyeCrop.crop_pre_mask(results, crop_box)
        img_shape = event_volume.shape
        results['event_volume'] = event_volume
        results['img_shape'] = img_shape[1:3]

        # translate box
        EyeCrop.translate_box(results, crop_box)

        return results

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        image_size = results['img_shape']
        
        pre_state_bound = self.boxes_bound(img_size=image_size, target_bboxes=results["pre_bboxes"])
        crop_box = self.get_crop_box(img_size=image_size, boxes_bound=pre_state_bound)
        results = self.crop_data(results, crop_box)
        return results


@EV_TRANSFORMS.register_module()
class ImgCenterCrop(EyeCrop):

    def __init__(self, min_scale, crop_width, crop_height, strict_crop=True):
        super().__init__(min_scale)
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.strict_crop = strict_crop

    def get_crop_box(self, img_size):
        if self.strict_crop:
            assert img_size[0] > self.crop_height and img_size[1] > self.crop_width
        center_x = img_size[1] // 2
        center_y = img_size[0] // 2
        x_right = int(center_x + self.crop_width // 2)
        y_bottom = int(center_y + self.crop_height // 2)
        x_left = x_right - self.crop_width
        y_top = y_bottom - self.crop_height

        return np.array([x_left, y_top, x_right, y_bottom])

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        image_size = results['img_shape']
        
        crop_box = self.get_crop_box(img_size=image_size)
        results = self.crop_data(results, crop_box)
        return results


@EV_TRANSFORMS.register_module()
class ImgEyeRegionRandCrop(EyeRandomCenterCrop):
    def __init__(self, max_width, max_height):
        super(ImgEyeRegionRandCrop, self).__init__(1, max_width, max_height)

    def boxes_bound(self, eye_region):
        """

        :param eye_region: [x_left, y_top, width, height]
        :return: [center_x, center_y, box_w, box_h]
        """

        return np.array([eye_region[0] + eye_region[2] / 2, eye_region[1] + eye_region[3] / 2,
                         eye_region[2], eye_region[3]])

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        image_size = results['img'].shape[:2]
        assert "eye_region" in results
        # eye region [x,y,width,height]
        eye_region_box = self.boxes_bound(results["eye_region"])
        crop_box = super().get_crop_box(image_size, eye_region_box)

        results = super().crop_data(results, crop_box)
        return results
