import logging
from pathlib import Path
from typing import Union

import math
import pandas as pd
import numpy as np
import cv2
import torch
from mmengine.structures import InstanceData
from mmrotate.structures import RotatedBoxes


origin_dataset_dir = Path("/path/to/data") # TODO modify required

origin_single_data_pattern = \
    str(origin_dataset_dir / "user{user_id}/{eye}/session_{session[0]}_{session[1]}_{session[2]}/events")

mini_dataset_dir = Path("/path/to/data") # TODO modify required

single_mini_data_pattern = \
    str(mini_dataset_dir / "user{user_id}/{eye}/session_{session[0]}_{session[1]}_{session[2]}/events")
    
landmark_label_file_pattern = single_mini_data_pattern + "/{eye}{session}user{user_id}_labelled.csv"

img_shape = (260, 346)

classes = ('pupil', 'iris')

base_meta_info = {
    "dataset_name": 'ev eye dataset',
    "task_name": "ev pupil tracking",
    "classes": classes,
    "img_shape": img_shape,
}

continuous_ann_segments_path = mini_dataset_dir / "data_segments.xlsx"
session_col_idx = {'101': [1, 2], '102': [4, 5], '201': [8, 9], '202': [11, 12]}


def parse_frame_filename(frame_filename: str, timestamp_offset: int = 0):
    # frame timestamp of ev-eye dataset usually use 4000 offset
    frame_filename = frame_filename.rstrip(".png")
    frame_idx, frame_timestamp = frame_filename.split("_")
    return int(frame_idx), int(frame_timestamp) + timestamp_offset


def parse_landmark_labels(landmark_label_file: Union[Path, str]):
    landmark_labels = pd.read_csv(landmark_label_file)
    valid_labels = landmark_labels[landmark_labels.region_count > 0].reset_index(drop=True)

    instance_list = []

    i = 0
    while i < len(valid_labels):
        frame_idx, frame_timestamp = parse_frame_filename(valid_labels.loc[i].filename)

        instance = InstanceData(metainfo={
            "filename": valid_labels.loc[i].filename,
            "frame_idx": frame_idx,
            "frame_timestamp": frame_timestamp})
        region_count = valid_labels.loc[i].region_count
        ellipses = []
        digit_labels = []
        labels = []

        for j in range(region_count):
            temp_ellipse = eval(valid_labels.loc[i + j].region_shape_attributes)
            ellipses.append(
                [temp_ellipse['cx'], temp_ellipse['cy'], temp_ellipse['rx'] * 2, temp_ellipse['ry'] * 2,
                 temp_ellipse['theta']])
            digit_labels.append(int(valid_labels.loc[i + j].region_id))
            labels.append(classes[valid_labels.loc[i + j].region_id])

        instance['ellipses'] = ellipses
        instance['labels'] = labels
        instance['digit_labels'] = digit_labels

        instance_list.append(instance)

        i = i + region_count

    return instance_list


def print_stats(data_list, stat_name):
    if len(data_list) > 0:
        print(
            f"mean {stat_name}: {data_list.mean()}, std {stat_name}: {data_list.std()}, max {stat_name}: {data_list.max()}, min {stat_name}: {data_list.min()}")
    else:
        logging.log(logging.WARNING, f"has no {stat_name}, may be result of no ground truth")


def ellipse_mask(mask_size, box_corners):
    '''
    :param mask_size
    :param box_corners

    :return: np.ndarray (H, W)
    '''
    mask_frame = np.zeros(mask_size, dtype=np.float32)

    cv2.ellipse(mask_frame,
                cv2.RotatedRect(box_corners[0, :], box_corners[1, :], box_corners[2, :]),
                1, -1)
    return mask_frame


def plot_ellipse(ellipse, plt_img=None):
    if plt_img is None:
        plt_img = np.zeros(img_shape)
    cv2.ellipse(plt_img, (int(ellipse[0]), int(ellipse[1])), (int(ellipse[2] / 2), int(ellipse[3] / 2)),
                ellipse[4] / np.pi * 180, 0, 360, 1, 1)
    return plt_img


def mask_iou(mask1, mask2, with_f1_score=False):
    intersection = np.sum((mask1 + mask2) == 2)  # intersection
    mask1_area = np.sum(mask1)
    mask2_area = np.sum(mask2)
    union = mask1_area + mask2_area - intersection

    iou = intersection / union
    if with_f1_score:
        f1_score = (2 * intersection) / (mask1_area + mask2_area)
        return iou, f1_score
    else:
        return iou


def ellipse_iou(ellipse_0, ellipse_1, mask_size=img_shape, with_f1_score=False):
    """
    
    :param ellipse_0: [x,y,w,h,t]
    :param ellipse_1: [x,y,w,h,t]
    :return:
    """
    box_corners_0 = RotatedBoxes.rbox2corner(torch.tensor(ellipse_0)).cpu().numpy()
    box_corners_1 = RotatedBoxes.rbox2corner(torch.tensor(ellipse_1)).cpu().numpy()

    ellipse_mask_0 = ellipse_mask(mask_size, box_corners_0)
    ellipse_mask_1 = ellipse_mask(mask_size, box_corners_1)

    return mask_iou(ellipse_mask_0, ellipse_mask_1, with_f1_score)


def ellipse_dist(ellipse_0, ellipse_1):
    dist = np.sqrt(np.sum((np.square(ellipse_0[:2] - ellipse_1[:2]))))
    return dist


def ellipse_point_sample(ellipse, sample_rads):
    """

    :param ellipse: [x,y,w,h,t] require not Tensor
    :param sample_rads:
    :return:
    """
    x, y, w, h, t = ellipse
    orin_sample_rads = sample_rads - t

    a, b = w / 2, h / 2
    r = np.sqrt(np.square(a * b) / (np.square(b * np.cos(orin_sample_rads)) + np.square(a * np.sin(orin_sample_rads))))
    sample_points = np.stack([r * np.cos(sample_rads) + x, r * np.sin(sample_rads) + y], axis=1)

    return sample_points


def ellipse_norm(ellipse, sample_rads):
    x, y, w, h, t = ellipse
    orin_sample_rads = sample_rads - t

    a, b = w / 2, h / 2

    r = np.sqrt(np.square(a * b) / (np.square(b * np.cos(orin_sample_rads)) + np.square(a * np.sin(orin_sample_rads))))
    origin_sample_points = np.stack([r * np.cos(orin_sample_rads), r * np.sin(orin_sample_rads)], axis=1)
    origin_norm = np.stack([origin_sample_points[:, 0], (a / b) ** 2 * origin_sample_points[:, 1]], axis=1)
    origin_norm = origin_norm / np.linalg.norm(origin_norm, axis=1, keepdims=True)

    # x = x * cos(t) - y * sin(t)
    # y = x * sin(t) + y * cos(t)

    trans_points = np.stack([origin_sample_points[:, 0] * np.cos(t) - origin_sample_points[:, 1] * np.sin(t) + x,
                             origin_sample_points[:, 0] * np.sin(t) + origin_sample_points[:, 1] * np.cos(t) + y],
                            axis=1)
    trans_norm = np.stack([origin_norm[:, 0] * np.cos(t) - origin_norm[:, 1] * np.sin(t),
                           origin_norm[:, 0] * np.sin(t) + origin_norm[:, 1] * np.cos(t)], axis=1)

    return orin_sample_rads, origin_sample_points, origin_norm, trans_points, trans_norm
