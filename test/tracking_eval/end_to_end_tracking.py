import argparse
import json
import logging
import os
import time
from pathlib import Path
import pickle
from typing import List

import math
import cv2
import numpy
import numpy as np
import pandas as pd
import torch
from mmdet.structures import DetDataSample
from mmengine import Config
from mmengine.fileio import load as load_ann
from mmrotate.structures import RotatedBoxes

from misc.ev_eye_dataset_utils import origin_single_data_pattern, mini_dataset_dir, single_mini_data_pattern, \
    continuous_ann_segments_path, session_col_idx, parse_frame_filename, ellipse_iou, img_shape
from misc.generate_event_threshold_dataset.check_valid_data import parse_patch_region
from test.test_utils import build_eval_model, ev_accum_tracking, parse_pupil_ellipse, event_patch_accum, ev_single_pred, \
    img_pupil_detect_with_multi_obj

from model.img_pupil_similarity import shape_based_similarity, EllipseOutOfBoundException
from test.tracking_eval.end_to_end_tracking_model_cfg import model_cfg_options
from configs._base_.data_split import train_user_list, test_user_list, val_user_list

# frame_timestamp_offset = 4000
frame_timestamp_offset = 0

class BaseEndToEndTracker:
    def __init__(self,
                 img_path_list, event_stream, tracking_start_timestamp, tracking_end_timestamp,  # tracking related param
                 ev_pupil_tracking_model, img_pupil_detection_model,  # model used
                 img_det_model_input_name, is_single_obj_detect_model, is_eye_region_crop_resize,
                 frame_target_size, eye_region,  # pupil detection model config
                 sample_rads, patch_size,  # tracking model config
                 # tracking param
                 event_accum_time,  # min accum time -> min time window
                 max_accum_frame_num,  
                 event_accum_num_threshold,
                 accum_with_overlap,
                 similarity_threshold,
                 frame_interval=40000, force_re_localization=False, re_localization_wo_similarity_gap=False):

        self.img_path_list = img_path_list
        self.tracking_start_timestamp = tracking_start_timestamp
        self.tracking_end_timestamp = tracking_end_timestamp

        self.event_stream = event_stream
        self.ev_pupil_tracking_model = ev_pupil_tracking_model
        self.img_pupil_detection_model = img_pupil_detection_model
        self.img_det_model_input_name = img_det_model_input_name
        self.is_single_obj_detect_model = is_single_obj_detect_model
        self.is_eye_region_crop_resize = is_eye_region_crop_resize
        self.frame_target_size = frame_target_size  # [width,height]
        if is_eye_region_crop_resize:
            self.eye_region = eye_region
            self.recover_scale_factor = self.eye_region[2] / self.frame_target_size[0], self.eye_region[3] / \
                                    self.frame_target_size[1]
        # event 
        self.sample_rads = sample_rads
        self.patch_size = patch_size

        self.event_accum_time = event_accum_time
        self.max_accum_frame_num = max_accum_frame_num
        self.event_accum_num_threshold = event_accum_num_threshold
        self.accum_with_overlap = accum_with_overlap
        # similarity
        self.similarity_threshold = similarity_threshold
        self.frame_interval = frame_interval

        self.force_re_localization = force_re_localization
        self.re_localization_wo_similarity_gap = re_localization_wo_similarity_gap

        self.tracking_result = []

        self.latest_pupil_state = None

        self.re_localization = True
        self.ev_tracking_start_timestamp = 0.0  # next tracking start timestamp

        self.next_frame_idx = 0
        _, self.next_frame_timestamp = parse_frame_filename(self.img_path_list[self.next_frame_idx].name,
                                                            timestamp_offset=frame_timestamp_offset)  

    def parse_img_single_pupil_detection(self, result):
        # cropped_img = img[eye_region[1]:eye_region[1] + eye_region[3],
        #               eye_region[0]:eye_region[0] + eye_region[2]]  # 2d
        # resized_img = cv2.resize(cropped_img, self.frame_target_size, interpolation=cv2.INTER_LINEAR)
        # recover_scale_factor = (eye_region[2] / self.frame_target_size[0], eye_region[3] / self.frame_target_size[1])
        # model_input = dict(
        #     input_volume=[torch.from_numpy(resized_img).unsqueeze(0)],
        #     data_samples=[DetDataSample()]
        # )

        pupil_box: RotatedBoxes = RotatedBoxes(result.pred_instances.bboxes)

        return pupil_box

    def parse_img_multi_obj_detect_result(self, result):
        # img = torch.from_numpy(img).unsqueeze(0)
        # img_data_sample = DetDataSample()
        # img_data_sample.set_metainfo(dict(
        #     img_shape=img_shape,
        #     ori_shape=img_shape
        # ))
        # result = self.img_pupil_detection_model.test_step(dict(
        #     inputs=[img],
        #     data_samples=[img_data_sample]
        # ))[0].cpu().detach()
        if len(result.pred_instances) != 0:
            bboxes = result.pred_instances.bboxes
            scores = result.pred_instances.scores
            idx = int(scores.argmax())
            pred_ellipse = bboxes[idx].unsqueeze(0)
            # if not isinstance(pred_ellipse, torch.Tensor):
            #     pred_ellipse = pred_ellipse.tensor
            # return pred_ellipse.squeeze().numpy()
            pupil_box: RotatedBoxes = RotatedBoxes(pred_ellipse)
            return pupil_box
        else:
            # no detected
            return None

    def img_pupil_detect(self, img: np.array):
        # pre process
        if self.is_eye_region_crop_resize:
            cropped_img = img[self.eye_region[1]:self.eye_region[1] + self.eye_region[3],
                          self.eye_region[0]:self.eye_region[0] + self.eye_region[2]]  # 2d
            input_img = cv2.resize(cropped_img, self.frame_target_size, interpolation=cv2.INTER_LINEAR)
        else:
            input_img = img
        # forward
        img_data_sample = DetDataSample()
        img_data_sample.set_metainfo(dict(
            img_shape=input_img.shape,
            ori_shape=input_img.shape
        ))
        input_img = torch.from_numpy(input_img).unsqueeze(0)
        result = self.img_pupil_detection_model.test_step({
            self.img_det_model_input_name: [input_img],
            "data_samples": [img_data_sample]}
        )[0].cpu().detach()
        # result post process
        if self.is_single_obj_detect_model:
            pred_pupil_box: RotatedBoxes = self.parse_img_single_pupil_detection(result)
        else:
            pred_pupil_box: RotatedBoxes = self.parse_img_multi_obj_detect_result(result)
            if pred_pupil_box is None:
                return None

        # post process
        if self.is_eye_region_crop_resize:
            # resize
            pred_pupil_box.rescale_(self.recover_scale_factor)
            # crop
            pred_pupil_box.translate_(self.eye_region[:2])

        return pred_pupil_box.tensor.squeeze().numpy()

    def frame_based_update(self):
        next_frame = cv2.imread(str(self.img_path_list[self.next_frame_idx]), flags=cv2.IMREAD_GRAYSCALE)
        if self.force_re_localization:
            self.frame_force_re_localization(next_frame, self.next_frame_timestamp)
        elif self.re_localization_wo_similarity_gap:
            self.frame_force_re_localization_with_low_similarity(next_frame, self.next_frame_timestamp)
        else:
            self.frame_re_localization(next_frame, self.next_frame_timestamp)

        self.next_frame_idx += 1
        _, self.next_frame_timestamp = parse_frame_filename(self.img_path_list[self.next_frame_idx].name,
                                                            timestamp_offset=frame_timestamp_offset)  # 下一帧的时间

    def frame_re_localization(self, next_frame, next_frame_timestamp):
        # frame based
        if self.re_localization:
            # if self.is_single_obj_detect_model:
            #     self.latest_pupil_state = self._img_single_pupil_detection(next_frame, self.eye_region)
            # else:
            #     self.latest_pupil_state = self._img_multi_obj_detect(next_frame)
            self.frame_force_re_localization(next_frame, next_frame_timestamp)
            if self.latest_pupil_state is not None:
                self.re_localization = False
            else:
                self.re_localization = True
        else:
            similarity = self.frame_check_similarity(next_frame, self.next_frame_timestamp)
            if similarity < self.similarity_threshold:
                self.re_localization = True

    def frame_force_re_localization(self, next_frame, next_frame_timestamp):
        # force frame based update
        self.latest_pupil_state = self.img_pupil_detect(next_frame)

        self.tracking_result.append(dict(
            method="img_detect", result_timestamp=next_frame_timestamp, pupil=self.latest_pupil_state
        ))
        self.ev_tracking_start_timestamp = next_frame_timestamp

    def frame_check_similarity(self, next_frame, next_frame_timestamp):
        try:
            similarity = shape_based_similarity(next_frame, self.latest_pupil_state)
        except EllipseOutOfBoundException as e:
            self.tracking_result.append(dict(
                method="invalid pupil for similarity", pupil=self.latest_pupil_state, result_timestamp=next_frame_timestamp
            ))
            similarity = 0
            print(e)
        else:
            # self.img_tracking_result[self.next_frame_timestamp] = dict(method="similarity", similarity=similarity)
            self.tracking_result.append(dict(
                method="img_similarity", result_timestamp=next_frame_timestamp, similarity=similarity,
                pupil=self.latest_pupil_state
            ))
        return similarity

    def frame_force_re_localization_with_low_similarity(self, next_frame, next_frame_timestamp):
        # frame based update when lost
        if self.re_localization:
            self.frame_force_re_localization(next_frame, next_frame_timestamp)
        else:
            similarity = self.frame_check_similarity(next_frame, next_frame_timestamp)
            if similarity < self.similarity_threshold:
                self.frame_force_re_localization(next_frame, next_frame_timestamp)
        if self.latest_pupil_state is not None:
            self.re_localization = False
        else:
            self.re_localization = True


class EndToEndTracker(BaseEndToEndTracker):
    def __init__(self, img_path_list, event_stream, tracking_start_timestamp, tracking_end_timestamp,
                 ev_pupil_tracking_model, img_pupil_detection_model, img_det_model_input_name,
                 is_single_obj_detect_model, is_eye_region_crop_resize, frame_target_size, eye_region, sample_rads,
                 patch_size, event_accum_time, max_accum_frame_num, event_accum_num_threshold, accum_with_overlap,
                 similarity_threshold, frame_interval=40000, force_re_localization=False,
                 re_localization_wo_similarity_gap=False):

        super().__init__(img_path_list, event_stream, tracking_start_timestamp, tracking_end_timestamp,
                         ev_pupil_tracking_model, img_pupil_detection_model, img_det_model_input_name,
                         is_single_obj_detect_model, is_eye_region_crop_resize, frame_target_size, eye_region,
                         sample_rads, patch_size, event_accum_time, max_accum_frame_num, event_accum_num_threshold,
                         accum_with_overlap, similarity_threshold, frame_interval, force_re_localization,
                         re_localization_wo_similarity_gap)

        self.latest_pupil_state: np.ndarray = None  # 5 array
        self.patch_mask = None
        self.patch_regions = None

        self.ev_sum_accum_num = 0
        self.ev_accum_interval_list = []

    def frame_force_re_localization(self, next_frame, next_frame_timestamp):
        super().frame_force_re_localization(next_frame, next_frame_timestamp)
        self.ev_accum_interval_list.clear()
        self.ev_sum_accum_num = 0

    def end_to_end_tracking(self):
        self.frame_based_update() # init

        while self.ev_tracking_start_timestamp < self.tracking_end_timestamp:
            if (self.next_frame_timestamp - self.ev_tracking_start_timestamp) < (self.event_accum_time / 2):
                if self.next_frame_idx == len(self.img_path_list) - 1:
                    return
                self.frame_based_update()
                continue

            accum_start_time = self.ev_tracking_start_timestamp
            accum_end_time = accum_start_time + self.event_accum_time

            if self.latest_pupil_state is not None:
                if len(self.ev_accum_interval_list) == 0:
                    self.patch_mask, self.patch_regions = parse_patch_region(self.latest_pupil_state, self.sample_rads,
                                                                             self.patch_size,
                                                                             with_overlap=self.accum_with_overlap)

                if np.any(((self.patch_regions[:, 2] - self.patch_regions[:, 0]) < self.patch_size // 2) | (
                        (self.patch_regions[:, 3] - self.patch_regions[:, 1]) < self.patch_size // 2)):
                    self.tracking_result.append(dict(
                        method="invalid img detect pupil", pupil=self.latest_pupil_state, result_timestamp=accum_end_time
                    ))
                    self.re_localization = True
                else:
                    event_accum_num = event_patch_accum(self.event_stream, accum_start_time, accum_end_time,
                                                        self.patch_mask)
                    self.ev_sum_accum_num += event_accum_num
                    self.ev_accum_interval_list.append((accum_start_time, accum_end_time, event_accum_num))

                    if self.ev_sum_accum_num >= self.event_accum_num_threshold:
                        self.latest_pupil_state = ev_single_pred(self.ev_pupil_tracking_model, self.event_stream,
                                                                 self.ev_accum_interval_list[0][0],
                                                                 self.ev_accum_interval_list[-1][1],
                                                                 self.latest_pupil_state, "pol_event_count")
                        self.tracking_result.append(dict(
                            method="ev_tracking_model", accum_start_timestamp=self.ev_accum_interval_list[0][0],
                            accum_end_timestamp=self.ev_accum_interval_list[-1][1],
                            event_accum_num=self.ev_sum_accum_num,
                            pupil=self.latest_pupil_state
                        ))

                        sub_mask = self.event_stream[:, 3] >= self.ev_accum_interval_list[-1][1]
                        self.event_stream = self.event_stream[sub_mask, :]

                        self.ev_accum_interval_list.clear()
                        self.ev_sum_accum_num = 0
                    else:
                        self.tracking_result.append(dict(
                            method="ev_accum_threshold", accum_start_timestamp=self.ev_accum_interval_list[0][0],
                            accum_end_timestamp=self.ev_accum_interval_list[-1][1],
                            inter_val_num=len(self.ev_accum_interval_list), event_accum_num=self.ev_sum_accum_num,
                            pupil=self.latest_pupil_state
                        ))
                        if self.max_accum_frame_num is not None and len(

                                self.ev_accum_interval_list) == self.max_accum_frame_num:
                            removed_events = self.ev_accum_interval_list.pop(0)
                            self.ev_sum_accum_num -= removed_events[2]
            else:
                self.tracking_result.append(dict(
                    method="no pre img detect pupil", pupil=self.latest_pupil_state, result_timestamp=accum_end_time
                ))
                assert self.re_localization == True

            self.ev_tracking_start_timestamp = accum_end_time


class EndToEndTrackerWithPreAccum(BaseEndToEndTracker): # process event use pre accum 
    

    def __init__(self, img_path_list, event_stream, tracking_start_timestamp, tracking_end_timestamp,
                 ev_pupil_tracking_model, img_pupil_detection_model, img_det_model_input_name,
                 is_single_obj_detect_model, is_eye_region_crop_resize, frame_target_size, eye_region, sample_rads,
                 patch_size, event_accum_time, max_accum_frame_num, event_accum_num_threshold, accum_with_overlap,
                 similarity_threshold, frame_interval=40000, force_re_localization=False,
                 re_localization_wo_similarity_gap=False):

        super().__init__(img_path_list, event_stream, tracking_start_timestamp, tracking_end_timestamp,
                         ev_pupil_tracking_model, img_pupil_detection_model, img_det_model_input_name,
                         is_single_obj_detect_model, is_eye_region_crop_resize, frame_target_size, eye_region,
                         sample_rads, patch_size, event_accum_time, max_accum_frame_num, event_accum_num_threshold,
                         accum_with_overlap, similarity_threshold, frame_interval, force_re_localization,
                         re_localization_wo_similarity_gap)
        self.latest_pupil_state: np.ndarray = None

    def end_to_end_tracking(self):
        self.frame_based_update()  # init

        while self.ev_tracking_start_timestamp < self.tracking_end_timestamp:
            if (self.next_frame_timestamp - self.ev_tracking_start_timestamp) < (self.event_accum_time / 2):
                if self.next_frame_idx == len(self.img_path_list) - 1:
                    return
                self.frame_based_update()
                continue

            ev_start_time = self.ev_tracking_start_timestamp  # latest pupil state
            ev_end_time = ev_start_time + self.event_accum_time

            # event tracking
            if self.latest_pupil_state is not None:
                patch_mask, patch_regions = parse_patch_region(self.latest_pupil_state, self.sample_rads,
                                                               self.patch_size,
                                                               with_overlap=self.accum_with_overlap)

                if np.any(((patch_regions[:, 2] - patch_regions[:, 0]) < self.patch_size // 2) | (
                        (patch_regions[:, 3] - patch_regions[:, 1]) < self.patch_size // 2)):
                    self.tracking_result.append(dict(
                        method="invalid img detect pupil", pupil=self.latest_pupil_state, result_timestamp=ev_end_time
                    ))
                    self.re_localization = True
                else:
                    event_accum_num = 0

                    for accum_frame in range(self.max_accum_frame_num):
                        accum_end_time = ev_end_time - accum_frame * self.event_accum_time
                        accum_start_time = accum_end_time - self.event_accum_time
                        event_accum_num += event_patch_accum(self.event_stream, accum_start_time, accum_end_time,
                                                             patch_mask)

                        if event_accum_num >= self.event_accum_num_threshold:
                            self.latest_pupil_state = ev_single_pred(self.ev_pupil_tracking_model, self.event_stream,
                                                                     accum_start_time,
                                                                     ev_end_time,
                                                                     self.latest_pupil_state, "pol_event_count")
                            self.tracking_result.append(dict(
                                method="ev_tracking_model", accum_start_timestamp=accum_start_time,
                                accum_end_timestamp=ev_end_time,
                                event_accum_num=event_accum_num,
                                pupil=self.latest_pupil_state
                            ))
                            break

                            # sub_mask = self.event_stream[:, 3] >= self.ev_accum_interval_list[-1][1]
                            # self.event_stream = self.event_stream[sub_mask, :]

                        if accum_frame == self.max_accum_frame_num - 1:
                            self.tracking_result.append(dict(
                                method="ev_accum_threshold", accum_start_timestamp=accum_start_time,
                                accum_end_timestamp=ev_end_time,
                                event_accum_num=event_accum_num,
                                pupil=self.latest_pupil_state
                            ))
            else:
                self.tracking_result.append(dict(
                    method="no pre img detect pupil", pupil=self.latest_pupil_state,result_timestamp=ev_end_time
                ))
                assert self.re_localization == True

            self.ev_tracking_start_timestamp = ev_end_time


class EndToEndTrackingEvaluator:
    def __init__(self,
                 user_list, eye_list, session_list,
                 tracking_frame_num,
                 continuous_ann_track: bool,
                 annotation_filename,
                 similarity_threshold,
                 model_cfg, device,
                 model_cfg_name,
                 pre_accum_tracking,
                 force_re_localization,
                 re_localization_wo_similarity_gap):

        base_rad = np.pi * 2 / model_cfg["patch_num"]
        self.sample_rads = base_rad * np.arange(model_cfg["patch_num"])

        img_pupil_detection_cfg = Config.fromfile(model_cfg["img_detect_cfg"])
        ev_pupil_tracking_cfg = Config.fromfile(model_cfg["ev_tracking_cfg"])

        self.img_pupil_detection_model = build_eval_model(img_pupil_detection_cfg, device,
                                                          model_cfg.get("img_detect_checkpoint", None))
        self.ev_pupil_tracking_model = build_eval_model(ev_pupil_tracking_cfg, device, model_cfg.get("event_tracking_checkpoint", None))
        self.user_list = user_list
        self.eye_list = eye_list
        self.session_list = session_list
        self.continuous_ann_track = continuous_ann_track
        if not continuous_ann_track:
            self.annotation_filename = Path(annotation_filename)
        self.tracking_frame_num = tracking_frame_num

        self.is_single_obj_detect_model = model_cfg["is_single_obj_detect_model"]
        self.is_eye_region_crop_resize = model_cfg["is_eye_region_crop_resize"]
        if self.is_eye_region_crop_resize:
            self.frame_target_size = model_cfg["frame_target_size"]
        else:
            self.frame_target_size = None
        self.img_det_model_input_name = model_cfg["input_name"]

        # event
        self.patch_size = model_cfg["patch_size"]

        self.event_accum_time = model_cfg["ev_accum_time"]
        self.max_accum_frame_num = model_cfg["max_accum_frame_num"]
        self.event_accum_num_threshold = model_cfg["event_accum_num_threshold"]
        self.accum_with_overlap = model_cfg["accum_with_overlap"]
        # similarity
        self.similarity_threshold = similarity_threshold

        self.model_cfg_name = model_cfg_name
        self.model_cfg_option = model_cfg_options[model_cfg_name]

        self.pre_accum_tracking = pre_accum_tracking
        self.force_re_localization = force_re_localization
        self.re_localization_wo_similarity_gap = re_localization_wo_similarity_gap

    def _continuous_ann_data(self, user, session, continuous_ann_segments, img_path_list):
        user_segments = continuous_ann_segments.loc[user + 1]
        session_start = user_segments.iloc[session_col_idx[session][0]]
        session_end = user_segments.iloc[session_col_idx[session][1]]

        start_frame = img_path_list[session_start - 1 - 1]
        start_id, start_timestamp = parse_frame_filename(start_frame.name, timestamp_offset=frame_timestamp_offset)

        end_frame = img_path_list[session_end - 1]
        end_id, end_timestamp = parse_frame_filename(end_frame.name, timestamp_offset=frame_timestamp_offset)

        tracking_seg_frames_list = [dict(
            tracking_start_frame_name=start_frame.name,
            tracking_end_frame_name=end_frame.name,
            tracking_start_timestamp=start_timestamp,
            tracking_end_timestamp=end_timestamp,
            img_list=img_path_list[session_start - 1 - 1:session_end]
        )]
        return tracking_seg_frames_list

    def _last_ann_tracking_frame_list(self, user, eye, session, event_dir, img_path_list):
        annotation_file_path = event_dir / self.annotation_filename
        annotation = load_ann(annotation_file_path)

        tracking_seg_frames_list = []
        for data in annotation["data_list"]:
            last_gt_ellipse = parse_pupil_ellipse(data)
            if last_gt_ellipse is None:
                logging.log(logging.WARN,
                            f"user {user}, eye{eye},session{session} data {data['img_filename']} has no pupil ann")
                continue

            last_frame_id, last_frame_timestamp = parse_frame_filename(data["img_filename"], timestamp_offset=frame_timestamp_offset)
            if last_frame_id - self.tracking_frame_num - 1 < 0:
                continue

            first_frame_path = img_path_list[last_frame_id - self.tracking_frame_num - 1]

            first_frame_id, first_frame_timestamp = parse_frame_filename(first_frame_path.name, timestamp_offset=frame_timestamp_offset)
            assert first_frame_id == last_frame_id - self.tracking_frame_num

            tracking_seg_frames_list.append(dict(
                tracking_start_frame_name=first_frame_path.name,
                tracking_end_frame_name=data["img_filename"],
                tracking_start_timestamp=first_frame_timestamp,
                tracking_end_timestamp=last_frame_timestamp,
                img_list=img_path_list[first_frame_id - 1:last_frame_id]
            ))

        return tracking_seg_frames_list

    def _data_preparation(self, user, eye, session):
        event_dir = Path(single_mini_data_pattern.format(user_id=user, eye=eye, session=session))
        event_file_path = event_dir / "events.npz"
        events_list = np.load(event_file_path)
        event_stream = np.stack((events_list['x'], events_list['y'], events_list['p'], events_list['t']),
                                axis=1)
        eye_region_data = pd.read_csv(event_dir / "eye_region.txt").loc[0]
        eye_region = [eye_region_data.x, eye_region_data.y, eye_region_data.width, eye_region_data.height]

        img_dir = Path(origin_single_data_pattern.format(user_id=user, eye=eye, session=session)) / "frames"
        img_path_list = list(img_dir.glob("*.png"))
        img_path_list.sort()

        if self.continuous_ann_track:
            continuous_ann_segments = pd.read_excel(continuous_ann_segments_path)
            tracking_seg_frames_list = self._continuous_ann_data(user, session, continuous_ann_segments,
                                                                 img_path_list)
        else:
            tracking_seg_frames_list = self._last_ann_tracking_frame_list(user, eye, session, event_dir, img_path_list)

        return event_stream, tracking_seg_frames_list, eye_region

    def _one_session_tracking(self, user, eye, session):
        # prepare data,
        event_stream, tracking_seg_frames_list, eye_region = self._data_preparation(user, eye, session)

        results = []
        for tracking_segment in tracking_seg_frames_list:
            event_mask = (event_stream[:, 3] >= tracking_segment['tracking_start_timestamp'] - self.model_cfg_option[
                "max_accum_frame_num"] * self.model_cfg_option["ev_accum_time"]) & (
                                 event_stream[:, 3] <= tracking_segment['tracking_end_timestamp'])
            if self.pre_accum_tracking:
                end_to_end_tracker = EndToEndTrackerWithPreAccum(tracking_segment["img_list"],
                                                                 event_stream[event_mask, :],
                                                                 tracking_segment['tracking_start_timestamp'],
                                                                 tracking_segment['tracking_end_timestamp'],
                                                                 self.ev_pupil_tracking_model,
                                                                 self.img_pupil_detection_model,
                                                                 self.img_det_model_input_name,
                                                                 self.is_single_obj_detect_model,
                                                                 self.is_eye_region_crop_resize,
                                                                 self.frame_target_size, eye_region,
                                                                 self.sample_rads, self.patch_size,
                                                                 self.event_accum_time, self.max_accum_frame_num,
                                                                 self.event_accum_num_threshold,
                                                                 self.accum_with_overlap,
                                                                 self.similarity_threshold,
                                                                 force_re_localization=self.force_re_localization,
                                                                 re_localization_wo_similarity_gap=self.re_localization_wo_similarity_gap)
            else:
                end_to_end_tracker = EndToEndTracker(tracking_segment["img_list"], event_stream[event_mask, :],
                                                     tracking_segment['tracking_start_timestamp'],
                                                     tracking_segment['tracking_end_timestamp'],
                                                     self.ev_pupil_tracking_model, self.img_pupil_detection_model,
                                                     self.img_det_model_input_name, self.is_single_obj_detect_model,
                                                     self.is_eye_region_crop_resize,
                                                     self.frame_target_size, eye_region,
                                                     self.sample_rads, self.patch_size,
                                                     self.event_accum_time, self.max_accum_frame_num,
                                                     self.event_accum_num_threshold, self.accum_with_overlap,
                                                     self.similarity_threshold,
                                                     force_re_localization=self.force_re_localization,
                                                     re_localization_wo_similarity_gap=self.re_localization_wo_similarity_gap
                                                     )
            end_to_end_tracker.end_to_end_tracking()

            results.append(dict(
                tracking_start_frame_name=tracking_segment["tracking_start_frame_name"],
                tracking_end_frame_name=tracking_segment["tracking_end_frame_name"],
                tracking_result=end_to_end_tracker.tracking_result
            ))
            print(
                f"time{time.asctime(time.localtime())} tracked u{user} e{eye} s{session}, segment {tracking_segment['tracking_end_frame_name']}")

        return results

    def track(self):
        for u in self.user_list:
            for e in self.eye_list:
                for s in self.session_list:
                    results = self._one_session_tracking(u, e, s)
                    result_dir = Path(single_mini_data_pattern.format(user_id=u, eye=e,
                                                                      session=s)) / "end_to_end_tracking" / self.model_cfg_name
                    if not result_dir.exists():
                        result_dir.mkdir(parents=True)
                        with open(result_dir / 'model_cfg_config.json', "x") as config_file:
                            json.dump(model_cfg_options[self.model_cfg_name], config_file, indent=2)

                    if self.pre_accum_tracking:
                        result_filename = "pre_accum_"
                    else:
                        result_filename = ""

                    if self.continuous_ann_track:
                        result_filename = result_filename + f"continuous_seg_track"
                    else:
                        result_filename = result_filename + f"{self.annotation_filename.stem}-track_pre_{self.tracking_frame_num}frames"

                    if self.force_re_localization:
                        result_filename += "force_re_localization.pickle"
                    else:
                        if self.re_localization_wo_similarity_gap:
                            result_filename += f"_with_{self.similarity_threshold}similarity_no_gap.pickle"
                            # result_filename += "_with_{self.similarity_threshold}similarity_no_gap.pickle"
                        else:
                            result_filename += f"_with_{self.similarity_threshold}similarity.pickle"
                            # result_filename += "_with_{self.similarity_threshold}similarity.pickle"

                    with open(result_dir / result_filename, "wb") as result_file:
                        pickle.dump(results, result_file)
                        print(f"save {result_dir / result_filename}")

                    # return results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default="1")
    parser.add_argument("--user_start", type=int, default=1)
    parser.add_argument("--user_end", type=int, default=2)
    parser.add_argument("--model_cfg_option", type=str, default="config-eye_crop_mbv3spreX_multi_anchor_det-s1_f2_n4_trans-pre_accum10_50_blink_exp5_0.5_rand")
    parser.add_argument("--tracking_frame_num", type=int, default=100)
    parser.add_argument("--continuous_ann_track", action="store_true")
    parser.add_argument("--annotation_filename", type=str, default="origin_labelled_pupil_dataset.json")
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    parser.add_argument("--pre_accum_tracking", action="store_true")
    parser.add_argument("--force_re_localization", action="store_true")
    parser.add_argument("--re_localization_wo_similarity_gap", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    device = torch.device(f"cuda:{args.device}")

    # user_list = range(args.user_start, args.user_end)
    user_list = [48]
    eye_list = ["left"]
    session_list = ["201"]

    evaluator = EndToEndTrackingEvaluator(user_list, eye_list, session_list, args.tracking_frame_num,
                                          args.continuous_ann_track, args.annotation_filename,
                                          args.similarity_threshold, model_cfg_options[args.model_cfg_option], device,
                                          args.model_cfg_option, args.pre_accum_tracking, args.force_re_localization,
                                          args.re_localization_wo_similarity_gap)

    results = evaluator.track()
