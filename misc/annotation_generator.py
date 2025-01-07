import json
from pathlib import Path
from typing import Dict

from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd

from mmengine.structures import InstanceData

import misc.ev_eye_dataset_utils as ev_eye_dataset_utils

def tracking_dataset_annotation(instance_list):
    assert len(instance_list) > 0
    pre_instance = None
    data_list = []

    for i in range(0, len(instance_list)):
        cur_instance = instance_list[i]
        cur_gt = {
            "img_id": cur_instance.frame_idx,
            "timestamp": cur_instance.frame_timestamp,
            "img_filename": cur_instance.filename,
            "instances": [
                {
                    "ellipse": e,
                    "ellipse_label": ev_eye_dataset_utils.classes[l]
                } for e, l in zip(cur_instance.ellipses, cur_instance.digit_labels)
            ]
        }
        if pre_instance is not None:
            if pre_instance.frame_idx == cur_instance.frame_idx - 1:
                
                valid_data = {"pre_gt": pre_gt, "cur_gt": cur_gt,
                              "events_between": cur_gt["timestamp"],
                              "img_shape": ev_eye_dataset_utils.img_shape}
                data_list.append(valid_data)

        pre_instance = cur_instance
        pre_gt = cur_gt

    return data_list


def detect_dataset_annotation(instance_list):
    data_list = []

    for instance in instance_list:
        data = {
            "img_shape": ev_eye_dataset_utils.img_shape,
            "img_id": instance.frame_idx,
            "timestamp": instance.frame_timestamp,
            "img_filename": instance.filename,
            "instances": [
                {
                    "ellipse": e,
                    "ellipse_label": l
                } for e, l in zip(instance.ellipses, instance.labels)
            ]
        }
        data_list.append(data)

    return data_list


def parse_common_meta_info(meta_info, user, eye, session):
    eye_region = pd.read_csv(ev_eye_dataset_utils.single_mini_data_pattern.format(
        user_id=user, eye=eye, session=session) + "/eye_region.txt")
    meta_info["eye_region"] = [eye_region.iloc[0,0].item(), eye_region.iloc[0,1].item(),
                               eye_region.iloc[0,2].item(),
                               eye_region.iloc[0,3].item()]
    return meta_info


def ev_pupil_model_detection_dataset(user_list, eye_list, session_list, frame_rate_list, data_list_generator,
                                     task_name="ev pupil tracking",
                                     confident_thr=0, output_filename=None):
    def _parse_ellipse_predict(ellipse_predict_dir):
        pupil_predicts = pd.read_csv(ellipse_predict_dir + "/ellipse_label.csv")
        frame_timestamps = np.loadtxt(ellipse_predict_dir + "/timestamp.txt")
        valid_predicts = pupil_predicts[pupil_predicts.confident > confident_thr]

        instance_list = []
        for i, label in valid_predicts.iterrows():
            frame_timestamp = frame_timestamps[i]
            instance = InstanceData(
                metainfo={
                    "filename": label.filename,
                    "frame_idx": i,
                    "frame_timestamp": frame_timestamp}
            )
            instance["ellipses"] = [[label.cx, label.cy, label.w, label.h, label.angle]]
            instance["digit_labels"] = [0]
            instance_list.append(instance)

        return instance_list

    output_filename = (output_filename if output_filename is not None else task_name) + ".json"
    for user in user_list:
        for eye in eye_list:
            for session in session_list:
                for frame_rate in frame_rate_list:
                    meta_info = ev_eye_dataset_utils.base_meta_info
                    meta_info["dataset_name"] = "ev_pupil_model_detection_dataset"
                    meta_info["task name"] = task_name
                    meta_info["user_id"] = user
                    meta_info["eye"] = eye
                    meta_info["session"] = session
                    meta_info["classes"] = ["pupil"]
                    meta_info["frame_rate"] = 25 if frame_rate == "origin" else frame_rate
                    meta_info = parse_common_meta_info(meta_info, user, eye, session)

                    single_session_path = ev_eye_dataset_utils.single_mini_data_pattern.format(
                        user_id=user, eye=eye, session=session) + f"/inter-{frame_rate}"

                    instance_list = _parse_ellipse_predict(single_session_path)
                    data_list = data_list_generator(instance_list)

                    ann = dict(
                        metainfo=meta_info,
                        data_list=data_list
                    )
                    ann_output_path = single_session_path + "/" + output_filename

                    with open(ann_output_path, mode="w") as ann_file:
                        json.dump(ann, ann_file, indent=2)

                    print(f"dumped user{user}, eye {eye}, session {session} to {ann_output_path}")


def parse_origin_landmark_label_filename(eye, session, user_id):
    return ev_eye_dataset_utils.landmark_label_file_pattern.format(eye=eye, session=session,
                                                                   user_id=user_id)

# def parse_copy_origin_landmark_label_filename(eye, session, user_id):
#     return ev_eye_dataset_utils.copy_landmark_label_file_pattern.format(eye=eye, session=session,
#                                                                         user_id=user_id)


def parse_origin_dataset_label_filename(eye, session, user_id):
    return ev_eye_dataset_utils.origin_dataset_label_file_pattern.format(eye_prefix="right_" if eye == "right" else "",
                                                                         session=session, user_id=user_id, eye=eye)


def origin_landmark_labelled_dataset(user_list, eye_list, session_list, label_filename_parser, data_list_generator,
                                     task_name, dataset_name="ev eye landmark labelled dataset",
                                     output_filename=None):
    output_filename = (output_filename if output_filename is not None else task_name) + ".json"

    for user in user_list:
        for eye in eye_list:
            for session in session_list:
                landmark_label_file = label_filename_parser(eye=eye, session=session, user_id=user)

                session_meta_info = ev_eye_dataset_utils.base_meta_info
                session_meta_info["dataset_name"] = dataset_name
                session_meta_info["task_name"] = task_name
                session_meta_info['frame_rate'] = 25
                session_meta_info["user_id"] = user
                session_meta_info['eye'] = eye
                session_meta_info["session"] = session

                session_meta_info = parse_common_meta_info(session_meta_info, user, eye, session)

                instance_list = ev_eye_dataset_utils.parse_landmark_labels(landmark_label_file)

                data_list = data_list_generator(instance_list)

                ann = dict(metainfo=session_meta_info, data_list=data_list)
                ann_output_path = ev_eye_dataset_utils.single_mini_data_pattern.format(
                    user_id=user, eye=eye, session=session) + "/" + output_filename

                with open(ann_output_path, mode="w") as ann_file:
                    json.dump(ann, ann_file, indent=2)

                print(f"dumped user{user}, eye {eye}, session {session}")
