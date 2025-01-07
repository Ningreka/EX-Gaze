import random

import pandas as pd
import numpy as np

from pathlib import Path
import json

from mmengine.structures import InstanceData

from misc import ev_eye_dataset_utils
from misc.annotation_generator import parse_common_meta_info


def valid_timestamp_pair_annotation(valid_timestamp_pairs, instance_list):
    data_list = []
    for timestamp_pair in valid_timestamp_pairs:
        pre_gt = instance_list[timestamp_pair['pre_state_frame_timestamp']]
        cur_gt = instance_list[timestamp_pair['cur_state_frame_timestamp']]
        if pre_gt is None or cur_gt is None:
            continue
        data_list.append(dict(
            pre_gt=pre_gt, cur_gt=cur_gt,
            events_between=f"{timestamp_pair['event_start_timestamp']}-{timestamp_pair['event_end_timestamp']}",
            img_shape=ev_eye_dataset_utils.img_shape
        ))
    return data_list


def parse_ellipse_predict(ellipse_predict_dir, confident_thr):
    pupil_predicts = pd.read_csv(ellipse_predict_dir + "/ellipse_label.csv")
    frame_timestamps = np.loadtxt(ellipse_predict_dir + "/timestamp.txt")
    valid_predicts = pupil_predicts[pupil_predicts.confident > confident_thr]

    instance_list = dict()
    for i, label in valid_predicts.iterrows():
        frame_timestamp = frame_timestamps[i]
        instance = {
            "img_id": i,
            "timestamp": frame_timestamp,
            "img_filename": label.filename,
            "instances": [
                {
                    "ellipse": [label.cx, label.cy, label.w, label.h, label.angle],
                    "ellipse_label": ev_eye_dataset_utils.classes[0]
                }]
        }
        instance_list[frame_timestamp] = instance

    return instance_list


def parse_valid_timestamp_pair_wo_density(valid_timestamp_file_path):
    valid_timestamp_csv = pd.read_csv(valid_timestamp_file_path)
    valid_timestamp_pair = []
    """
    {
        pre_state_frame_timestamp:,
        cur_state_frame_timestamp:
        event_start_timestamp:,
        event_end_timestamp
    }
    """
    for i, label in valid_timestamp_csv.iterrows():
        valid_timestamp_pair.append(dict(
            pre_state_frame_timestamp=label.start_timestamp,
            cur_state_frame_timestamp=label.end_timestamp,
            event_start_timestamp=label.start_timestamp,
            event_end_timestamp=label.end_timestamp
        ))
    return valid_timestamp_pair


def parse_valid_timestamp_pair_with_density(valid_timestamp_file_path, repeat_filer_prob):
    valid_timestamp_csv = pd.read_csv(valid_timestamp_file_path)
    valid_timestamp_pair = []
    pre_event_num_accum = 0

    for i, label in valid_timestamp_csv.iterrows():
        event_start_timestamp = label.event_start_timestamp
        event_end_timestamp = label.event_end_timestamp
        event_num_accum = label.event_num_accum
        if i > 0 and (valid_timestamp_pair[-1]["event_start_timestamp"] == event_start_timestamp and
                      valid_timestamp_pair[-1]["event_end_timestamp"] == event_end_timestamp and
                      event_num_accum == pre_event_num_accum):
            
            if random.random() <= repeat_filer_prob:
                continue
        
        valid_timestamp_pair.append(dict(
            pre_state_frame_timestamp=label.iter_start_timestamp,
            cur_state_frame_timestamp=event_end_timestamp,
            event_start_timestamp=event_start_timestamp,
            event_end_timestamp=event_end_timestamp
        ))
        pre_event_num_accum = event_num_accum
    return valid_timestamp_pair


def continuous_frame_event_accum_tracking_dataset(user_list, eye_list, session_list, frame_rate_list,
                                                  valid_timestamp_file_path, is_with_density, additional_meta_info,
                                                  task_name="ev pupil tracking with multi frame",
                                                  confident_thr=0, output_filename=None):

    output_filename = (output_filename if output_filename is not None else task_name) + ".json"
    for user in user_list:
        for eye in eye_list:
            for session in session_list:
                for frame_rate in frame_rate_list:
                    meta_info = ev_eye_dataset_utils.base_meta_info
                    meta_info["dataset_name"] = "continuous_frame_event_accum_tracking_dataset"
                    meta_info["task name"] = task_name
                    meta_info["user_id"] = user
                    meta_info["eye"] = eye
                    meta_info["session"] = session
                    meta_info["classes"] = ["pupil"]
                    meta_info["frame_rate"] = 25 if frame_rate == "origin" else frame_rate
                    meta_info = parse_common_meta_info(meta_info, user, eye, session)
                    for k, v in additional_meta_info.items():
                        meta_info[k] = v

                    single_session_path = ev_eye_dataset_utils.single_mini_data_pattern.format(
                        user_id=user, eye=eye, session=session) + f"/inter-{frame_rate}_frame_endtime"
                    single_valid_timestamp_file_path = single_session_path + "/" + valid_timestamp_file_path

                    instance_list = parse_ellipse_predict(single_session_path, confident_thr)
                    if is_with_density:
                        valid_timestamp_pair = parse_valid_timestamp_pair_with_density(single_valid_timestamp_file_path, 0.95)
                    else:
                        valid_timestamp_pair = parse_valid_timestamp_pair_wo_density(single_valid_timestamp_file_path)

                    data_list = valid_timestamp_pair_annotation(valid_timestamp_pair, instance_list)

                    ann = dict(
                        metainfo=meta_info,
                        data_list=data_list
                    )
                    ann_output_path = single_session_path + "/" + output_filename

                    with open(ann_output_path, mode="w") as ann_file:
                        json.dump(ann, ann_file, indent=2)

                    print(f"dumped user{user}, eye {eye}, session {session} to {ann_output_path}")


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--patch_num", type=int, default=8)
    arg_parser.add_argument("--patch_size", type=int, default=16)
    arg_parser.add_argument("--max_accum_frame_num", type=int, default=None)
    arg_parser.add_argument("--confident_threshold", type=float, default=0.9)
    arg_parser.add_argument("--event_num_threshold", type=int, default=40)
    arg_parser.add_argument("--accum_with_overlap", action="store_true")
    arg_parser.add_argument("--use_blink_segments", action="store_true")
    arg_parser.add_argument("--blink_seg_expand", type=int, default=20, help="expand blink segments in ms")

    args = arg_parser.parse_args()

    user_list = range(1, 49)
    # user_list = range(1, 2)
    eye_list = ["left", "right"]
    # eye_list = ["left"]
    session_list = ["101", "102", "201", "202"]
    # session_list = ["101"]
    frame_rate_list = [2000]

    additional_meta_info = dict(
        patch_num=args.patch_num,
        patch_size=args.patch_size,
        confident_threshold=args.confident_threshold,
        event_num_threshold=args.event_num_threshold,
    )

    if args.use_blink_segments:
        save_to_dir = (f"blink_seg_exp{args.blink_seg_expand}_patch_n{args.patch_num}_s{args.patch_size}"
         + f"_ev_{'overlap_' if args.accum_with_overlap else ''}accum{args.event_num_threshold}")
        additional_meta_info["blink_seg_expand"] = args.blink_seg_expand
    else:
        save_to_dir = f"open{args.confident_threshold}_patch_n{args.patch_num}_s{args.patch_size}_ev_accum{args.event_num_threshold}"

    is_with_density = False
    if args.max_accum_frame_num:
        save_to_dir = save_to_dir + f"_continuous{args.max_accum_frame_num}frame"
        is_with_density = True
        additional_meta_info.update(
            max_accum_frame_num=args.max_accum_frame_num
        )
    valid_timestamp_pair_path = save_to_dir + "/valid_timestamp_pair.csv"

    continuous_frame_event_accum_tracking_dataset(
        user_list, eye_list, session_list, frame_rate_list, valid_timestamp_pair_path, is_with_density,
        additional_meta_info, confident_thr=args.confident_threshold,
        output_filename=save_to_dir + "/continuous_frame_event_accum_tracking_dataset"
    )
