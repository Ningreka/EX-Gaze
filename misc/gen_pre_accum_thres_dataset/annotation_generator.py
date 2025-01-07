"""
filter samples according to accumulated event num use "check_sample_event_num.py"
generate annotations for trainning
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from misc import ev_eye_dataset_utils
from misc.annotation_generator import parse_common_meta_info
from misc.ev_eye_dataset_utils import mini_dataset_dir, single_mini_data_pattern
from misc.generate_event_threshold_dataset.annotation_generator import parse_ellipse_predict
from misc.generate_event_threshold_dataset.check_valid_data import is_blink


def events_between_annotation(interval_event_nums, instance_list, blink_segments):
    data_list = []
    for sample_interval_event_nums in interval_event_nums:
        prior_instance = instance_list.get(sample_interval_event_nums["prior_frame_timestamp"])
        post_instance = instance_list.get(sample_interval_event_nums["post_frame_timestamp"])
        if prior_instance is None or post_instance is None:
            continue
        if (is_blink(0, 0, prior_instance["timestamp"], blink_segments) or
                is_blink(0, 0, post_instance["timestamp"], blink_segments)):
            continue

        if sample_interval_event_nums["total_event_num"] >= args.event_num_threshold:
            accum_event_num = 0
            interval_list = []
            for event_num in sample_interval_event_nums["event_nums"]:
                accum_event_num += event_num["event_num"]
                interval_list.append((event_num['start_timestamp'], event_num['end_timestamp']))
                if accum_event_num >= args.event_num_threshold:
                    break
            data_list.append(dict(
                pre_gt=prior_instance, cur_gt=post_instance,
                events_between=f"{interval_list[-1][0]}-{interval_list[0][1]}",
                accum_event_num=accum_event_num,
                img_shape=ev_eye_dataset_utils.img_shape
            ))

    return data_list

def interval_list_annotation(interval_event_nums, instance_list, blink_segments):
    data_list = []
    for sample_interval_event_nums in interval_event_nums:
        prior_instance = instance_list.get(sample_interval_event_nums["prior_frame_timestamp"])
        post_instance = instance_list.get(sample_interval_event_nums["post_frame_timestamp"])
        if prior_instance is None or post_instance is None:
            continue
        if (is_blink(0, 0, prior_instance["timestamp"], blink_segments) or
                is_blink(0, 0, post_instance["timestamp"], blink_segments)):
            continue

        if sample_interval_event_nums["total_event_num"] >= args.event_num_threshold:
            accum_event_num = 0
            interval_list = []
            for event_num in sample_interval_event_nums["event_nums"]:
                accum_event_num += event_num["event_num"]
                interval_list.append(f"{event_num['start_timestamp']}-{event_num['end_timestamp']}")
                if accum_event_num >= args.event_num_threshold:
                    break
            data_list.append(dict(
                pre_gt=prior_instance, cur_gt=post_instance,
                interval_list=interval_list,
                accum_event_num=accum_event_num,
                img_shape=ev_eye_dataset_utils.img_shape
            ))

    return data_list


def main():
    for user in user_list:
        for eye in eye_list:
            for session in session_list:
                data_dir = Path(single_mini_data_pattern.format(user_id=user, eye=eye, session=session))
                insert_frame_dir = data_dir / inter_frame_dir

                segments_csv = pd.read_csv(data_dir / "frames_segment/blink_segments.csv")
                blink_segments = np.column_stack(
                    [segments_csv.blink_start_timestamp, segments_csv.blink_end_timestamp])
                blink_segments[:, 0] -= (args.blink_seg_expand * 1000)
                blink_segments[:, 1] += (args.blink_seg_expand * 1000)

                result_dir = insert_frame_dir / (
                        f"patch_n{args.patch_num}_s{args.patch_size}_pre_max{args.max_accum_frame_num}"
                        + f"{'_with_overlap' if args.accum_with_overlap else ''}")

                interval_event_nums = pd.read_pickle(result_dir / "interval_event_num.pickle")

                instance_list = parse_ellipse_predict(str(insert_frame_dir), 0)

                if args.interval_list_dataset:
                    data_list = interval_list_annotation(instance_list, interval_event_nums, blink_segments)
                else:
                    data_list = events_between_annotation(interval_event_nums, instance_list, blink_segments)

                meta_info = ev_eye_dataset_utils.base_meta_info
                meta_info["dataset_name"] = "continuous_frame_event_pre_accum_tracking_dataset"
                meta_info["task name"] = "ev pupil tracking with multi frame"
                meta_info["user_id"] = user
                meta_info["eye"] = eye
                meta_info["session"] = session
                meta_info["classes"] = ["pupil"]
                meta_info["frame_rate"] = args.frame_rate
                meta_info = parse_common_meta_info(meta_info, user, eye, session)

                ann = dict(
                    metainfo=meta_info,
                    data_list=data_list
                )
                ann_output_path = result_dir / (
                    f"blink_seg_exp{args.blink_seg_expand}_cont_frame_event_pre_accum_thr{args.event_num_threshold}"
                    f"{'_inter_list' if args.interval_list_dataset else ''}_tracking_dataset.json")

                with open(ann_output_path, mode="w") as ann_file:
                    json.dump(ann, ann_file, indent=2)

                print(f"dumped user{user}, eye {eye}, session {session} to {ann_output_path}")


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--user_start", type=int, default=1)
    arg_parser.add_argument("--user_end", type=int, default=49)
    arg_parser.add_argument("--frame_rate", type=int, default=1000)
    arg_parser.add_argument("--patch_num", type=int, default=8)
    arg_parser.add_argument("--patch_size", type=int, default=16)
    arg_parser.add_argument("--max_accum_frame_num", type=int, default=None)
    arg_parser.add_argument("--event_num_threshold", type=int, default=40)
    arg_parser.add_argument("--accum_with_overlap", action="store_true")
    arg_parser.add_argument("--interval_list_dataset", action="store_true")
    arg_parser.add_argument("--blink_seg_expand", type=int, default=20, help="expand blink segments in ms")

    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    user_list = range(args.user_start, args.user_end)
    # user_list = range(1, 2)
    eye_list = ["left", "right"]
    # eye_list = ["left"]
    session_list = ["101", "102", "201", "202"]
    # session_list = ["101", "102"]

    inter_frame_dir = f"inter-{args.frame_rate}_frame_endtime"
    event_format = "pol_event_count"

    main()
