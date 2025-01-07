import argparse
import concurrent.futures
from pathlib import Path
import re
import numpy as np

from misc.ev_eye_dataset_utils import single_mini_data_pattern, img_shape
from misc.event_representations import serialization
import json

from misc.event_representations.event_count import events2event_count_v2

"""accumulate the events with static num threshold and dynamic window size"""

def parse_volume_timestamps(annotation_data_list):
    event_start_timestamps = []
    event_end_timestamps = []
    for data in annotation_data_list:
        event_between = data['events_between']
        match = re.match("(\d+.\d)-(\d+.\d)", event_between)
        event_start_timestamps.append(float(match.group(1)))
        event_end_timestamps.append(float(match.group(2)))
    return event_start_timestamps, event_end_timestamps


def single_session_representation(user, eye, session, annotation_filename, saved_file_name, format_option):
    data_dir = Path(single_mini_data_pattern.format(user_id=user, eye=eye, session=session))
    events_file = data_dir / "events.npz"
    insert_frame_dir = data_dir / inter_frame_dir

    result_dir = insert_frame_dir / (
            f"patch_n{args.patch_num}_s{args.patch_size}_pre_max{args.max_accum_frame_num}"
            + f"{'_with_overlap' if args.accum_with_overlap else ''}")
    with open(result_dir / annotation_filename) as annotation_file:
        annotation = json.load(annotation_file)

    saved_file = result_dir / saved_file_name
    
    save_to = serialization.Save2hdf5(saved_file, user=user, eye=eye, session=session)

    events = np.load(events_file)

    event_start_timestamps, event_end_timestamps = parse_volume_timestamps(annotation['data_list'])

    time_mask = np.logical_and(events["t"] >= event_start_timestamps[0],
                               events["t"] < event_end_timestamps[-1])

    events2event_count_v2(events["t"][time_mask], events["x"][time_mask],
                          events["y"][time_mask], events["p"][time_mask],
                          event_start_timestamps, event_end_timestamps,
                          frame_height=img_shape[0], frame_width=img_shape[1],
                          format_option=format_option, save_to=save_to)
    save_to.close()
    print("Saved events to {}".format(saved_file))


def separate_representation(user_list, eye_list, session_list, format_option, **kwargs):
    annotation_filename = f"blink_seg_exp{args.blink_seg_expand}_cont_frame_event_pre_accum_thr{args.event_num_threshold}_tracking_dataset.json"
    saved_filename = f"event_accum_thr{args.event_num_threshold}_{format_option}{'_b' + str(kwargs['B']) if format_option == 'voxel_grid' else ''}.hdf5"
    for user in user_list:
        for eye in eye_list:
            for session in session_list:
                single_session_representation(user, eye, session, annotation_filename, saved_filename, format_option)


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

    separate_representation(user_list, eye_list, session_list, event_format)
