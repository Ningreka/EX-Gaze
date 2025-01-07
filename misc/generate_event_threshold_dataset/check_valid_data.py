import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd

from misc.ev_eye_dataset_utils import single_mini_data_pattern, ellipse_point_sample, img_shape

def parse_patch_region(pre_state, sample_rads, patch_size, with_overlap=False):
    ellipse_sample_point = ellipse_point_sample(pre_state, sample_rads)
    sample_points = np.ceil(ellipse_sample_point).astype(np.int32)
    sample_regions = np.concatenate([np.clip(sample_points - np.array([patch_size // 2]), 0, None),
                                     sample_points + np.ceil([patch_size / 2]).astype(np.int32)],
                                    axis=1)  # [x0,y0,x1,y1]
    sample_regions[:, [0, 2]] = np.clip(sample_regions[:, [0, 2]], 0, img_shape[1])
    sample_regions[:, [1, 3]] = np.clip(sample_regions[:, [1, 3]], 0, img_shape[0])

    sample_region_mask = np.zeros(img_shape, dtype=int)
    for region in sample_regions:
        if with_overlap:
            sample_region_mask[region[1]:region[3], region[0]:region[2]] += 1
        else:
            sample_region_mask[region[1]:region[3], region[0]:region[2]] = 1

    return sample_region_mask, sample_regions


def check_wo_density(event_stream, frame_timestamps, ellipse_labels, start_index, confident_threshold,
                     event_num_threshold, patch_mask, blink_segments=None):
    total_frame_num = len(frame_timestamps)
    pre_timestamp = frame_timestamps[start_index]
    event_num_accum = 0
    for i in range(start_index + 1, total_frame_num):
        if is_blink(confident_threshold, ellipse_labels.loc[i].confident, frame_timestamps[i], blink_segments):
            return None

        cur_timestamp = frame_timestamps[i]
        sub_event_mask = (event_stream[:, 3] >= pre_timestamp) & (event_stream[:, 3] < cur_timestamp)
        sub_event_stream = event_stream[sub_event_mask, :]
        for event in sub_event_stream:
            event_num_accum += patch_mask[event[1], event[0]]

        if event_num_accum >= event_num_threshold:
            return cur_timestamp, event_num_accum

        pre_timestamp = cur_timestamp
    return None


def check_with_density(event_stream, frame_timestamps, ellipse_labels, start_index, max_accum_frame_num,
                       confident_threshold, event_num_threshold, patch_mask, blink_segments=None):
    total_frame_num = len(frame_timestamps)
    pre_timestamp = frame_timestamps[start_index]
    event_index = np.sum(event_stream[:,3] < pre_timestamp)

    interval_list = []
    sum_event_accum = 0
    for i in range(start_index + 1, total_frame_num):
        if is_blink(confident_threshold, ellipse_labels.loc[i].confident, frame_timestamps[i], blink_segments):
            return None

        cur_timestamp = frame_timestamps[i]

        event_num_accum = 0
        while event_index < len(event_stream) and event_stream[event_index, 3] < cur_timestamp:
            event = event_stream[event_index, :]
            if event[3] >= pre_timestamp:
                event_num_accum += patch_mask[event[1], event[0]]
            event_index += 1

        interval_list.append((pre_timestamp, cur_timestamp, event_num_accum))
        sum_event_accum += event_num_accum
        if sum_event_accum >= event_num_threshold:
            return interval_list[0][0], interval_list[-1][1], sum_event_accum
        else:
            if len(interval_list) == max_accum_frame_num:
                removed_events = interval_list.pop(0)
                sum_event_accum -= removed_events[2]

        pre_timestamp = cur_timestamp

    return None


def is_blink(blink_conf_thr, confident, time_point, blink_segments):
    if blink_segments is not None:
        for segment in blink_segments:
            if segment[0] <= time_point <= segment[1]:
                return True
        return False
    else:
        return np.isnan(confident) or confident < blink_conf_thr


def check_valid_data(event_stream, frame_timestamps, ellipse_labels, sample_rads, patch_size,
                     confident_threshold, event_num_threshold, blink_segments=None, max_accum_frame_num=None,
                     accum_with_overlap=False):
    total_frame_num = len(frame_timestamps)
    
    valid_timestamp_pair = []
    for i in range(total_frame_num - 1):
        if is_blink(confident_threshold, ellipse_labels.loc[i].confident, frame_timestamps[i], blink_segments):
            continue

        start_ellipse = np.array(ellipse_labels.loc[i][1:6])
        patch_mask, _ = parse_patch_region(start_ellipse, sample_rads, patch_size, with_overlap=accum_with_overlap)
        if max_accum_frame_num is None:
            one_take = check_wo_density(event_stream, frame_timestamps, ellipse_labels, i, confident_threshold,
                                        event_num_threshold, patch_mask, blink_segments=blink_segments)
        else:
            one_take = check_with_density(event_stream, frame_timestamps, ellipse_labels, i, max_accum_frame_num,
                                          confident_threshold, event_num_threshold, patch_mask,
                                          blink_segments=blink_segments)
        if one_take is not None:
            valid_timestamp_pair.append((i, frame_timestamps[i], *one_take))

    valid_timestamp_pair = np.array(valid_timestamp_pair)
    if max_accum_frame_num is None:
        if len(valid_timestamp_pair) == 0:
            data = dict(start_frame_file=[], start_timestamp=[], end_timestamp=[], event_num_accum=[])
        else:
            data = dict(
                start_frame_file=ellipse_labels.filename[valid_timestamp_pair[:, 0]],
                start_timestamp=valid_timestamp_pair[:, 1],
                end_timestamp=valid_timestamp_pair[:, 2],
                event_num_accum=valid_timestamp_pair[:, 3].astype(int)
            )
    else:
        if len(valid_timestamp_pair) == 0:
            data = dict(start_frame_file=[], iter_start_timestamp=[], event_start_timestamp=[], event_end_timestamp=[],
                        event_num_accum=[])
        else:
            data = dict(
                start_frame_file=ellipse_labels.filename[valid_timestamp_pair[:, 0]],
                iter_start_timestamp=valid_timestamp_pair[:, 1],
                event_start_timestamp=valid_timestamp_pair[:, 2],
                event_end_timestamp=valid_timestamp_pair[:, 3],
                event_num_accum=valid_timestamp_pair[:, 4].astype(int)
            )
    timestamp_pair_csv = pd.DataFrame(data, index=valid_timestamp_pair[:, 0].astype(int))
    return timestamp_pair_csv


def main(user, eye, session):
    print(f"time {time.asctime(time.localtime())} process user{user} and eye{eye} session{session}")
    data_dir = Path(single_mini_data_pattern.format(user_id=user, eye=eye, session=session))
    insert_frame_dir = data_dir / f"inter-{args.frame_rate}_frame_endtime"

    events_list = np.load(data_dir / "events.npz")
    frame_timestamps = np.loadtxt(insert_frame_dir / "timestamp.txt")
    ellipse_labels = pd.read_csv(insert_frame_dir / "ellipse_label.csv")

    segments = None
    if args.use_blink_segments:
        segments_csv = pd.read_csv(data_dir / "frames_segment/blink_segments.csv")
        segments = np.column_stack(
            [segments_csv.blink_start_timestamp, segments_csv.blink_end_timestamp])
        segments[:, 0] -= (args.blink_seg_expand * 1000)
        segments[:, 1] += (args.blink_seg_expand * 1000)

    assert len(frame_timestamps) == len(ellipse_labels)

    events_mask = (events_list["t"] >= frame_timestamps[0]) & (events_list["t"] <= frame_timestamps[-1])
    event_stream = np.stack((events_list["x"][events_mask],
                             events_list["y"][events_mask],
                             events_list["p"][events_mask],
                             events_list["t"][events_mask]), axis=1)

    timestamp_pair_csv = check_valid_data(event_stream, frame_timestamps, ellipse_labels, sample_rads,
                                          args.patch_size, args.confident_threshold,
                                          args.event_num_threshold, blink_segments=segments,
                                          max_accum_frame_num=args.max_accum_frame_num,
                                          accum_with_overlap=args.accum_with_overlap)

    if segments is not None:
        save_to_dir = (f"blink_seg_exp{args.blink_seg_expand}_patch_n{args.patch_num}_s{args.patch_size}"
                       + f"_ev_{'overlap_' if args.accum_with_overlap else ''}accum{args.event_num_threshold}")
    else:
        save_to_dir = f"open{args.confident_threshold}_patch_n{args.patch_num}_s{args.patch_size}_ev_accum{args.event_num_threshold}"
    if args.max_accum_frame_num:
        save_to_dir = save_to_dir + f"_continuous{args.max_accum_frame_num}frame"
    save_to_csv = insert_frame_dir / save_to_dir / "valid_timestamp_pair.csv"
    
    save_to_csv.parent.mkdir(exist_ok=True)
    timestamp_pair_csv.to_csv(save_to_csv)
    print(f"time {time.asctime(time.localtime())} save to {save_to_csv}")

if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--user_start", type=int, default=1)
    arg_parser.add_argument("--user_end", type=int, default=49)
    arg_parser.add_argument("--frame_rate", type=int, default=1000)
    arg_parser.add_argument("--patch_num", type=int, default=8)
    arg_parser.add_argument("--patch_size", type=int, default=16)
    arg_parser.add_argument("--max_accum_frame_num", type=int, default=None)
    arg_parser.add_argument("--confident_threshold", type=float, default=0.9)
    arg_parser.add_argument("--event_num_threshold", type=int, default=40)
    arg_parser.add_argument("--accum_with_overlap", action="store_true")
    arg_parser.add_argument("--use_blink_segments", action="store_true")
    arg_parser.add_argument("--blink_seg_expand", type=int, default=20, help="expand blink segments in ms")

    args = arg_parser.parse_args()

    base_rad = np.pi * 2 / args.patch_num
    sample_rads = base_rad * np.arange(args.patch_num)

    for user in range(args.user_start, args.user_end):
        # for user in range(5, 6):
        for eye in ("left", "right"):
            # for eye in ["right"]:
            for session in ["101", "102", "201", "202"]:
                # for session in ["102"]:
                main(user, eye, session)

