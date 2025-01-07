"""
preprocess event num for each sample in event patches
"""
import shutil
from pathlib import Path
import argparse
import concurrent.futures

import h5py
import numpy as np
import pandas as pd

from misc.ev_eye_dataset_utils import mini_dataset_dir, single_mini_data_pattern
from misc.generate_event_threshold_dataset.check_valid_data import parse_patch_region


def process_one_session_pre_multi_interval_event_num(user, eye, session, args):
    event_representation_file = h5py.File(
        mini_dataset_dir / f"all_user_continuous_seg_ev_representation/{inter_frame_dir}_{event_format}.hdf5",
        mode="r")

    data_dir = Path(single_mini_data_pattern.format(user_id=user, eye=eye, session=session))
    insert_frame_dir = data_dir / inter_frame_dir

    frame_timestamps = np.loadtxt(insert_frame_dir / "timestamp.txt")
    ellipse_labels = pd.read_csv(insert_frame_dir / "ellipse_label.csv")

    result_dir = insert_frame_dir / (f"patch_n{args.patch_num}_s{args.patch_size}_pre_max{args.max_accum_frame_num}"
                                     + f"{'_with_overlap' if args.accum_with_overlap else ''}")
    interval_event_num = []
    total_frame_num = len(frame_timestamps)
    assert total_frame_num == len(ellipse_labels)
    
    interval_frame_list = []
    for i in range(total_frame_num - 1):
        
        prior_frame_timestamp = frame_timestamps[i]
        post_frame_timestamp = frame_timestamps[i + 1]

        interval_frame_list.append(dict(
            start_timestamp=prior_frame_timestamp,
            end_timestamp=post_frame_timestamp,
            interval_frame=np.array(
                event_representation_file[f"{user}/{eye}/{session}/{prior_frame_timestamp}-{post_frame_timestamp}"])
        ))
        if len(interval_frame_list) > args.max_accum_frame_num:
            interval_frame_list.pop(0)

        if np.isnan(ellipse_labels.loc[i].confident):
            continue

        sample_interval_event_num = dict(
            piror_frame_file=ellipse_labels.loc[i].filename,
            prior_frame_id=i, prior_frame_timestamp=prior_frame_timestamp,
            post_frame_file=ellipse_labels.loc[i + 1].filename,
            post_frame_id=i + 1, post_frame_timestamp=post_frame_timestamp,
            event_nums=[]
        )

        latest_ellipse = np.array(ellipse_labels.loc[i][1:6])
        
        patch_mask, _ = parse_patch_region(latest_ellipse, sample_rads, args.patch_size,
                                           with_overlap=args.accum_with_overlap)

        total_event_num = 0
        for window in range(args.max_accum_frame_num):
            if i - window < 0:
                break
            start_timestamp = frame_timestamps[i - window]
            end_timestamp = frame_timestamps[i + 1 - window]

            interval = interval_frame_list[-1 - window]
            if not (interval["start_timestamp"] == start_timestamp and interval["end_timestamp"] == end_timestamp):
                raise Exception(
                    f"start timestamp {start_timestamp} and end timestamp {end_timestamp} but interval_list is {interval_frame_list}")
            interval_event_count_frame = interval["interval_frame"]
            # interval_event_count_frame = np.array(
            #     event_representation_file[f"{user}/{eye}/{session}/{start_timestamp}-{end_timestamp}"])

            sum_event = np.sum(interval_event_count_frame * patch_mask)
            total_event_num += sum_event
            sample_interval_event_num["event_nums"].append(dict(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                event_num=sum_event
            ))

        sample_interval_event_num["total_event_num"] = total_event_num
        interval_event_num.append(sample_interval_event_num)

    if result_dir.exists():
        shutil.rmtree(result_dir)
        result_dir.mkdir()
    else:
        result_dir.mkdir()
    pd.to_pickle(interval_event_num, result_dir / "interval_event_num.pickle")
    print(f"save to {result_dir} with {total_frame_num} frames")


def check_sample_pre_multi_interval_event_num():
    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        future_list = []
        for user in user_list:
            for eye in eye_list:
                for session in session_list:
                    future = executor.submit(process_one_session_pre_multi_interval_event_num, user, eye, session, args)
                    print(f"submit {user}, {eye}, {session}")
                    future_list.append([user, eye, session, future])

        for u, e, s, future in future_list:
            future.result()

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--user_start", type=int, default=1)
    arg_parser.add_argument("--user_end", type=int, default=49)
    arg_parser.add_argument("--patch_num", type=int, default=8)
    arg_parser.add_argument("--patch_size", type=int, default=16)
    arg_parser.add_argument("--max_accum_frame_num", type=int, default=None)
    arg_parser.add_argument("--accum_with_overlap", action="store_true")

    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    base_rad = np.pi * 2 / args.patch_num
    sample_rads = base_rad * np.arange(args.patch_num)

    user_list = [48]
    eye_list = ["left"]
    session_list = ["102"]

    inter_frame_dir = "inter-2000_frame_endtime"
    event_format = "pol_event_count"

    check_sample_pre_multi_interval_event_num()
