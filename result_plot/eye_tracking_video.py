import torch
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from misc.ev_eye_dataset_utils import origin_single_data_pattern, single_mini_data_pattern, parse_frame_filename, \
    img_shape
from misc.event_representations.event_count import to_pol_event_count


def parse_ex_gaze_result(ex_gaze_result):
    ex_img_result_list = []
    ex_event_result_list = []
    ex_img_timestamps = []
    ex_event_timestamps = []
    for item in ex_gaze_result[0]['tracking_result']:
        if item['method'] in ["img_detect", "img_similarity"]:
            ex_img_result_list.append(item)
            ex_img_timestamps.append(item['result_timestamp'])
        else:
            ex_event_result_list.append(item)
            ex_event_timestamps.append(item['accum_end_timestamp'])
    return ex_img_result_list, np.array(ex_img_timestamps), ex_event_result_list, np.array(ex_event_timestamps)

def ex_gaze_tracking(result, result_timestamps: np.ndarray, target_timestamp, frame_or_event_result=0):
    nearest_idx = np.argmin(np.abs(result_timestamps - target_timestamp))
    return result[nearest_idx]['pupil']

if __name__ == '__main__':
    # load data
    # user48_eye_left_sess_201
    # data segment 700-750 000700_1658487163817602.png-000750_1658487165817604

    sample_data_root = Path(origin_single_data_pattern.format(user_id=48, eye='left', session='201'))
    sample_result_root = Path(single_mini_data_pattern.format(user_id=48, eye='left', session='201'))
    img_list = []

    for img in (sample_data_root / "frames").glob("*.png"):
        frame_id = parse_frame_filename(img.name, 4000)[0]
        if 700 <= frame_id <= 750:
            img_list.append(img)

    img_list.sort(key=lambda x: parse_frame_filename(x.name, 4000)[0])
    img_timestamps = np.array([parse_frame_filename(item.name, 4000)[1] for item in img_list])
    start_timestamp = img_timestamps[0]
    end_timestamp = img_timestamps[-1]

    events = np.load(sample_result_root / "events.npz")
    time_mask = np.logical_and(events['t'] >= start_timestamp, events['t'] <= end_timestamp)
    events = np.column_stack(
        [events['x'][time_mask], events['y'][time_mask], events['p'][time_mask], events["t"][time_mask]])

    # load result ---------------
    # ex-gaze result
    ex_gaze_result = pd.read_pickle(
        sample_result_root / "end_to_end_tracking/config-eye_crop_mbv3spreX_multi_anchor_det-s1_f2_n4_trans-pre_accum10_50_blink_exp5_0.5_rand/pre_accum_continuous_seg_track_with_0.8similarity.pickle")
    ex_img_result_list, ex_img_timestamps, ex_event_result_list, ex_event_timestamps = parse_ex_gaze_result(
        ex_gaze_result)

    # create video
    frame_video_writer = cv2.VideoWriter("result_plot/frames_tracking_video.avi",
                                         cv2.VideoWriter.fourcc('I', '4', '2', '0'),
                                         25, (346, 260), isColor=True)
    events_video_writer = cv2.VideoWriter("result_plot/events_tracking_video.avi",
                                          cv2.VideoWriter.fourcc('I', '4', '2', '0'),
                                          25, (346, 260), isColor=True)
    timestamp = start_timestamp
    # frame interval 40,000
    interval = 4000

    event_frame_num = 0
    insert_frame = None

    while timestamp <= end_timestamp and timestamp + interval <= end_timestamp:
        events_mask = np.logical_and(events[:, 3] >= timestamp, events[:, 3] < (timestamp + interval))
        event_count = to_pol_event_count(events[events_mask, :], 260, 346)
        event_count = (event_count > 0) * 255

        temp_event_frame = np.zeros([260, 346, 3], dtype=np.uint8)
        temp_event_frame[:, :, 1] = event_count[0, :, :]
        temp_event_frame[:, :, 2] = event_count[1, :, :]

        ex_ellipse = ex_gaze_tracking(ex_event_result_list, ex_event_timestamps, timestamp + interval)
        cv2.ellipse(temp_event_frame, (int(ex_ellipse[0]), int(ex_ellipse[1])),
                    (int(ex_ellipse[2] / 2), int(ex_ellipse[3] / 2)),
                    ex_ellipse[4] / np.pi * 180, 0, 360, color=[255, 255,255])

        if event_frame_num % 10 == 0:
            insert_frame = cv2.imread(str(img_list[event_frame_num // 10]))
            ex_frame_ellipse = ex_gaze_tracking(ex_img_result_list, ex_img_timestamps, timestamp + interval)
            cv2.ellipse(insert_frame, (int(ex_frame_ellipse[0]), int(ex_frame_ellipse[1])),
                        (int(ex_frame_ellipse[2] / 2), int(ex_frame_ellipse[3] / 2)),
                        ex_frame_ellipse[4] / np.pi * 180, 0, 360, color=[255,255,255])

        timestamp += interval
        event_frame_num += 1

        events_video_writer.write(temp_event_frame)
        frame_video_writer.write(insert_frame)

    events_video_writer.release()
    frame_video_writer.release()