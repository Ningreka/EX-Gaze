import numpy as np
from tqdm import tqdm

from misc.event_representations import serialization


def to_pol_event_count(sub_stream, frame_height, frame_width):
    """
    :return:
    """
    pos_mask = sub_stream[:, 2] == 1
    neg_mask = sub_stream[:, 2] != 1
    pos_x, pos_y = sub_stream[pos_mask, 0], sub_stream[pos_mask, 1]
    neg_x, neg_y = sub_stream[neg_mask, 0], sub_stream[neg_mask, 1]
    
    hist1, _, _ = np.histogram2d(x=pos_y, y=pos_x, bins=[frame_height, frame_width],
                                 range=[[0, frame_height], [0, frame_width]])
    hist2, _, _ = np.histogram2d(x=neg_y, y=neg_x, bins=[frame_height, frame_width],
                                 range=[[0, frame_height], [0, frame_width]])
    return np.stack((hist1, hist2))


def to_abs_event_count(sub_stream, frame_height, frame_width):
    """
    :return:
    """
    x, y = sub_stream[:, 0], sub_stream[:, 1]
    
    hist, _, _ = np.histogram2d(x=y, y=x, bins=[frame_height, frame_width],
                                range=[[0, frame_height], [0, frame_width]])
    return np.expand_dims(hist, 0)


def to_pol_event_sum(sub_stream, frame_height, frame_width):
    """
    :return:
    """
    pol_event_count = to_pol_event_count(sub_stream, frame_height, frame_width)
    return np.expand_dims(pol_event_count[0, :, :] - pol_event_count[1, :, :], 0)


def to_event_binary(sub_stream, frame_height, frame_width):
    event_count = to_abs_event_count(sub_stream, frame_height, frame_width)
    return (event_count > 0).astype(np.int8)


def get_format_function(format_option):
    assert format_option in ["pol_event_count", "abs_event_count", "pol_event_sum", "event_binary"]
    if format_option == "pol_event_count":
        format_function = to_pol_event_count
    elif format_option == "abs_event_count":
        format_function = to_abs_event_count
    elif format_option == "pol_event_sum":
        format_function = to_pol_event_sum
    elif format_option == "event_binary":
        format_function = to_event_binary
    else:
        raise Exception
    return format_function


def events2event_count(ts, xs, ys, ps, sample_times, frame_height: int = 128, frame_width: int = 128,
                       format_option="pol_event_count", save_to=serialization.Save2Memory()):
    format_function = get_format_function(format_option)

    events = np.stack((xs, ys, ps, ts), axis=1)
    for i in tqdm(range(1, len(sample_times))):
        begin_time = sample_times[i - 1]
        end_time = sample_times[i]
        mask = np.logical_and(ts >= begin_time, ts < end_time)
        sub_stream = events[mask, :]
        voxel = format_function(sub_stream, frame_height=frame_height, frame_width=frame_width)

        save_to.save(str(end_time), voxel)


def events2event_count_v2(ts, xs, ys, ps, event_start_timestamps, event_end_timestamps, frame_height: int = 128,
                          frame_width: int = 128, format_option="pol_event_count", save_to=serialization.Save2Memory()):
    format_function = get_format_function(format_option)

    events = np.stack((xs, ys, ps, ts), axis=1)
    for start_time, end_time in tqdm(zip(event_start_timestamps, event_end_timestamps)):
        voxel_key = str(start_time) + "-" + str(end_time)
        if save_to.is_exists(voxel_key):
            continue
        mask = np.logical_and(ts >= start_time, ts < end_time)
        sub_stream = events[mask, :]
        voxel = format_function(sub_stream, frame_height=frame_height, frame_width=frame_width)

        save_to.save(voxel_key, voxel)


def events2event_count_v3(ts, xs, ys, ps, event_start_timestamps, event_end_timestamps, frame_height: int = 128,
                          frame_width: int = 128, format_option="pol_event_count"):
    save_to = serialization.Save2Memory()
    events2event_count_v2(ts, xs, ys, ps, event_start_timestamps, event_end_timestamps, frame_height=frame_height,
                          frame_width=frame_width, format_option=format_option, save_to=save_to)
    return save_to
