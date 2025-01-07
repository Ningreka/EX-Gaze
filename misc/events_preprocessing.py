import numpy

import numpy as np
from pathlib import Path

from misc import ev_eye_dataset_utils

"""
convert data to npz and filter bad point in ev-eye dataset
"""

def load_from_txt(events_file, array_format={'t': 0, 'x': 1, 'y': 2, 'p': 3}, filter_points=None):
    events = np.loadtxt(events_file)
    events_mask = np.ones(len(events), dtype=bool)
    if filter_points is not None:
        for point in filter_points:
            point_mask = np.logical_and(events[:, array_format['x']] == point[0],
                                        events[:, array_format['y']] == point[1])
            events_mask = np.logical_and(events_mask, np.logical_not(point_mask))

    return events[events_mask, array_format['t']], events[events_mask, array_format['x']], events[events_mask, array_format['y']], \
        events[events_mask, array_format['p']]


def save2npz(saved_file, t, x, y, p):
    np.savez_compressed(saved_file, t=np.asarray(t, dtype=np.int64), x=np.asarray(x, dtype=np.int16),
                        y=np.asarray(y, dtype=np.int16), p=np.asarray(p, dtype=np.int8))


def check_bad_point(event_stream, points):
    points_num = {}
    for point in points:
        point_mask = np.logical_and(event_stream[1] == point[0], event_stream[2] == point[1])
        points_num[f"[{point[0]},{point[1]}]"] = np.sum(point_mask)

    return points_num


# if __name__ == '__main__':
#     filter_point = [[158, 27], [324, 27]]  # bad points in ev-eye dataset left camera
#     for u in range(40, 49):
#         for e in ["left", "right"]:
#         # for e in ["left"]:
#             for s in ["101", "102", "201", "202"]:
#             # for s in ["101"]:
#                 events_file = Path(
#                     ev_eye_dataset_utils.origin_single_data_pattern.format(user_id=u, eye=e, session=s)) / "events.txt"
#                 saved_file = Path(
#                     ev_eye_dataset_utils.single_mini_data_pattern.format(user_id=u, eye=e, session=s)) / "events.npz"
#                 if e == "left":
#                     events = load_from_txt(events_file, dict(zip(['t', 'x', 'y', 'p'], [0, 1, 2, 3])),filter_points=filter_point)
#                 else:
#                     events = load_from_txt(events_file, dict(zip(['t', 'x', 'y', 'p'], [0, 1, 2, 3])),
#                                            filter_points=None)
#                 # points_num = check_bad_point(events,filter_point)
#                 # print(f"u{u}_e{e}_s{s} points: {points_num}")

#                 save2npz(saved_file, events[0], events[1], events[2], events[3])
#                 print(str(saved_file))
