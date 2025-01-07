import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from misc.ev_eye_dataset_utils import single_mini_data_pattern, continuous_ann_segments_path, session_col_idx, \
    parse_frame_filename, img_shape, print_stats
from test.test_utils import parse_pupil_ellipse
from misc.ev_eye_dataset_utils import ellipse_iou

from mmengine.fileio import load as load_ann

from configs._base_.data_split import train_user_list, val_user_list, test_user_list


def unpack_one_tracking_result(tracking_result):
    result_timestamp = []
    result_pupil = []
    result_method = []
    result_similarity = []
    for single_pred in tracking_result:
        pupil = single_pred['pupil']
        if pupil is None:
            pupil = [None] * 5
        result_method.append(single_pred['method'])
        result_pupil.append(pupil)
        if 'result_timestamp' in single_pred:
            result_timestamp.append(single_pred['result_timestamp'])
        elif 'accum_end_timestamp' in single_pred:
            result_timestamp.append(single_pred['accum_end_timestamp'])
        if single_pred['method'] == 'img_similarity':
            result_similarity.append(single_pred['similarity'])
    result_timestamp = np.array(result_timestamp)
    result_pupil = np.array(result_pupil)
    result_method = np.array(result_method)
    result_similarity = np.array(result_similarity)

    ev_result_mask = np.logical_or(result_method == 'ev_tracking_model',
                                   result_method == 'ev_accum_threshold')
    ev_result_timestamp = result_timestamp[ev_result_mask]
    ev_result_pupil = result_pupil[ev_result_mask]

    img_det_result_mask = result_method == 'img_detect'
    img_det_result_timestamp = result_timestamp[img_det_result_mask]
    img_det_result_pupil = result_pupil[img_det_result_mask]

    img_similarity_mask = result_method == 'img_similarity'
    img_similarity_timestamp = result_timestamp[img_similarity_mask]
    img_similarity_pupil = result_pupil[img_similarity_mask]

    return (result_timestamp, result_pupil, result_method, result_similarity,
            ev_result_timestamp, ev_result_pupil,
            img_det_result_timestamp, img_det_result_pupil,
            img_similarity_timestamp, img_similarity_pupil)


class EndToEndTrackingAnalyzer:
    def __init__(self, user_list, eye_list, session_list,
                 continuous_ann_track: bool,
                 annotation_filename,
                 model_cfg_name, result_filename):
        self.user_list = user_list
        self.eye_list = eye_list
        self.session_list = session_list
        self.continuous_ann_track = continuous_ann_track
        self.annotation_filename = annotation_filename
        self.model_cfg_name = model_cfg_name
        self.result_filename = result_filename

    def _fetch_results(self, user, eye, session):
        result_file_path = Path(single_mini_data_pattern.format(user_id=user, eye=eye,
                                                                session=session)) / "end_to_end_tracking" / self.model_cfg_name / self.result_filename
        if not result_file_path.exists():
            logging.log(logging.WARN, f"result_file_path {result_file_path} not exists")
            return None
        else:
            with open(result_file_path, "rb") as result_file:
                results = pickle.load(result_file)
            return results

    def _load_continuous_track_annotations(self, user, eye, session):
        continuous_ann_segments = pd.read_excel(continuous_ann_segments_path)
        user_segments = continuous_ann_segments.loc[user + 1]
        session_start_id = user_segments.iloc[session_col_idx[session][0]]
        session_end_id = user_segments.iloc[session_col_idx[session][1]]

        return self._load_target_annotation(user, eye, session, session_start_id, session_end_id)

    def _load_target_annotation(self, user, eye, session, start_id=None, end_id=None):
        assert (start_id is None and end_id is None) or (start_id is not None and end_id is not None)
        event_dir = Path(single_mini_data_pattern.format(user_id=user, eye=eye, session=session))
        annotation_file_path = event_dir / self.annotation_filename
        annotation = load_ann(annotation_file_path)

        frame_ids = []
        frame_timestamps = []
        gts = []
        for data in annotation["data_list"]:
            frame_id, frame_timestamp = parse_frame_filename(data["img_filename"], timestamp_offset=4000)
            if start_id is not None:
                if frame_id < start_id or frame_id > end_id:
                    continue
            frame_ids.append(frame_id)
            frame_timestamps.append(frame_timestamp)
            gts.append(parse_pupil_ellipse(data))

        return dict(frame_ids=frame_ids, frame_timestamps=frame_timestamps, gts=gts)

    def _one_session_analyze(self, user, eye, session):
        
        results = self._fetch_results(user, eye, session)
        if self.continuous_ann_track:
            target_gt = self._load_continuous_track_annotations(user, eye, session)
        else:
            target_gt = self._load_target_annotation(user, eye, session)

        if results is not None:
            iou_list = []
            f1_score_list = []
            dist_list = []
            img_similarity = []
            img_detect_num = 0
            img_similarity_num = 0
            ev_tracking_num = 0
            ev_threshold_num = 0

            if self.continuous_ann_track:
                assert len(results) == 1
                (result_timestamps, result_pupils, result_methods, result_similarities,
                 ev_result_timestamps, ev_result_pupils,
                 img_det_result_timestamps, img_det_result_pupils,
                 img_similarity_timestamps, img_similarity_pupils) = unpack_one_tracking_result(
                    results[0]['tracking_result'])
                img_similarity.append(result_similarities)
                img_detect_num += len(img_det_result_pupils)
                img_similarity_num += len(result_similarities)
                ev_tracking_num += np.sum(result_methods == "ev_tracking_model")
                ev_threshold_num += np.sum(result_methods == "ev_accum_threshold")
                for gt_id, gt_timestamp, gt_pupil in zip(target_gt["frame_ids"], target_gt["frame_timestamps"],
                                                         target_gt['gts']):
                    nearest_result_idx = np.argmin(np.abs(ev_result_timestamps - gt_timestamp))
                    ev_pupil = ev_result_pupils[nearest_result_idx].astype(np.float32)
                    if ev_pupil[0] is None:
                        logging.log(logging.WARNING, "No pupil detected")
                        continue
                    iou, f1_score = ellipse_iou(gt_pupil, ev_pupil, img_shape, with_f1_score=True)
                    if iou < 0.1:
                        logging.log(logging.WARNING,f"u{user} e {eye} s{session} {gt_id}_{gt_timestamp} {gt_pupil} with {ev_pupil}")
                    dist = np.sqrt(np.sum((np.square(gt_pupil[:2] - ev_pupil[:2]))))
                    iou_list.append(iou)
                    f1_score_list.append(f1_score)
                    dist_list.append(dist)
            else:
                for one_tracking_result in results:
                    (result_timestamps, result_pupils, result_methods, result_similarities,
                     ev_result_timestamps, ev_result_pupils,
                     img_det_result_timestamps, img_det_result_pupils,
                     img_similarity_timestamps, img_similarity_pupils) = unpack_one_tracking_result(
                        one_tracking_result['tracking_result'])
                    img_similarity.append(result_similarities)
                    img_detect_num += len(img_det_result_pupils)
                    img_similarity_num += len(result_similarities)
                    ev_tracking_num += np.sum(result_methods == "ev_tracking_model")
                    ev_threshold_num += np.sum(result_methods == "ev_accum_threshold")
                    nearest_gt_idx = np.argmin(np.abs(target_gt['frame_timestamps'] - ev_result_timestamps[-1]))
                    nearest_gt_pupil = target_gt['gts'][nearest_gt_idx]
                    iou, f1_score = ellipse_iou(nearest_gt_pupil, result_pupils[-1],
                                                img_shape, with_f1_score=True)
                    dist = np.sqrt(np.sum((np.square(nearest_gt_pupil[:2] - result_pupils[-1][:2]))))
                    iou_list.append(iou)
                    f1_score_list.append(f1_score)
                    dist_list.append(dist)

            return iou_list, f1_score_list, dist_list, img_similarity, img_detect_num, img_similarity_num, ev_tracking_num, ev_threshold_num
        else:
            logging.log(logging.WARNING,
                        f"results of user{user} and eye {eye} session {session} {self.result_filename} not found")
            return None

    def analyze(self):
        total_iou = []
        total_f1_score = []
        total_pixel_error = []
        total_img_det = 0
        total_ev_tracking = 0
        for u in self.user_list:
            user_iou = []
            user_f1_score = []
            user_pixel_error = []
            for e in self.eye_list:
                eye_iou = []
                eye_f1_score = []
                eye_pixel_error = []
                for s in self.session_list:
                    results = self._one_session_analyze(u, e, s)
                    if results is not None:
                        iou_list: np.ndarray = np.array(results[0])
                        f1_score_list: np.ndarray = np.array(results[1])
                        dist_list: np.ndarray = np.array(results[2])
                        img_similarity: np.array = np.concatenate(results[3])
                        img_detect_num = results[4]
                        img_similarity_num = results[5]
                        ev_tracking_num = results[6]
                        ev_threshold_num = results[7]

                        total_img_det += img_detect_num
                        total_ev_tracking += ev_tracking_num

                        print(f"---------------- stat of user{u} and eye {e} seesion {s} ---------------")
                        print_stats(iou_list, "IOU")
                        print_stats(f1_score_list, "f1_score")
                        print_stats(dist_list, "pixel error")
                        print_stats(img_similarity, "img_similarity")

                        print(
                            f"img_detect_num: {img_detect_num}, img_similarity_num: {img_similarity_num}, ev_tracking_num: {ev_tracking_num}, ev_threshold_num {ev_threshold_num}")
                        eye_iou.append(iou_list)
                        eye_f1_score.append(f1_score_list)
                        eye_pixel_error.append(dist_list)

                eye_iou = np.concatenate(eye_iou)
                eye_f1_score = np.concatenate(eye_f1_score)
                eye_pixel_error = np.concatenate(eye_pixel_error)

                print(f"************* stat of user{u} and eye {e} ********************")
                print_stats(eye_iou, "IOU")
                print_stats(eye_f1_score, "f1 score")
                print_stats(eye_pixel_error, "pixel error")

                user_iou.append(eye_iou)
                user_f1_score.append(eye_f1_score)
                user_pixel_error.append(eye_pixel_error)

            user_iou = np.concatenate(user_iou)
            user_f1_score = np.concatenate(user_f1_score)
            user_pixel_error = np.concatenate(user_pixel_error)

            print(f"++++++++++++++ stat of user{u} ++++++++++++++++++")
            print_stats(user_iou, "IOU")
            print_stats(user_f1_score, "f1 score")
            print_stats(user_pixel_error, "pixel error")

            total_iou.append(user_iou)
            total_f1_score.append(user_f1_score)
            total_pixel_error.append(user_pixel_error)

        total_iou = np.concatenate(total_iou)
        total_f1_score = np.concatenate(total_f1_score)
        total_pixel_error = np.concatenate(total_pixel_error)

        print(len(total_iou))
        print(np.sum((total_pixel_error >= 25)))
        filtered_iou = total_iou[total_pixel_error < 34]
        filtered_f1_score = total_f1_score[total_pixel_error < 34]
        filtered_pixel_error = total_pixel_error[total_pixel_error < 34]
        print("=================== total stats =============================")
        print_stats(total_iou, "IOU")
        print_stats(total_f1_score, "f1 score")
        print_stats(total_pixel_error, "pixel error")
        print_stats(filtered_iou,"filtered iou")
        print_stats(filtered_f1_score,"filtered f1 score")
        print_stats(filtered_pixel_error,"filtered pixel error")

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(f"total img det {total_img_det},total ev tracking {total_ev_tracking}")

        print(self.model_cfg_name)


# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument("--model_cfg_option", type=str, default="config-eye_crop_mbv3spreX_multi_anchor_det-s1_f2_n4_trans-pre_accum10_50_blink_exp5_0.5_rand")
#     # parser.add_argument("--result_filename", type=str, default="pre_accum_continuous_seg_track_with_0.8similarity.pickle")
#     parser.add_argument("--continuous_ann_track", action="store_true")

#     args = parser.parse_args()
#     return args


if __name__ == '__main__':
    user_list = [48]
    eye_list = ["left"]
    session_list = ['201']
    continuous_ann_track = True
    # continuous_ann_track = False
    # annotation_filename = "origin_labelled_pupil_dataset.json"
    annotation_filename = "origin_landmark_detection_dataset_ann.json"

    model_cfg_name = "config-eye_crop_mbv3spreX_multi_anchor_det-s1_f2_n4_trans-pre_accum10_50_blink_exp5_0.5_rand"

    result_filename = "pre_accum_continuous_seg_track_with_0.8similarity.pickle"

    analyzer = EndToEndTrackingAnalyzer(user_list, eye_list, session_list,
                                        continuous_ann_track, annotation_filename,
                                        model_cfg_name, result_filename)
    analyzer.analyze()
