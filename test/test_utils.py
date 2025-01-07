from pathlib import Path
import torch
import numpy as np

import mmcv
from mmdet.structures import DetDataSample
from mmengine.runner import load_checkpoint

from misc.ev_eye_dataset_utils import img_shape
from misc.event_representations.event_count import to_abs_event_count, to_pol_event_count, to_pol_event_sum, \
    to_event_binary

import registry
from misc.generate_event_threshold_dataset.check_valid_data import parse_patch_region


def img_pupil_detect_with_multi_obj(model, img_path, img_shape):
    # load img
    img = torch.from_numpy(mmcv.imread(img_path, flag="grayscale"))
    img = img.unsqueeze(0)
    return img_pupil_detect_with_multi_obj_imgin(model,img,img_shape)

def img_pupil_detect_with_multi_obj_imgin(model, img, img_shape):
    img_data_sample = DetDataSample()
    img_data_sample.set_metainfo(dict(
        img_shape=img_shape,
        ori_shape=img_shape
    ))
    result = model.test_step(dict(
        inputs=[img],
        data_samples=[img_data_sample]
    ))[0].cpu()
    if len(result.pred_instances) != 0:
        bboxes = result.pred_instances.bboxes
        scores = result.pred_instances.scores
        idx = int(scores.argmax())
        return bboxes[idx], scores[idx]
    else:
        # no target
        return None

def parse_pupil_ellipse(data):
    for instance in data["instances"]:
        if instance["ellipse_label"] == "pupil":
            return np.array(instance["ellipse"])
    return None


def event_patch_accum(event_stream, accum_start_time, accum_end_time, patch_mask):
    sub_event_mask = (event_stream[:, 3] >= accum_start_time) & (event_stream[:, 3] < accum_end_time)
    sub_event_stream = event_stream[sub_event_mask, :]
    event_num_accum = 0
    for event in sub_event_stream:
        event_num_accum += patch_mask[event[1], event[0]]
    return event_num_accum


def ev_single_pred(model, event_stream, event_start_time, event_end_time, pre_state, event_format):
    mask = np.logical_and(event_stream[:, 3] >= event_start_time,
                          event_stream[:, 3] < event_end_time)
    sub_stream = event_stream[mask, :]
    if event_format == "abs_event_count":
        event_volume = to_abs_event_count(sub_stream, img_shape[0], img_shape[1])
    elif event_format == "pol_event_count":
        event_volume = to_pol_event_count(sub_stream, img_shape[0], img_shape[1])
    elif event_format == "pol_event_sum":
        event_volume = to_pol_event_sum(sub_stream, img_shape[0], img_shape[1])
    elif event_format == "event_binary":
        event_volume = to_event_binary(sub_stream, img_shape[0], img_shape[1])

    event_volume = event_volume.astype(dtype=np.float32)
    event_volume = torch.from_numpy(event_volume).unsqueeze(0)

    pre_state = torch.from_numpy(pre_state).unsqueeze(dim=0)
    result = model.test_step(dict(
        input_volume=event_volume,
        pre_state=pre_state,
        data_samples=[DetDataSample(metainfo=dict(ori_shape=img_shape))]
    ))
    ev_pred_ellipse = result[0].pred_instances["bboxes"].detach().cpu().numpy().squeeze()
    return ev_pred_ellipse


def ev_accum_tracking(model, pre_img_timestamp, cur_img_timestamp, insert_frame_num, tracking_frame_num, event_stream,
                      event_format, pre_state, max_accum_frame_num, event_num_threshold, sample_rads, patch_size,
                      with_overlap=False):
    total_interval_num = (insert_frame_num + 1) * tracking_frame_num
    frame_time_span = (cur_img_timestamp - pre_img_timestamp) / total_interval_num
    time_list = pre_img_timestamp + np.arange(0, total_interval_num + 1) * frame_time_span

    pred_ellipse_list = [pre_state]

    patch_mask, _ = parse_patch_region(pre_state, sample_rads, patch_size,
                                       with_overlap=with_overlap)
    sum_event_accum = 0
    interval_list = []
    for i in range(total_interval_num):
        accum_start_time = time_list[i]
        accum_end_time = time_list[i + 1]
        event_accum_num = event_patch_accum(event_stream, accum_start_time, accum_end_time, patch_mask)

        sum_event_accum += event_accum_num
        interval_list.append((accum_start_time, accum_end_time, event_accum_num))

        if sum_event_accum >= event_num_threshold:
 
            ev_pred_ellipse = ev_single_pred(model, event_stream, interval_list[0][0], interval_list[-1][1], pre_state,
                                             event_format)

            pred_ellipse_list.append(ev_pred_ellipse)
            pre_state = ev_pred_ellipse
            # update patch_mask
            patch_mask, patch_regions = parse_patch_region(pre_state, sample_rads, patch_size)
            if np.any(((patch_regions[:, 2] - patch_regions[:, 0]) < patch_size) | (
                    (patch_regions[:, 3] - patch_regions[:, 1]) < patch_size)):
                break  # out of range
            interval_list.clear()
            sum_event_accum = 0
        else:
            if max_accum_frame_num is not None and len(interval_list) == max_accum_frame_num:
                removed_events = interval_list.pop(0)
                sum_event_accum -= removed_events[2]

    return pred_ellipse_list

def build_eval_model(train_cfg, device, checkpoint=None):
    model = registry.EV_MODELS.build(train_cfg["model"])
    model.to(device=device)

    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(train_cfg["work_dir"]) / checkpoint
    else:
        with open(Path(train_cfg["work_dir"]) / "last_checkpoint") as last_checkpoint:
            checkpoint_path = last_checkpoint.readline()
    load_checkpoint(model, str(checkpoint_path), map_location=device)
    model.eval()
    return model