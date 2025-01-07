from pathlib import Path

import torch
from mmengine.config import Config
import mmengine
from mmengine.runner import Runner
from mmdeploy.apis.onnx.passes.optimize_onnx import optimize_onnx
from test.test_utils import build_eval_model
import onnxoptimizer
import onnxsim
from thop import profile

from registry import *

def dummy_profile(model, dummy_input):
    macs, params = profile(model, inputs=(dummy_input,))

    print('macs = ' + str(macs / 1000 ** 2) + 'M')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

pupil_detector_cfg = Config.fromfile("configs/train_config/full_eye_pupil_detector/mbv3spreX_head_retina_img_pupil_det_eye_region_crop.py")
input_shape = (2, 1, 160, 256)
pupil_detector = build_eval_model(pupil_detector_cfg,"cuda","/path/to/model_weight/frame_based_model.pth") # TODO require absolute path
torch.onnx.export(pupil_detector, torch.randn(input_shape).to("cuda"),
                  f"deploy/models/full_eye_pupil_detector/mbv3spreX_head_retina_img_pupil_det_eye_region_crop_x2.onnx",
                  verbose=True, input_names=["input_volume"], output_names=["cls","reg"], opset_version=17)
dummy_profile(pupil_detector,torch.randn(input_shape).to("cuda"))

ev_tracking_cfg = Config.fromfile(
    "configs/train_config/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre0.5.py")
input_shape = (2, 8, 2, 16, 16) # b,p,c,h,w
ev_tracker= build_eval_model(ev_tracking_cfg,"cuda","/path/to/model_weight/event_based_model.pth") # TODO require absolute path

torch.onnx.export(ev_tracker, torch.randn(input_shape).to("cuda"),
                  f"deploy/models/efficient_trans/ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre0.5x2.onnx",
                  verbose=True, input_names=["patches"], output_names=["pred"], opset_version=17)
dummy_profile(ev_tracker,torch.randn(input_shape).to("cuda"))

