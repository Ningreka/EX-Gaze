model_cfg_options = {
    "config-eye_crop_mbv3spreX_multi_anchor_det-s1_f2_n4_trans-pre_accum10_50_blink_exp5_0.5_rand": dict(
        img_detect_cfg="configs/train_config/full_eye_pupil_detector/mbv3spreX_head_retina_img_pupil_det_eye_region_crop.py",
        img_detect_checkpoint="/path/to/model_weight/frame_based_model.pth",  # TODO require absolute path
        input_name="input_volume",
        is_single_obj_detect_model=False, is_eye_region_crop_resize=True, frame_target_size=[256, 160],
        ev_tracking_cfg="configs/train_config/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre0.5.py",
        event_tracking_checkpoint="/path/to/model_weight/event_based_model.pth", # TODO require absolute path
        event_accum_num_threshold=50, ev_accum_time=500,  
        max_accum_frame_num=10, accum_with_overlap=True, event_format="pol_event_count",
        patch_num=8, patch_size=16,
    ),
}
