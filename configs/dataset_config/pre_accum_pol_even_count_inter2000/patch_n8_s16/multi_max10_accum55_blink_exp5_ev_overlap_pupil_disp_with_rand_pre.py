from mmengine.config import read_base

with read_base():
    from configs.dataset_config.pol_event_count_inter2000.patch_n8_s16.multi_max10_accum30_ev_overlap_pupil_disp_with_rand_pre import \
        train_dataloader, val_dataloader, test_dataloader

inter_accum_frame_ann_file_name = "inter-2000_frame_endtime/patch_n8_s16_pre_max10_with_overlap/blink_seg_exp5_cont_frame_event_pre_accum_thr55_tracking_dataset.json"
event_representation_filename = "inter-2000_frame_endtime/patch_n8_s16_pre_max10_with_overlap/event_accum_thr55_pol_event_count.hdf5"


train_dataloader["dataset"].update(
    ann_file_name=inter_accum_frame_ann_file_name,
    event_representation_filename=event_representation_filename,
)

val_dataloader["dataset"].update(
    ann_file_name=inter_accum_frame_ann_file_name,
    event_representation_filename=event_representation_filename,
)


test_dataloader["dataset"].update(
    ann_file_name=inter_accum_frame_ann_file_name,
    event_representation_filename=event_representation_filename,
)