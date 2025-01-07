from mmengine.config import read_base

with read_base():
    from configs._base_.data_split import train_user_list, val_user_list, test_user_list
    from configs.dataset_config.pol_event_count_inter1000.multi_interval_ev_pupil_disp import train_dataloader, \
        val_dataloader, test_dataloader

inter_accum_frame_ann_file_name = "inter-2000_frame_endtime/blink_seg_exp20_patch_n8_s16_ev_overlap_accum30_continuous10frame/continuous_frame_event_accum_tracking_dataset.json"
event_representation_filename = "inter-2000_frame_endtime/blink_seg_exp20_patch_n8_s16_ev_overlap_accum30_continuous10frame/pol_event_count.hdf5"

train_dataloader["dataset"].update(
    user_list=train_user_list,
    ann_file_name=inter_accum_frame_ann_file_name,
    event_representation_filename=event_representation_filename,
    filter_dist=0,
)

val_dataloader["dataset"].update(
    user_list=val_user_list,
    ann_file_name=inter_accum_frame_ann_file_name,
    event_representation_filename=event_representation_filename
)

test_dataloader["dataset"].update(
    user_list=test_user_list,
    ann_file_name=inter_accum_frame_ann_file_name,
    event_representation_filename=event_representation_filename
)
