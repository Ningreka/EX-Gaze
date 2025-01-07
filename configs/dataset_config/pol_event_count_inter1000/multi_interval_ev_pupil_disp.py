from mmengine.config import read_base

with read_base():
    from configs.dataset_config.pol_event_count_inter1000.base_ev_pupil_displace import train_dataloader, val_dataloader, \
        test_dataloader

inter_accum_frame_ann_file_name = "inter-1000/open0.9_patch_n8_s16_ev_accum40/continuous_frame_event_accum_tracking_dataset.json"
event_representation_filename = "inter-1000/open0.9_patch_n8_s16_ev_accum40/pol_event_count.hdf5"

train_dataloader["dataset"].update(
    ann_file_name=inter_accum_frame_ann_file_name,
    event_representation_filename=event_representation_filename,
    filter_dist=0.2,
    filter_prob=0.2
)

val_dataloader["dataset"].update(
    ann_file_name=inter_accum_frame_ann_file_name,
    event_representation_filename=event_representation_filename
)

test_dataloader["dataset"].update(
    ann_file_name=inter_accum_frame_ann_file_name,
    event_representation_filename=event_representation_filename
)
