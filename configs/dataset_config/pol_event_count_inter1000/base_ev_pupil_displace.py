from mmengine.config import read_base

with read_base():
    from configs.dataset_config.zhu_b4_inter500.base_ev_pupil_displace import train_dataloader, val_dataloader, \
        test_dataloader

inter_ann_file_name = "inter-1000/open_90_eye_tracking_dataset.json"
event_representation_filename = "inter-1000/pol_event_count.hdf5"

train_dataloader["dataset"].update(
    ann_file_name=inter_ann_file_name,
    event_representation_filename=event_representation_filename,
    filter_dist=0.2,
    filter_prob=0.1
)

val_dataloader["dataset"].update(
    ann_file_name=inter_ann_file_name,
    event_representation_filename=event_representation_filename
)

test_dataloader["dataset"].update(
    ann_file_name=inter_ann_file_name,
    event_representation_filename=event_representation_filename
)
