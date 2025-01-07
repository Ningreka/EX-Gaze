from mmengine.config import read_base

inter_ann_file_name = "inter-500/open_90_eye_tracking_dataset.json"
event_representation_filename = "inter-500/voxel_grid_b4.hdf5"

with read_base():
    from configs.dataset_config.zhu_b8_u1_3.base_ev_pupil_displace import train_dataloader, val_dataloader, \
        test_dataloader

train_dataloader.update(
    batch_size=512,
    num_workers=8
)
train_dataloader["dataset"].update(
    user_list=range(1, 41),
    ann_file_name=inter_ann_file_name,
    event_representation_filename=event_representation_filename,
    filter_dist=0.2,
    filter_prob=0.1
)
train_dataloader["dataset"]["dataset_pipeline"][2].update(distance_as_weight=False)

val_dataloader.update(
    batch_size=512, num_workers=8
)
val_dataloader["dataset"].update(
    user_list=range(41, 44),
    ann_file_name=inter_ann_file_name,
    event_representation_filename=event_representation_filename
)

test_dataloader.update(
    batch_size=512, num_workers=8
)
test_dataloader["dataset"].update(
    user_list=range(44, 49),
    ann_file_name=inter_ann_file_name,
    event_representation_filename=event_representation_filename
)
