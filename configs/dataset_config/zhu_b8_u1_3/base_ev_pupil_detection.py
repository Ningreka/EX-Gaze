import torch.utils.data

from dataset.datasets import EvEyeDataset
from dataset.transforms.loading import LoadTrackingTargetEllipse2BBox
from dataset.transforms.loading import ParseDistance
from dataset.transforms.formatting import MapResultItems
from dataset.transforms.formatting import PackEvPupilTrackingInputs
from dataset.transforms.utils import DictPath
from dataset.wrapped_dataset import WrappedEvPupilDataset

from mmengine.dataset import pseudo_collate, default_collate, DefaultSampler

inter_ann_file_name = "inter-origin/open_90_eye_tracking_dataset.json"
orin_ann_file_name = "origin_landmark_tracking_dataset_ann.json"
event_representation_filename = "zhu_voxel_grid_b8.hdf5"

base_pipeline = [
    None, # place holder
    dict(type=ParseDistance, square_dist=True),
    dict(type=LoadTrackingTargetEllipse2BBox, target_gt="cur_gt", target_labels=[0], distance_as_weight=True),
    dict(type=MapResultItems, source=dict(type=DictPath, key="img_shape"), target=dict(type=DictPath, key="ori_shape")),
    dict(type=MapResultItems,
         source=dict(type=DictPath, key="cur_gt", children=[dict(type=DictPath, key="img_path")], merge_children=False),
         target=dict(type=DictPath, key="img_path", unfold_value=True)),
    dict(type=MapResultItems,
         source=dict(type=DictPath, key="cur_gt", children=[dict(type=DictPath, key="img_id")], merge_children=False),
         target=dict(type=DictPath, key="img_id", unfold_value=True)),
    dict(type=PackEvPupilTrackingInputs,
         input_keys=dict(input_volume="event_volume"),
         meta_keys=('ori_shape', 'img_id', 'img_path')
         )
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    collate_fn=dict(type=default_collate),
    sampler=dict(type=DefaultSampler, shuffle=True),
    drop_last=True,
    dataset=dict(type=WrappedEvPupilDataset,
                 user_list=range(1, 3),
                 eye_list=["left", "right"],
                 session_list=["101", "102", "201", "202"],
                 dataset_pipeline=base_pipeline, ann_file_name=inter_ann_file_name,
                 dataset_cls=EvEyeDataset, tracking_classes=["pupil"],
                 load_event_representation=True,
                 event_representation_filename=event_representation_filename)
)

val_dataloader = train_dataloader.copy()
test_dataloader = train_dataloader.copy()

val_dataloader.update(
    batch_size=64,
    sampler=dict(type=DefaultSampler, shuffle=False),
    drop_last=True)
val_dataloader["dataset"].update(
    type=WrappedEvPupilDataset, user_list=range(3, 6), eye_list=["left", "right"],
    session_list=["101", "102", "201", "202"],
    dataset_pipeline=base_pipeline, ann_file_name=orin_ann_file_name,
    event_representation_filename=event_representation_filename
)

test_dataloader.update(
    batch_size=64,
    sampler=dict(type=DefaultSampler, shuffle=False),
    drop_last=True)
test_dataloader["dataset"].update(
    type=WrappedEvPupilDataset, user_list=range(6, 12), eye_list=["left", "right"],
    session_list=["101", "102", "201", "202"],
    dataset_pipeline=base_pipeline, ann_file_name=orin_ann_file_name,
    event_representation_filename=event_representation_filename
)
