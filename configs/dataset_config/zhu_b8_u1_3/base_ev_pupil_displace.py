from mmengine.config import read_base

from dataset.transforms.loading import LoadTrackingTargetEllipse2BBox
from dataset.transforms.loading import ParseDistance
from dataset.transforms.formatting import MapResultItems
from dataset.transforms.formatting import PackEvPupilTrackingInputs
from dataset.transforms.utils import DictPath
from dataset.wrapped_dataset import WrappedEvPupilDataset

with read_base():
    from .base_ev_pupil_detection import train_dataloader, val_dataloader, test_dataloader

pre_pupil_instance_path = dict(type=DictPath, key="pre_gt",
                               children=[dict(type=DictPath, key="instances", merge_children=False,
                                              children=[dict(type=DictPath, key=0, merge_children=False,
                                                             children=[dict(type=DictPath, key="ellipse")])])])

base_pipeline = [
    None,  # 用于load event volume的占位
    dict(type=ParseDistance, square_dist=True),
    dict(type=LoadTrackingTargetEllipse2BBox, target_gt="cur_gt", target_labels=[0], distance_as_weight=True),
    dict(type=MapResultItems, source=pre_pupil_instance_path,
         target=dict(type=DictPath, key="pre_state", unfold_value=True)),
    dict(type=MapResultItems, source=dict(type=DictPath, key="img_shape"), target=dict(type=DictPath, key="ori_shape")),
    dict(type=MapResultItems,
         source=dict(type=DictPath, key="cur_gt", children=[dict(type=DictPath, key="img_path")], merge_children=False),
         target=dict(type=DictPath, key="img_path", unfold_value=True)),
    dict(type=MapResultItems,
         source=dict(type=DictPath, key="cur_gt", children=[dict(type=DictPath, key="img_id")], merge_children=False),
         target=dict(type=DictPath, key="img_id", unfold_value=True)),
    dict(type=PackEvPupilTrackingInputs,
         input_keys=dict(input_volume="event_volume", pre_state="pre_state"),
         meta_keys=('ori_shape', 'img_id', 'img_path')
         )
]

train_dataloader["dataset"].update(dataset_pipeline=base_pipeline)

val_dataloader["dataset"].update(dataset_pipeline=base_pipeline)

test_dataloader["dataset"].update(dataset_pipeline=base_pipeline)