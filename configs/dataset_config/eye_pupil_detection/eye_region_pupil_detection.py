import copy

from mmcv import LoadImageFromFile
from mmengine.dataset import DefaultSampler

from dataset.eye_pupil_dataset import EyePupilDataset
from dataset.transforms.loading import LoadEyeRegionPupilBBox
from dataset.transforms.formatting import PackEyeRegionPupilDetInputs
from dataset.wrapped_dataset import WrappedEvPupilDataset
from configs._base_.data_split import train_user_list, val_user_list, test_user_list

ann_filename = "origin_landmark_eye_region_pupil_dataset_ann.json"

data_pipeline = [
    dict(type=LoadImageFromFile, color_type="grayscale"),
    dict(type=LoadEyeRegionPupilBBox),
    dict(type=PackEyeRegionPupilDetInputs,
         input_keys=dict(input_volume="img"),
         meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(type=WrappedEvPupilDataset,
                 user_list=train_user_list,
                 eye_list=["left", "right"],
                 session_list=["101", "102", "201", "202"],
                 dataset_pipeline=data_pipeline, ann_file_name=ann_filename,
                 dataset_cls=EyePupilDataset
                 ))

val_dataloader = copy.deepcopy(train_dataloader)

val_dataloader.update(drop_last=False)
val_dataloader["dataset"].update(user_list=val_user_list, test_mode=True)

test_dataloader = copy.deepcopy(val_dataloader)
test_dataloader["dataset"].update(user_list=test_user_list, test_mode=True)
