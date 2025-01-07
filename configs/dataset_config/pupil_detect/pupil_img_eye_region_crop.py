import mmcv

from dataset.datasets import EyeDetectionDataset
from dataset.wrapped_dataset import WrappedEvPupilDataset

from dataset.transforms.formatting import PackEvPupilTrackingInputs
from dataset.transforms.loading import LoadTrackingTargetEllipse2BBox
from dataset.transforms.transform import ImgEyeRegionRandCrop

from mmcv import LoadImageFromFile
from mmdet.datasets.transforms.transforms import RandomFlip, RandomShift, FixShapeResize
from mmrotate.datasets.transforms.transforms import RandomRotate

from mmengine.dataset import pseudo_collate, default_collate, DefaultSampler

from configs._base_.data_split import train_user_list,test_user_list,val_user_list

orin_ann_file_name = "origin_landmark_detection_dataset_ann.json"
resize_width = 256
resize_height = 160

train_pipeline = [
    dict(type=LoadImageFromFile, color_type="grayscale"),
    dict(type=LoadTrackingTargetEllipse2BBox, target_gt=None, target_labels=[0]),
    dict(type=ImgEyeRegionRandCrop, max_width=280, max_height=180),
    dict(type=FixShapeResize, width=resize_width, height=resize_height, clip_object_border=False),
    # rotate
    # dict(type=RandomRotate, rect_obj_labels=[0]),
    # shift
    dict(type=RandomShift),
    # flip
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackEvPupilTrackingInputs,
         input_keys=dict(input_volume="img"),
         meta_keys=('ori_shape', 'img_shape', 'img_id', 'img_path', 'flip', 'flip_direction', 'scale', 'scale_factor'))
    # pack detect inputs
]

test_pipeline = [
    dict(type=LoadImageFromFile, color_type="grayscale"),
    dict(type=LoadTrackingTargetEllipse2BBox, target_gt=None, target_labels=[0]),
    dict(type=ImgEyeRegionRandCrop, max_width=resize_width, max_height=resize_height),
    dict(type=FixShapeResize, width=resize_width, height=resize_height, clip_object_border=False),
    dict(type=PackEvPupilTrackingInputs,
         input_keys=dict(input_volume="img"),
         meta_keys=('ori_shape', 'img_shape', 'img_id', 'img_path', 'scale', 'scale_factor'))
    # pack detect inputs
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    collate_fn=dict(type=pseudo_collate),
    sampler=dict(type=DefaultSampler, shuffle=True),
    drop_last=True,
    dataset=dict(type=WrappedEvPupilDataset,
                 user_list=train_user_list,
                #  eye_list=["left", "right"],
                 eye_list=["left"],
                #  session_list=["101", "102", "201", "202"],
                 session_list=["201"],
                 dataset_pipeline=train_pipeline, ann_file_name=orin_ann_file_name,
                 dataset_cls=EyeDetectionDataset, detect_classes=["pupil"]
                 )
)

val_dataloader = train_dataloader.copy()
test_dataloader = train_dataloader.copy()

val_dataloader.update(
    batch_size=16,
    sampler=dict(type=DefaultSampler, shuffle=False),
    drop_last=True,
    dataset=dict(type=WrappedEvPupilDataset, user_list=val_user_list,
                #  eye_list=["left", "right"],
                 eye_list=["left"],
                #  session_list=["101", "102", "201", "202"],
                 session_list=["201"],
                 dataset_pipeline=test_pipeline,
                 ann_file_name=orin_ann_file_name,
                 dataset_cls=EyeDetectionDataset, detect_classes=["pupil"])
)

test_dataloader.update(
    batch_size=16,
    sampler=dict(type=DefaultSampler, shuffle=False),
    drop_last=True,
    dataset=dict(type=WrappedEvPupilDataset, user_list=test_user_list,
                #  eye_list=["left", "right"],
                 eye_list=["left"],
                #  session_list=["101", "102", "201", "202"],
                 session_list=["201"],
                 dataset_pipeline=test_pipeline,
                 ann_file_name=orin_ann_file_name,
                 dataset_cls=EyeDetectionDataset, detect_classes=["pupil"])
)
