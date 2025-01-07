from pathlib import Path
from typing import Sequence

from mmengine.dataset import BaseDataset
from mmengine.dataset.dataset_wrapper import ConcatDataset

from dataset.datasets import EvEyeDataset, EyeDetectionDataset
from dataset.eye_pupil_dataset import EyePupilDataset
from dataset.eye_region_dataset import EyeRegionDataset
from dataset.transforms.loading import LoadEventVolume

from misc.ev_eye_dataset_utils import single_mini_data_pattern, origin_single_data_pattern
from registry import EV_DATASETS


@EV_DATASETS.register_module()
class WrappedEvPupilDataset(ConcatDataset):
    """used to wrap multi session datasets"""

    def __init__(self, user_list, eye_list, session_list, dataset_pipeline, ann_file_name,
                 load_event_representation=True, event_representation_loader_cls=LoadEventVolume, events_filename=None,
                 event_representation_filename=None, B=4, dataset_cls=EvEyeDataset,
                 pass_list=None, **kwargs):
        assert dataset_cls in [EyeDetectionDataset, EvEyeDataset, EyeRegionDataset, EyePupilDataset]
        dataset_list = []

        for u in user_list:
            for e in eye_list:
                for s in session_list:
                    if pass_list is not None and f"{u}/{e}/{s}" in pass_list:
                        continue
                    ann_file = single_mini_data_pattern.format(user_id=u, eye=e, session=s) + f"/{ann_file_name}"
                    data_root = origin_single_data_pattern.format(user_id=u, eye=e, session=s) + "/frames"
                    if dataset_cls is EvEyeDataset:
                        if load_event_representation:
                            if Path(event_representation_filename).is_absolute():
                                event_volume_file = Path(event_representation_filename)
                            else:
                                event_volume_file = single_mini_data_pattern.format(
                                    user_id=u, eye=e, session=s) + f"/{event_representation_filename}"
                            dataset_pipeline[0] = dict(type=event_representation_loader_cls,
                                                       events_volume_file=event_volume_file)
                            dataset_list.append(dataset_cls(ann_file=ann_file,
                                                            data_root=data_root,
                                                            pipeline=dataset_pipeline, **kwargs))
                        else:
                            events_file = single_mini_data_pattern.format(
                                user_id=u, eye=e, session=s) + f"/{events_filename}"
                            dataset_pipeline[0] = dict(type=VoxelGridRepresentation, B=B)
                            dataset_list.append(dataset_cls(ann_file=ann_file,
                                                            data_root=data_root,
                                                            pipeline=dataset_pipeline,
                                                            events_file=events_file,
                                                            **kwargs))
                    else:
                        dataset_list.append(dataset_cls(ann_file=ann_file,
                                                        data_root=data_root,
                                                        pipeline=dataset_pipeline, **kwargs))

        super().__init__(dataset_list, ignore_keys=["user_id", "eye", "session", "eye_region"])
