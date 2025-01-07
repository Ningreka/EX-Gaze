from typing import Optional, Sequence, Union

import torch
import numpy as np
from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine import is_seq_of
from mmengine.model import BaseDataPreprocessor, stack_batch

from registry import EV_MODELS
from misc.ev_eye_dataset_utils import ellipse_point_sample


@EV_MODELS.register_module()
class EvPupilPatchPreprocessor(BaseDataPreprocessor):

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 patch_size: int = 16,
                 patch_num: int = 8,
                 pad_value: Union[float, int] = 0,
                 empty_filter=False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False):
        super().__init__(non_blocking)
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None:
            assert len(mean) == len(std)
            self._enable_normalize = True
            self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        self.patch_size = patch_size
        self.patch_num = patch_num
        self.base_rad = np.pi * 2 / self.patch_num
        self.sample_rads = self.base_rad * np.arange(self.patch_num)
        self.pad_size = [patch_size // 2] * 4
        self.pad_value = pad_value
        self.boxtype2tensor = boxtype2tensor
        # self.empty_filter = empty_filter

    def ellipse_patchify(self, batch_inputs: Union[Sequence[torch.Tensor], torch.Tensor], batch_pre_state):
        assert len(batch_inputs) == len(batch_pre_state)
        _batch_pre_state = batch_pre_state.squeeze(1)
        if is_seq_of(batch_inputs, torch.Tensor):
            batch_inputs = torch.cat(batch_inputs)
        batch_inputs = torch.nn.functional.pad(batch_inputs, self.pad_size, mode="constant", value=self.pad_value)

        batch_patches = []
        batch_sample_regions = []
        for input_volume, pre_state in zip(batch_inputs, _batch_pre_state):
            assert input_volume.dim() == 3
            assert len(pre_state) == 5  # cx, cy, w, h, t
            ellipse_sample_point = ellipse_point_sample(pre_state.cpu().numpy(), self.sample_rads)
            sample_points = np.ceil(ellipse_sample_point).astype(np.int32) + self.patch_size // 2  # 由于pad
            sample_regions = np.concatenate([sample_points - np.array([self.patch_size // 2]),
                                             sample_points + np.ceil([self.patch_size / 2]).astype(np.int32)], axis=1)
            sample_regions = torch.from_numpy(sample_regions)
            # [x1,y1,x2,y2]
            patches = []
            for region in sample_regions:
                patches.append(input_volume[:, region[1]:region[3], region[0]:region[2]])  # C x H x W
            # P x C x H x W
            batch_sample_regions.append(sample_regions - self.patch_size // 2)
            batch_patches.append(torch.stack(patches))
        batch_sample_regions = torch.stack(batch_sample_regions)
        batch_patches = torch.stack(batch_patches)
        # if self.empty_filter:
        #     mask = batch_patches == 0
        #     batch_mask = mask.any(-1).any(-1).any(-1).any(-1)
        #     batch_patches = batch_patches[batch_mask]
        #     batch_sample_regions = batch_sample_regions[batch_mask]
        #     batch_pre_state = batch_pre_state[batch_mask]
        if self._enable_normalize:
            batch_patches = (batch_patches - self.mean) / self.std
        return batch_patches, batch_sample_regions, batch_pre_state

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = super().forward(data)

        _batch_input_volume = data.pop('input_volume')

        _batch_pre_state = data["pre_state"]
        if is_seq_of(_batch_pre_state, torch.Tensor):
            _batch_pre_state = torch.cat(_batch_pre_state)
            data["pre_state"] = _batch_pre_state

        data['batch_inputs'], _batch_sample_regions, batch_pre_state \
            = self.ellipse_patchify(_batch_input_volume, _batch_pre_state)
        data["pre_state"] = batch_pre_state

        data_samples = data.setdefault('data_samples', None)

        if data_samples is not None:

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            for i, sample_regions in enumerate(_batch_sample_regions):
                data_samples[i].sample_regions = sample_regions

        return data
