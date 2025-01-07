from typing import Optional, Union, List, Sequence
import math

import torch
import torchvision
import mmengine
from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine import ConfigDict
from mmengine.model.base_model import BaseDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.utils.misc import is_seq_of

from registry import EV_MODELS


@EV_MODELS.register_module()
class EvPupilDataPreprocessor(BaseDataPreprocessor):

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 boxtype2tensor: bool = True,
                 img_aug: bool = False,
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

        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.boxtype2tensor = boxtype2tensor
        if img_aug:
            self.img_augmentation = torchvision.transforms.RandomApply([
                torchvision.transforms.RandomChoice([
                    torchvision.transforms.GaussianBlur([5, 9]),
                    torchvision.transforms.RandomAdjustSharpness(2)
                ]),
                torchvision.transforms.RandomPosterize(6),
                torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.5,hue=0.5)
            ],p=0.7)
        self.img_aug = img_aug

    def pad_inputs(self, batch_inputs, norm_input, aug_input):
        # Process data with `pseudo_collate`.
        if is_seq_of(batch_inputs, torch.Tensor):
            # Pad and stack Tensor.
            assert batch_inputs[0].dim() == 3

            # Normalization.
            if aug_input:
                batch_inputs = [self.img_augmentation(_img) for _img in batch_inputs]
            if norm_input:
                _batch_inputs = []
                for _batch_input in batch_inputs:
                    # efficiency
                    _batch_input = _batch_input.float()
                    _batch_input = (_batch_input - self.mean) / self.std
                    _batch_inputs.append(_batch_input)
                batch_inputs = _batch_inputs

            padded_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                        self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(batch_inputs, torch.Tensor):
            assert batch_inputs.dim() == 4
            if aug_input:
                batch_inputs = self.img_augmentation(batch_inputs)
            if norm_input:
                batch_inputs = (batch_inputs - self.mean) / self.std
            h, w = batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            padded_inputs = torch.nn.functional.pad(batch_inputs, (0, pad_w, 0, pad_h),
                                                    'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(batch_inputs)}： {batch_inputs}')
        return padded_inputs

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = super().forward(data)

        _batch_input_volume = data['input_volume']
        batch_input_volume = self.pad_inputs(_batch_input_volume, self._enable_normalize & True,
                                             self.img_aug & training)
        data['input_volume'] = batch_input_volume

        _batch_pre_mask = data.setdefault('pre_mask', None)
        if _batch_pre_mask is not None:
            batch_pre_mask = self.pad_inputs(_batch_pre_mask, False, False)
            assert (batch_input_volume.shape[0] == batch_pre_mask.shape[0]
                    and batch_input_volume.shape[2:] == batch_pre_mask.shape[2:])
            data["pre_mask"] = batch_pre_mask
        _batch_pre_state = data.setdefault("pre_state", None)
        if _batch_pre_state is not None:
            if is_seq_of(_batch_pre_state, torch.Tensor):
                data["pre_state"] = torch.cat(_batch_pre_state)
        else:
            data.pop("pre_state")

        data_samples = data.setdefault('data_samples', None)

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(batch_input_volume.shape[2:])
            for data_sample in data_samples:
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': batch_input_shape,
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)  #
        return data
