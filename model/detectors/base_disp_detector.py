import abc
from typing import Union

import torch
from mmdet.models import BaseDetector
from mmdet.models.detectors.base import ForwardResults
from mmdet.structures import OptSampleList, SampleList


class BaseDispDetector(BaseDetector):

    def forward(self,
                batch_inputs: torch.Tensor,
                pre_state: torch.Tensor = None,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(batch_inputs=batch_inputs, pre_state=pre_state, batch_data_samples=data_samples)
        elif mode == 'predict':
            return self.predict(batch_inputs=batch_inputs, pre_state=pre_state, batch_data_samples=data_samples)
        elif mode == 'tensor':
            return self._forward(batch_inputs=batch_inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, batch_inputs: torch.Tensor, pre_state: torch.Tensor, batch_data_samples: SampleList) -> Union[
        dict, tuple]:
        feat = self.extract_feat(batch_inputs)
        return self.bbox_head.loss(feat, pre_state, batch_data_samples)

    def predict(self, batch_inputs: torch.Tensor, pre_state: torch.Tensor,
                batch_data_samples: SampleList) -> SampleList:
        feat = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(feat, pre_state, batch_data_samples)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: torch.Tensor):
        feat = self.extract_feat(batch_inputs)
        return self.bbox_head(feat)

    @abc.abstractmethod
    def extract_feat(self, batch_inputs: torch.Tensor):
        pass
