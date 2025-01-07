from typing import Union, List, Optional

import mmengine
import mmdet
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.detectors.base import BaseDetector,ForwardResults
from mmdet.utils import ConfigType, OptConfigType
from torch import Tensor, nn

from registry import EV_MODELS

# CHECKED
@EV_MODELS.register_module()
class BasePupilDetector(BaseDetector):
    """
    
        input->backbones->detector head->output
    input:
        1. event only
        2. event + pre state mask
    output:
       1. pupil ellipse parameter
       2. pupil confidence
    """

    def __init__(self,
                 stem: ConfigType,
                 backbone: ConfigType,
                 bbox_head: ConfigType,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone = EV_MODELS.build(backbone)
        stem.update(stem_channels=self.backbone.stem_channels)
        self.stem = EV_MODELS.build(stem)
        bbox_head.update(in_channels=self.backbone.output_channels)
        self.bbox_head = EV_MODELS.build(bbox_head)

    def forward(self,
                input_volume: Tensor, pre_mask: Optional[Tensor] = None,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(input_volume=input_volume, pre_mask=pre_mask, batch_data_samples=data_samples)
        elif mode == 'predict':
            return self.predict(input_volume=input_volume, pre_mask=pre_mask, batch_data_samples=data_samples)
        elif mode == 'tensor':
            return self._forward(input_volume=input_volume, pre_mask=pre_mask)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, input_volume: Tensor,
             batch_data_samples: SampleList,
             pre_mask: Optional[Tensor] = None) -> Union[dict, tuple]:
        feat = self.extract_feat(input_volume, pre_mask)
        return self.bbox_head.loss(feat, batch_data_samples)

    def predict(self, input_volume: Tensor,
                batch_data_samples: SampleList,
                pre_mask: Optional[Tensor] = None) -> SampleList:
        feat = self.extract_feat(input_volume, pre_mask)
        results_list = self.bbox_head.predict(
            feat, batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, input_volume: Tensor, pre_mask: Optional[Tensor] = None):
        feat = self.extract_feat(input_volume, pre_mask)
        return self.bbox_head(feat)

    def extract_feat(self, input_volume: Tensor, pre_mask: Optional[Tensor] = None):
        if self.stem.num_input == 1:
            stem_feature = self.stem(input_volume)
        else:
            stem_feature = self.stem(pre_mask, input_volume)
        return self.backbone(stem_feature)
