from typing import Union, Callable

import torch
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORM
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import Registry
from torch import nn

ev_eye_scope = "EV_EYE"

EV_MODELS = Registry('model', parent=MMENGINE_MODELS, scope=ev_eye_scope, locations=['model'])

EV_MODELS.register_module('Relu', module=torch.nn.ReLU)
EV_MODELS.register_module('Hardswish', module=torch.nn.Hardswish)

EV_DATASETS = Registry('dataset', parent=MMENGINE_DATASETS, scope=ev_eye_scope, locations=['dataset'])

EV_METRICS = Registry('metric', parent=MMENGINE_METRICS, scope=ev_eye_scope, locations=['metrics'])

EV_TRANSFORMS = Registry("transform", parent=MMENGINE_TRANSFORM, scope=ev_eye_scope,
                         locations=['dataset/transforms'])

EV_TASK_UTILS = Registry("task util", parent=MMENGINE_TASK_UTILS, scope=ev_eye_scope,
                         locations=['model'])

OptionLayerType = Union[Callable[..., nn.Module], str]