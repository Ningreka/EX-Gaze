from model.backbones.mobilenet import MobileNetBackbone

mobilenet_v3_small = dict(
    type=MobileNetBackbone,
    net_version="v3_small"
)