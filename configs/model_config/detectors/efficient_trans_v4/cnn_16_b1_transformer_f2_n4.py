from mmengine.config import read_base

from model.detectors.efficient_trans_vit_v4 import EfficientTransVit

with read_base():
    from configs.model_config.backbones.efficient_v4_transformer_f2_n4 import efficient_v4_transformer_encoder
    from configs.model_config.heads.displace_heads.trans_transformer_head import trans_transformer_head
    from configs.model_config.data_preprocessors.ev_pupil_patch_preprocessor import ev_pupil_patch_preprocessor

trans_transformer_head.update(
    hidden_dim=64
)
ev_pupil_patch_preprocessor.update(
    mean=[0.0008481635243511221, 0.0007458575739646381],
    std=[0.029362320982337497, 0.02762292878853001],
    patch_size=16
)

efficient_trans_vit = dict(
    type=EfficientTransVit,
    patch_num=8,
    patch_size=16,
    in_channels=2,
    cnn_encoder_config="CNNEncoderConfig_16_b1",
    transformer_encoder_config=efficient_v4_transformer_encoder,
    bbox_head=trans_transformer_head,
    data_preprocessor=ev_pupil_patch_preprocessor
)
