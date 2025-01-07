from mmengine.config import read_base

with read_base():
    from configs.model_config.detectors.efficient_trans_v4.cnn_16_b1_transformer_f2_n4 import efficient_trans_vit

efficient_trans_vit.update(
    cnn_encoder_config="CNNEncoderConfig_16_b0",
)
