from torch import nn

from model.blocks.seperable_self_attention import LinearAttnEncoder

efficient_v4_transformer_encoder = dict(
    type=LinearAttnEncoder,
    patch_num=8,
    attn_unit_dim=64,
    ffn_multiplier=2.0,
    n_attn_blocks=4,
    attn_dropout=0.0,
    dropout=0.1,
    ffn_dropout=0.0,
    act_layer=nn.ReLU,
    learnable_position_embeddings=True
)
