from mmrotate.models import DeltaXYWHTRBBoxCoder, GDLoss
from mmrotate.structures import RotatedBoxes

from model.heads.trans_transformer_head import TransTransformerHead

trans_transformer_head = dict(
    type=TransTransformerHead,
    hidden_dim=768,
    pool_dim=None,
    is_distil=False,
    norm_layer=None,
    bbox_coder=dict(
        type=DeltaXYWHTRBBoxCoder,
        angle_version="le90",
        norm_factor=None,
        edge_swap=True,
        proj_xy=True,
        target_means=(.0, .0, .0, .0, .0),
        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
    is_delta_coder=True,
    ref_bbox_shape=[256, 256],
    use_crop=False,
    bbox_cls=RotatedBoxes,
    loss_decoded_bbox=True,
    loss_bbox=dict(type=GDLoss, loss_type='kld', tau=1)
)
