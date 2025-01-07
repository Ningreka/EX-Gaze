from metrics.EllipseMetric import EllipseMetric

val_evaluator = dict(type=EllipseMetric, mask_size=(260, 346), prefix="ev_eye")
test_evaluator = val_evaluator.copy()

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_begin=5, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type="ReduceOnPlateauLR",
        monitor="ev_eye/mDist",
        rule="greater",
        patience=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=21,
        by_epoch=True,
        milestones=[10, 20],
        gamma=0.3)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type="Adam",
        lr=1e-5,
        betas=[0.9, 0.999],
        eps=1e-3
    ),
    clip_grad=dict(max_norm=35, norm_type=2))
