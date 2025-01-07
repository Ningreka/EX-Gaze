from mmengine.config import read_base
from dataset.transforms.transform import RandomEllipseGtTransform

with read_base():
    from configs.dataset_config.pol_event_count_inter2000.patch_n8_s16.multi_max10_accum30_ev_overlap_pupil_disp import train_dataloader, \
        val_dataloader, test_dataloader

train_dataloader.update(batch_size=128)
train_dataloader["dataset"]["dataset_pipeline"].insert(1, dict(type=RandomEllipseGtTransform,
                                                               transform_target='pre_gt',
                                                               sync_transform=True, random_dis_range=3,
                                                               random_rot_range=0.3,
                                                               random_scale_range=0.1))

val_dataloader["dataset"]["dataset_pipeline"].insert(1, dict(type=RandomEllipseGtTransform,
                                                               transform_target='pre_gt',
                                                               sync_transform=True, random_dis_range=1,
                                                               random_rot_range=0.3,
                                                               random_scale_range=0.1))

test_dataloader["dataset"]["dataset_pipeline"].insert(1, dict(type=RandomEllipseGtTransform,
                                                               transform_target='pre_gt',
                                                               sync_transform=True, random_dis_range=1,
                                                               random_rot_range=0.3,
                                                               random_scale_range=0.1))
