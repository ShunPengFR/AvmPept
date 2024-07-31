import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import AdamW,SGD
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmengine.runner import Runner

import sys
sys.path.append(".")
from projects.models.res18_fpn import ParkModel
# from projects.models.res50_deeplabv3 import ParkModel
from projects.dataset.pld_data import PldData
from projects.metric.pld_metric import PldMetric


### data preprocess
norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(**norm_cfg)
                                ])
target_transform = transforms.Lambda(
        lambda x: torch.tensor(np.array(x), dtype=torch.long))

### data set
train_set = PldData(
    r'E:\dnn\data\Boden_AVM_002000_019999\train',
    img_folder='images',
    mask_folder='fs_labels',
    label_folder='labels',
    transform=transform,
    target_transform=target_transform)
valid_set = PldData(
    r'E:\dnn\data\Boden_AVM_002000_019999\test',
    img_folder='images',
    mask_folder='fs_labels',
    label_folder='labels',
    transform=transform,
    target_transform=target_transform)

### data loader
train_dataloader = dict(
    batch_size=2,
    dataset=train_set,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='park_collate'))
val_dataloader = dict(
    batch_size=2,
    dataset=valid_set,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='park_collate'))

### model runner
task_name = 'pld'
# task_name = 'freespace'
# task_name = 'multi_task'
runner = Runner(
    model=ParkModel(task_name),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(type=OptimWrapper, optimizer=dict(type=AdamW, lr=2e-4)),
    train_cfg=dict(by_epoch=True, max_epochs=50, val_interval=5),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=PldMetric),
    # custom_hooks=[SegVisHook(r'G:\dnn\data\soiling_dataset\train')],
    default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=5)),
    test_dataloader=val_dataloader,
    test_cfg=dict(),
    test_evaluator=dict(type=PldMetric),
    # load_from='checkpoints/pld/epoch_25.pth'
)


### train or test
runner.train()
# runner.test()

debug = 0
