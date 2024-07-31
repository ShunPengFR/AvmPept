from mmengine.model import BaseModel
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn.functional as F


class FreespaceModel(BaseModel): 
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet_fpn_backbone('resnet18', pretrained=True, returned_layers=[2, 3, 4])
        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1, padding=0),
        )

    def forward(self, imgs, data_samples=None, mode='tensor'):
        x = self.backbone(imgs)
        seg = self.seg_head(x['0'])
        if mode == 'loss':
            return {'loss': F.cross_entropy(seg, data_samples['labels'])}
        elif mode == 'predict':
            return seg, data_samples
        else:
            return seg