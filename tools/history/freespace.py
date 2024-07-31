# 实现继承自 VisionDataset 的 SoildData 数据类。
# 在这个类中，我们重写了__getitem__和__len__方法，以确保每个索引返回一个包含图像和标签的字典。
# 此外，我们还实现了color_to_class字典，将 mask 的颜色映射到类别索引。

import os
import numpy as np
import pycocotools.coco
import pycocotools.mask
from torchvision.datasets import VisionDataset
from PIL import Image, ImageDraw
import csv

import json
import pycocotools
import cv2

# # 假设json_file_path是你的JSON文件的路径
# json_path = r'G:\dnn\data\Boden_AVM_002000_019999\train\fs_labels\007356.json'
# img_path = r'G:\dnn\data\Boden_AVM_002000_019999\train\images\007356.jpg'
# # 使用with语句确保文件正确关闭
# with open(json_path, 'r', encoding='utf-8') as file:
#     mask_dct = json.load(file) 
# # 现在data包含了JSON文件中的数据，它是一个字典
# # print(mask_dct)
# img = Image.open(img_path).convert('RGB')

# objects = mask_dct['objects'][0]
# pt_list = np.array(objects['point_list'][0],dtype='int32')
# pt_list1 = np.array(objects['point_list'][1],dtype='int32')

# draw = ImageDraw.Draw(img)

# draw.point((600,300), fill='red')
# draw.polygon(pt_list, width=6)

# mask = np.zeros_like(img)
# cv2.fillConvexPoly(mask, pt_list, (2,2,2))
# mask1 = np.zeros_like(img)
# cv2.fillConvexPoly(mask1, pt_list1, (5,5,5))

# img_data = np.array(img)
# img_data1 = img_data + mask
# img1 = Image.fromarray(img_data1)
# img_data2 = img_data + mask1
# img2 = Image.fromarray(img_data2)
# # img.save('points.png')
# img1.save('points1.png')
# img2.save('points2.png')

def create_palette(csv_filepath):
    color_to_class = {}
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_class[(r, g, b)] = idx
    return color_to_class

class ParkData(VisionDataset):
    def __init__(self,
                 root,
                 root2,
                 img_folder,
                 mask_folder,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root, transform=transform, target_transform=target_transform)
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.images = list(
            sorted([f for f in os.listdir(os.path.join(self.root, img_folder)) if f.endswith('.jpg')]))
        self.masks = list(
            sorted([f for f in os.listdir(os.path.join(self.root, mask_folder)) if f.endswith('.json')]))
        # self.root2 = root2
        # self.images.extend(os.listdir(os.path.join(self.root2, img_folder)))
        # # self.images = self.images[0:4]
        # self.masks.extend(os.listdir(os.path.join(self.root2, mask_folder)))
        self.color_to_class = create_palette(
            os.path.join(self.root, 'class_dict.csv'))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_folder, self.images[index])
        mask_path = os.path.join(self.root, self.mask_folder, self.masks[index])
        if '011951' in self.images[index]:
            img_path = os.path.join(self.root, self.img_folder, self.images[0])
            mask_path = os.path.join(self.root, self.mask_folder, self.masks[0])
        # mask_path = os.path.join(self.root, self.mask_folder, '012523.json')
        # print(self.masks[index])
        img = Image.open(img_path).convert('RGB')
        with open(mask_path, 'r', encoding='utf-8') as file:
            mask_dct = json.load(file)
        objects = mask_dct['objects'][0]
        mask_data = np.zeros_like(img)
        if 'Multi' in objects['feature_type']:
            pt_list = np.array(objects['point_list'][0],dtype='int32')
            cv2.fillConvexPoly(mask_data, pt_list, (10,10,10))
            for j in range(1, len(objects['point_list'])):
                pt_list1 = np.array(objects['point_list'][j],dtype='int32')
                cv2.fillConvexPoly(mask_data, pt_list1, (50,50,50))
        else:
            pt_list = np.array(objects['point_list'],dtype='int32')
            cv2.fillConvexPoly(mask_data, pt_list, (10,10,10))
        mask = Image.fromarray(mask_data)

        input_size = (512, 512)
        input_size = (256, 256)
        img = img.resize(input_size)
        mask = mask.resize(input_size, Image.NEAREST)

        if self.transform is not None:
            img = self.transform(img)

        # Convert the RGB values to class indices
        mask = np.array(mask)
        mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
        labels = np.zeros_like(mask, dtype=np.int64)
        for color, class_index in self.color_to_class.items():
            rgb = color[0] * 65536 + color[1] * 256 + color[2]
            labels[mask == rgb] = class_index

        # labels = labels>=2
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        data_samples = dict(
            labels=labels, img_path=img_path, mask_path=mask_path)
        return img, data_samples

    def __len__(self):
        return len(self.images)
    

# 基于 CamVid 数据类，选择相应的数据增强方式，构建 train_dataloader 和 val_dataloader，供后续 runner 使用
import torch
import torchvision.transforms as transforms

norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**norm_cfg)
             ])

target_transform = transforms.Lambda(
        lambda x: torch.tensor(np.array(x), dtype=torch.long))

train_set = ParkData(
    r'G:\dnn\data\Boden_AVM_002000_019999\train',
    r'G:\dnn\data\soiling_dataset\ps_test\train',
    img_folder='images',
    mask_folder='fs_labels',
    transform=transform,
    target_transform=target_transform)

valid_set = ParkData(
    r'G:\dnn\data\Boden_AVM_002000_019999\test',
    r'G:\dnn\data\soiling_dataset\ps_test\test',
    img_folder='images',
    mask_folder='fs_labels',
    transform=transform,
    target_transform=target_transform)

train_dataloader = dict(
    batch_size=2,
    dataset=train_set,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'))

val_dataloader = dict(
    batch_size=6,
    dataset=valid_set,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'))


# 定义一个名为MMDeeplabV3的模型类。该类继承自BaseModel，并集成了DeepLabV3架构的分割模型。
# MMDeeplabV3 重写了forward方法，以处理输入图像和标签，并支持在训练和预测模式下计算损失和返回预测结果。
from mmengine.model import BaseModel
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F
from torchvision._internally_replaced_utils import load_state_dict_from_url
import models.my_resnet, models.resnet_seg
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import retinanet
import torch.nn as nn


class ParkModel(BaseModel): 
    def __init__(self, num_classes):
        super().__init__()
        self.net = deeplabv3_resnet50()
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
            nn.Conv2d(128, 2, kernel_size=1, padding=0),
        )
        debug = 0

    def forward(self, imgs, data_samples=None, mode='tensor'):
        # x1 = self.net(imgs)
        x = self.backbone(imgs)
        x1 = x['0']
        seg = self.seg_head(x['0'])
        # x = x.sigmoid()
        if mode == 'loss':
            return {'loss': F.cross_entropy(seg, data_samples['labels'])}
        elif mode == 'predict':
            return seg, data_samples
        

# 在使用 Runner 进行训练之前，我们需要实现 IoU（交并比）指标来评估模型的性能。
from mmengine.evaluator import BaseMetric
class IoU(BaseMetric):
    def process(self, data_batch, data_samples):
        preds, labels = data_samples[0], data_samples[1]['labels']
        preds = torch.argmax(preds, dim=1)
        preds = preds.cpu()
        labels = labels.cpu()
        # intersect = (labels == preds).sum()
        # union = (torch.logical_or(preds, labels)).sum()
        # iou = (intersect / union).cpu()
        ious = [] # 记录每一类的iou
        iou_sum = 0
        cnt = 0
        for c in range(2):
            label_c = (labels == c) # label_c为true/false矩阵
            pred_c = (preds == c)
            intersection = torch.logical_and(pred_c, label_c).sum()
            union = torch.logical_or(pred_c, label_c).sum()
            if union == 0:
                ious.append(float('nan'))  
            else:
                ious.append(intersection / union)
                iou_sum = iou_sum + (intersection / union)
                cnt = cnt + 1
        iou = 0
        if cnt >= 1:
            iou = iou_sum / cnt
        preds_acc = torch.sum((preds.reshape(preds.size(0), -1) >= 1), dim = 1) / (256 * 256)
        labels_acc = torch.sum((labels.reshape(labels.size(0), -1) >= 1), dim = 1) / (256 * 256)
        preds_acc = preds_acc > 0.1
        labels_acc = labels_acc > 0.1
        correct_num = sum(preds_acc == labels_acc)
        self.results.append(
            dict(batch_size=len(labels), iou=iou * len(labels), correct_num = correct_num))
    def compute_metrics(self, results):
        total_iou = sum(result['iou'] for result in self.results)
        num_samples = sum(result['batch_size'] for result in self.results)
        total_correct = sum(result['correct_num'] for result in results)
        return dict(iou=total_iou / num_samples, accuracy=100 * total_correct / num_samples)




from torch.optim import AdamW,SGD
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmengine.runner import Runner


num_classes = 2  # Modify to actual number of categories.
runner = Runner(
    model=ParkModel(num_classes),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(
        type=OptimWrapper, optimizer=dict(type=AdamW, lr=2e-4)),
    train_cfg=dict(by_epoch=True, max_epochs=50, val_interval=5),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=IoU),
    # custom_hooks=[SegVisHook(r'G:\dnn\data\soiling_dataset\train')],
    default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=5)),
    test_dataloader=val_dataloader,
    test_cfg=dict(),
    test_evaluator=dict(type=IoU),
    # load_from='checkpoints/seg/epoch_25.pth'
)
runner.train()
# runner.val()
# runner.test()



debug = 0
