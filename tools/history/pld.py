# 实现继承自 VisionDataset 的 SoildData 数据类。
# 在这个类中，我们重写了__getitem__和__len__方法，以确保每个索引返回一个包含图像和标签的字典。
# 此外，我们还实现了color_to_class字典，将 mask 的颜色映射到类别索引。

import os
import numpy as np
import torch.nn.functional
from torchvision.datasets import VisionDataset
from PIL import Image, ImageDraw
import csv
import json
import cv2


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
                 label_folder,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root, transform=transform, target_transform=target_transform)
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.label_folder = label_folder
        self.images = list(
            sorted([f for f in os.listdir(os.path.join(self.root, img_folder)) if f.endswith('.jpg')]))
        self.masks = list(
            sorted([f for f in os.listdir(os.path.join(self.root, mask_folder)) if f.endswith('.json')]))
        self.labels = list(
            sorted([f for f in os.listdir(os.path.join(self.root, label_folder)) if f.endswith('.json')]))
        # self.root2 = root2
        # self.images.extend(os.listdir(os.path.join(self.root2, img_folder)))
        # # self.images = self.images[0:4]
        # self.masks.extend(os.listdir(os.path.join(self.root2, mask_folder)))
        self.color_to_class = create_palette(
            os.path.join(self.root, 'class_dict.csv'))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_folder, self.images[index])
        mask_path = os.path.join(self.root, self.mask_folder, self.masks[index])
        label_path = os.path.join(self.root, self.label_folder, self.labels[index])
        if '011951' in self.images[index] or '008953' in self.images[index] or '014768' in self.images[index]:
            img_path = os.path.join(self.root, self.img_folder, self.images[0])
            mask_path = os.path.join(self.root, self.mask_folder, self.masks[0])
            label_path = os.path.join(self.root, self.label_folder, self.labels[0])
        # mask_path = os.path.join(self.root, self.mask_folder, '007356.json')
        # print(self.masks[index])
        img = Image.open(img_path).convert('RGB')
        # with open(mask_path, 'r', encoding='utf-8') as file:
        #     mask_dct = json.load(file)
        # objects = mask_dct['objects'][0]
        # mask_data = np.zeros_like(img)
        # if 'Multi' in objects['feature_type']:
        #     pt_list = np.array(objects['point_list'][0],dtype='int32')
        #     cv2.fillConvexPoly(mask_data, pt_list, (10,10,10))
        #     for j in range(1, len(objects['point_list'])):
        #         pt_list1 = np.array(objects['point_list'][j],dtype='int32')
        #         cv2.fillConvexPoly(mask_data, pt_list1, (50,50,50))
        # else:
        #     pt_list = np.array(objects['point_list'],dtype='int32')
        #     cv2.fillConvexPoly(mask_data, pt_list, (10,10,10))
        # mask = Image.fromarray(mask_data)
        with open(label_path, 'r', encoding='utf-8') as file:
            label_dct = json.load(file)
        slots = label_dct['objects']

        ori_size = img.size
        input_size = (512, 512)
        input_size = (256, 256)
        scale = input_size[0] / ori_size[0]
        img = img.resize(input_size)
        # mask = mask.resize(input_size, Image.NEAREST)
        pld_cls = []
        pld_pts = []
        for i in range(len(slots)):
            if slots[i]['attributes']['status'] == 'occupied':
                slot_cls = 1
            if slots[i]['attributes']['status'] == 'empty':
                slot_cls = 2
            pld_cls.append(slot_cls)
            slot_pts = slots[i]['point_list']
            slot_pts = [[item[0]*scale, item[1]*scale] for item in slot_pts]
            pld_pts.append(slot_pts)
        # pld_cls = tuple([pld_cls])
        # pld_pts = tuple([pld_pts])

        if self.transform is not None:
            img = self.transform(img)

        # # Convert the RGB values to class indices
        # mask = np.array(mask)
        # mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
        # fs_labels = np.zeros_like(mask, dtype=np.int64)
        # for color, class_index in self.color_to_class.items():
        #     rgb = color[0] * 65536 + color[1] * 256 + color[2]
        #     fs_labels[mask == rgb] = class_index

        # # labels = labels>=2
        # if self.target_transform is not None:
        #     fs_labels = self.target_transform(fs_labels)

        data_samples = dict(pld_cls=pld_cls, pld_pts=pld_pts,
                            img_path=img_path, mask_path=mask_path, label_path=label_path)
        return img, data_samples

    def __len__(self):
        return len(self.images)
    
from mmengine.registry import FUNCTIONS
@FUNCTIONS.register_module()
def park_collate(data_batch):
    imgs = torch.cat([item[0].unsqueeze(0) for item in data_batch], dim=0)
    # fs_labels = torch.cat([item[1]['fs_labels'].unsqueeze(0) for item in data_batch], dim=0)
    pld_cls = []
    pld_pts = []
    img_path = []
    mask_path = []
    label_path = []
    for item in data_batch:
        pld_cls.append(item[1]['pld_cls'])
        pld_pts.append(item[1]['pld_pts'])
        img_path.append(item[1]['img_path'])
        mask_path.append(item[1]['mask_path'])
        label_path.append(item[1]['label_path'])
    data_samples = dict(pld_cls=pld_cls, pld_pts=pld_pts,
                            img_path=img_path, mask_path=mask_path, label_path=label_path)

    return imgs, data_samples

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
    label_folder='labels',
    transform=transform,
    target_transform=target_transform)

valid_set = ParkData(
    r'G:\dnn\data\Boden_AVM_002000_019999\test',
    r'G:\dnn\data\soiling_dataset\ps_test\test',
    img_folder='images',
    mask_folder='fs_labels',
    label_folder='labels',
    transform=transform,
    target_transform=target_transform)

train_dataloader = dict(
    batch_size=2,
    dataset=train_set,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='park_collate'))

val_dataloader = dict(
    batch_size=6,
    dataset=valid_set,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='park_collate'))


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
from torch import Tensor
from torchvision.ops import focal_loss
import math


class ParkModel(BaseModel): 
    def __init__(self, num_classes):
        super().__init__()
        # self.net = deeplabv3_resnet50()
        self.backbone = resnet_fpn_backbone('resnet18', pretrained=True, returned_layers=[2, 3, 4])
        conv = []
        for _ in range(2):
            conv.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.pld_conv = nn.Sequential(*conv)
        self.pld_cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=3, padding=1, bias=False))
        self.pld_reg1_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=3, padding=1, bias=False))
        self.pld_reg2_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=3, padding=1, bias=False))
        self.pld_angle_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=3, padding=1, bias=False))
        # self.seg_head = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 2, kernel_size=1, padding=0),
        # )
        debug = 0

    def forward(self, imgs, data_samples=None, mode='tensor'):
        # x1 = self.net(imgs)
        x = self.backbone(imgs)
        pld_fea = self.pld_conv(x['2'])
        pld_cls = self.pld_cls_head(pld_fea)
        pld_pt0 = self.pld_reg1_head(pld_fea)
        pld_pt1 = self.pld_reg2_head(pld_fea)
        pld_angle = self.pld_angle_head(pld_fea)
        # seg = self.seg_head(x['0'])
        # x = x.sigmoid()
        input_size = (imgs.shape[3], imgs.shape[2])
        featmap_size = (pld_fea.shape[3], pld_fea.shape[2])
        if mode == 'loss':
            # loss_seg = self.cal_seg_loss(seg, data_samples['fs_labels'])
            loss_seg = 0

            cls_preds = pld_cls
            pts_preds = torch.cat((pld_pt0.sigmoid(), pld_pt1.tanh(), pld_angle.tanh()), dim=1)
            cls_gts, pts_gts = self.get_gt_map(data_samples, input_size, featmap_size)
            loss_pld = self.cal_pld_loss(cls_preds, pts_preds, cls_gts, pts_gts)
            return {'loss': 1.0*loss_seg + 1.0*loss_pld}
        elif mode == 'predict':
            seg = 0
            return seg, data_samples
        
    def cal_pld_loss(self, cls_preds: Tensor, pts_preds: Tensor,
                     cls_gts: Tensor, pts_gts: Tensor):
        cls_preds = cls_preds.permute(0, 2, 3, 1).reshape(-1,3)
        pts_preds = pts_preds.permute(0, 2, 3, 1).reshape(-1,6)
        cls_gts = cls_gts.reshape(-1)
        pts_gts = pts_gts.permute(0, 2, 3, 1).reshape(-1,6)
        cls_onehot = F.one_hot(cls_gts, num_classes=3).to(torch.float32)
        loss_cls = focal_loss.sigmoid_focal_loss(cls_preds, cls_onehot, alpha=0.25, gamma=2)
        loss_cls = loss_cls.mean()

        pos_idxs = (cls_gts > 0).nonzero().squeeze(1)
        if len(pos_idxs) > 0:
            pt0_preds = pts_preds[pos_idxs, :2]
            pt1_preds = pts_preds[pos_idxs, 2:4]
            angle_preds = pts_preds[pos_idxs, 4:]
            pt0_gts = pts_gts[pos_idxs, :2]
            pt1_gts = pts_gts[pos_idxs, 2:4]
            angle_gts = pts_gts[pos_idxs, 4:]

            # L1Loss = nn.L1Loss()
            loss_pt0 = F.l1_loss(pt0_preds, pt0_gts)
            loss_pt1 = F.l1_loss(pt1_preds, pt1_gts)
            loss_angle = F.l1_loss(angle_preds, angle_gts)
            # loss_angle = F.MSELoss(angle_preds, angle_gts)
        else:
            loss_pt0, loss_pt1, loss_angle = 0, 0, 0
        
        loss_pld = 5.0*loss_cls + 1.0*loss_pt0 + 4.0*loss_pt1 + 1.0*loss_angle

        return loss_pld

    def cal_seg_loss(self, seg_pres: Tensor, seg_labels: Tensor):
        loss_seg = F.cross_entropy(seg_pres, seg_labels)
        return loss_seg
    
    def get_gt_map(self, data_samples, input_size, featmap_size):
        grid_size = input_size[0] / featmap_size[0]
        pts_gts = torch.zeros(len(data_samples['pld_cls']), 6, featmap_size[0], featmap_size[1], device = 'cuda')
        cls_gts = torch.zeros(len(data_samples['pld_cls']), featmap_size[0], featmap_size[1], device = 'cuda', dtype = torch.long)
        for i in range(len(data_samples['pld_cls'])):
            for j in range(len(data_samples['pld_cls'][i])):
                ## gen gt_map
                pt = data_samples['pld_pts'][i][j]
                col = math.floor(pt[3][0] / grid_size)
                row = math.floor(pt[3][1] / grid_size)
                col = min(col, featmap_size[0] - 1)
                row = min(row, featmap_size[1] - 1)
                pts_gts[i, 0, row, col] = pt[3][0] / grid_size - col
                pts_gts[i, 1, row, col] = pt[3][1] / grid_size - row
                pts_gts[i, 2, row, col] = (pt[0][0] - pt[3][0]) / input_size[0]
                pts_gts[i, 3, row, col] = (pt[0][1] - pt[3][1]) / input_size[1]
                pts_gts[i, 4, row, col] = (pt[2][0] - pt[3][0]) / input_size[0]
                pts_gts[i, 5, row, col] = (pt[2][1] - pt[3][1]) / input_size[1]
                # radian = math.atan2(pld_marker[5] * featmap_size[1],
                #             pld_marker[4] * featmap_size[0])
                # gt_maps[i, 4, col, row] = math.cos(radian)
                # gt_maps[i, 5, col, row] = math.sin(radian)
                cls_gts[i, col, row] = data_samples['pld_cls'][i][j]
        return cls_gts, pts_gts
        

# 在使用 Runner 进行训练之前，我们需要实现 IoU（交并比）指标来评估模型的性能。
from mmengine.evaluator import BaseMetric
class IoU(BaseMetric):
    def process(self, data_batch, data_samples):
        debug = 0
        # preds, labels = data_samples[0], data_samples[1]['labels']
        # preds = torch.argmax(preds, dim=1)
        # preds = preds.cpu()
        # labels = labels.cpu()
        # # intersect = (labels == preds).sum()
        # # union = (torch.logical_or(preds, labels)).sum()
        # # iou = (intersect / union).cpu()
        # ious = [] # 记录每一类的iou
        # iou_sum = 0
        # cnt = 0
        # for c in range(2):
        #     label_c = (labels == c) # label_c为true/false矩阵
        #     pred_c = (preds == c)
        #     intersection = torch.logical_and(pred_c, label_c).sum()
        #     union = torch.logical_or(pred_c, label_c).sum()
        #     if union == 0:
        #         ious.append(float('nan'))  
        #     else:
        #         ious.append(intersection / union)
        #         iou_sum = iou_sum + (intersection / union)
        #         cnt = cnt + 1
        # iou = 0
        # if cnt >= 1:
        #     iou = iou_sum / cnt
        # preds_acc = torch.sum((preds.reshape(preds.size(0), -1) >= 1), dim = 1) / (256 * 256)
        # labels_acc = torch.sum((labels.reshape(labels.size(0), -1) >= 1), dim = 1) / (256 * 256)
        # preds_acc = preds_acc > 0.1
        # labels_acc = labels_acc > 0.1
        # correct_num = sum(preds_acc == labels_acc)
        # self.results.append(
        #     dict(batch_size=len(labels), iou=iou * len(labels), correct_num = correct_num))
    def compute_metrics(self, results):
        # total_iou = sum(result['iou'] for result in self.results)
        # num_samples = sum(result['batch_size'] for result in self.results)
        # total_correct = sum(result['correct_num'] for result in results)
        # return dict(iou=total_iou / num_samples, accuracy=100 * total_correct / num_samples)
        return dict(iou=0)




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
