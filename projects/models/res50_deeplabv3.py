from mmengine.model import BaseModel
import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision._internally_replaced_utils import load_state_dict_from_url
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import focal_loss
import math


class ParkModel(BaseModel): 
    def __init__(self, task_name):
        super().__init__()
        self.task_name = task_name
        self.model = deeplabv3_resnet50(pretrained=False, num_classes=2)
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth', progress=True)
        self.model.backbone.load_state_dict(state_dict, strict=False)
        self.model.classifier[0].load_state_dict(state_dict, strict=False)
        if ('pld' in task_name) or ('multi_task' in task_name):
            self.model.downsample1 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
                )
            self.model.downsample2 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
                )
            # conv = []
            # for _ in range(2):
            #     conv.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            #     conv.append(nn.ReLU())
            # self.pld_conv = nn.Sequential(*conv)
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
        if ('freespace' in task_name) or ('multi_task' in task_name):
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
        # backbone
        x = self.model.backbone(imgs)
        x = x["out"]
        # ASPP
        x = self.model.classifier[0](x)
        input_size = (imgs.shape[3], imgs.shape[2])
        featmap_size = (int(imgs.shape[3]/32), int(imgs.shape[2]/32))
        # pld head
        if ('pld' in self.task_name) or ('multi_task' in self.task_name):
            pld_fea = self.model.downsample1(x)
            pld_fea = self.model.downsample2(pld_fea)
            pld_cls = self.pld_cls_head(pld_fea)
            pld_pt0 = self.pld_reg1_head(pld_fea)
            pld_pt1 = self.pld_reg2_head(pld_fea)
            pld_angle = self.pld_angle_head(pld_fea)
            scores = pld_cls
            pts_preds = torch.cat((pld_pt0.sigmoid(), pld_pt1.tanh(), pld_angle.tanh()), dim=1)
        # seg head
        if ('freespace' in self.task_name) or ('multi_task' in self.task_name):
            seg = self.model.classifier[1](x)
            seg = self.model.classifier[2](seg)
            seg = self.model.classifier[3](seg)
            seg = self.seg_head(seg)
        # loss or predict or tensor
        if mode == 'loss':
            if ('multi_task' in self.task_name):
                # pld loss
                cls_gts, pts_gts = self.get_gt_map(data_samples, input_size, featmap_size)
                loss_pld = self.cal_pld_loss(scores, pts_preds, cls_gts, pts_gts)
                # seg loss
                loss_seg = self.cal_seg_loss(seg, data_samples['fs_labels'])
                loss = 1.0*loss_seg + 1.0*loss_pld
            if ('pld' in self.task_name):
                cls_gts, pts_gts = self.get_gt_map(data_samples, input_size, featmap_size)
                loss = self.cal_pld_loss(scores, pts_preds, cls_gts, pts_gts)
            if ('freespace' in self.task_name):
                loss = self.cal_seg_loss(seg, data_samples['fs_labels'])
            return {'loss': loss}
        elif mode == 'predict':
            seg = 0
            return seg, data_samples
        else:
            if ('pld' in self.task_name) or ('multi_task' in self.task_name):
                ref_y = torch.linspace(0, featmap_size[1] - 1.0, featmap_size[1]) / featmap_size[1]
                ref_y = ref_y.reshape(featmap_size[1], 1).expand(featmap_size[0], featmap_size[1])
                ref_x = torch.linspace(0, featmap_size[0] - 1.0, featmap_size[0]) / featmap_size[0]
                ref_x = ref_x.reshape(1, featmap_size[0]).expand(featmap_size[0], featmap_size[1])
                ref_2d = torch.stack((ref_x, ref_y), -1)
                ref_2d = ref_2d.reshape(featmap_size[0] * featmap_size[1], -1)

                scores = scores.cpu().sigmoid()
                pts_preds = pts_preds.cpu()
                preds = []
                for i in range(pld_cls.shape[0]):
                    score = scores[i, :, :, :]
                    pts_pred = pts_preds[i, :, :, :]
                    score = score.permute(1, 2, 0).reshape(featmap_size[0] * featmap_size[1], -1)
                    pts_pred = pts_pred.permute(1, 2, 0).reshape(featmap_size[0] * featmap_size[1], -1)
                    confi, cls_pred = torch.max(score, dim=1)
                    keep_idx = ((cls_pred>0) & (confi>0.3)).nonzero().squeeze(1)
                    cls_pred = cls_pred[keep_idx]
                    confi = confi[keep_idx]
                    pts_pred = pts_pred[keep_idx, :]
                    ref_pt = ref_2d[keep_idx, :]
                    pts_pred[:,0] = pts_pred[:,0]/featmap_size[0] + ref_pt[:,0]
                    pts_pred[:,1] = pts_pred[:,1]/featmap_size[1] + ref_pt[:,1]
                    pts_pred[:,2:4] = pts_pred[:,2:4] + pts_pred[:,0:2]
                    pts_pred[:,4:] = pts_pred[:,4:] + pts_pred[:,0:2]
                    preds.append(dict(cls_pred=cls_pred, confi=confi, pts_pred=pts_pred))

            if ('multi_task' in self.task_name):
                return preds, seg
            if ('pld' in self.task_name):
                return preds
            if ('freespace' in self.task_name):
                return seg
        
    def cal_pld_loss(self, scores: Tensor, pts_preds: Tensor,
                     cls_gts: Tensor, pts_gts: Tensor):
        scores = scores.permute(0, 2, 3, 1).reshape(-1,3)
        pts_preds = pts_preds.permute(0, 2, 3, 1).reshape(-1,6)
        cls_gts = cls_gts.reshape(-1)
        pts_gts = pts_gts.permute(0, 2, 3, 1).reshape(-1,6)
        cls_onehot = F.one_hot(cls_gts, num_classes=3).to(torch.float32)
        loss_cls = focal_loss.sigmoid_focal_loss(scores, cls_onehot, alpha=0.25, gamma=2)
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
        
        loss_pld = 10.0*loss_cls + 1.0*loss_pt0 + 4.0*loss_pt1 + 1.0*loss_angle

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
                # gt_maps[i, 4, row, col] = math.cos(radian)
                # gt_maps[i, 5, row, col] = math.sin(radian)
                cls_gts[i, row, col] = data_samples['pld_cls'][i][j]
        return cls_gts, pts_gts