from mmengine.model import BaseModel
import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import focal_loss
import math

from projects.loss.lovasz_losses import lovasz_softmax


class ParkModel(BaseModel): 
    def __init__(self, task_name, num_seg_cls):
        super().__init__()
        self.task_name = task_name
        self.backbone = resnet_fpn_backbone('resnet18', pretrained=True, returned_layers=[2, 3, 4])
        if ('pld' in task_name) or ('multi_task' in task_name):
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
                nn.Conv2d(128, num_seg_cls, kernel_size=1, padding=0),
            )
        debug = 0

    def forward(self, imgs, data_samples=None, mode='tensor'):
        # backbone + neck
        x = self.backbone(imgs)
        input_size = (imgs.shape[3], imgs.shape[2])
        featmap_size = (x['2'].shape[3], x['2'].shape[2])
        # pld head
        if ('pld' in self.task_name) or ('multi_task' in self.task_name):
            pld_fea = self.pld_conv(x['2'])
            pld_cls = self.pld_cls_head(pld_fea)
            pld_pt0 = self.pld_reg1_head(pld_fea)
            pld_pt1 = self.pld_reg2_head(pld_fea)
            pld_angle = self.pld_angle_head(pld_fea)
            scores = pld_cls
            pts_preds = torch.cat((pld_pt0.sigmoid(), pld_pt1.tanh(), pld_angle.tanh()), dim=1)
        # seg head
        if ('freespace' in self.task_name) or ('multi_task' in self.task_name):
            seg = self.seg_head(x['0'])
        # loss or predict or tensor
        if mode == 'loss':
            if ('multi_task' in self.task_name):
                # pld loss
                cls_gts, pts_gts = self.get_gt_map(data_samples, input_size, featmap_size)
                loss_pld = self.cal_pld_loss(scores, pts_preds, cls_gts, pts_gts, featmap_size)
                # seg loss
                loss_seg = self.cal_seg_loss(seg, data_samples['fs_labels'])
                loss = 1.0*loss_seg + 1.0*loss_pld
            if ('pld' in self.task_name):
                cls_gts, pts_gts = self.get_gt_map(data_samples, input_size, featmap_size)
                loss = self.cal_pld_loss(scores, pts_preds, cls_gts, pts_gts, featmap_size)
            if ('freespace' in self.task_name):
                loss = self.cal_seg_loss(seg, data_samples['fs_labels'])
            return {'loss': loss}
        # elif mode == 'predict':
        #     seg = 0
        #     return seg, data_samples
        else:
            if ('pld' in self.task_name) or ('multi_task' in self.task_name):
                ref_y = torch.linspace(0, featmap_size[1] - 1.0, featmap_size[1]) / featmap_size[1]
                ref_y = ref_y.reshape(featmap_size[1], 1).expand(featmap_size[0], featmap_size[1])
                ref_x = torch.linspace(0, featmap_size[0] - 1.0, featmap_size[0]) / featmap_size[0]
                ref_x = ref_x.reshape(1, featmap_size[0]).expand(featmap_size[0], featmap_size[1])
                ref_2d = torch.stack((ref_x, ref_y), -1)
                ref_2d = ref_2d.reshape(featmap_size[0] * featmap_size[1], -1)

                scores = scores.cpu().detach().sigmoid()
                pts_preds = pts_preds.cpu().detach()
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
                    # pts_pred[:,2:4] = pts_pred[:,2:4] + ref_pt[:,0:2]
                    # pts_pred[:,4:] = pts_pred[:,4:] + ref_pt[:,0:2]
                    preds.append(dict(cls_pred=cls_pred, confi=confi, pts_pred=pts_pred))

            if ('multi_task' in self.task_name):
                return preds, seg.cpu().detach(), data_samples
            if ('pld' in self.task_name):
                return preds, data_samples
            if ('freespace' in self.task_name):
                return seg.cpu().detach(), data_samples
        
    def cal_pld_loss(self, scores: Tensor, pts_preds: Tensor,
                     cls_gts: Tensor, pts_gts: Tensor, featmap_size):
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
            pt3_preds = pts_preds[pos_idxs, 4:]
            pt0_gts = pts_gts[pos_idxs, :2]
            pt1_gts = pts_gts[pos_idxs, 2:4]
            pt3_gts = pts_gts[pos_idxs, 4:]

            # # L1Loss = nn.L1Loss()
            # loss_pt0 = F.l1_loss(pt0_preds, pt0_gts)
            # loss_pt1 = F.l1_loss(pt1_preds, pt1_gts)
            # loss_angle = F.l1_loss(pt3_preds, pt3_gts)
            loss_pt0 = F.smooth_l1_loss(pt0_preds, pt0_gts)
            loss_pt1 = F.smooth_l1_loss(pt1_preds, pt1_gts)
            angle01_gts = torch.atan2(pt1_gts[:,1]-pt0_gts[:,1]/featmap_size[1], pt1_gts[:,0]-pt0_gts[:,0]/featmap_size[0])
            angle01_preds = torch.atan2(pt1_preds[:,1]-pt0_preds[:,1]/featmap_size[1], pt1_preds[:,0]-pt0_preds[:,0]/featmap_size[0])
            loss_angle01 = (F.smooth_l1_loss(angle01_preds.cos(), angle01_gts.cos()) +
                          F.smooth_l1_loss(angle01_preds.sin(), angle01_gts.sin()))
            loss_pt3 = F.smooth_l1_loss(pt3_preds, pt3_gts)
            angle03_gts = torch.atan2(pt3_gts[:,1]-pt0_gts[:,1]/featmap_size[1], pt3_gts[:,0]-pt0_gts[:,0]/featmap_size[0])
            angle03_preds = torch.atan2(pt3_preds[:,1]-pt0_preds[:,1]/featmap_size[1], pt3_preds[:,0]-pt0_preds[:,0]/featmap_size[0])
            loss_angle03 = (F.smooth_l1_loss(angle03_preds.cos(), angle03_gts.cos()) +
                          F.smooth_l1_loss(angle03_preds.sin(), angle03_gts.sin()))

            loss_pld = 10.0*loss_cls + 4.0*loss_pt0 + 4.0*loss_pt1 + 2.0*loss_angle01 + 2.0*loss_angle03 + 0.2*loss_pt3
        else:
            loss_pld = 0

        return loss_pld

    def cal_seg_loss(self, seg_pres: Tensor, seg_labels: Tensor):
        ce_loss = F.cross_entropy(seg_pres, seg_labels)
        lovasz_loss = lovasz_softmax(seg_pres, seg_labels)
        loss_seg = 1.0*ce_loss + 1.0*lovasz_loss
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
                # pts_gts[i, 2, row, col] = (pt[0][0] - pt[3][0]) / input_size[0]
                # pts_gts[i, 3, row, col] = (pt[0][1] - pt[3][1]) / input_size[1]
                # pts_gts[i, 4, row, col] = (pt[2][0] - pt[3][0]) / input_size[0]
                # pts_gts[i, 5, row, col] = (pt[2][1] - pt[3][1]) / input_size[1]
                pts_gts[i, 2, row, col] = (pt[0][0] - col*grid_size) / input_size[0]
                pts_gts[i, 3, row, col] = (pt[0][1] - row*grid_size) / input_size[1]
                pts_gts[i, 4, row, col] = (pt[2][0] - col*grid_size) / input_size[0]
                pts_gts[i, 5, row, col] = (pt[2][1] - row*grid_size) / input_size[1]
                cls_gts[i, row, col] = data_samples['pld_cls'][i][j]
        return cls_gts, pts_gts