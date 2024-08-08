import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from projects.metric.metric_fun import pld_metric, seg_confu_mat, cal_iou, cal_acc

class MultiMetric(BaseMetric):
    def process(self, data_batch, data_samples):
        input_size = (data_batch[0].shape[3], data_batch[0].shape[2])
        scale = np.zeros(6)
        scale[0::2] = input_size[0]
        scale[1::2] = input_size[1]
        bs = len(data_samples[0])
        num_seg_cls = data_samples[1].shape[1]
        for b_idx in range(bs):
            cls_preds = np.array(data_samples[0][b_idx]['cls_pred'])
            pt_preds = np.array(data_samples[0][b_idx]['pts_pred']) * scale
            confi = np.array(data_samples[0][b_idx]['confi'])
            seg_pred = np.array(torch.argmax(data_samples[1][b_idx,:,:,:], dim=0))
            cls_gts = np.array(data_samples[2]['pld_cls'][b_idx])
            pt_gts = np.array(data_samples[2]['pld_pts'][b_idx])
            seg_gt = np.array(data_samples[2]['fs_labels'][b_idx,:,:].cpu())

            confus_mat, error_mat = pld_metric(cls_preds, pt_preds, confi, cls_gts, pt_gts)
            confus_mat_seg = seg_confu_mat(seg_pred.flatten(), seg_gt.flatten(), num_seg_cls)

            if len(self.results) == 0:
                self.results.append(confus_mat)
                self.results.append(error_mat)
                self.results.append(confus_mat_seg)
            else:
                self.results[0] = self.results[0] + confus_mat
                self.results[1] = self.results[1] + error_mat
                self.results[2] = self.results[2] + confus_mat_seg

            ### print
            recall_pld = confus_mat[2,2] / (np.sum(confus_mat[2,:]) + 0.0001) if np.sum(confus_mat[2,:])!=0 else 1.0
            precision_pld = confus_mat[2,2] / (np.sum(confus_mat[:,2]) + 0.0001) if np.sum(confus_mat[:,2])!=0 else 1.0
            FN_pld, FP_pld = np.sum(confus_mat[2,0:2]), np.sum(confus_mat[0:2,2])
            pld_log_flag = ((FN_pld + FP_pld) >= 5 or recall_pld < 0.45 or precision_pld < 0.6
                          or error_mat[1,1] > 9 or error_mat[1,2] > 36 or error_mat[1,4] > 0.17)
            if (pld_log_flag and 0):
                print(data_samples[1]['img_path'][b_idx][-10:], end=";")
                print("FN_pld = %d" % FN_pld, end=";")
                print("FP_pld = %d" % FP_pld, end=";")
                print("dist_pt0 = %.2f" % error_mat[1,1], end=";")
                print("dist_pt1 = %.2f" % error_mat[1,2], end=";")
                print("dist_pt3 = %.2f" % error_mat[1,3], end=";")
                print("error_angle = %.2f" % error_mat[1,4], end="\n")
        debug = 0


    def compute_metrics(self, results):
        confus_mat, error_mat, confus_mat_seg = self.results
        # pld
        recall_pld = confus_mat[2,2] / (np.sum(confus_mat[2,:]) + 0.0001)
        precision_pld = confus_mat[2,2] / (np.sum(confus_mat[:,2]) + 0.0001)
        recall_slot = np.sum(confus_mat[1:,1:]) / (np.sum(confus_mat[1:,:]) + 0.0001)
        precision_slot = np.sum(confus_mat[1:,1:]) / (np.sum(confus_mat[:,1:]) + 0.0001)
        # seg
        acc_seg, acc_cls_seg = cal_acc(confus_mat_seg)
        iou_seg, miou_seg = cal_iou(confus_mat_seg)

        return dict(recall_pld=recall_pld, precision_pld=precision_pld, recall_slot=recall_slot, precision_slot=precision_slot,
                    acc_seg=acc_seg, miou_seg=miou_seg, acc_cls_seg=acc_cls_seg, iou_seg=iou_seg,
                    error_mat=error_mat, confus_mat=confus_mat, confus_mat_seg=confus_mat_seg)
    
