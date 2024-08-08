import numpy as np


def pld_metric(cls_preds, pt_preds, confi, cls_gts, pt_gts):
    confus_mat = np.zeros((3,3))
    error_list = []
    not_matched = [True] * pt_preds.shape[0]
    for i in range(pt_gts.shape[0]):
        match_id = -1
        max_confi = 0
        for j in range(pt_preds.shape[0]):
            # tmp = (pt_preds[j,0:2] - pt_gts[i,3,:])**2
            dist_pt0 = np.sum((pt_preds[j,0:2] - pt_gts[i,3,:])**2)
            if (dist_pt0 > 10):
                continue
            error_confi = (1 - confi[j]) if cls_gts[i]==cls_preds[j] else 1.0
            dist_pt1 = np.sum((pt_preds[j,2:4] - pt_gts[i,0,:])**2)
            dist_pt3 = np.sum((pt_preds[j,4:] - pt_gts[i,2,:])**2)
            # if dist_pt3 > 100:
            #     debug = 0
            line_pred = pt_preds[j,4:] - pt_preds[j,0:2]
            line_gt = pt_gts[i,2,:] - pt_gts[i,3,:]
            angle03_pred = np.arctan2(line_pred[1], line_pred[0])
            angle03_gt = np.arctan2(line_gt[1], line_gt[0])
            error_angle03 = abs(angle03_pred-angle03_gt)
            error_angle03 = error_angle03 if error_angle03 < np.pi else (2*np.pi - error_angle03)
            error_angle03_sin = abs(np.sin(angle03_pred) - np.sin(angle03_gt))
            error_angle03_cos = abs(np.cos(angle03_pred) - np.cos(angle03_gt))
            match_flag = dist_pt1<40
            if (match_flag and max_confi < confi[j]):
                max_confi = confi[j]
                match_id = j
                not_matched[j] = False
                confus_mat[cls_gts[i],cls_preds[j]] = confus_mat[cls_gts[i],cls_preds[j]] + 1
                error_list.append([error_confi,dist_pt0,dist_pt1,dist_pt3,error_angle03,error_angle03_sin,error_angle03_cos])
            # error_list.append([error_confi,dist_pt0,dist_pt1,dist_pt3,error_angle03,error_angle03_sin,error_angle03_cos])
        if (match_id == -1):
            confus_mat[cls_gts[i],0] = confus_mat[cls_gts[i],0] + 1
    for j in range(len(not_matched)):
        if not_matched[j]:
            confus_mat[0,cls_preds[j]] = confus_mat[0,cls_preds[j]] + 1
    
    error_mat = np.zeros((2,7))
    if len(error_list) != 0:
        error_list = np.array(error_list)
        error_mat[0,:] = np.mean(error_list, axis=0)
        error_mat[1,:] = np.max(error_list, axis=0)

    return confus_mat, error_mat


def seg_confu_mat(seg_pred, seg_gt, num_cls):
    mask = (seg_gt >= 0)
    label = num_cls * seg_gt[mask].astype('int') + seg_pred[mask].astype('int')
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_cls**2)
    confus_mat = count.reshape(num_cls, num_cls)
    return confus_mat

def cal_acc(confus_mat):
    acc = np.diag(confus_mat).sum() / confus_mat.sum()
    acc_cls = np.diag(confus_mat) / confus_mat.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    return acc, acc_cls


def cal_iou(confus_mat):
    iou = np.diag(confus_mat) / (np.sum(confus_mat, axis=1) + np.sum(confus_mat, axis=0) - np.diag(confus_mat))
    miou = np.nanmean(iou) #跳过0值求mean

    return iou, miou