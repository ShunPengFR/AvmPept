from mmengine.evaluator import BaseMetric

class PldMetric(BaseMetric):
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