import os
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from PIL import Image, ImageDraw
import cv2
import json
import random

from projects.dataset.data_aug import rotate_data, flip_data


class ParkData(VisionDataset):
    def __init__(self,
                 mode,
                 root,
                 img_folder,
                 mask_folder,
                 label_folder,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root, transform=transform, target_transform=target_transform)
        self.mode = mode
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.label_folder = label_folder
        self.seg_folder = 'seg_labels'
        self.images = list(
            sorted([f for f in os.listdir(os.path.join(self.root, img_folder)) if f.endswith('.jpg')]))
        self.masks = list(
            sorted([f for f in os.listdir(os.path.join(self.root, mask_folder)) if f.endswith('.json')]))
        self.labels = list(
            sorted([f for f in os.listdir(os.path.join(self.root, label_folder)) if f.endswith('.json')]))
        self.seg_labels = list(
            sorted([f for f in os.listdir(os.path.join(self.root, self.seg_folder)) if f.endswith('.json')]))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_folder, self.images[index])
        # mask_path = os.path.join(self.root, self.mask_folder, self.masks[index])
        label_path = os.path.join(self.root, self.label_folder, self.labels[index])
        # seg_path = os.path.join(self.root, self.seg_folder, self.seg_labels[index])
        if '011951' in self.images[index] or '008953' in self.images[index] or '014768' in self.images[index]:
            img_path = os.path.join(self.root, self.img_folder, self.images[0])
            # mask_path = os.path.join(self.root, self.mask_folder, self.masks[0])
            label_path = os.path.join(self.root, self.label_folder, self.labels[0])
            # seg_path = os.path.join(self.root, self.seg_folder, self.seg_labels[0])
        # print(self.masks[index])
        img = Image.open(img_path).convert('RGB')
        ori_size = img.size
        input_size = (512, 512)
        input_size = (256, 256)
        scale = input_size[0] / ori_size[0]
        img = img.resize(input_size)

        ### pld gt
        with open(label_path, 'r', encoding='utf-8') as file:
            label_dct = json.load(file)
        slots = label_dct['objects']
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

        labels = np.zeros(input_size, dtype=np.uint8)
        if self.mode == 'train':
            ### data aug
            r1 = random.randrange(40, 140) / 100
            img = (np.array(img) * r1).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)

            if random.random() < 0.5:
                img, labels, pld_pts = flip_data(img, labels, pld_pts)

            prob = random.random()
            if prob < 0.4:
                angle = 90
                if random.random() < 0.5:
                    angle = 270
                img, labels, pld_pts = rotate_data(img, labels, pld_pts, angle)
            elif prob < 0.7:
                angle = random.randrange(-90, 90)
                angle = (360 - angle) if angle < 0 else angle
                img, labels, pld_pts = rotate_data(img, labels, pld_pts, angle)

        labels = labels.astype(np.int64)
        if self.transform is not None:
            img = self.transform(img)

        data_samples = dict(pld_cls=pld_cls, pld_pts=pld_pts,
                            img_path=img_path, label_path=label_path)
        
        return img, data_samples

    def __len__(self):
        return len(self.images)



from mmengine.registry import FUNCTIONS
@FUNCTIONS.register_module()
def park_collate(data_batch):
    imgs = torch.cat([item[0].unsqueeze(0) for item in data_batch], dim=0)
    pld_cls = []
    pld_pts = []
    img_path = []
    label_path = []
    for item in data_batch:
        pld_cls.append(item[1]['pld_cls'])
        pld_pts.append(item[1]['pld_pts'])
        img_path.append(item[1]['img_path'])
        label_path.append(item[1]['label_path'])
    data_samples = dict(pld_cls=pld_cls, pld_pts=pld_pts,
                        img_path=img_path, label_path=label_path)

    return imgs, data_samples
