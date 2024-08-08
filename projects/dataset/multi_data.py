import os
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from PIL import Image, ImageDraw
import cv2
import json
import random
from labelme.utils import shape_to_mask

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
        # self.images = self.images[0:30]
        self.masks = list(
            sorted([f for f in os.listdir(os.path.join(self.root, mask_folder)) if f.endswith('.json')]))
        self.labels = list(
            sorted([f for f in os.listdir(os.path.join(self.root, label_folder)) if f.endswith('.json')]))
        self.seg_labels = list(
            sorted([f for f in os.listdir(os.path.join(self.root, self.seg_folder)) if f.endswith('.json')]))

    def __getitem__(self, index):
        ### get file path, including img, fs_label, pld_label, seg_label
        img_path = os.path.join(self.root, self.img_folder, self.images[index])
        mask_path = os.path.join(self.root, self.mask_folder, self.masks[index])
        label_path = os.path.join(self.root, self.label_folder, self.labels[index])
        seg_path = os.path.join(self.root, self.seg_folder, self.seg_labels[index])
        if '011951' in self.images[index] or '008953' in self.images[index] or '014768' in self.images[index]:
            img_path = os.path.join(self.root, self.img_folder, self.images[0])
            mask_path = os.path.join(self.root, self.mask_folder, self.masks[0])
            label_path = os.path.join(self.root, self.label_folder, self.labels[0])
            seg_path = os.path.join(self.root, self.seg_folder, self.seg_labels[0])
        ### load img
        img = Image.open(img_path).convert('RGB')
        ori_size = img.size
        input_size = (512, 512)
        # input_size = (256, 256)
        scale = input_size[0] / ori_size[0]
        img = img.resize(input_size)
        ### fs & seg label
        labels = np.zeros(input_size, dtype=np.uint8)
        ### load freespace gt
        with open(mask_path, 'r', encoding='utf-8') as file:
            mask_dct = json.load(file)
        objects = mask_dct['objects'][0]
        if 'Multi' in objects['feature_type']:
            points = [[min(item[0]*scale,input_size[0]-1), min(item[1]*scale,input_size[1]-1)] for item in objects['point_list'][0]]
            mask_idx = shape_to_mask(img.size, points)
            for j in range(1, len(objects['point_list'])):
                points = [[min(item[0]*scale,input_size[0]-1), min(item[1]*scale,input_size[1]-1)] for item in objects['point_list'][j]]
                mask_idx1 = shape_to_mask(img.size, points)
                mask_idx = np.logical_and(mask_idx, np.logical_not(mask_idx1))
        else:
            points = [[min(item[0]*scale,input_size[0]-1), min(item[1]*scale,input_size[1]-1)] for item in objects['point_list']]
            mask_idx = shape_to_mask(img.size, points)
        labels[mask_idx] = 1
        ### load seg gt
        with open(seg_path, 'r', encoding='utf-8') as file:
            seg_dct = json.load(file)
        objects = seg_dct['objects']
        ego_flag = True
        for i in range(len(objects)):
            if 'Multi' in objects[i]['feature_type']:
                points = [[min(item[0]*scale,input_size[0]-1), min(item[1]*scale,input_size[1]-1)] for item in objects[i]['point_list'][0]]
                mask_idx = shape_to_mask(img.size, points)
                for j in range(1, len(objects[i]['point_list'])):
                    points = [[min(item[0]*scale,input_size[0]-1), min(item[1]*scale,input_size[1]-1)] for item in objects[i]['point_list'][j]]
                    mask_idx1 = shape_to_mask(img.size, points)
                    mask_idx = np.logical_or(mask_idx, mask_idx1)
            else:
                points = [[min(item[0]*scale,input_size[0]-1), min(item[1]*scale,input_size[1]-1)] for item in objects[i]['point_list']]
                mask_idx = shape_to_mask(img.size, points)
            if objects[i]['category']=='speed_bump':
                labels[mask_idx] = 2
            if objects[i]['category'] == 'road_sign':
                labels[mask_idx] = 3
            if objects[i]['category'] in ['slot_line', 'lane']:
                labels[mask_idx] = 4
            if objects[i]['category']=='ego':
                ego_mask_idx = shape_to_mask(img.size, points)
                ego_flag = False
        if ego_flag:
            print(seg_path)
            object_pts = [[444.19,342.6694],[582.0035,342.926],[582.3517,685.95],[443.98, 686.34]]
            points = [[min(item[0]*scale,input_size[0]-1), min(item[1]*scale,input_size[1]-1)] for item in object_pts]
            mask_idx = shape_to_mask(img.size, points)
            labels[mask_idx] = 0
        else:
            labels[ego_mask_idx] = 0
        ### load pld gt
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
        ### data augmentation
        if self.mode == 'train':
            ## Brightness adjustment
            r1 = random.randrange(40, 140) / 100
            img = (np.array(img) * r1).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            ## flip
            if random.random() < 0.5:
                img, labels, pld_pts = flip_data(img, labels, pld_pts)
            ## rotate
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
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        if self.transform is not None:
            img = self.transform(img)

        data_samples = dict(fs_labels=labels, pld_cls=pld_cls, pld_pts=pld_pts,
                            img_path=img_path, mask_path=mask_path, label_path=label_path)
        return img, data_samples

    def __len__(self):
        return len(self.images)
    
from mmengine.registry import FUNCTIONS
@FUNCTIONS.register_module()
def park_collate(data_batch):
    imgs = torch.cat([item[0].unsqueeze(0) for item in data_batch], dim=0)
    fs_labels = torch.cat([item[1]['fs_labels'].unsqueeze(0) for item in data_batch], dim=0)
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
    data_samples = dict(fs_labels=fs_labels, pld_cls=pld_cls, pld_pts=pld_pts,
                            img_path=img_path, mask_path=mask_path, label_path=label_path)

    return imgs, data_samples