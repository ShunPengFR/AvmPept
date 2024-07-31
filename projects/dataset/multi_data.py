import os
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from PIL import Image, ImageDraw
import json
import csv
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
        with open(label_path, 'r', encoding='utf-8') as file:
            label_dct = json.load(file)
        slots = label_dct['objects']

        ori_size = img.size
        input_size = (512, 512)
        input_size = (256, 256)
        scale = input_size[0] / ori_size[0]
        img = img.resize(input_size)
        mask = mask.resize(input_size, Image.NEAREST)
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

        # Convert the RGB values to class indices
        mask = np.array(mask)
        mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
        fs_labels = np.zeros_like(mask, dtype=np.int64)
        for color, class_index in self.color_to_class.items():
            rgb = color[0] * 65536 + color[1] * 256 + color[2]
            fs_labels[mask == rgb] = class_index

        # labels = labels>=2
        if self.target_transform is not None:
            fs_labels = self.target_transform(fs_labels)

        data_samples = dict(fs_labels=fs_labels, pld_cls=pld_cls, pld_pts=pld_pts,
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