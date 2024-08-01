import numpy as np
import torch
import torchvision.transforms as transforms
import os
from PIL import Image, ImageDraw
import cv2
import json
from labelme.utils import shape_to_mask

input_size = (512, 512)
# input_size = (256, 256)
scale = 2

data_root = r'G:\dnn\data\Boden_AVM_002000_019999\test'
image_files = sorted(os.listdir(os.path.join(data_root, 'images')))
label_files = sorted(os.listdir(os.path.join(data_root, 'labels')))
mask_files = sorted(os.listdir(os.path.join(data_root, 'fs_labels')))
for id in range(len(image_files)):
    img_file = os.path.join(data_root, 'images', image_files[id])
    label_file = os.path.join(data_root, 'labels', label_files[id])
    mask_file = os.path.join(data_root, 'fs_labels', mask_files[id])

    img = Image.open(img_file).convert('RGB')
    img = img.resize(input_size)
    draw = ImageDraw.Draw(img)

    with open(mask_file, 'r', encoding='utf-8') as file:
        mask_dct = json.load(file)
    objects = mask_dct['objects'][0]
    points = [[min(item[0]/scale,input_size[0]-1), min(item[1]/scale,input_size[1]-1)] for item in objects['point_list']]
    mask_idx = shape_to_mask(img.size, points)
    mask_data = np.zeros_like(img)
    mask_data[mask_idx,:] = np.ones(3) * 255
    mask_show = Image.fromarray(mask_data)
    for i in range(len(objects['point_list'])-1):
        line = objects['point_list'][i] + objects['point_list'][i+1]
        line = [min(item/scale,input_size[0]-1) for item in line]
        draw.line(line, fill='white', width=2)
    # img.show()
    # mask_show.show()
    debug = 0

    with open(label_file, 'r', encoding='utf-8') as file:
        label_dct = json.load(file)
    slots = label_dct['objects']
    for i in range(len(slots)):
        pt = [slots[i]['point_list'][3][0]/scale, slots[i]['point_list'][3][1]/scale]
        line1 = slots[i]['point_list'][3] + slots[i]['point_list'][0]
        line1 = [item/scale for item in line1]
        line2 = slots[i]['point_list'][3] + slots[i]['point_list'][2]
        line2 = [item/scale for item in line2]
        line3 = slots[i]['point_list'][0] + slots[i]['point_list'][1]
        line3 = [item/scale for item in line3]
        width = 0
        cls_text = 'occupied'
        if slots[i]['attributes']['status'] == 'empty':
            width = 5
            cls_text = 'empty'
        draw.line(line1, fill='red', width=width)
        draw.line(line2, fill='green', width=width)
        draw.line(line3, fill='blue', width=width)
        draw.text(pt, cls_text, fill='red')

    # im = Image.blend(img, mask_show, 0.8)
    img.show()

    debug = 0