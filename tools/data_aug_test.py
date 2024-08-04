import os
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from PIL import Image, ImageDraw
import cv2
import json
import random
import math


pt_list = [[1,2],[3,4],[5,6]]
pt_list1 = pt_list[::-1]

input_size = (512, 512)
input_size = (256, 256)
img_path = r'F:\data\Boden_AVM_000000_001999\images\images\000793.jpg'
img = Image.open(img_path).convert('RGB')
img = img.resize(input_size)

# img.show()
img_flip = cv2.flip(np.array(img), 0)
img_flip = Image.fromarray(img_flip)
# img_flip.show()
labels = np.zeros(input_size, dtype=np.uint8)
labels[0,0] = 1
labels[1,1] = 2
labels[1,2] = 3
labels_flip = np.flip(labels, axis=0)

# img.show()
# r1 = random.randrange(40, 140) / 100
# img_dark = (np.array(img) * 0.5).clip(0, 255).astype(np.uint8)
# img_dark = Image.fromarray(img_dark)
# img_dark.show()



# labels = np.zeros(input_size, dtype=np.int64)
labels = np.zeros(input_size, dtype=np.uint8)
labels[0,0] = 1
labels[1,1] = 2
labels[1,2] = 3
# img.show()
### data aug
img = np.array(img)
angle = 90
rotation_matrix = cv2.getRotationMatrix2D((input_size[0]/2, input_size[1]/2), angle, 1)
rotated = cv2.warpAffine(img, rotation_matrix, (input_size))
rotated_img = Image.fromarray(rotated)
# img.show()
point = np.array([10,20])
labels1 = cv2.warpAffine(labels, rotation_matrix, input_size)

angle_rad = math.pi * angle / 180
angle_sin = math.sin(angle_rad)
angle_cos = math.cos(angle_rad)

pt = [1, 1]
pt_x = pt[0] - (input_size[0] /2)
pt_y = pt[1] - (input_size[1] /2)
x = pt_x * angle_cos + pt_y * angle_sin
y = -pt_x * angle_sin + pt_y * angle_cos
x = x + (input_size[0] /2)
y = y + (input_size[1] /2)

aug_probel = random.random()
angle_probel = random.random()

debug = 0