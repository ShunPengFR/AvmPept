import os
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from PIL import Image, ImageDraw
import cv2
import json
import random
import math


def rotate_data(img, labels, pld_pts, angle):
    input_size = img.size
    img = np.array(img)
    rotation_matrix = cv2.getRotationMatrix2D((input_size[0]/2, input_size[1]/2), angle, 1)
    rotated = cv2.warpAffine(img, rotation_matrix, (input_size))
    rotated = Image.fromarray(rotated)
    labels = cv2.warpAffine(labels, rotation_matrix, input_size)

    # angle_rad = math.pi * angle_degree / 180
    #         xval = vector[0]*math.cos(angle_rad) + \
    #             vector[1]*math.sin(angle_rad)
    #         yval = -vector[0]*math.sin(angle_rad) + \
    #             vector[1]*math.cos(angle_rad)
    angle_rad = math.pi * angle / 180
    angle_sin = math.sin(angle_rad)
    angle_cos = math.cos(angle_rad)

    pld_pts_out = []
    for i in range(len(pld_pts)):
        pts_out = []
        for pt in pld_pts[i]:
            pt_x = pt[0] - (input_size[0] / 2)
            pt_y = pt[1] - (input_size[1] / 2)
            x = pt_x * angle_cos + pt_y * angle_sin
            y = -pt_x * angle_sin + pt_y * angle_cos
            x = x + (input_size[0] / 2)
            y = y + (input_size[1] / 2)
            pts_out.append([x, y])
        pld_pts_out.append(pts_out)

    return rotated, labels, pld_pts_out

def flip_data(img, labels, pld_pts):
    input_size = img.size
    labels_flip = labels
    pld_pts_out = pld_pts
    if random.random() < 0.5:
        img_flip = cv2.flip(np.array(img), 0)
        img = Image.fromarray(img_flip)
        labels_flip = np.flip(labels, axis=0)
        pld_pts_out = []
        for i in range(len(pld_pts)):
            pts_out = []
            for pt in pld_pts[i]:
                pt[1] = input_size[1] - 1 - pt[1]
                pt[1] = max(0, pt[1])
                pt[1] = min(input_size[1]-1, pt[1])
                pts_out.append(pt)
            pts_out = pts_out[::-1]
            pld_pts_out.append(pts_out)
    if random.random() < 0.5:
        img_flip = cv2.flip(np.array(img), 1)
        img = Image.fromarray(img_flip)
        labels_flip = np.flip(labels, axis=1)
        pld_pts_out = []
        for i in range(len(pld_pts)):
            pts_out = []
            for pt in pld_pts[i]:
                pt[0] = input_size[0] - 1 - pt[0]
                pt[0] = max(0, pt[0])
                pt[0] = min(input_size[0]-1, pt[0])
                pts_out.append(pt)
            pts_out = pts_out[::-1]
            pld_pts_out.append(pts_out)

    return img, labels_flip, pld_pts_out