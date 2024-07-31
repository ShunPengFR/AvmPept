import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from projects.models.res18_fpn import ParkModel
# from projects.models.res50_deeplabv3 import ParkModel


task_name = 'freespace'
model = ParkModel(task_name)
params_dict = torch.load(r'G:\dnn\mmengine\checkpoints\fs\epoch_10.pth')
# params_dict = torch.load(r'G:\dnn\mmengine\checkpoints\fs_res50\epoch_10.pth')
model.load_state_dict(params_dict['state_dict'])

input_size = (512, 512)
norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(**norm_cfg)
                                ])


img_path = r'G:\dnn\data\Boden_AVM_002000_019999\train\images\015000.jpg'
img = Image.open(img_path).convert('RGB')
img = img.resize(input_size)
imgs = transform(img).unsqueeze(0)

model.eval()
pred = model(imgs)
pred = torch.argmax(pred, dim=1)
# pred = pred.repeat(3, 1, 1).to(torch.float32)
pred_show = torch.zeros(3, pred.shape[1], pred.shape[2])
for i in range(pred.shape[1]):
    for j in range(pred.shape[2]):
        if pred[0,i,j]==0:
            continue
        for [i_tmp, j_tmp] in [[i-1,j],[i,j-1],[i+1,j],[i,j+1]]:
            if (i_tmp<0 or i_tmp>=pred.shape[1] or j_tmp<0 or j_tmp>=pred.shape[2]):
                break
            if pred[0,i_tmp,j_tmp]==0:
                pred_show[:,i,j]=torch.ones(3)
                break
trans_pil = transforms.ToPILImage()
img_pred = trans_pil(pred_show)

# img_pred.show()
im = Image.blend(img, img_pred, 0.3)
# im.show()

new_img = Image.new("RGB", (img.width + im.width, img.height))
new_img.paste(img, (0, 0))
new_img.paste(im, (img.width, 0))
new_img.show()




debug = 0
