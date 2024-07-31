import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from projects.models.res18_fpn import ParkModel
# from projects.models.res50_deeplabv3 import ParkModel


task_name = 'multi_task'
model = ParkModel(task_name)
params_dict = torch.load(r'G:\dnn\mmengine\checkpoints\multi\epoch_40.pth')
model.load_state_dict(params_dict['state_dict'])

input_size = (512, 512)
# input_size = (256, 256)
norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(**norm_cfg)
                                ])


img_path = r'G:\dnn\data\Boden_AVM_002000_019999\test\images\016000.jpg'
img = Image.open(img_path).convert('RGB')
img = img.resize(input_size)
imgs = transform(img).unsqueeze(0)

model.eval()
pred = model(imgs)
pld = pred[0][0]
draw = ImageDraw.Draw(img)
scale = torch.zeros(4)
scale[0::2] = input_size[0]
scale[1::2] = input_size[1]
for i in range(pld['cls_pred'].shape[0]):
    pt = pld['pts_pred'][i, [0,1]] * scale[0:2]
    # draw.point(pt.tolist(), fill='red')
    line1 = pld['pts_pred'][i, 0:4] * scale
    line2 = pld['pts_pred'][i, [0,1,4,5]] * scale
    width = 0
    cls_text = 'occupied'
    if pld['cls_pred'][i]==2:
        width = 5
        cls_text = 'empty'
    draw.line(line1.tolist(), fill='red', width=width)
    draw.line(line2.tolist(), fill='green', width=width)
    draw.text(pt.tolist(), cls_text, fill='red')

seg = pred[1]
seg = torch.argmax(seg, dim=1)
# pred = pred.repeat(3, 1, 1).to(torch.float32)
seg_show = torch.zeros(3, seg.shape[1], seg.shape[2])
for i in range(seg.shape[1]):
    for j in range(seg.shape[2]):
        if seg[0,i,j]==0:
            continue
        for [i_tmp, j_tmp] in [[i-1,j],[i,j-1],[i+1,j],[i,j+1]]:
            if (i_tmp<0 or i_tmp>=seg.shape[1] or j_tmp<0 or j_tmp>=seg.shape[2]):
                break
            if seg[0,i_tmp,j_tmp]==0:
                seg_show[:,i,j]=torch.ones(3)
                break
trans_pil = transforms.ToPILImage()
seg_pred = trans_pil(seg_show)

# img_pred.show()
im = Image.blend(img, seg_pred, 0.3)
im.show()

# new_img = Image.new("RGB", (img.width + im.width, img.height))
# new_img.paste(img, (0, 0))
# new_img.paste(im, (img.width, 0))
# new_img.show()




debug = 0
