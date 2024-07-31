import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

import sys
sys.path.append(".")
from projects.models.res18_fpn import ParkModel
# from projects.models.res50_deeplabv3 import ParkModel


task_name = 'pld'
model = ParkModel(task_name)
params_dict = torch.load(r'G:\dnn\mmengine\checkpoints\pld\epoch_50.pth')
model.load_state_dict(params_dict['state_dict'])

input_size = (512, 512)
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
pred = pred[0]
draw = ImageDraw.Draw(img)
scale = torch.zeros(4)
scale[0::2] = input_size[0]
scale[1::2] = input_size[1]
for i in range(pred['cls_pred'].shape[0]):
    pt = pred['pts_pred'][i, [0,1]] * scale[0:2]
    # draw.point(pt.tolist(), fill='red')
    line1 = pred['pts_pred'][i, 0:4] * scale
    line2 = pred['pts_pred'][i, [0,1,4,5]] * scale
    width = 0
    cls_text = 'occupied'
    if pred['cls_pred'][i]==2:
        width = 5
        cls_text = 'empty'
    draw.line(line1.tolist(), fill='red', width=width)
    draw.line(line2.tolist(), fill='green', width=width)
    draw.text(pt.tolist(), cls_text, fill='red')

img.show()



debug = 0
