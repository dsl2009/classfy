#coding=utf-8
from model.native_senet_c import se_resnext50_32x4d
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
import glob
from progressbar import *
import shutil
from torch.nn import functional as F
from PIL import Image
import json
import numpy as np
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
trans = transforms.Compose([
        transforms.Resize(size=896),
        transforms.CenterCrop(896),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

js = json.loads(open('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/check/round2.json').read())
lbs = js['label_ix']
new_lbs = dict()
for x in lbs:
    new_lbs[lbs[x]] = x
print(new_lbs)
pred_path = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/testb_round/step2'
base_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/guangdong_round2_test_b_20181106'
#base_dr = 'D:/deep_learn_data/luntai/crop/crop'
model = se_resnext50_32x4d(num_classes=1000, pretrained=None)
model.fc1 = nn.Linear(2048, 2)
model.fc2 = nn.Linear(2, 2)
model.load_state_dict(torch.load('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/check/round2.pth'), strict=False)
model.cuda()
model.eval()
for k in glob.glob(os.path.join(base_dr,'*.*')):
    imag = Image.open(k)
    score = []
    for x in range(1):
        ig = trans(imag)
        ig = ig.unsqueeze(0)
        ig = torch.autograd.Variable(ig.cuda())
        fc1, output = model(ig)
        output = F.softmax(output,dim=1)
        pred = output.data.squeeze(dim=0).cpu().numpy()
        score.append(pred)
    score = np.asarray(score)
    score = np.sum(score,axis=0)/1
    print(score)

    pred_lb = np.argmax(score)
    sc = np.max(score)
    pred_label = new_lbs[pred_lb]
    name = k.replace('\\', '/').split('/')[-1]

    new_path = os.path.join(pred_path, pred_label)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    shutil.copy(k, new_path + '/' +  name)









