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
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def run(json_pth, save_pth, img_pth, check_dr ):

    js = json.loads(open(json_pth).read())
    lbses = js['label_ix']
    new_lbs = dict()
    for x in lbses:
        new_lbs[lbses[x]] = x

    pred_pathes = save_pth
    base_dr = img_pth
    model = se_resnext50_32x4d(num_classes=1000, pretrained=None)
    model.fc1 = nn.Linear(2048, 2)
    model.fc2 = nn.Linear(2, len(new_lbs))
    model.load_state_dict(torch.load(check_dr),
                          strict=False)
    model.cuda()
    model.eval()
    for ip in glob.glob(os.path.join(base_dr, '*.*')):
        imag = Image.open(ip)
        score = []
        for x in range(1):
            ig = trans(imag)
            ig = ig.unsqueeze(0)
            ig = torch.autograd.Variable(ig.cuda())
            fc1, output = model(ig)
            output = F.softmax(output, dim=1)
            pred = output.data.squeeze(dim=0).cpu().numpy()
            score.append(pred)
        score = np.asarray(score)
        score = np.sum(score, axis=0) / 1

        pred_lb = np.argmax(score)
        sc = np.max(score)
        pred_label = new_lbs[pred_lb]
        name = ip.replace('\\', '/').split('/')[-1]

        new_path = os.path.join(pred_pathes, pred_label)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        shutil.copy(ip, new_path + '/' + name)


pred_path = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/pred'
check_dirs = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/log'
lbs = ['train']
for k in lbs:
    js_pth1 = os.path.join(check_dirs,k+'.json')
    check_dr1 = os.path.join(check_dirs, k + '.pth')
    img_pth1 = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/test/test/images'
    save_pth1 = os.path.join(pred_path,'step1')
    run(js_pth1, save_pth1, img_pth1, check_dr1)
lbs = ['樱桃', '辣椒', '玉米', '桃子', '苹果', '马铃薯', '柑桔', '番茄', '草莓', '葡萄']

for k in lbs:
    js_pth2 = os.path.join(check_dirs,k+'.json')
    check_dr2 = os.path.join(check_dirs, k + '.pth')
    img_pth2 = os.path.join(pred_path,'step1',k)
    save_pth2 = os.path.join(pred_path,'step2')
    run(js_pth2, save_pth2, img_pth2, check_dr2)






