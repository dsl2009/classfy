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
from dsl_data.utils import resize_image_fixed_size
ImageFile.LOAD_TRUNCATED_IMAGES = True
trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def run(json_pth, save_pth, image_pth ,check_dr):
    js = json.loads(open(json_pth).read())
    lbses = js['label_ix']
    new_lbs = dict()
    for x in lbses:
        new_lbs[lbses[x]] = x
    print(new_lbs)

    base_dr = image_pth
    # base_dr = 'D:/deep_learn_data/luntai/crop/crop'
    model = se_resnext50_32x4d(num_classes=1000, pretrained=None)
    model.fc1 = nn.Linear(2048, 2)
    model.fc2 = nn.Linear(2, len(new_lbs))
    model.load_state_dict(torch.load(check_dr),
                          strict=False)
    model.cuda()
    model.eval()
    for k1 in glob.glob(os.path.join(base_dr, '*.*')):
        imag = Image.open(k1)
        ig = np.asarray(imag)
        img, _, _, _, _ = resize_image_fixed_size(ig, [512,512])
        ig = Image.fromarray(img)
        ig = trans(ig)
        ig = ig.unsqueeze(0)

        ig = torch.autograd.Variable(ig.cuda())
        fc1, output = model(ig)
        output = F.softmax(output, dim=1)
        pred = output.data.squeeze(dim=0).cpu().numpy()


        pred_lb = np.argmax(pred)
        sc = np.max(pred)
        pred_label = new_lbs[pred_lb]
        name = k.replace('\\', '/').split('/')[-1]

        new_path = os.path.join(save_pth, pred_label)

        if not os.path.exists(new_path):
            os.makedirs(new_path)
        shutil.copy(k1, new_path )

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


lbs = ['樱桃白粉病', '辣椒疮痂病', '玉米锈病', '玉米叶斑病', '玉米灰斑病', '桃子疮痂病', '苹果黑星病', '苹果雪松锈病',
       '马铃薯晚疫病', '马铃薯早疫病', '柑桔黄龙病', '番茄斑点病', '番茄早疫病', '番茄叶霉病', '番茄黄化曲叶病毒病',
       '番茄白粉病', '番茄红蜘蛛损伤', '番茄斑枯病', '番茄晚疫病菌', '草莓叶枯病', '葡萄褐斑病', '葡萄黑腐病', '葡萄轮斑病']

for k in lbs:
    js_pth3 = os.path.join(check_dirs,k+'.json')
    check_dr3 = os.path.join(check_dirs, k + '.pth')
    img_pth3 = os.path.join(pred_path,'step2',k)
    save_pth3 = os.path.join(pred_path,'step3')
    run(js_pth3, save_pth3, img_pth3, check_dr3)

