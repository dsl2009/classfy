from model.native_senet import se_resnext50_32x4d
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch
import glob
from progressbar import *
import shutil
from PIL import Image
import json
trans = transforms.Compose([
        transforms.Resize(448),
        transforms.RandomCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

js = json.loads(open('log/step24.json').read())
lbs = js['label_ix']
new_lbs = dict()
for x in lbs:
    new_lbs[lbs[x]] = x
print(new_lbs)
pred_path = 'D:/deep_learn_data/luntai/pred/total1'
#base_dr = 'D:/deep_learn_data/luntai/guangdong_round1_test_a_20180916/guangdong_round1_test_a_20180916'
base_dr = 'D:/deep_learn_data/luntai/crop/crop'
model = se_resnext50_32x4d(num_classes=1000, pretrained=None)
model.last_linear = nn.Linear(2048, 12)
model.load_state_dict(torch.load('log/step24.pth'), strict=False)
model.cuda()
model.eval()
for k in glob.glob(os.path.join(base_dr,'*.*')):
    imag = Image.open(k)
    score = []
    for x in range(15):
        ig = trans(imag)
        ig = ig.unsqueeze(0)
        ig = torch.autograd.Variable(ig.cuda())
        output = model(ig)
        output = F.softmax(output,dim=1)
        pred = output.data.squeeze(dim=0).cpu().numpy()
        score.append(pred)
    score = np.asarray(score)
    score = np.sum(score,axis=0)/15
    print(score)

    pred_lb = np.argmax(score)
    print(pred_lb)
    new_path = os.path.join(pred_path, new_lbs[pred_lb])
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    shutil.copy(k, new_path)







