from focal_loss import FocalLoss
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.nn import functional as F
from progressbar import *
from matplotlib import pyplot as plt
import numpy as np
import time
import json
from torch.optim import lr_scheduler
from pretrainedmodels import inceptionv4
from model.native_senet import se_resnext50_32x4d
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(384),
        transforms.RandomCrop(384),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])}

def run(trainr,name,cls_num,idx):

    imagenet_data = ImageFolder(trainr,
                                transform=data_transforms['train'])
    data_loader = DataLoader(imagenet_data, batch_size=6, shuffle=True)
    model = inceptionv4(num_classes=1000,pretrained=None)
    model.avg_pool = nn.AvgPool2d(13, count_include_pad=False)
    model.load_state_dict(torch.load('D:/deep_learn_data/check/se_resnext50_32x4d-a260b3a4.pth'), strict=False)
    model.last_linear = nn.Linear(1536, cls_num)

    model.cuda()
    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    state['label_ix'] = imagenet_data.class_to_idx
    state['cls_name'] = name

    state['best_accuracy'] = 0
    sch = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    ll = len(data_loader.dataset)
    focal_loss = FocalLoss(gamma=2)
    focal_loss.cuda()
    def train():
        model.train()
        loss_avg = 0.0
        progress = ProgressBar()
        ip1_loader = []
        idx_loader = []
        correct = 0
        for (data, target) in progress(data_loader):
            data.detach().numpy()
            if data.size(0) != 6:
                break
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            output = model(data)
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())
            optimizer.zero_grad()
            loss = focal_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            print(correct, ll, loss_avg)

        state['train_accuracy'] = correct / len(data_loader.dataset)

        state['train_loss'] = loss_avg

    best_accuracy = 0.0
    for epoch in range(100):
        state['epoch'] = epoch
        train()
        sch.step(state['train_accuracy'])

        if best_accuracy < state['train_accuracy']:
            state['best_accuracy'] =  state['train_accuracy']
            torch.save(model.state_dict(), os.path.join('./log', idx+'.pth'))
        with open(os.path.join('./log', idx+'.json'),'w') as f:
            f.write(json.dumps(state))
            f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])
        if best_accuracy == 1.0 or state['train_accuracy']>0.98:
            break

if __name__ == '__main__':
    train_dr = 'D:/deep_learn_data/luntai/train/瑕疵样本'
    run(train_dr,'luntai',12, 'step24')
