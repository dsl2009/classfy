from model.resnet_18 import resnet18
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.nn import functional as F
from progressbar import *
from matplotlib import pyplot as plt
import numpy as np
import json
from losses.center_loss import CenterLoss
from torch.optim import lr_scheduler
from focal_loss import FocalLoss
from PIL import ImageFile
import glob
from chan_data_loader import get_batch,Dataset
import os
from torch import optim
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def run(train_sets,valid_sets, cls_num,idx):
    batch_size = 16
    train_gen = get_batch(batch_size= batch_size,data_set=train_sets, image_size=train_sets.image_size)
    valid_gen = get_batch(batch_size= 1,data_set=valid_sets, image_size=train_sets.image_size)
    model = resnet18(num_classes=1000,pretrained=None)
    model.load_state_dict(torch.load('/home/dsl/all_check/resnet18-5c106cde.pth'), strict=False)
    model.fc = nn.Linear(512,1)
    model.cuda()
    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    state['label_ix'] = train_sets.cls_map
    state['cls_name'] = idx
    centerloss = CenterLoss(cls_num,2)
    centerloss.cuda()
    optimzer_center = torch.optim.SGD(centerloss.parameters(), lr=0.3)
    state['best_accuracy'] = 0
    sch = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    ll = train_sets.len()
    focal_loss = FocalLoss(gamma=2)
    focal_loss.cuda()

    bc_loss = nn.BCEWithLogitsLoss()

    def train():
        model.train()
        loss_avg = 0.0
        progress = ProgressBar()
        ip1_loader = []
        idx_loader = []
        correct = 0
        for b in range(int(train_sets.len()/batch_size)):
            images, labels = next(train_gen)
            images = np.transpose(images,[0,3,1,2])
            images = torch.from_numpy(images)
            labels = torch.from_numpy(labels).float()
            data, target = torch.autograd.Variable(images.cuda()), torch.autograd.Variable(labels.cuda())
            output = model(data)

            output = output.squeeze()
            ot = F.sigmoid(output)
            pred = ot.ge(0.5).float()

            correct += float(pred.eq(target.data).sum())

            optimizer.zero_grad()

            loss = bc_loss(output, target)
            loss.backward()
            optimizer.step()



            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            print(correct, ll, loss_avg)
        state['train_accuracy'] = correct / train_sets.len()
        state['train_loss'] = loss_avg
    def test():
        with torch.no_grad():
            model.eval()
            loss_avg = 0.0
            correct = 0
            for i in range(valid_sets.len()):
                images, labels = next(valid_gen)
                images = np.transpose(images, [0,3,1,2])
                images = torch.from_numpy(images)
                labels = torch.from_numpy(labels).float()
                data, target = torch.autograd.Variable(images.cuda()), torch.autograd.Variable(labels.cuda())
                output = model(data)
                output = output.squeeze()
                ot = F.sigmoid(output)
                pred = ot.ge(0.5).float()
                correct += float(pred.eq(target.data).sum())
                state['test_accuracy'] = correct / valid_sets.len()
            print(state['test_accuracy'])

    best_accuracy = 0.0
    for epoch in range(40):
        state['epoch'] = epoch
        train()
        test()
        sch.step(state['train_accuracy'])
        best_accuracy = (state['train_accuracy']+state['test_accuracy'])/2

        if best_accuracy > state['best_accuracy']:
            state['best_accuracy'] =  best_accuracy
            torch.save(model.state_dict(), os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/be', idx+'.pth'))
            with open(os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/be', idx+'.json'),'w') as f:
                f.write(json.dumps(state))
                f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])

        if state['train_accuracy']-state['test_accuracy']>0.06 and epoch>30:
            break


def run_all(train_dr,test_dr):
    save_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/be'
    image_size = [512, 512]
    train_set = Dataset(root=train_dr, image_size=image_size)
    valid_set = Dataset(root=test_dr, image_size=image_size, is_trans=False, cls_map=train_set.cls_map)
    num_cls = len(train_set.cls_map)
    name = train_dr.split('/')[-1]
    print(num_cls, train_set.cls_map)
    if os.path.exists(os.path.join(save_dr, name+'.pth')):
        pass
    run(train_set, valid_set, num_cls, name)

if __name__ == '__main__':

    for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/pytorch/step3/train/*/*'):
        print(x)
        traindr = x
        validdr = x.replace('train','valid')
        run_all(traindr, validdr)

