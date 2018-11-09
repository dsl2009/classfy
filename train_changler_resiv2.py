from pretrainedmodels import inceptionresnetv2
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
ImageFile.LOAD_TRUNCATED_IMAGES = True
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=299),
        transforms.CenterCrop(size=299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


def run(train_sets,valid_sets, idx, save_dr):
    batch_size = 8
    imagenet_data = ImageFolder(train_sets,
                                transform=data_transforms['train'])
    test_data = ImageFolder(valid_sets,
                            transform=data_transforms['val'])
    data_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    cls_num = len(imagenet_data.class_to_idx)
    model = inceptionresnetv2(num_classes=1001,pretrained=None)
    model.load_state_dict(torch.load('/home/dsl/all_check/inceptionresnetv2-520b38e4.pth'), strict=True)
    model.last_linear =  nn.Linear(1536, cls_num)
    model.cuda()
    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    #optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                #weight_decay=state['decay'], nesterov=True)

    optimizer = torch.optim.Adam(model.parameters(), state['learning_rate'],
                                weight_decay=state['decay'],amsgrad=True)

    state['label_ix'] = imagenet_data.class_to_idx
    state['cls_name'] = idx

    state['best_accuracy'] = 0
    sch = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.9, patience=3)

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
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            output = model(data)
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())
            optimizer.zero_grad()
            loss = focal_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            print(correct, len(data_loader.dataset), loss_avg)
        state['train_accuracy'] = correct / len(data_loader.dataset)
        state['train_loss'] = loss_avg
    def test():
        with torch.no_grad():
            model.eval()
            loss_avg = 0.0
            correct = 0
            for (data, target) in test_data_loader:

                data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
                output = model(data)
                loss = F.cross_entropy(output, target)
                pred = output.data.max(1)[1]
                correct += float(pred.eq(target.data).sum())
                loss_avg += float(loss)
                state['test_loss'] = loss_avg / len(test_data_loader.dataset)
                state['test_accuracy'] = correct / len(test_data_loader.dataset)
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
            torch.save(model.state_dict(), os.path.join(save_dr, idx+'.pth'))
            with open(os.path.join(save_dr, idx+'.json'),'w') as f:
                f.write(json.dumps(state))
                f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])

        if state['test_accuracy']==1 and epoch>10:
            break


def run_all(train_dr,test_dr):
    save_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/log_resv2'

    name = train_dr.split('/')[-1]

    if os.path.exists(os.path.join(save_dr, name+'.pth')):
        return
    run(train_dr, test_dr,  name, save_dr)

if __name__ == '__main__':

    for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/pytorch/step2/train/*'):
        print(x)
        traindr = x
        validdr = x.replace('train','valid')

        run_all(traindr, validdr)

    d = []
    for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/pytorch/step3/train/*/*'):
        print(x)
        traindr = x
        validdr = x.replace('train','valid')
        d.append(x.split('/')[-1])
        run_all(traindr, validdr)

    traindr ='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/pytorch/step1/train'
    validdr = traindr.replace('train','valid')
    run_all(traindr, validdr)

