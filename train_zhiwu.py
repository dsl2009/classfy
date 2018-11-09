from model.native_senet_c import se_resnext50_32x4d
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
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def show(data):
    ig = data.numpy()
    ig = ig[0]
    ig = np.transpose(ig, [1, 2, 0])
    ig = ig / 2 + 0.5
    plt.imshow(ig)
    plt.show()
def visualize(feat, labels, epoch, nums):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999', ]
    plt.clf()
    for i in range(nums):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i%10])
    plt.legend(['0', '1'], loc = 'upper right')
    plt.xlim(xmin=-5,xmax=5)
    plt.ylim(ymin=-5,ymax=5)
    plt.text(-4.8,4.6,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)


def run(trainr,test_dr, name,svdr):
    print(test_dr)
    batch_size = 16
    imagenet_data = ImageFolder(trainr,
                                transform=data_transforms['train'])

    test_data = ImageFolder(test_dr,transform=data_transforms['val'])
    cls_num = len(imagenet_data.class_to_idx)

    data_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    model = se_resnext50_32x4d(num_classes=1000,pretrained=None)
    model.load_state_dict(torch.load('/home/dsl/all_check/se_resnext50_32x4d-a260b3a4.pth'), strict=False)
    model.fc1 = nn.Linear(2048, 2)
    model.fc2 = nn.Linear(2, cls_num)
    model.cuda()
    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    state['label_ix'] = imagenet_data.class_to_idx
    state['cls_name'] = name
    centerloss = CenterLoss(cls_num,2)
    centerloss.cuda()
    optimzer_center = torch.optim.SGD(centerloss.parameters(), lr=0.3)
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

            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            f1, output = model(data)
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            optimizer.zero_grad()
            optimzer_center.zero_grad()

            loss = focal_loss(output, target)+ centerloss(target, f1)*0.4
            loss.backward()
            optimizer.step()
            optimzer_center.step()

            ip1_loader.append(f1)
            idx_loader.append((target))

            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            print(correct, ll, loss_avg)

        state['train_accuracy'] = correct / len(data_loader.dataset)
        feat = torch.cat(ip1_loader, 0)
        labels = torch.cat(idx_loader, 0)
        state['train_loss'] = loss_avg
    def test():
        with torch.no_grad():
            model.eval()
            loss_avg = 0.0
            correct = 0
            print(test_data_loader)
            for batch_idx, (data, target) in enumerate(test_data_loader):

                data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
                f1, output = model(data)
                loss = F.cross_entropy(output, target)
                pred = output.data.max(1)[1]
                correct += float(pred.eq(target.data).sum())
                loss_avg += float(loss)
                state['test_loss'] = loss_avg / len(test_data_loader)
                state['test_accuracy'] = correct / len(test_data_loader.dataset)
            print(state['test_accuracy'])

    best_accuracy = 0.0
    for epoch in range(50):
        state['epoch'] = epoch
        train()
        test()
        sch.step(state['train_accuracy'])

        best_accuracy = (state['train_accuracy'] + state['test_accuracy']) / 2

        if best_accuracy > state['best_accuracy']:
            state['best_accuracy'] = best_accuracy
            torch.save(model.state_dict(), os.path.join(svdr, name + '.pth'))
            with open(os.path.join(svdr, name + '.json'), 'w') as f:
                f.write(json.dumps(state))
                f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])
        if best_accuracy == 1.0:
            break

def run_all(train_dr,test_dr):
    save_dr = 'log'

    name = train_dr.split('/')[-1]
    if not os.path.exists(save_dr):
        os.makedirs(save_dr)
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
    for x in glob.glob('/data/pytorch/step3/train/*/*'):
        print(x)
        traindr = x
        validdr = x.replace('train','valid')
        d.append(x.split('/')[-1])
        run_all(traindr, validdr)

    traindr ='/data/pytorch/step1/train'
    validdr = traindr.replace('train','valid')
    run_all(traindr, validdr)