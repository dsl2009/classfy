from model.native_iv3_center import inception_v3
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
from torch.optim import lr_scheduler
from pretrainedmodels import inceptionv4
from losses.center_loss import CenterLoss
from focal_loss import FocalLoss

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((448,598)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(598),
        transforms.CenterCrop(598),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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


def run(trainr,name,cls_num,idx):

    imagenet_data = ImageFolder(trainr,
                                transform=data_transforms['train'])
    data_loader = DataLoader(imagenet_data, batch_size=6, shuffle=True)
    model = inception_v3(num_classes=1000,pretrained=None,aux_logits=False)
    model.load_state_dict(torch.load('D:/deep_learn_data/check/inception_v3_google-1a9a5a14.pth'), strict=False)
    model.fc1 = nn.Linear(2048, 4)
    model.fc2 = nn.Linear(4, 12)
    #model.load_state_dict(torch.load('D:/deep_learn_data/luntai/check/1.pth'), strict=False)
    model.cuda()
    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    state['label_ix'] = imagenet_data.class_to_idx
    state['cls_name'] = name
    centerloss = CenterLoss(cls_num, 4)
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
            data.detach().numpy()
            if data.size(0) != 6:
                break
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            f1, output = model(data)
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            optimizer.zero_grad()
            optimzer_center.zero_grad()

            loss = focal_loss(output, target)+ centerloss(target, f1)*0.3
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
        visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch, cls_num)
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
        if best_accuracy == 1.0 or state['train_accuracy']>0.99:
            break

if __name__ == '__main__':
    train_dr = 'D:/deep_learn_data/luntai/guangdong_round1_train2_20180916/guangdong_round1_train2_20180916/瑕疵样本'
    run(train_dr,'luntai',12, 'iv3')
