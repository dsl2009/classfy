from model.native_iv4 import inceptionv4
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
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(448, 598)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(448, 598)),
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
    test_data = ImageFolder('D:/deep_learn_data/luntai/pred/other',
                                transform=data_transforms['val'])
    data_loader = DataLoader(imagenet_data, batch_size=6, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=6, shuffle=True)
    model = inceptionv4(num_classes=1001, pretrained=None)
    #model.load_state_dict(torch.load('D:/deep_learn_data/check/inceptionv4-8e4777a0.pth'), strict=False)
    model.last_linear = nn.Linear(1536, cls_num)
    #model.fc2 = nn.Linear(4, cls_num)
    model.load_state_dict(torch.load('log/1006_iv_other.pth'), strict=False)
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
            output = model(data)
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            optimizer.zero_grad()
            optimzer_center.zero_grad()

            loss = focal_loss(output, target)
            loss.backward()
            optimizer.step()
            optimzer_center.step()
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            print(correct, ll, loss_avg)

        state['train_accuracy'] = correct / len(data_loader.dataset)
        state['train_loss'] = loss_avg
    def test():
        with torch.no_grad():
            model.eval()
            loss_avg = 0.0
            correct = 0
            for batch_idx, (data, target) in enumerate(test_data_loader):
                data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
                output = model(data)
                loss = F.cross_entropy(output, target)
                pred = output.data.max(1)[1]
                correct += float(pred.eq(target.data).sum())
                loss_avg += float(loss)
                state['test_loss'] = loss_avg / len(test_data_loader)
                state['test_accuracy'] = correct / len(test_data_loader.dataset)
            print(state['test_accuracy'])

    best_accuracy = 0.0
    for epoch in range(100):
        state['epoch'] = epoch
        train()
        test()
        sch.step(state['train_accuracy'])

        if best_accuracy < state['test_accuracy']:
            state['best_accuracy'] =  state['test_accuracy']
            torch.save(model.state_dict(), os.path.join('./log', idx+'.pth'))
        with open(os.path.join('./log', idx+'.json'),'w') as f:
            f.write(json.dumps(state))
            f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])
        if best_accuracy == 1.0:
            break

if __name__ == '__main__':
    train_dr = 'D:/deep_learn_data/luntai/guangdong_round1_train2_20180916/guangdong_round1_train2_20180916/瑕疵样本'
    run(train_dr,'luntai',12, '1006_iv_other')