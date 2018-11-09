import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from losses.center_loss import CenterLoss
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*5*5, 2)
        self.ip2 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu2_1(self.conv2_1(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu3_1(self.conv3_1(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*5*5)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        return ip1, F.log_softmax(ip2, dim=1)

def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(2):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    plt.xlim(xmin=-5,xmax=5)
    plt.ylim(ymin=-5,ymax=5)
    plt.text(-4.8,4.6,"epoch=%d" % epoch)
    plt.savefig('../images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)


def train(epoch):
    print ("Training... Epoch = " ,epoch)
    ip1_loader = []
    idx_loader = []
    for i,(data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()

        ip1, pred = model(data)
        loss = nllloss(pred, target) + loss_weight * centerloss(target, ip1)

        optimizer4nn.zero_grad()
        optimzer4center.zero_grad()

        loss.backward()

        optimizer4nn.step()
        optimzer4center.step()

        ip1_loader.append(ip1)
        idx_loader.append((target))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
# Dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(320,320)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}
imagenet_data = ImageFolder('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step3/桃子/桃子疮痂病',
                            transform=data_transforms['train'])

data_loader = DataLoader(imagenet_data, batch_size=8, shuffle=True)


# Model
model = Net().cuda()

# NLLLoss
nllloss = nn.NLLLoss().cuda() #CrossEntropyLoss = log_softmax + NLLLoss
# CenterLoss
loss_weight = 1
centerloss = CenterLoss(2, 2).cuda()

# optimzer4nn
optimizer4nn = optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)

# optimzer4center
optimzer4center = optim.SGD(centerloss.parameters(), lr =0.5)

for epoch in range(100):
    sheduler.step()
    # print optimizer4nn.param_groups[0]['lr']
    train(epoch+1)