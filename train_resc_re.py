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
from PIL import Image
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(448, 598)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3,hue=0.2),
        transforms.RandomGrayscale(0.3),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(448, 598)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def load_dat(batch_size, trainr):
    imagenet_data = ImageFolder(trainr,
                                transform=data_transforms['train'])
    data_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=True)
    new_lbs = dict()
    for x in imagenet_data.class_to_idx:
        new_lbs[imagenet_data.class_to_idx[x]] = x

    return data_loader, imagenet_data,new_lbs


def run(trainr,testdr, name,cls_num,idx):
    batch_size = 8
    data_loader, imagenet_data,new_lbs = load_dat(batch_size, trainr)

    model = se_resnext50_32x4d(num_classes=1000,pretrained=None)
    model.load_state_dict(torch.load('/home/dsl/all_check/se_resnext50_32x4d-a260b3a4.pth'), strict=False)
    model.fc1 = nn.Linear(2048, 4)
    model.fc2 = nn.Linear(4, cls_num)
    #model.load_state_dict(torch.load('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/check/1009_res_total.pth'), strict=False)
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
    state['train_accuracy'] =0
    def train():
        model.train()
        loss_avg = 0.0
        progress = ProgressBar()
        ip1_loader = []
        idx_loader = []
        correct = 0
        for (data, target) in progress(data_loader):
            data.detach().numpy()
            if data.size(0) != batch_size:
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


            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            print(correct, ll, loss_avg)
        state['train_accuracy'] = correct / len(data_loader.dataset)


        state['train_loss'] = loss_avg
    def test():
        with torch.no_grad():
            model.eval()
            loss_avg = 0.0
            correct = 0
            for k in glob.glob(os.path.join(testdr,'*.jpg')):
                imag = Image.open(k)
                ig = data_transforms['val'](imag)
                ig = ig.unsqueeze(0)
                ig = torch.autograd.Variable(ig.cuda())
                f1, output = model(ig)
                output = F.softmax(output, dim=1)
                pred = output.data.squeeze(dim=0).cpu().numpy()
                score = np.asarray(pred)
                score = np.sum(score, axis=0)
                pred_lb = np.argmax(score)
                sc = np.max(score)
                print(k)
                lbs = new_lbs[pred_lb]
                if sc>0.66:
                    shutil.copy(k,os.path.join(train_dr, lbs))
                else:
                    try:
                        nn_name = k.split('/')[-1]
                        os.remove(os.path.join(train_dr, lbs, nn_name))
                    except:
                        pass

    best_accuracy = 0.0
    for epoch in range(100):
        state['epoch'] = epoch
        train()
        test()
        data_loader, imagenet_data, new_lbs = load_dat(batch_size, trainr)
        sch.step(state['train_accuracy'])

        if best_accuracy < state['train_accuracy']:
            state['best_accuracy'] =  state['train_accuracy']
            torch.save(model.state_dict(), os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/check', idx+'.pth'))
        with open(os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/check', idx+'.json'),'w') as f:
            f.write(json.dumps(state))
            f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])
        if best_accuracy == 1.0:
            break





if __name__ == '__main__':
    train_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/train/1009_bk'
    test_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/test2/guangdong_round1_test_b_20181009'
    run(train_dr,test_dr, 'luntai',12, '1009_rr')