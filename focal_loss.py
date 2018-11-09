import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0).cuda()
    index = index.view(*view)
    ones = 1.
    return mask.scatter_(1, index, ones)

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -1 * y * torch.log(logit)
        loss = loss * (1 - logit) ** self.gamma
        return loss.sum(0).mean()