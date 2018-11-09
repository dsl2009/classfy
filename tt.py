import glob
import torch
from torch import nn
loss = nn.BCEWithLogitsLoss()

inputs = torch.randn(3, requires_grad=True)
print(inputs)
target = torch.empty(3).random_(2)
print(target)
output = loss(inputs, target)
print(output)
