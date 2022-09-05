import torch
import torch.nn as nn
from btcn import TemporalConvNet
import numpy as np

# input = torch.randn(2,25,18)
# btcn = TemporalConvNet(25, [32,16,8,4], kernel_size=2, dropout=0)

input = torch.randn(2,32,25)
btcn = TemporalConvNet(32, [32,16,8,4,1], kernel_size=2, dropout=0)
linear = nn.Linear(25, 1)

x = btcn(input)
x = np.squeeze(x,axis=1)
x = linear(x)

print(x.size())