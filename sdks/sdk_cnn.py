from pathlib import Path
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import optim
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets

class sdk_CNN(nn.Module):
    def __init__(self):
        super(sdk_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=21)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=21)

        self.fc1 = nn.Linear(in_features=12*17*17, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
    
    def forward(self, t):
        # input
        t = t

        # conv1
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # conv2
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # ravel tensor
        t = t.reshape(-1, 12*17*17)

        # fc1
        t = F.relu(self.fc1(t))
        # fc2
        t = F.relu(self.fc2(t))
        # out
        t = self.out(t)

        return t