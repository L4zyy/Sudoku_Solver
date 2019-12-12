import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torchsummary import summary

class ConvBNRelu(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size=(3, 3), padding=1, stride=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGG_like(nn.Module):
    def __init__(self, scale):
        super(VGG_like, self).__init__()

        self.scale = scale

        self.conv1_1 = ConvBNRelu(3, scale)
        self.conv1_2 = ConvBNRelu(scale, scale)

        self.conv2_1 = ConvBNRelu(scale, 2*scale)
        self.conv2_2 = ConvBNRelu(2*scale, 2*scale)

        self.conv3_1 = ConvBNRelu(2*scale, 4*scale)
        self.conv3_2 = ConvBNRelu(4*scale, 4*scale)
        self.conv3_3 = ConvBNRelu(4*scale, 4*scale, kernel_size=(1, 1), padding=0)

        self.conv4_1 = ConvBNRelu(4*scale, 8*scale)
        self.conv4_2 = ConvBNRelu(8*scale, 8*scale)
        self.conv4_3 = ConvBNRelu(8*scale, 8*scale, kernel_size=(1, 1), padding=0)

        self.conv5_1 = ConvBNRelu(8*scale, 8*scale)
        self.conv5_2 = ConvBNRelu(8*scale, 8*scale)
        self.conv5_3 = ConvBNRelu(8*scale, 8*scale, kernel_size=(1, 1), padding=0)

        self.fc1 = nn.Sequential(
            nn.Linear(8*self.scale*2*2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.out = nn.Linear(64, 10)
    
    def forward(self, x):
        # input
        out = x

        # block 1
        out = self.conv1_1(out)
        out = self.conv1_2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)

        # block 2
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)

        # block 3
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)

        # block 4
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)

        # block 5
        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)

        # FCs
        out = out.reshape(-1, 32*self.scale)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.out(out)

        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = VGG_like(16).to(device)
    summary(model, (1, 64, 64))