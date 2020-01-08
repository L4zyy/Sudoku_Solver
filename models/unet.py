import torch
import torch.nn as nn
import torch.nn.functional as F
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

class EncoderBlock(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size=(3, 3), padding=1, stride=1):
        super(EncoderBlock, self).__init__()
        self.encode = nn.Sequential(
            ConvBNRelu(inChannels, outChannels),
            ConvBNRelu(outChannels, outChannels)
        )

    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small

class DecoderBlock(nn.Module):
    def __init__(self,catChannels, inChannels, outChannels, kernel_size=(3, 3), padding=1, stride=1):
        super(DecoderBlock, self).__init__()

        self.decode = nn.Sequential(
            ConvBNRelu(catChannels+inChannels, outChannels),
            ConvBNRelu(outChannels, outChannels)
        )
    
    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.interpolate(x, size=(height, width), mode='bilinear')
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x

class UNet(nn.Module):
    def __init__(self, inShape, scale):
        super(UNet, self).__init__()
        channels, height, width = inShape

        # 512
        self.down1 = EncoderBlock(channels, scale) #256
        self.down2 = EncoderBlock(scale, 2*scale) #128
        self.down3 = EncoderBlock(2*scale, 4*scale) #64
        self.down4 = EncoderBlock(4*scale, 8*scale) #32

        self.center = nn.Sequential(
            ConvBNRelu(8*scale, 8*scale)
        )

        #32
        self.up4 = DecoderBlock(8*scale, 8*scale, 4*scale) #64
        self.up3 = DecoderBlock(4*scale, 4*scale, 2*scale) #128
        self.up2 = DecoderBlock(2*scale, 2*scale, scale) #256
        self.up1 = DecoderBlock(scale, scale, scale) #512

        self.classify = nn.Conv2d(scale, 1, kernel_size=1, bias=True)

    def forward(self, x):
        out = x
        down1_out, out = self.down1(out)
        down2_out, out = self.down2(out)
        down3_out, out = self.down3(out)
        down4_out, out = self.down4(out)

        out = self.center(out)

        out = self.up4(out, down4_out)
        out = self.up3(out, down3_out)
        out = self.up2(out, down2_out)
        out = self.up1(out, down1_out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = UNet((3, 512, 512), 8).to(device)

    summary(model, (3, 512, 512), 8)