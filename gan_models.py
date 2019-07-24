import os
import torch
import torch.nn as nn
from skimage import io
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

##################################################################################################
## Unet Instance norm model
class double_conv_in(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv_in, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_in(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down_in, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_in(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up_in(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_in, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_in(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class GaussianNoise(nn.Module):

    def __init__(self, device, sigma=1):
        super().__init__()
        self.sigma = sigma
        self.device = device
    def forward(self, x):
        if self.training and self.sigma != 0:
            sampled_noise = torch.rand(x.size()).to(self.device) * self.sigma
            x = x + sampled_noise
        return x 

class unet_in_generator(nn.Module):

    def __init__(self, device):
        super(unet_in_generator,self).__init__()
        
        # self.train = train
        self.device = device

        self.inc = double_conv_in(3, 64)
        self.down1 = down_in(64, 128)
        self.down2 = down_in(128, 256)
        self.down3 = down_in(256, 512)
        self.down4 = down_in(512, 512)
        self.noise = GaussianNoise(self.device)
        self.up1 = up_in(1024, 256)
        self.up2 = up_in(512, 128)
        self.up3 = up_in(256, 64)
        self.up4 = up_in(128, 64)
        self.out1 = nn.Conv2d(64, 3, 3, padding=1)

        # self.sigmoid = nn.Sigmoid()	
        self.tanh = nn.Tanh()

    def forward(self,x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out1(x)

        # x = self.sigmoid(x)
        x = self.tanh(x)

        return x
##################################################################################################

class Discriminator(nn.Module):
    def __init__(self, inImageSize):
        super(Discriminator, self).__init__()

        inImage_xdim = int(inImageSize[1])
        inImage_ydim = int(inImageSize[2]) 

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 1, 1), nn.MaxPool2d(2,2), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        self.adv_layer = nn.Sequential(nn.Linear( (inImage_xdim * inImage_ydim)//2 , 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity 

##################################################################################################