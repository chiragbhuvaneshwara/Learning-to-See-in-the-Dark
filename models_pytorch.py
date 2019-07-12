import os
import torch
import torch.nn as nn
from skimage import io
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F



##################################################################################################
## Unet model
class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class unet(nn.Module):

    def __init__(self):
        super(unet,self).__init__()
        
        # https://github.com/alishdipani/U-net-Pytorch/blob/master/train_Unet.py
        self.inc = double_conv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out1 = nn.Conv2d(64, 3, 3, padding=1)
        #self.out2 = DepthToSpace(2)

    def forward(self,x):
        
        #print('#################################')
        #print(x.size())
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
        #x = self.out2(x)
        #print(x.size())

        return x
##################################################################################################
class simpleUNET(nn.Module):

    def __init__(self):
        super(simpleUNET,self).__init__()

        #Input Tensor Dimensions = 256x256x3
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=5,stride=1, padding=2)
        nn.init.xavier_uniform(self.conv1.weight) #Xaviers Initialisation
        self.activ_1= nn.ELU()
        #Pooling 1
        self.pool1= nn.MaxPool2d(kernel_size=2, return_indices=True)
        #Output Tensor Dimensions = 128x128x16


        #Input Tensor Dimensions = 128x128x16
        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.conv2.weight)
        self.activ_2 = nn.ELU()
        #Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        #Output Tensor Dimensions = 64x64x32

        #Input Tensor Dimensions = 64x64x32
        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.activ_3 = nn.ELU()
        #Output Tensor Dimensions = 64x64x64

        # 32 channel output of pool2 is concatenated
        
        #https://www.quora.com/How-do-you-calculate-the-output-dimensions-of-a-deconvolution-network-layer
        #Input Tensor Dimensions = 64x64x96
        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=96,out_channels=32,kernel_size=3,padding=1) ##
        nn.init.xavier_uniform(self.deconv1.weight)
        self.activ_4=nn.ELU()
        #UnPooling 1
        self.unpool1=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 128x128x32

        #16 channel output of pool1 is concatenated

        #Input Tensor Dimensions = 128x128x48
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=48,out_channels=16,kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.deconv2.weight)
        self.activ_5=nn.ELU()
        #UnPooling 2
        self.unpool2=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 256x256x16

        # 3 Channel input is concatenated

        #Input Tensor Dimensions= 256x256x19
        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=19,out_channels=3,kernel_size=5,padding=2)
        nn.init.xavier_uniform(self.deconv3.weight)
        self.activ_6=nn.Sigmoid()
        ##Output Tensor Dimensions = 256x256x1

    def forward(self,x):

        out_1 = x
        out = self.conv1(x)
        out = self.activ_1(out)
        size1 = out.size()
        out,indices1=self.pool1(out)
        out_2 = out
        out = self.conv2(out)
        out = self.activ_2(out)
        size2 = out.size()
        out,indices2=self.pool2(out)
        out_3 = out
        out = self.conv3(out)
        out = self.activ_3(out)

        out=torch.cat((out,out_3),dim=1)
        
        
        out=self.deconv1(out)
        out=self.activ_4(out)
        out=self.unpool1(out,indices2,size2)

        
        out=torch.cat((out,out_2),dim=1) 


        out=self.deconv2(out)
        out=self.activ_5(out)
        out=self.unpool2(out,indices1,size1)

        out=torch.cat((out,out_1),dim=1)
         
        out=self.deconv3(out)
        out=self.activ_6(out)
        out=out
        return out
##################################################################################################

