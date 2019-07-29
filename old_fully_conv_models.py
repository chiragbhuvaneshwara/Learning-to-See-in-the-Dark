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
        x = F.sigmoid(x)
        return x
##################################################################################################

##################################################################################################
## Unet Batch norm model
class double_conv_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class down_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_bn, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_bn(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class up_bn(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_bn, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_bn(in_ch, out_ch)

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


class unet_bn(nn.Module):

    def __init__(self):
        super(unet_bn,self).__init__()
        
        # https://github.com/alishdipani/U-net-Pytorch/blob/master/train_Unet.py
        self.inc = double_conv_bn(3, 64)
        self.down1 = down_bn(64, 128)
        self.down2 = down_bn(128, 256)
        self.down3 = down_bn(256, 512)
        self.down4 = down_bn(512, 512)
        self.up1 = up_bn(1024, 256)
        self.up2 = up_bn(512, 128)
        self.up3 = up_bn(256, 64)
        self.up4 = up_bn(128, 64)
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
        x = F.sigmoid(x)
        return x
##################################################################################################

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


class unet_in(nn.Module):

    def __init__(self):
        super(unet_in,self).__init__()
        
        # https://github.com/alishdipani/U-net-Pytorch/blob/master/train_Unet.py
        self.inc = double_conv_in(3, 64)
        self.down1 = down_in(64, 128)
        self.down2 = down_in(128, 256)
        self.down3 = down_in(256, 512)
        self.down4 = down_in(512, 512)
        self.up1 = up_in(1024, 256)
        self.up2 = up_in(512, 128)
        self.up3 = up_in(256, 64)
        self.up4 = up_in(128, 64)
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
        x = F.sigmoid(x)
        return x
##################################################################################################

##################################################################################################
## Unet dropout model
class double_conv_d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.Dropout2d(p=0.3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.Dropout2d(p=0.3),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class down_d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_d, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_d(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class up_d(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_d, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_d(in_ch, out_ch)

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


class unet_d(nn.Module):

    def __init__(self):
        super(unet_d,self).__init__()
        
        # https://github.com/alishdipani/U-net-Pytorch/blob/master/train_Unet.py
        self.inc = double_conv_d(3, 64)
        self.down1 = down_d(64, 128)
        self.down2 = down_d(128, 256)
        self.down3 = down_d(256, 512)
        self.down4 = down_d(512, 512)
        self.up1 = up_d(1024, 256)
        self.up2 = up_d(512, 128)
        self.up3 = up_d(256, 64)
        self.up4 = up_d(128, 64)
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
        x = F.sigmoid(x)
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

        return out
##################################################################################################

## FPN
'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        # print("Bottleneck")
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        # print("init")
        # print(block)
        # print(num_blocks)
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer3 = nn.Conv2d(64,3,kernel_size=1,stride=1,padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        p2 = self.toplayer2(p2)
        p2 = self.up(p2)
        p2 = self.toplayer3(p2)
        p2 = self.up(p2)
        # return p2, p3, p4, p5
        p2 = F.sigmoid(p2)
        return p2


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    #  print("FPN101")
    return FPN(Bottleneck, [2,2,2,2])


def test():
    net = FPN101()
    fms = net(Variable(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())

# test()
