import os
import torch
import torch.nn as nn
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable
import torch.nn.functional as F

class SeeingIntTheDarkDataset(Dataset):

    def __init__(self, x_dir, y_dir, transform=None):

        self.x_dir =  x_dir
        self.x_dir_files = sorted(os.listdir(x_dir))[1:]

        self.y_dir = y_dir
        self.y_dir_files = sorted(os.listdir(y_dir))[1:]

        self.transform = transform

    def __len__(self):
        return len(self.x_dir_files)

    def __getitem__(self, idx):
        inp_img_name = os.path.join(self.x_dir, self.x_dir_files[idx])
        out_img_name = os.path.join(self.y_dir, self.y_dir_files[idx])

        in_image = io.imread(inp_img_name)
        out_image = io.imread(out_img_name)

        if self.transform:
            in_image = self.transform(in_image)
            out_image = self.transform(out_image)

        return [in_image, out_image]


# sitd_dataset = SeeingIntTheDarkDataset('dataset/Sony/short_temp_down/', 'dataset/Sony/long_temp_down/', transforms.ToTensor())
sitd_dataset = SeeingIntTheDarkDataset('dataset/Sony/short_down/', 'dataset/Sony/long_down/', transforms.ToTensor())
print(sitd_dataset[0][0].size())

n = 1234
np.random.seed(n)
torch.cuda.manual_seed_all(n)
torch.manual_seed(n)

def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, mean=0.0, std=1e-3)
        m.bias.data.fill_(0.0)
        
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data) 
        m.bias.data.fill_(0.0)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#### final params
# num_training= 20
# num_validation = 6
# num_test = 8

#### dev params
input_size = 32 * 32 * 3
layer_config= [512, 256]
num_classes = 10
num_epochs = 30
learning_rate = 0.01 #1e-3
learning_rate_decay = 0.99
reg=0#0.001

num_training= 20
num_validation = 6
num_test = 8
batch_size = 2

mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(sitd_dataset, mask)

mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(sitd_dataset, mask)

mask = list(range(num_training + num_validation, num_training + num_validation + num_test))
test_dataset = torch.utils.data.Subset(sitd_dataset, mask)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

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
        self.inc = double_conv(3, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(768, 256)
        self.up2 = up(384, 128)
        self.up3 = up(192, 64)
        self.up4 = up(96, 32)
        self.out1 = nn.Conv2d(32, 12, 3, padding=1)
        self.out2 = DepthToSpace(2)

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
        x = self.out2(x)

        return out
##################################################################################################
class simpleUNET(nn.Module):

    def __init__(self):
        super(unet,self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3,stride=1, padding=2)
        self.activ_1= nn.LeakyReLU(inplace=True),
        self.pool1= nn.MaxPool2d(kernel_size=2, return_indices=True)
       
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
        self.activ_2 = nn.LeakyReLU(inplace=True),
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        self.activ_3 = nn.LeakyReLU(inplace=True),
        

        self.deconv1=nn.ConvTranspose2d(in_channels=96,out_channels=32,kernel_size=3,padding=1) 
        self.activ_4=nn.LeakyReLU(inplace=True),
        self.unpool1=nn.MaxUnpool2d(kernel_size=2)
        
        self.deconv2=nn.ConvTranspose2d(in_channels=48,out_channels=16,kernel_size=3,padding=1)
        self.activ_5=nn.LeakyReLU(inplace=True),
        self.unpool2=nn.MaxUnpool2d(kernel_size=2)
        
        self.deconv3=nn.ConvTranspose2d(in_channels=19,out_channels=3,kernel_size=3,padding=2)
        self.activ_6=nn.LeakyReLU(inplace=True),
        

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
        # out=self.activ_6(out)
        # out=out
        return out
##################################################################################################

# Initialize the model for this run
model= unet()
model.apply(weights_init)

# Print the model we just instantiated
print(model)

model.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
Loss = []                           #to save all the model losses
valMSE = []
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (in_images, exp_images) in enumerate(train_loader):
        # Move tensors to the configured device
        in_images = in_images.type(torch.FloatTensor).to(device)
        exp_images = exp_images.type(torch.FloatTensor).to(device)

        # Forward pass
        outputs = model(in_images)

        loss = criterion(outputs, exp_images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        Loss.append(loss)               #save the loss so we can get accuracies later
    
    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)

    with torch.no_grad():

        for in_images, exp_images in val_loader:
            in_images = in_images.to(device)
            exp_images = exp_images.to(device)
            outputs = model(in_images)
            MSE = torch.sum((outputs - exp_images) ** 2)
            
        best_model = None
        
        current_MSE = MSE
        valMSE.append(current_MSE)
        if current_MSE <= np.amin(valMSE):
            torch.save(model.state_dict(),'models/ESmodel'+str(epoch+1)+'.ckpt')

        print('Validataion MSE is: {} '.format(current_MSE))

