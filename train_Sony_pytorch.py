import os
import torch
import torch.nn as nn
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable

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


sitd_dataset = SeeingIntTheDarkDataset('dataset/Sony/short_temp_down/', 'dataset/Sony/long_temp_down/', transforms.ToTensor())

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

class unet(nn.Module):

    def __init__(self):
        super(unet,self).__init__()

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
        self.deconv3=nn.ConvTranspose2d(in_channels=19,out_channels=1,kernel_size=5,padding=2)
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

        print(in_images.size())
        print(outputs.size())
        print(exp_images.size())
        print('###########################################')

        loss = criterion(outputs, exp_images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
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
            torch.save(model.state_dict(),'model'+str(epoch+1)+'.ckpt')

        print('Validataion accuracy is: {} %'.format(current_MSE))

