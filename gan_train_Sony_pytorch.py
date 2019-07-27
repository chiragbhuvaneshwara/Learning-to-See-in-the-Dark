import os
import torch
import torch.nn as nn
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
# from torchvision.transforms.transforms import ToPILImage as trans
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.measure import compare_ssim

# from gan_models import *
from datasetLoader_pytorch import SeeingIntTheDarkDataset

trans = transforms.ToPILImage()

path = ''

#n = 1234
#np.random.seed(n)
#torch.cuda.manual_seed_all(n)
#torch.manual_seed(n)


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


#sitd_dataset = SeeingIntTheDarkDataset(path+'dataset/Sony/short_temp_down/', path+'dataset/Sony/long_temp_down/', transforms.ToTensor())
sitd_dataset = SeeingIntTheDarkDataset(path+'dataset/Sony/short_down/', path+'dataset/Sony/long_down/', transforms.ToTensor())
print('Input Image Size:')
print(sitd_dataset[0][0].size())

inImageSize = sitd_dataset[0][0].size()
inImage_xdim = int(inImageSize[1])
inImage_ydim = int(inImageSize[2])

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(torch.cuda.current_device())
   
    torch.cuda.set_device(4)
    print(torch.cuda.current_device())

print('Using device: %s'%device)

#### final params
num_training= 2100
num_validation = 200
num_test = 396

num_epochs = 20
learning_rate = .5e-3
learning_rate_decay = 0.7
reg = 0.001
batch_size = 2

# ### dev params
#num_training= 20
#num_validation = 7
#num_test = 7

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

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        #self.is_relative_detach = is_relative_detach
        #self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            #x = x.detach() if self.is_relative_detach else x
            #scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = torch.rand(x.size()).to(device) * self.sigma
            #sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

class unet_in_generator(nn.Module):

    def __init__(self):
        super(unet_in_generator,self).__init__()
        
        # self.train = train

        # https://github.com/alishdipani/U-net-Pytorch/blob/master/train_Unet.py
        self.inc = double_conv_in(3, 64)
        self.down1 = down_in(64, 128)
        self.down2 = down_in(128, 256)
        self.down3 = down_in(256, 512)
        self.down4 = down_in(512, 512)
        self.noise = GaussianNoise()
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
        #print('########################')
        #print(x5.size())
        #print(z.size())
        
        x5_modified = self.noise(x5)
        
        # if self.train == True:
        #     x5 += torch.rand(x5.size())

        x = self.up1(x5_modified, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out1(x)
        #x = self.out2(x)
        #print(x.size())

        return x
##################################################################################################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

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
        #print(img.size())
        out = out.view(out.shape[0], -1)
        #print('Here',out.size())
        validity = self.adv_layer(out)

        return validity 

def trainAndTestModel(name):

    generator = unet_in_generator()
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=reg, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=reg, betas=(0.5, 0.999))

    # Train the model
    lr = learning_rate
    GLoss = []                           #to save all the model losses
    DLoss = []
    valMSE = []
    valSSIM = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        generator.train()
        for i, (in_images, exp_images) in enumerate(train_loader):
          
            # Move tensors to the configured device
            in_images = in_images.type(torch.FloatTensor).to(device)
            exp_images = exp_images.type(torch.FloatTensor).to(device)

            # Adversarial ground truths
            valid = torch.ones([batch_size, 1]).to(device) # Discriminator Label to real
            fake = torch.zeros([batch_size, 1]).to(device) # Discriminator Lab
            # valid = Variable(torch.Tensor(1).fill_(1.0), requires_grad=False)
            # fake = Variable(torch.Tensor(1).fill_(0.0), requires_grad=False)

            ###############################
            # Generator update
            # Forward pass
            #z = Variable(torch.rand((batch_size, 512, 8, 12))).to(device)

            gen_images = generator(in_images)
            g_loss = criterion(discriminator(gen_images), valid)
            GLoss.append(g_loss)               #save the loss so we can get accuracies later

            # Backward and optimize
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            ###############################
            # Discriminator update
            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(discriminator(exp_images), valid)
            fake_loss = criterion(discriminator( generator(in_images) ), fake)
            
            d_loss = real_loss + fake_loss
            DLoss.append(d_loss)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            if (i+1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Gen_Loss: {:.6f}, Disc_Loss: {:.6f}'
                    .format(epoch+1, num_epochs, i+1, total_step, g_loss.item(), d_loss.item() ) )
        
        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer_D, lr)
        update_lr(optimizer_G, lr)

        with torch.no_grad():
            generator.eval()
            
            overallSSIM = 0
            MSE = 0
            for in_images, exp_images in val_loader:
                in_images = in_images.to(device)
                exp_images = exp_images.to(device)
                #z = torch.zeros((batch_size, 512, 8, 12)).to(device)
                outputs = generator(in_images)

                MSE += torch.sum((outputs - exp_images) ** 2)
                #outputs = outputs.cpu()
                outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy() 
                exp_images_np = exp_images.permute(0,2,3,1).cpu().numpy()

                SSIM = 0
                for i in range(len(outputs_np)):
                    SSIM += compare_ssim(exp_images_np[i], outputs_np[i], multichannel=True)

                overallSSIM += SSIM

            
            total = len(val_dataset)
            valSSIM.append(overallSSIM/total)

            current_MSE = MSE/total
            valMSE.append(current_MSE)
            if current_MSE <= np.amin(valMSE):
                torch.save(generator.state_dict(),path+'models/ESmodel'+str(epoch+1)+'.ckpt')

            print('Avg Validation MSE on all the {} Val images is: {} '.format(total, current_MSE))
            print('Avg Validation SSIM on all the {} Val images is: {} '.format(total, overallSSIM/total))

    # Training loss and Val MSE curves
    plt.plot(valMSE)
    title='AvgValMSE_vs_Epochs_Generator'
    plt.ylabel('Avg Validation MSE')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    #plt.show()
    plt.close()

    plt.plot(valSSIM)
    title='AvgValSSIM_vs_Epochs_Generator'
    plt.ylabel('Avg Validation SSIM')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    #plt.show()
    plt.close()

    plt.plot(GLoss)
    title='Loss_vs_Iterations_Generator'
    plt.ylabel('Generator Loss')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    #plt.show()
    plt.close()

    plt.plot(DLoss)
    title='Loss_vs_Iterations_Discriminator'
    plt.ylabel('Discriminator Loss')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    #plt.show()
    plt.close()

    print('Testing ..............................')

    # last_model = model
    best_id = np.argmin(valMSE)
    bestESmodel = generator

    bestESmodel.load_state_dict(torch.load(path+'models/ESmodel'+str(best_id+1)+'.ckpt'))
    bestESmodel = bestESmodel.to(device)


    # last_model.eval()
    bestESmodel.eval()
    
        
    with torch.no_grad():

        overallSSIM = 0
        MSE = 0
        count = 0 
        for in_images, exp_images in test_loader:
            
            in_images = in_images.to(device)
            exp_images = exp_images.to(device)
            
            #z = torch.zeros((batch_size, 512, 8, 12)).to(device)
            outputs = bestESmodel(in_images)

            MSE += torch.sum((outputs - exp_images) ** 2)

            # Visualize the output of the best model against ground truth
            in_images_py = in_images.cpu()
            outputs_py = outputs.cpu()
            exp_images_py = exp_images.cpu()
            
            reqd_size = int(in_images.size()[0])

            for i in range(reqd_size):

                img = in_images_py[i].numpy()
                nonZero = np.count_nonzero(img)
                count += 1 
                f, axarr = plt.subplots(1,3)
                title='Input ('+str(round((nonZero*100)/(192*128*3), 2))+'% Non Zero) vs Model Output vs Ground truth'
                plt.suptitle(title)
                axarr[0].imshow(trans(in_images_py[i]))
                axarr[1].imshow(trans(outputs_py[i]))
                axarr[2].imshow(trans(exp_images_py[i]))
                
                plt.savefig(path+'images/'+name+'_%d.png'%(count))
                plt.close()

                if count % 10 == 0:
                    print('Saving image_%d.png'%(count))

            outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
            exp_images_np = exp_images.permute(0, 2, 3, 1).cpu().numpy()

            SSIM = 0
            for i in range(len(outputs_np)):
                SSIM += compare_ssim(exp_images_np[i], outputs_np[i], multichannel=True)

            overallSSIM += SSIM

        total = len(test_dataset)
        print('Avg Test MSE of the best ES network on all the {} test images: {} '.format(total, MSE/total))
        print('Avg Test SSIM of the best ES network on all the {} test images: {} '.format(total, overallSSIM/total))
        print("Best Epoch with lowest Avg Validation MSE: ", best_id+1)

    torch.save(bestESmodel.state_dict(), path+'models/bestESModel_'+name+'.ckpt')


###############################################################################################################################################
# parameters to select different models ==> Just change here. 
name = 'gan'

trainAndTestModel(name)


