import os
import torch
import torch.nn as nn
# from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.measure import compare_ssim
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
# from gan_models import *
from datasetLoader import SeeingIntTheDarkDataset
from perceptual_loss_models import VggModelFeatures
trans = transforms.ToPILImage()

path = '/media/chirag/Chirag/Learning-to-See-in-the-Dark/'
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

forwardTransform = transforms.Compose([  transforms.ToTensor(),
                                         transforms.Normalize(  mean = [ 0.5, 0.5, 0.5],
                                                                std = [ 0.5, 0.5, 0.5]    )
                                    ])

inverseTransform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

# sitd_dataset = SeeingIntTheDarkDataset(path+'dataset/Sony/short_temp_down/', path+'dataset/Sony/long_temp_down/', forwardTransform)
sitd_dataset = SeeingIntTheDarkDataset(path+'dataset/Sony/short_down/', path+'dataset/Sony/long_down/', forwardTransform)
print('Input Image Size:')
print(sitd_dataset[0][0].size())
print('#################################################')
print('min: ',torch.min(sitd_dataset[0][0]))
print('max: ',torch.max(sitd_dataset[0][0]))

inImageSize = sitd_dataset[0][0].size()
inImage_xdim = int(inImageSize[1])
inImage_ydim = int(inImageSize[2])

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Initial GPU:',torch.cuda.current_device())
   
    torch.cuda.set_device(1)
    print('Selected GPU:', torch.cuda.current_device())

print('Using device: %s'%device)

# #### final params
num_training= 2100
num_validation = 200
num_test = 397

num_epochs = 20
learning_rate = 1e-5
learning_rate_decay = 0.7
reg = 0.001
batch_size = 1

### dev params
# num_training= 20
# num_validation = 7
# num_test = 7

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

    def __init__(self, sigma=1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma != 0:
            sampled_noise = torch.rand(x.size()).to(device) * self.sigma
            x = x + sampled_noise
        return x 

class unet_in_generator(nn.Module):

    def __init__(self):
        super(unet_in_generator,self).__init__()
        
        # self.train = train

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
        out = out.view(out.shape[0], -1)
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
    criterion_2 = nn.MSELoss()
    vgg_feature_extractor = VggModelFeatures(feature_extracting=True)
    vgg_feature_extractor.to(device) # Send the model to GPU
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=reg, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=reg, betas=(0.5, 0.999))

    # Train the model
    lr = learning_rate
    GLoss = []                           
    DLoss = []
    valMSE = []
    valSSIM = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        
        for i, (in_images, exp_images) in enumerate(train_loader):
            generator.train()

            # Move tensors to the configured device
            in_images = in_images.type(torch.FloatTensor).to(device)
            exp_images = exp_images.type(torch.FloatTensor).to(device)

            # Adversarial ground truths
            valid = torch.ones([batch_size, 1]).to(device) # Discriminator Label to real
            fake = torch.zeros([batch_size, 1]).to(device) # Discriminator Label to fake

            ###############################
            # Generator update
            # Forward pass
            gen_images = generator(in_images)

            # Vgg Features
            gen_images_vgg_features = vgg_feature_extractor(gen_images)
            exp_images_vgg_features = vgg_feature_extractor(exp_images)

            g_loss =  (criterion(discriminator(gen_images), valid) 
                + criterion_2(gen_images_vgg_features.relu1_2, exp_images_vgg_features.relu1_2)
                + criterion_2(gen_images_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                + criterion_2(gen_images_vgg_features.relu3_3, exp_images_vgg_features.relu3_3)
                + criterion_2(gen_images_vgg_features.relu4_3, exp_images_vgg_features.relu4_3)    
            )          
            
            GLoss.append(g_loss)               

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
                outputs = generator(in_images)

                MSE += torch.sum((outputs - exp_images) ** 2)
                
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
            
            outputs = bestESmodel(in_images)

            MSE += torch.sum((outputs - exp_images) ** 2)

            outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
            exp_images_np = exp_images.permute(0, 2, 3, 1).cpu().numpy()

            SSIM = 0
            for i in range(len(outputs_np)):
                SSIM += compare_ssim(exp_images_np[i], outputs_np[i], multichannel=True)

            overallSSIM += SSIM

            # Visualize the output of the best model against ground truth
            # in_images_py = in_images.cpu()
            # outputs_py = outputs.cpu()
            # exp_images_py = exp_images.cpu()
            
            reqd_size = int(in_images.size()[0])

            for i in range(reqd_size):

                # img = in_images_py[i].numpy()
                # nonZero = np.count_nonzero(img)
                count += 1 
                
                # title='Input ('+str(round((nonZero*100)/(inImage_xdim*inImage_ydim*3) , 2))+'% Non Zero) vs Model Output vs Ground truth'
                title = '###Input### vs ###Model_Output### vs ###Ground_truth###'
                plt.title(title)
                
                # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

                plt.imshow(np.transpose( vutils.make_grid([in_images[i], outputs[i], exp_images[i]], padding=5, normalize=True).cpu() , (1,2,0)))

                # axarr[0].imshow(trans(in_images_py[i]))
                # axarr[1].imshow(trans(outputs_py[i]))
                # axarr[2].imshow(trans(exp_images_py[i]))
                plt.tight_layout()
                plt.savefig(path+'images/'+name+'_%d.png'%(count))
                plt.close()

                if count % 10 == 0:
                    print('Saving image_%d.png'%(count))


        total = len(test_dataset)
        print('Avg Test MSE of the best ES network on all the {} test images: {} '.format(total, MSE/total))
        print('Avg Test SSIM of the best ES network on all the {} test images: {} '.format(total, overallSSIM/total))
        print("Best Epoch with lowest Avg Validation MSE: ", best_id+1)

    torch.save(bestESmodel.state_dict(), path+'models/bestESModel_'+name+'.ckpt')


###############################################################################################################################################
# parameters to select different models ==> Just change here. 
name = 'gan'

trainAndTestModel(name)


