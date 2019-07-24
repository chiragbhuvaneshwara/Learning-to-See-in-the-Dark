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
from gan_models import unet_in_generator, Discriminator
from fully_conv_models import simpleUNET, unet, unet_bn, unet_d, unet_in, FPN, Bottleneck
from datasetLoader import SeeingIntTheDarkDataset
from perceptual_loss_models import VggModelFeatures
trans = transforms.ToPILImage()

def testModel(path, nameOfSavedModel, model, subset_loader):

    model.load_state_dict(torch.load(path+'models/'+nameOfSavedModel))
    model.to(device)

    model.eval()
       
    with torch.no_grad():

        overallSSIM = 0
        MSE = 0
        count = 0 
        for in_images, exp_images in subset_loader:
            
            in_images = in_images.to(device)
            exp_images = exp_images.to(device)
            
            outputs = model(in_images)

            MSE += torch.sum((outputs - exp_images) ** 2)

            outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
            exp_images_np = exp_images.permute(0, 2, 3, 1).cpu().numpy()

            SSIM = 0
            for i in range(len(outputs_np)):
                SSIM += compare_ssim(exp_images_np[i], outputs_np[i], multichannel=True)

            overallSSIM += SSIM

        total = len(test_dataset)
        
        avgMSE = MSE/total
        avgSSIM = overallSSIM/total

        print('Results of {} on {} images:'.format(nameOfSavedModel[:-5], total))
        print('Avg MSE  : {} '.format(avgMSE))
        print('Avg SSIM : {} '.format(avgSSIM))

    return avgMSE, avgSSIM

def testModelOnAllSets(path, nameOfSavedModel, model, train_loader, val_loader, test_loader):
    
    print('###############################################################')
    print('Train Set:')
    MSE1, SSIM1 = testModel(path, nameOfSavedModel, generator, train_loader)

    print('###############################################################')
    print('Validation Set:')
    MSE2, SSIM2 = testModel(path, nameOfSavedModel, generator, val_loader)

    print('###############################################################')
    print('Test Set')
    MSE3, SSIM3 = testModel(path, nameOfSavedModel, generator, test_loader)

    print('###############################################################')
    print('Overall: ')
    overallMSE = (MSE1 + MSE2 + MSE3)/3
    overallSSIM = (SSIM1 + SSIM2 + SSIM3)/3
    print('Avg MSE : {}'.format(overallMSE))
    print('Avg SSIM: {}'.format(overallSSIM))

# path = '/media/chirag/Chirag/Learning-to-See-in-the-Dark/'
path = ''
#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Initial GPU:',torch.cuda.current_device())
   
    torch.cuda.set_device(1)
    print('Selected GPU:', torch.cuda.current_device())

print('Using device: %s'%device)

forwardTransform = transforms.Compose([  transforms.ToTensor(),
                                         transforms.Normalize(  mean = [ 0.5, 0.5, 0.5],
                                                                std = [ 0.5, 0.5, 0.5]    )
                                    ])

# sitd_dataset = SeeingIntTheDarkDataset(path+'dataset/Sony/short_temp_down/', path+'dataset/Sony/long_temp_down/', forwardTransform)
sitd_dataset = SeeingIntTheDarkDataset(path+'dataset/Sony/short_down/', path+'dataset/Sony/long_down/', forwardTransform)
print('Input Image Size: ',sitd_dataset[0][0].size())
print('Min image value: ',int(torch.min(sitd_dataset[0][0])) )
print('Max image value: ',int(torch.max(sitd_dataset[0][0])) )

inImageSize = sitd_dataset[0][0].size()
inImage_xdim = int(inImageSize[1])
inImage_ydim = int(inImageSize[2])

#### final params
num_training= 2100
num_validation = 200
num_test = 397
batch_size = 1

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


#############################################################################################################
generator = unet_in_generator()
testModelOnAllSets(path, 'bestESModel_gan.ckpt', generator, train_loader, val_loader, test_loader)

# model = simpleUNET()
# testModelOnAllSets(path, 'bestESModel_simpleUNET.ckpt', model, train_loader, val_loader, test_loader)

# model = unet()
# testModelOnAllSets(path, 'bestESModel_unet.ckpt', model, train_loader, val_loader, test_loader)

# model = unet_bn()
# testModelOnAllSets(path, 'bestESModel_unet_bn.ckpt', model, train_loader, val_loader, test_loader)

# model = unet_in()
# testModelOnAllSets(path, 'bestESModel_unet_in.ckpt', model, train_loader, val_loader, test_loader)

# model = unet_d()
# testModelOnAllSets(path, 'bestESModel_unet_d.ckpt', model, train_loader, val_loader, test_loader)

# model = FPN(Bottleneck, [2,2,2,2])
# testModelOnAllSets(path, 'bestESModel_FPN.ckpt', model, train_loader, val_loader, test_loader)

