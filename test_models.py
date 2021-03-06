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

# tanh models
from gan_models import unet_in_generator, Discriminator
from fully_conv_models import simpleUNET, unet, unet_bn, unet_d, unet_in, FPN 

# sigmoid models
from fully_conv_models_sigmoid import unet, unet_d, unet_in as old_unet
from fully_conv_models_sigmoid import unet_d as old_unet_d
from fully_conv_models_sigmoid import unet_in as old_unet_in

from datasetLoader import SeeingIntTheDarkDataset
from perceptual_loss_models import VggModelFeatures
# trans = transforms.ToPILImage()

def testModel(path, device, nameOfSavedModel, model, subset_loader, subset, save_images = False):

    name = nameOfSavedModel[:-5]
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

            if save_images:
                # Visualize the output of the best model against ground truth
                reqd_size = int(in_images.size()[0])
                for i in range(reqd_size):
                    count += 1 

                    title = '###Input### vs ###Model_Output### vs ###Ground_truth###'
                    plt.title(title)
                    plt.imshow(np.transpose( vutils.make_grid([in_images[i], outputs[i], exp_images[i]], padding=5, normalize=True).cpu() , (1,2,0)))
                    plt.tight_layout()
                    plt.savefig(path+'images/'+name+'_%d.png'%(count))
                    plt.close()
                    
                    # plt.savefig(path+'images/'+name+'_%d.png'%(count))
                    # plt.close()

                    if count % 100 == 0:
                        print('Saving image_%d.png'%(count))

        total = len(subset)
        
        avgMSE = MSE/total
        avgSSIM = overallSSIM/total

        print('Results of {} on {} images:'.format(nameOfSavedModel[:-5], total))
        print('Avg MSE  : {} '.format(avgMSE))
        print('Avg SSIM : {} '.format(avgSSIM))

    return avgMSE, avgSSIM

def testModelOnAllSets(path, device, nameOfSavedModel, model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, save_images=False):
    
    print('###############################################################')
    print('Train Set:')
    MSE1, SSIM1 = testModel(path, device, nameOfSavedModel, model, train_loader, train_dataset)

    print('###############################################################')
    print('Validation Set:')
    MSE2, SSIM2 = testModel(path, device, nameOfSavedModel, model, val_loader, val_dataset)

    print('###############################################################')
    print('Test Set')
    MSE3, SSIM3 = testModel(path, device, nameOfSavedModel, model, test_loader, test_dataset, save_images)

    print('###############################################################')
    print('Overall: ')
    overallMSE = (MSE1 + MSE2 + MSE3)/3
    overallSSIM = (SSIM1 + SSIM2 + SSIM3)/3
    print('Avg MSE : {}'.format(overallMSE))
    print('Avg SSIM: {}'.format(overallSSIM))

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


def prepare_data(forwardTransform):
    
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
    batch_size = 5

    mask = list(range(num_training))
    train_dataset = torch.utils.data.Subset(sitd_dataset, mask)

    mask = list(range(num_training, num_training + num_validation))
    val_dataset = torch.utils.data.Subset(sitd_dataset, mask)

    mask = list(range(num_training + num_validation, num_training + num_validation + num_test))
    test_dataset = torch.utils.data.Subset(sitd_dataset, mask)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

#############################################################################################################
# Results from models before presentation

forwardTransform = transforms.Compose([  transforms.ToTensor(),
                                        #  transforms.Normalize(  mean = [ 0.5, 0.5, 0.5],
                                        #                          std = [ 0.5, 0.5, 0.5]    )
                                    ])

train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_data(forwardTransform)

model = old_unet()
testModelOnAllSets(path, device, 'old_bestESModel_unet.ckpt', model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, save_images= True)

model = old_unet_in()
testModelOnAllSets(path, device, 'old_bestESModel_unet_in.ckpt', model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, save_images= True)

model = old_unet_d()
testModelOnAllSets(path, device, 'old_bestESModel_unet_d.ckpt', model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, save_images= True)

#############################################################################################################
# Results from new models after presentation

forwardTransform = transforms.Compose([  transforms.ToTensor(),
                                         transforms.Normalize(  mean = [ 0.5, 0.5, 0.5],
                                                                 std = [ 0.5, 0.5, 0.5]    )
                                    ])

train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_data(forwardTransform)

generator = unet_in_generator(device)
testModelOnAllSets(path, device, 'bestESModel_gan_mse_loss.ckpt', generator, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)

generator = unet_in_generator(device)
testModelOnAllSets(path, device, 'bestESModel_gan_perceptual_loss.ckpt', generator, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)

# model = simpleUNET()
# testModelOnAllSets(path, device, 'bestESModel_simpleUNET.ckpt', model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)

# model = unet()
# testModelOnAllSets(path, device, 'bestESModel_unet.ckpt', model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)

# model = unet_bn()
# testModelOnAllSets(path, device, 'bestESModel_unet_bn.ckpt', model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)

# model = unet_in()
# testModelOnAllSets(path, device, 'bestESModel_unet_in.ckpt', model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)

# model = unet_d()
# testModelOnAllSets(path, device, 'bestESModel_unet_d.ckpt', model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)

# model = FPN(Bottleneck, [2,2,2,2])
# testModelOnAllSets(path, device, 'bestESModel_FPN.ckpt', model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
