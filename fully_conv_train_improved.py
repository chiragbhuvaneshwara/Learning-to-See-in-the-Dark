import os
import torch
import torch.nn as nn
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.utils as vutils
# from torchvision.transforms.transforms import ToPILImage as trans
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.measure import compare_ssim

from fully_conv_models import simpleUNET, unet, unet_bn, unet_d, unet_in, FPN, Bottleneck
from datasetLoader import SeeingIntTheDarkDataset
from perceptual_loss_models import VggModelFeatures
from utils_train import weights_init, update_lr
# trans = transforms.ToPILImage()

def trainModel_withGradAccum(name, path, device, num_epochs, learning_rate, learning_rate_decay, reg, train_loader, val_loader, train_dataset, val_dataset, accumulation_steps=5, use_perceptual_loss = True):

    print('Effective Batch Size :',batch_size*accumulation_steps)

    # Initialize the model for this run
    if name == 'simpleUNET':
        model = simpleUNET()

    elif name == 'unet':
        model = unet()

    elif name == 'FPN':
        model = FPN(Bottleneck, [2,2,2,2])

    elif name == 'unet_bn':
        model = unet_bn()

    elif name == 'unet_in':
        model = unet_in()
    
    elif name == 'unet_d':
        model = unet_d()

    else:
        print('Name variable passed is incorrect')
        return None

    model.apply(weights_init)

    # Print the model we just instantiated
    print(model)

    model.to(device)
    
    if use_perceptual_loss:
        vgg_feature_extractor = VggModelFeatures(feature_extracting=True)
        vgg_feature_extractor.to(device) # Send the model to GPU      

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    # Train the model
    lr = learning_rate
    Loss = []                          
    valMSE = []
    valSSIM = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for i, (in_images, exp_images) in enumerate(train_loader):
            # Move tensors to the configured device
            in_images = in_images.type(torch.FloatTensor).to(device)
            exp_images = exp_images.type(torch.FloatTensor).to(device)

            # Forward pass
            outputs = model(in_images)

            if name != 'FPN' and use_perceptual_loss:
                # Vgg Features
                outputs_vgg_features = vgg_feature_extractor(outputs)
                exp_images_vgg_features = vgg_feature_extractor(exp_images)              
                loss = (  criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)  )

            elif name == 'FPN' and use_perceptual_loss:
                # Vgg Features
                outputs_vgg_features = vgg_feature_extractor(outputs)
                exp_images_vgg_features = vgg_feature_extractor(exp_images)              
                loss = (  criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)  )

            elif (name == 'FPN') and (use_perceptual_loss == False ): # using simple MSE Loss for FPN
                loss = criterion(outputs, exp_images)
            
            else:  # using simple MSE Loss for all unet models
                loss = criterion(outputs, exp_images)

            Loss.append(loss)               

            # Backward and optimize
            loss.backward()

            if (i+1) % accumulation_steps == 0:  
                optimizer.step()
                optimizer.zero_grad()
            
            if (i+1) % 200 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer, lr)

        with torch.no_grad():

            overallSSIM = 0
            MSE = 0
            for in_images, exp_images in val_loader:
                in_images = in_images.to(device)
                exp_images = exp_images.to(device)
                outputs = model(in_images)
                
                MSE += torch.sum((outputs - exp_images) ** 2)
                
                outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy() 
                exp_images_np = exp_images.permute(0,2,3,1).cpu().numpy()

                SSIM = 0
                for i in range(len(outputs_np)):
                    SSIM += compare_ssim(exp_images_np[i], outputs_np[i], multichannel=True)

                overallSSIM += SSIM

            
            total = len(val_dataset)
            current_SSIM = overallSSIM/total
            valSSIM.append(current_SSIM)

            current_MSE = MSE/total
            valMSE.append(current_MSE)

            # if current_MSE <= np.amin(valMSE):
            if current_SSIM >= np.amax(valSSIM):
                torch.save(model.state_dict(),path+'models/ESmodel'+str(epoch+1)+'.ckpt')

            print('Results on Validation set of {} images:'.format(total))
            print('Avg Validation MSE : {} '.format(current_MSE))
            print('Avg Validation SSIM: {} '.format(current_SSIM))

    # Training loss and Val MSE curves
    plt.plot(valMSE)
    title='AvgValMSE_vs_Epochs'
    plt.ylabel('Avg Validation MSE')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    # plt.show()
    plt.close()

    plt.plot(valSSIM)
    title='AvgValSSIM_vs_Epochs'
    plt.ylabel('Avg Validation SSIM')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    # plt.show()
    plt.close()

    plt.plot(Loss)
    title='Loss_vs_Iterations'
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    # plt.show()
    plt.close()

    return model, valSSIM

def trainModel(name, path, device, num_epochs, learning_rate, learning_rate_decay, reg, train_loader, val_loader, train_dataset, val_dataset, use_perceptual_loss = True):

    # Initialize the model for this run
    if name == 'simpleUNET':
        model = simpleUNET()

    elif name == 'unet':
        model = unet()

    elif name == 'FPN':
        model = FPN(Bottleneck, [2,2,2,2])

    elif name == 'unet_bn':
        model = unet_bn()

    elif name == 'unet_in':
        model = unet_in()
    
    elif name == 'unet_d':
        model = unet_d()

    else:
        print('Name variable passed is incorrect')
        return None

    model.apply(weights_init)

    # Print the model we just instantiated
    print(model)

    model.to(device)
    
    if use_perceptual_loss:
        vgg_feature_extractor = VggModelFeatures(feature_extracting=True)
        vgg_feature_extractor.to(device) # Send the model to GPU      

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    # Train the model
    lr = learning_rate
    Loss = []                          
    valMSE = []
    valSSIM = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (in_images, exp_images) in enumerate(train_loader):
            # Move tensors to the configured device
            in_images = in_images.type(torch.FloatTensor).to(device)
            exp_images = exp_images.type(torch.FloatTensor).to(device)

            # Forward pass
            outputs = model(in_images)

            if name != 'FPN' and use_perceptual_loss:
                # Vgg Features
                outputs_vgg_features = vgg_feature_extractor(outputs)
                exp_images_vgg_features = vgg_feature_extractor(exp_images)              
                loss = (  criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)  )

            elif name == 'FPN' and use_perceptual_loss:
                # Vgg Features
                outputs_vgg_features = vgg_feature_extractor(outputs)
                exp_images_vgg_features = vgg_feature_extractor(exp_images)              
                loss = (  criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                        + criterion(outputs_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)  )

            elif (name == 'FPN') and (use_perceptual_loss == False ): # using simple MSE Loss for FPN
                loss = criterion(outputs, exp_images)
            
            else:  # using simple MSE Loss for all unet models
                loss = criterion(outputs, exp_images)

            Loss.append(loss)               

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer, lr)

        with torch.no_grad():

            overallSSIM = 0
            MSE = 0
            for in_images, exp_images in val_loader:
                in_images = in_images.to(device)
                exp_images = exp_images.to(device)
                outputs = model(in_images)
                
                MSE += torch.sum((outputs - exp_images) ** 2)
                
                outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy() 
                exp_images_np = exp_images.permute(0,2,3,1).cpu().numpy()

                SSIM = 0
                for i in range(len(outputs_np)):
                    SSIM += compare_ssim(exp_images_np[i], outputs_np[i], multichannel=True)

                overallSSIM += SSIM

            
            total = len(val_dataset)
            current_SSIM = overallSSIM/total
            valSSIM.append(current_SSIM)

            current_MSE = MSE/total
            valMSE.append(current_MSE)

            # if current_MSE <= np.amin(valMSE):
            if current_SSIM >= np.amax(valSSIM):
                torch.save(model.state_dict(),path+'models/ESmodel'+str(epoch+1)+'.ckpt')

            print('Results on Validation set of {} images:'.format(total))
            print('Avg Validation MSE : {} '.format(current_MSE))
            print('Avg Validation SSIM: {} '.format(current_SSIM))

    # Training loss and Val MSE curves
    plt.plot(valMSE)
    title='AvgValMSE_vs_Epochs'
    plt.ylabel('Avg Validation MSE')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    # plt.show()
    plt.close()

    plt.plot(valSSIM)
    title='AvgValSSIM_vs_Epochs'
    plt.ylabel('Avg Validation SSIM')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    # plt.show()
    plt.close()

    plt.plot(Loss)
    title='Loss_vs_Iterations'
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    # plt.show()
    plt.close()

    return model, valSSIM

def testModelAndSaveOutputs(name, path, device, model, valSSIM, test_loader, test_dataset):

    best_id = np.argmax(valSSIM)
    bestESmodel = model

    bestESmodel.load_state_dict(torch.load(path+'models/ESmodel'+str(best_id+1)+'.ckpt'))
    bestESmodel = bestESmodel.to(device)

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
            reqd_size = int(in_images.size()[0])
            for i in range(reqd_size):
                count += 1 

                title = '###Input### vs ###Model_Output### vs ###Ground_truth###'
                plt.title(title)
                plt.imshow(np.transpose( vutils.make_grid([in_images[i], outputs[i], exp_images[i]], padding=5, normalize=True).cpu() , (1,2,0)))
                plt.tight_layout()
                plt.savefig(path+'images/'+name+'_%d.png'%(count))
                plt.close()
                
                plt.savefig(path+'images/'+name+'_%d.png'%(count))
                plt.close()

                if count % 100 == 0:
                    print('Saving image_%d.png'%(count))



        total = len(test_dataset)
        
        print('Results on Test set of {} images:'.format(total))
        print('Avg Validation MSE : {} '.format(MSE/total))
        print('Avg Validation SSIM: {} '.format(overallSSIM/total))

        print("Best Epoch wrt Avg Validation SSIM: ", best_id+1)

    torch.save(bestESmodel.state_dict(), path+'models/bestESModel_'+name+'.ckpt')
###############################################################################################################################################

path = ''

n = 1234
np.random.seed(n)
torch.cuda.manual_seed_all(n)
torch.manual_seed(n)


forwardTransform = transforms.Compose([     transforms.ToTensor(),
                                            transforms.Normalize(  mean = [ 0.5, 0.5, 0.5], std = [ 0.5, 0.5, 0.5]  )   ])

# sitd_dataset = SeeingIntTheDarkDataset(path+'dataset/Sony/short_temp_down/', path+'dataset/Sony/long_temp_down/', forwardTransform)
sitd_dataset = SeeingIntTheDarkDataset(path+'dataset/Sony/short_down/', path+'dataset/Sony/long_down/', forwardTransform)
print('Input Image Size:', sitd_dataset[0][0].size())
print('#################################################')
print('Min image value: ',int(torch.min(sitd_dataset[0][0])) )
print('Max image value: ',int(torch.max(sitd_dataset[0][0])) )
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

## final params
num_training= 2100
num_validation = 200
num_test = 397

num_epochs = 2
learning_rate = 1e-4
learning_rate_decay = 0.9
reg = 0.005
batch_size = 2

# ### dev params
# num_training= 20
# num_validation = 7
# num_test = 7


mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(sitd_dataset, mask)

mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(sitd_dataset, mask)

mask = list(range(num_training + num_validation, num_training + num_validation + num_test))
test_dataset = torch.utils.data.Subset(sitd_dataset, mask)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
###############################################################################################################################################
# parameters to select different models ==> Just change here. 
# name = 'simpleUNET'
# name = 'unet'
# name = 'unet_bn'
# name = 'unet_in'
# name = 'unet_d'
name = 'FPN'
print(name)
model, list_valSSIM = trainModel_withGradAccum(name, path, device, num_epochs, learning_rate, learning_rate_decay, reg, train_loader, val_loader, train_dataset, val_dataset, accumulation_steps=5, use_perceptual_loss = True)
print('Testing ..............................')
testModelAndSaveOutputs(name, path, device, model, list_valSSIM, test_loader, test_dataset)
