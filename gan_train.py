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
from datasetLoader import SeeingIntTheDarkDataset
from perceptual_loss_models import VggModelFeatures
from utils_train import weights_init, update_lr

def trainGanModel_withGradAccum(name, path, device, num_epochs, learning_rate, learning_rate_decay, reg, train_loader, val_loader, train_dataset, val_dataset, inImageSize, accumulation_steps=5, use_perceptual_loss = True):
    
    print('Effective Batch Size :',batch_size*accumulation_steps)

    generator = unet_in_generator(device)
    discriminator = Discriminator(inImageSize)

    generator.to(device)
    discriminator.to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss and optimizer
    criterion = nn.BCELoss()
    criterion_2 = nn.MSELoss()

    if use_perceptual_loss:
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
        unet_in_generator.zero_grad()
        discriminator.zero_grad()
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
            
            if use_perceptual_loss:
                # Vgg Features
                gen_images_vgg_features = vgg_feature_extractor(gen_images)
                exp_images_vgg_features = vgg_feature_extractor(exp_images)

                percept_loss =  (     criterion_2(gen_images_vgg_features.relu1_2, exp_images_vgg_features.relu1_2)
                                    + criterion_2(gen_images_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                                    + criterion_2(gen_images_vgg_features.relu3_3, exp_images_vgg_features.relu3_3)
                                    + criterion_2(gen_images_vgg_features.relu4_3, exp_images_vgg_features.relu4_3)    ) / 4   
                
                g_loss =  (criterion(discriminator(gen_images), valid) +  percept_loss)                    
            
            else: # use just MSE Loss
                g_loss = criterion(discriminator(gen_images), valid) + criterion_2(gen_images, exp_images)
            
            g_loss = g_loss/accumulation_steps
            
            GLoss.append(g_loss)

            # Backward and optimize
            g_loss.backward()

            if (i+1) % accumulation_steps == 0:  
                
                optimizer_G.step()
                unet_in_generator.zero_grad()

            ###############################
            # Discriminator update
            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(discriminator(exp_images), valid)
            fake_loss = criterion(discriminator( generator(in_images) ), fake)
            
            d_loss = real_loss + fake_loss
            d_loss = d_loss / accumulation_steps

            DLoss.append(d_loss)
            
            d_loss.backward()

            if (i+1) % accumulation_steps == 0:             
                optimizer_D.step()
                discriminator.zero_grad()

            if (i+1) % 200 == 0:
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
            current_SSIM = overallSSIM/total
            valSSIM.append(current_SSIM)

            current_MSE = MSE/total
            valMSE.append(current_MSE)

            # if current_MSE <= np.amin(valMSE):
            if current_SSIM >= np.amax(valSSIM):
                torch.save(generator.state_dict(),path+'models/ESmodel'+str(epoch+1)+'.ckpt')

            print('Results on Validation set of {} images:'.format(total))
            print('Avg Validation MSE : {} '.format(current_MSE))
            print('Avg Validation SSIM: {} '.format(current_SSIM))

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

    return generator, valSSIM

def trainGanModel(name, path, device, num_epochs, learning_rate, learning_rate_decay, reg,train_loader, val_loader, train_dataset, val_dataset, inImageSize, use_perceptual_loss = True):
    
    generator = unet_in_generator(device)
    discriminator = Discriminator(inImageSize)

    generator.to(device)
    discriminator.to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss and optimizer
    criterion = nn.BCELoss()
    criterion_2 = nn.MSELoss()

    if use_perceptual_loss:
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
            
            if use_perceptual_loss:
                # Vgg Features
                gen_images_vgg_features = vgg_feature_extractor(gen_images)
                exp_images_vgg_features = vgg_feature_extractor(exp_images)

                percept_loss =  (     criterion_2(gen_images_vgg_features.relu1_2, exp_images_vgg_features.relu1_2)
                                    + criterion_2(gen_images_vgg_features.relu2_2, exp_images_vgg_features.relu2_2)
                                    + criterion_2(gen_images_vgg_features.relu3_3, exp_images_vgg_features.relu3_3)
                                    + criterion_2(gen_images_vgg_features.relu4_3, exp_images_vgg_features.relu4_3)    ) / 4   
                
                g_loss =  (criterion(discriminator(gen_images), valid) +  percept_loss)            
            else: # use just MSE Loss
                g_loss = criterion(discriminator(gen_images), valid) + criterion_2(gen_images, exp_images)
            
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
            current_SSIM = overallSSIM/total
            valSSIM.append(current_SSIM)

            current_MSE = MSE/total
            valMSE.append(current_MSE)

            # if current_MSE <= np.amin(valMSE):
            if current_SSIM >= np.amax(valSSIM):
                torch.save(generator.state_dict(),path+'models/ESmodel'+str(epoch+1)+'.ckpt')

            print('Results on Validation set of {} images:'.format(total))
            print('Avg Validation MSE : {} '.format(current_MSE))
            print('Avg Validation SSIM: {} '.format(current_SSIM))

    # Training loss and Val MSE curves
    plt.plot( range(1,len(valMSE)+1), valMSE)
    title='AvgValMSE_vs_Epochs_Generator'
    plt.ylabel('Avg Validation MSE')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig(path+'plots/'+name+title+'.png')
    #plt.show()
    plt.close()

    plt.plot(range(1,len(valSSIM)+1), valSSIM)
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

    return generator, valSSIM
    
def testModelAndSaveOutputs(name, path, device, model, valSSIM, test_loader, test_dataset, use_perceptual_loss = False):

    best_id = np.argmax(valSSIM)
    bestESmodel = model

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
            reqd_size = int(in_images.size()[0])
            for i in range(reqd_size):
                count += 1 
                
                title = '###Input### vs ###Model_Output### vs ###Ground_truth###'
                plt.title(title)
                plt.imshow(np.transpose( vutils.make_grid([in_images[i], outputs[i], exp_images[i]], padding=5, normalize=True).cpu() , (1,2,0)))
                plt.tight_layout()
                plt.savefig(path+'images/'+name+str(use_perceptual_loss)+'_%d.png'%(count))
                plt.close()

                if count % 10 == 0:
                    print('Saving image_%d.png'%(count))


        total = len(test_dataset)

        print('Results on Test set of {} images:'.format(total))
        print('Avg Validation MSE : {} '.format(MSE/total))
        print('Avg Validation SSIM: {} '.format(overallSSIM/total))

        print("Best Epoch wrt Avg Validation SSIM: ", best_id+1)

    torch.save(bestESmodel.state_dict(), path+'models/bestESModel_'+name+str(use_perceptual_loss)+'.ckpt')

###############################################################################################################################################
# trans = transforms.ToPILImage()
# path = '/media/chirag/Chirag/Learning-to-See-in-the-Dark/'

path = ''

n = 2
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

### final params
num_training= 2100
num_validation = 200
num_test = 397

num_epochs = 20
learning_rate = 1e-4
learning_rate_decay = 0.7
reg = 0.001
batch_size = 2

# # ### dev params
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
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
###############################################################################################################################################
#parameters to select different models ==> Just change here. 

### mse loss
print('##########################################################################################################################################')
name = 'gan_'
ploss = False
print(name+str(ploss))
model, list_valSSIM = trainGanModel_withGradAccum(name, path, device, num_epochs, learning_rate, learning_rate_decay, reg, train_loader, val_loader, train_dataset, val_dataset, inImageSize, accumulation_steps=5, use_perceptual_loss = ploss)
print('Testing ..............................')
testModelAndSaveOutputs(name, path, device, model, list_valSSIM, test_loader, test_dataset, use_perceptual_loss = ploss)


### perceptual loss
print('##########################################################################################################################################')
name = 'gan_'
ploss = True
print(name+str(ploss))
model, list_valSSIM = trainGanModel_withGradAccum(name, path, device, num_epochs, learning_rate, learning_rate_decay, reg, train_loader, val_loader, train_dataset, val_dataset, inImageSize, accumulation_steps=5, use_perceptual_loss = ploss)
print('Testing ..............................')
testModelAndSaveOutputs(name, path, device, model, list_valSSIM, test_loader, test_dataset, use_perceptual_loss = ploss)
