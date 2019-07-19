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

from gan_models import *
from datasetLoader_pytorch import SeeingIntTheDarkDataset

trans = transforms.ToPILImage()

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

#sitd_dataset = SeeingIntTheDarkDataset('dataset/Sony/short_temp_down/', 'dataset/Sony/long_temp_down/', transforms.ToTensor())
sitd_dataset = SeeingIntTheDarkDataset('dataset/Sony/short_down/', 'dataset/Sony/long_down/', transforms.ToTensor())
print('Input Image Size:')
print(sitd_dataset[0][0].size())

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.cuda.set_device(1)

print('Using device: %s'%device)

#### final params
num_training= 2100
num_validation = 200
num_test = 397

num_epochs = 100
learning_rate = 1e-5
learning_rate_decay = 0.9
reg = 0.001
batch_size = 10

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

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def trainAndTestModel(name):

    generator = unet_in_generator(train=True)
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=reg)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=reg)

    # Train the model
    lr = learning_rate
    GLoss = []                           #to save all the model losses
    DLoss = []
    valMSE = []
    valSSIM = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
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
            z = torch.rand((batch_size, 30, 45, 512))

            gen_images = generator(in_images, z)
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
            fake_loss = criterion(discriminator(gen_images.detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2
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

            overallSSIM = 0
            MSE = 0
            for in_images, exp_images in val_loader:
                in_images = in_images.to(device)
                exp_images = exp_images.to(device)
                z = torch.zeros((batch_size, 30, 45, 512))
                outputs = generator(in_images, z)

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
                torch.save(generator.state_dict(),'models/ESmodel'+str(epoch+1)+'.ckpt')

            print('Avg Validation MSE on all the {} Val images is: {} '.format(total, current_MSE))
            print('Avg Validation SSIM on all the {} Val images is: {} '.format(total, overallSSIM/total))

    # Training loss and Val MSE curves
    plt.plot(valMSE)
    title='AvgValMSE_vs_Epochs'
    plt.ylabel('Avg Validation MSE')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig('plots/'+name+title+'.png')
    plt.close()

    plt.plot(valSSIM)
    title='AvgValSSIM_vs_Epochs'
    plt.ylabel('Avg Validation SSIM')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig('plots/'+name+title+'.png')
    plt.close()

    plt.plot(GLoss)
    title='GeneratorLoss_vs_Iterations'
    plt.ylabel('Generator Loss')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.savefig('plots/'+name+title+'.png')
    plt.close()

    plt.plot(DLoss)
    title='DiscriminatorLoss_vs_Iterations'
    plt.ylabel('Discriminator Loss')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.savefig('plots/'+name+title+'.png')
    plt.close()

    print('Testing ..............................')

    # last_model = model
    best_id = np.argmin(valMSE)
    bestESmodel = generator

    bestESmodel.load_state_dict(torch.load('models/ESmodel'+str(best_id+1)+'.ckpt'))
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
                title='Input ('+str(nonZero)+'Px) vs Model Output vs Ground truth'
                plt.suptitle(title)
                axarr[0].imshow(trans(in_images_py[i]))
                axarr[1].imshow(trans(outputs_py[i]))
                axarr[2].imshow(trans(exp_images_py[i]))
                
                plt.savefig('images/'+name+'_%d.png'%(count))
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

    torch.save(bestESmodel.state_dict(), 'models/bestESModel_'+name+'.ckpt')

###############################################################################################################################################
# parameters to select different models ==> Just change here. 
name = 'unet'

trainAndTestModel(name)


