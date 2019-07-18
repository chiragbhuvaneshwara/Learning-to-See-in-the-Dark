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

from models_pytorch import *
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
learning_rate = 1e-4
learning_rate_decay = 0.99
reg=0.001
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

    # Initialize the model for this run
    if name == 'simpleUNET':
        model = simpleUNET()
        # selectedModel = simpleUNET()

    elif name == 'unet':
        model = unet()
        # selectedModel = unet()
    elif name == 'FPN':
        model = FPN(Bottleneck, [2,2,2,2])
    else:
        print('Name variable passed is incorrect')
        return None

    model.apply(weights_init)

    # Print the model we just instantiated
    print(model)

    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    # Train the model
    lr = learning_rate
    Loss = []                           #to save all the model losses
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

            loss = criterion(outputs, exp_images)
            Loss.append(loss)               #save the loss so we can get accuracies later

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
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
                torch.save(model.state_dict(),'models/ESmodel'+str(epoch+1)+'.ckpt')

            print('Avg Validation MSE on all the {} Val images is: {} '.format(total, current_MSE))
            print('Avg Validation SSIM on all the {} Val images is: {} '.format(total, overallSSIM/total))

    # Training loss and Val MSE curves
    plt.plot(valMSE)
    title='AvgValMSE_vs_Epochs'
    plt.ylabel('Avg Validation MSE')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig('plots/'+name+title+'.png')

    plt.plot(valSSIM)
    title='AvgValSSIM_vs_Epochs'
    plt.ylabel('Avg Validation SSIM')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig('plots/'+name+title+'.png')

    plt.plot(Loss)
    title='Loss_vs_Iterations'
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.savefig('plots/'+name+title+'.png')

    print('Testing ..............................')

    # last_model = model
    best_id = np.argmin(valMSE)
    bestESmodel = model

    bestESmodel.load_state_dict(torch.load('models/ESmodel'+str(best_id+1)+'.ckpt'))
    bestESmodel = bestESmodel.to(device)


    # last_model.eval()
    bestESmodel.eval()
    
        
    with torch.no_grad():

        overallSSIM = 0
        MSE = 0
        count = 0 
        for in_images, exp_images in test_loader:
            count += 1
            in_images = in_images.to(device)
            exp_images = exp_images.to(device)
            outputs = bestESmodel(in_images)
            
            MSE += torch.sum((outputs - exp_images) ** 2)
            # Visualize the output of the best model against ground truth
            outputs_py = outputs.cpu()
            exp_images_py = exp_images.cpu()
            
            f, axarr = plt.subplots(1,2)
            title='Output of the best model vs Ground truth Image'
            plt.suptitle(title)
            axarr[0].imshow(trans(outputs_py[0]))
            axarr[1].imshow(trans(exp_images_py[0]))
            print('Writing model predictions to disk:')
            print('Saving image_%d.png'%(count))
            plt.savefig('images/'+name+'_%d.png'%(count))

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
# name = 'simpleUNET'
# name = 'unet'
name = 'FPN'
trainAndTestModel(name)


