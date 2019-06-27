# import numpy as np
# from skimage import io, exposure, img_as_uint, img_as_float
# import rawpy

# im2 = io.imread('dataset/Sony/short_temp_down/00002_08_0.1s.png')
# print(im2.dtype)

# def downsample(in_path):

#     with rawpy.imread(in_path) as raw:
#         rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)

#     print(rgb.shape)
#     print(rgb.shape[0]/4, rgb.shape[1]/4)
#     #rgb_downsample = resize(rgb, (1280 , 720), anti_aliasing=True)
    
#     #rgb = Image.fromarray(rgb)
#     #rgb_downsample = rgb.resize((1280 , 720), resample=Image.LANCZOS)

#     #rgb_np = np.array(rgb)
#     #print(rgb_np.shape)
#     #plt.imshow(rgb)
#     #plt.show()

#     #return rgb_downsample

# #in_path = 'dataset/Sony/short/00001_00_0.1s.ARW'
# #downsample(in_path)
# import os
# d1 = sorted(os.listdir('dataset/Sony/short_temp_down/'))
# d2 = sorted(os.listdir('dataset/Sony/long_temp_down/'))

# print(d1[1].split('_'))

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

for j, (in_images, exp_images) in enumerate(train_loader):
    in_images = in_images.squeeze(0).permute(1,2,0)
    exp_images = exp_images.squeeze(0).permute(1,2,0)

    print('in',in_images.size())
    print('exp',exp_images.size())


    images= [in_images.numpy(), exp_images.numpy()]
    titles = ["lowLight", "corrected"]
    fig=plt.figure(figsize=(16, 16))

    columns = 2
    rows = 1

    for i in range(1,3):

        ax = plt.subplot(rows, columns, i)
        ax.set_title(titles[i-1])
        ax.imshow(images[i-1])
    plt.show()

    
    if j == 4:
        break
    
