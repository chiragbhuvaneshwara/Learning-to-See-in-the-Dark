from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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


d = SeeingIntTheDarkDataset('dataset/Sony/short_temp_down/', 'dataset/Sony/long_temp_down/')
print(d[0][0].shape, d[0][1].shape)