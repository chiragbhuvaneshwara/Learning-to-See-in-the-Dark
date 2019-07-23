import torch
import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class VggModelFeatures(nn.Module):
    def __init__(self, feature_extracting, pretrained=True):
        super(VggModelFeatures, self).__init__()
        
        #load the pretrained model ==> only the feature extractor
        vgg11_bn = models.vgg19_bn(pretrained=pretrained)
        self.model = nn.Sequential()
        self.model.features = vgg11_bn.features
        set_parameter_requires_grad(self.model, feature_extracting)

    def forward(self, x):

        x = self.model.features(x)
        x = x.squeeze()

        return x

