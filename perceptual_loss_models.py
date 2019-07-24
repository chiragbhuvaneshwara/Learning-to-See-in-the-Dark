from collections import namedtuple

import torch
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class VggModelFeatures(torch.nn.Module):
    def __init__(self, feature_extracting, pretrained=True):
        super(VggModelFeatures, self).__init__()
        
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        if feature_extracting:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


# class Interpolate(nn.Module):
#     def __init__(self, size, mode):
#         super(Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode
        
#     def forward(self, x):
#         x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
#         return x

# class VggModelFeatures(nn.Module):
#     def __init__(self, feature_extracting, pretrained=True):
#         super(VggModelFeatures, self).__init__()
        
#         #load the pretrained model ==> only the feature extractor
#         vgg11_bn = models.vgg19_bn(pretrained=pretrained)
#         self.model = nn.Sequential()
#         self.model.features = vgg11_bn.features
#         set_parameter_requires_grad(self.model, feature_extracting)

#     def forward(self, x):

#         x = self.model.features(x)
#         x = x.squeeze()

#         return x

