import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#%matplotlib inline

import cv2
from timeit import default_timer as timer



# Residual block in bottleneck style
# input: input depth
# output with same shape as input
class ResidualBottleNeckBlock(nn.Module):
    def __init__(self, input_channels, bottleneck_ratio):
        super(ResidualBottleNeckBlock, self).__init__()
        self.reduction = nn.Conv2d(input_channels,input_channels//bottleneck_ratio,1)
        self.batch_red = nn.BatchNorm2d(input_channels//bottleneck_ratio)
        self.conv = nn.Conv2d(input_channels//bottleneck_ratio,input_channels//bottleneck_ratio,3,padding=1)
        self.batch_conv = nn.BatchNorm2d(input_channels//bottleneck_ratio)
        self.expansion = nn.Conv2d(input_channels//bottleneck_ratio,input_channels,1)
        self.batch_exp = nn.BatchNorm2d(input_channels)


    def forward(self, x):

        out = F.relu(self.batch_red(self.reduction(x)))
        out = F.relu(self.batch_conv(self.conv(out)))
        out = self.batch_exp(self.expansion(out))

        return F.relu(out+x)

# Squeeze and Excitation block
class SEblock(nn.Module):
    def __init__(self, input_channels, height, width, ratio):
        super(SEblock, self).__init__()
        self.input_channels = input_channels

        self.squeeze = nn.AvgPool2d((height,width),(height,width))
        self.fc = nn.Linear(input_channels,input_channels//ratio)
        self.excite = nn.Linear(input_channels//ratio,input_channels)

    def forward(self, x):

        se = self.squeeze(x)
        se = se.view(-1,self.input_channels)
        se = F.relu(self.fc(se))
        se = torch.sigmoid(self.excite(se))
        se = se.view(-1,self.input_channels,1,1)

        x = x * se.expand_as(x)

        return x


#Squeee and Excitation Residual block with bottleneck style
class SE_ResidualBottleNeckBlock(nn.Module):
    def __init__(self, input_channels, height, width, bottleneck_ratio, se_ratio):
        super(SE_ResidualBottleNeckBlock, self).__init__()
        self.reduction = nn.Conv2d(input_channels,input_channels//bottleneck_ratio,1)
        self.batch_red = nn.BatchNorm2d(input_channels//bottleneck_ratio)
        self.conv = nn.Conv2d(input_channels//bottleneck_ratio,input_channels//bottleneck_ratio,3,padding=1)
        self.batch_conv = nn.BatchNorm2d(input_channels//bottleneck_ratio)
        self.expansion = nn.Conv2d(input_channels//bottleneck_ratio,input_channels,1)
        self.batch_exp = nn.BatchNorm2d(input_channels)

        self.se = SEblock(input_channels, height, width, se_ratio)


    def forward(self, x):

        out = F.relu(self.batch_red(self.reduction(x)))
        out = F.relu(self.batch_conv(self.conv(out)))
        out = self.batch_exp(self.expansion(out))

        # SE must be applied prior to the identity addition
        out = self.se(out)

        return F.relu(out+x)

class DenseBlock(nn.Module):
    def __init__(self, input_channels, inner_output, layers):
        super(DenseBlock, self).__init__()

        self.layers = layers
        self.reduction = nn.ModuleList([nn.Conv2d(input_channels+i*inner_output, inner_output,1) for i in range(layers)])
        self.conv = nn.ModuleList([nn.Conv2d(inner_output, inner_output,3,padding=1) for i in range(layers)])
        self.norm1 = nn.ModuleList([nn.BatchNorm2d(inner_output) for i in range(layers)])
        self.norm2 = nn.ModuleList([nn.BatchNorm2d(inner_output) for i in range(layers)])


    def forward(self, x):

        for i in range(self.layers):
            out = self.reduction[i](x)
            out = self.norm1[i](out)
            out = F.relu(out)
            out = self.conv[i](out)
            out = self.norm2[i](out)
            out = F.relu(out)
            x = torch.cat((x,out),1)

        return x