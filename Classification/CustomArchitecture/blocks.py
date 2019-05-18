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

class ResidualBottleNeckBlock(nn.Module):
    def __init__(self, input_channels):
    #def __init__(self, input_channels):
        super(ResidualBottleNeckBlock, self).__init__()
        self.reduction = nn.Conv2d(input_channels,input_channels//4,1)
        self.batch_red = nn.BatchNorm2d(input_channels//4)
        self.conv = nn.Conv2d(input_channels//4,input_channels//4,3,padding=1)
        self.batch_conv = nn.BatchNorm2d(input_channels//4)
        self.expansion = nn.Conv2d(input_channels//4,input_channels,1)
        self.batch_exp = nn.BatchNorm2d(input_channels)


    def forward(self, x):

        out = F.relu(self.batch_red(self.reduction(x)))
        out = F.relu(self.batch_conv(self.conv(out)))
        out = self.batch_exp(self.expansion(out))

        return F.relu(out+x)