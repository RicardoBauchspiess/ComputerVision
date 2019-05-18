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

import blocks
from blocks import *


# define the CNN architecture
class BottleNeckResNet(nn.Module):
    def __init__(self, initial_depth, layers_per_scale, scales):
        super(BottleNeckResNet, self).__init__()
        # convolutional layer (sees 32x32x1 image tensor

        self.layers = layers_per_scale
        self.scales = scales
        self.residualblocks = nn.ModuleList([nn.ModuleList([ResidualBottleNeckBlock(initial_depth*(2**j)) for i in range(layers_per_scale)]) for j in range(scales)])
        self.scaleblocks = nn.ModuleList([nn.Conv2d(initial_depth*(2**(i-1)), initial_depth*(2**(i)),3,stride = 2, padding = 1) for i in range(1,scales)])

    def forward(self, x):
        # add sequence of convolutional and max pooling layers

        for i in range(self.scales):
        	for j in range(self.layers):
        		x = self.residualblocks[i][j](x)
        	if (i < self.scales-1):
        		x = self.scaleblocks[i](x)

        return x
