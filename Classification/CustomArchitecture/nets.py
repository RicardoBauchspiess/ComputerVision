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

# inputs: initial_depth = first input layer depth
#		  layers_per_scale = number os residual layers per scale
#		  scales = number of scale, scales-1 = number of dimensions reductions
#		  bottleneck_ratio = reduction ratio for the bottleneck in the residual block
class BottleNeckResNet(nn.Module):
    def __init__(self, initial_depth, layers_per_scale, scales, bottleneck_ratio):
        super(BottleNeckResNet, self).__init__()

        self.layers = layers_per_scale
        self.scales = scales
        self.residualblocks = nn.ModuleList([nn.ModuleList([ResidualBottleNeckBlock(initial_depth*(2**j),bottleneck_ratio) for i in range(layers_per_scale)]) for j in range(scales)])
        self.scaleblocks = nn.ModuleList([nn.Conv2d(initial_depth*(2**(i-1)), initial_depth*(2**(i)),3,stride = 2, padding = 1) for i in range(1,scales)])

    def forward(self, x):
    	# apply sequence of residualblocks, then reduce scale with convolution with stride 2
        for i in range(self.scales):
        	for j in range(self.layers):
        		x = self.residualblocks[i][j](x)
        	if (i < self.scales-1):
        		x = self.scaleblocks[i](x)

        return x

# inputs: initial_depth = first input layer depth
#		  height = first input layer height
#		  width = first input layer width
#		  layers_per_scale = number os residual layers per scale
#		  scales = number of scale, scales-1 = number of dimensions reductions
#		  bottleneck_ratio = reduction ratio for the bottleneck in the residual block
#		  se_ratio = activation ratio in the SE block
class SE_BottleNeckResNet(nn.Module):
    def __init__(self, initial_depth, height, width, layers_per_scale, scales, bottleneck_ratio, se_ratio):
        super(SE_BottleNeckResNet, self).__init__()

        self.layers = layers_per_scale
        self.scales = scales
        self.se_residualblocks = nn.ModuleList([nn.ModuleList([SE_ResidualBottleNeckBlock(
        	initial_depth*(2**j),height//(2**j), width//(2**j), bottleneck_ratio, se_ratio) for i in range(layers_per_scale)]) for j in range(scales)])
        self.scaleblocks = nn.ModuleList([nn.Conv2d(initial_depth*(2**(i-1)), initial_depth*(2**(i)),3,stride = 2, padding = 1) for i in range(1,scales)])

    def forward(self, x):
    	# apply sequence of residualblocks, then reduce scale with convolution with stride 2
        for i in range(self.scales):
        	for j in range(self.layers):
        		x = self.se_residualblocks[i][j](x)
        	if (i < self.scales-1):
        		x = self.scaleblocks[i](x)

        return x


# inputs: initial_depth = first input layer depth
#		  inner_depth = output_depth of each convolution in the dense block
#	      layers_per_block = number of 1x1+3x3 convolution layers per dense block
#		  layers_per_scale = number of dense blocks per scale
#		  scales = number of scale, scales-1 = number of dimensions reductions
class DenseNet(nn.Module):
    def __init__(self, initial_depth, inner_depth, layers_per_block, layers_per_scale, scales):
        super(DenseNet, self).__init__()

        self.layers = layers_per_scale
        self.scales = scales
        self.denseblocks = nn.ModuleList([nn.ModuleList([DenseBlock(initial_depth*(2**j),inner_depth,layers_per_block) for i in range(layers_per_scale)]) for j in range(scales)])
        self.transition = nn.ModuleList([nn.ModuleList([
        						nn.Conv2d(initial_depth*(2**j)+inner_depth*layers_per_block,initial_depth*(2**j),1) for i in range(layers_per_scale-1)]) for j in range(scales)])
        self.scale = nn.ModuleList([nn.Conv2d(initial_depth*(2**(i-1))+inner_depth*layers_per_block, initial_depth*(2**(i)),3,1) for i in range(1,scales)])
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        for i in range(self.scales):
        	for j in range(self.layers):
        		x = self.denseblocks[i][j](x)
        		if (j < self.layers-1):
        			x = self.transition[i][j]
        	if (i < self.scales-1):
        		x = F.relu(self.scale[i](x))
        		x = self.pool(x)

        return x