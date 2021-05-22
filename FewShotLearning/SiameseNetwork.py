import torch
import torch.nn as nn
import torch.nn.functional as F

# Classification models are imported, so they may be used as alternative backbones for the siamese network
import sys
sys.path.append('../')
from Classification.Models.ResNet import *


class Linearizer(nn.Module):
	def __init__(self,input_features, output_features):
		super(Linearizer, self).__init__()
		self.fc = nn.Linear(input_features, output_features)
	def forward(self, x):
		x = x.view(x.size()[0],-1)
		return self.fc(x)



class SiameseNetwork(nn.Module):
	def __init__(self, feature_extractor = None, exit_features = 0, compare_features = 4096):
		super(SiameseNetwork, self).__init__()

		# Enables loading custom networks for the feature_extractor layer
		if feature_extractor is not None and exit_features > 0:
			# Use feature extractor layers from custom network
			self.features = feature_extractor

			# Custom feature extractor may be used for input images of unknown sizes,
			# so global average pooling is used here
			self.fc = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				Linearizer(exit_features,compare_features),
				nn.Sigmoid()
				)
		else:
			# network from paper
			self.features = nn.Sequential(
				nn.Conv2d(1,64,10),
				nn.ReLU(inplace = True),
				nn.MaxPool2d(2),
				nn.Conv2d(64,128,7),
				nn.ReLU(inplace = True),
				nn.MaxPool2d(2),
				nn.Conv2d(128,128,4),
				nn.ReLU(inplace = True),
				nn.MaxPool2d(2),
				nn.Conv2d(128,256,4),
				nn.ReLU(inplace = True)
				)
			self.fc = nn.Sequential(
				Linearizer(9216,compare_features),
				nn.Sigmoid()
				)

		self.classifier = nn.Linear(compare_features, 1)

	def forward(self, x1, x2):
		x1 = self.fc(self.features(x1))
		x2 = self.fc(self.features(x2))
		dif = torch.abs(x1-x2)
		return self.classifier(dif)
