import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import os

from torchvision import datasets
import torchvision.transforms as transforms

from random import randrange

class NshotNwayOmniglot(datasets.Omniglot):

	def __init__(self, root, download = False, background = True, transform=None, shots = 1, nWay = 1):
		super(NshotNwayOmniglot, self).__init__(root = root, background = background, download = download, transform = transform)
	
	def __len__(self):
		return super(NshotNwayOmniglot, self).__len__()
	
	# get anchor image and classes
	def __getitem__(self,idx):
		return super(NshotNwayOmniglot, self).__getitem__(idx)

	# load data matching or not with target classes
	def getimages(self, classes, target):
		
		imgs = []
		for i in range(classes.size()[0]):
			if target[i] == 1:
				idx = randrange(len(self._character_images[classes[i]]))
				image_name = self._character_images[classes[i]][idx][0]
				idx_c = classes[i]
			else:
				idx_c = classes[i]
				while(idx_c == classes[i]):
					idx_c = randrange(len(self._character_images))

				idx = randrange(len(self._character_images[idx_c]))
				image_name = self._character_images[idx_c][idx][0]
			image_path = os.path.join(self.target_folder, self._characters[idx_c], image_name)
			image = Image.open(image_path, mode='r').convert('L')
			if self.transform:
				image = self.transform(image)
			imgs.append(image)
		return torch.cat(imgs).unsqueeze(1)

	# load positive and negative images for triplet loss
	def gettriplet(self, classes):
		positive = []
		negative = []
		for i in range(classes.size()[0]):
			# get positive image
			idx = randrange(len(self._character_images[classes[i]]))
			image_name = self._character_images[classes[i]][idx][0]
			idx_c = classes[i]
			image_path = os.path.join(self.target_folder, self._characters[idx_c], image_name)
			image = Image.open(image_path, mode='r').convert('L')
			if self.transform:
				image = self.transform(image)
			positive.append(image)

			# get negative image
			idx_c = classes[i]
			while(idx_c == classes[i]):
				idx_c = randrange(len(self._character_images))

			idx = randrange(len(self._character_images[idx_c]))
			image_name = self._character_images[idx_c][idx][0]
			image_path = os.path.join(self.target_folder, self._characters[idx_c], image_name)
			image = Image.open(image_path, mode='r').convert('L')
			if self.transform:
				image = self.transform(image)
			positive.append(image)

		return torch.cat(positive).unsqueeze(1), torch.cat(negative).unsqueeze(1)

	# load positive, negative and second negative images for quadruplet loss
	def getquadruplet(self, classes):
		positive = []
		negative = []
		extra = []
		return positive, negative, extra


class ContrastiveLoss(nn.Module):
	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, x, target):
		return torch.mean(target*x*x+(1-target)*torch.pow(torch.clamp(self.margin-x,min=0.0),2) )/2

class TripletLoss(nn.Module):
	def __init__(self, margin=2.0):
		super(TripletLoss, self).__init__()
		self.margin = margin

	def forward(self, x, negative, positive):
		d1 = (x-positive).pow(2).sum(1)
		d2 = (x-negative).pow(2).sum(1)
		return torch.clamp(self.margin+d1-d2,min=0.0).mean()

class QuadrupletLoss(nn.Module):
	def __init__(self, margin1=2.0, margin2 = 1.0):
		super(QuadrupletLoss, self).__init__()
		self.margin1 = margin1
		self.margin2 = margin2

	def forward(self, x, negative, positive, extra):
		d1 = (x-positive).pow(2).sum(1)
		d2 = (x-negative).pow(2).sum(1)
		d3 = (negative-extra).pow(2).sum(1)
		return (torch.clamp(self.margin1+d1-d2,min=0.0)+torch.clamp(self.margin2+d1-d3,min=0.0)).mean()

