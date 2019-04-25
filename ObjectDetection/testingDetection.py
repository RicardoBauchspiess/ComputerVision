
import torch
import numpy as np

import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
import torch.utils.data as data

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#%matplotlib inline

import cv2
from timeit import default_timer as timer




# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


#--------------------------------------------------------------------------------
#--------------- START DATA LOADERS ---------------------------------------------
#----------------------------------------------------------------------------

# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 10

# convert data to a normalized torch.FloatTensor
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#use data augmentation for the training set
augmented_transform = transforms.Compose([
    transforms.Compose([
        transforms.RandomAffine((-15,15),translate = (0.1,0.1),scale=(0.8,1.2),shear=(-3,3)),
        transforms.RandomHorizontalFlip()
        ]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

denormalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

#print(dir(datasets))
#print(torchvision.__file__)



# choose the training and test datasets
#train_data = datasets.VOCDetection('VOCDetectionData', image_set = 'train',
#                              download=True, transform=augmented_transform)
test_data = datasets.VOCDetection('VOCDetectionData', image_set = 'val',
                              download=True, transform=base_transform)

#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

print(len(test_data))

for i in range(len(test_data)):
    img, target = test_data.__getitem__(i)


    image = img
    image = denormalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


    annotation = target['annotation']['object']

    if isinstance(annotation,dict):
        xmin = int(annotation['bndbox']['xmin'])
        ymin = int(annotation['bndbox']['ymin'])
        xmax = int(annotation['bndbox']['xmax'])
        ymax = int(annotation['bndbox']['ymax'])
        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        name = annotation['name']
        cv2.putText(image,name,(xmin,ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        #print('1')
        #print(annotation['name'],":",annotation['bndbox'] )
    else:
        #print(len(annotation))
        for ann in annotation:
            xmin = int(ann['bndbox']['xmin'])
            ymin = int(ann['bndbox']['ymin'])
            xmax = int(ann['bndbox']['xmax'])
            ymax = int(ann['bndbox']['ymax'])
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
            name = ann['name']
            cv2.putText(image,name,(xmin,ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
            #print(ann['name'],":",ann['bndbox'] )


    #detection = target['annotation']['object']
    #print(len(target['annotation']['object']['name']),"\n",target['annotation']['object']['name'],"\n")
    #print(target['annotation']['object'])


    '''
    for detection in target['annotation']['object']:
        print(detection," ",target['annotation']['object'][detection])
    '''

    cv2.imshow("image",image)
    key = cv2.waitKey(0)
    if (key == 27):
        break

