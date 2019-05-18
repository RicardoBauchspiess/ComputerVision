
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
    print('CUDA is not available.  Testing on CPU ...')
else:
    print('CUDA is available!  Testing on GPU ...')

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


#--------------------------------------------------------------------------------
#--------------- START DATA LOADERS ---------------------------------------------
#----------------------------------------------------------------------------

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 1

# convert data to a normalized torch.FloatTensor
test_data_path = "./images/test/"
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

denormalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])


test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=base_transform)
test_loader  = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4) 


# specify the image classes
classes = ["drums","flute","guitar","piano","trombone","violin"]

num_classes = 6

#experiment name
experiment_name = "0"

#---------------define the CNN architecture------------------------

# ---   vgg16 with batch normalization
model = torchvision.models.vgg16_bn(pretrained=True)
#freeze initial layers
i=0
for name, param in model.features.named_parameters():
    #print(name)
    i+=1
    if i < 47:
        param.requires_grad = False

#create classifier
classifier = nn.Sequential(nn.Linear(25088, 512),
                           nn.ReLU(), 
                           nn.Dropout(p=0.5),
                           nn.Linear(512, num_classes),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier


print(model)


# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()




model.load_state_dict(torch.load('models/model_'+experiment_name+'.pt'))
model.eval()
# iterate over test data
print (len(test_loader))
index = 0
for data, target in test_loader:

    group = np.zeros((548,448,3),float)

    features = data
    image = data[0,:,:,:]
    image = denormalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    show_image = cv2.copyMakeBorder(image, top=25, bottom=25, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    correct_index = target.numpy()[0]
    cv2.putText(show_image, classes[correct_index], (0, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA) 
    
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target, features = data.cuda(), target.cuda(), features.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    prediction = np.argmax(output.cpu().detach().numpy())
    
    #write prediction on image
    if (prediction == correct_index):
        color = (0,255,0)
    else:
        color = (0,0,255)
    cv2.putText(show_image, classes[prediction], (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, lineType=cv2.LINE_AA) 
    #cv2.imshow("image",show_image)

    group[0:274,0:224,:] = show_image

    index = 0
    skip = 0

    #Get feature maps
    for module_name, module in model._modules.items():
        if (module_name == 'features'):
            num_modules = len(module._modules.items())
            for sub_module_name, sub_module in module._modules.items():
                index += 1
                #forward one layer
                features = sub_module(features)
                num_maps = features.shape[1]
                feature_list = features[0,:,:,:]
                for i in range(num_maps):
                    feature_map = feature_list[i,:,:].cpu().detach().numpy()
                    feature_map = feature_map.clip(0, 1)
                    feature_map = cv2.resize(feature_map,(int(224),int(224)))
                    feature_map = cv2.cvtColor(feature_map,cv2.COLOR_GRAY2BGR)
                    
                    feature_image = image*feature_map

                    #show feature map
                    feature_map = cv2.copyMakeBorder(feature_map, top=25, bottom=25, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
                    cv2.putText(feature_map, 'Layer: '+str(index)+'/'+str(num_modules), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), lineType=cv2.LINE_AA)
                    cv2.putText(feature_map, 'Map: '+str(i)+'/'+str(num_maps), (0, feature_map.shape[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), lineType=cv2.LINE_AA) 
                    #cv2.imshow("feature map",feature_map)
                    group[274:,0:224,:] = feature_map


                    #show feature on image
                    feature_image = cv2.copyMakeBorder(feature_image, top=25, bottom=25, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
                    cv2.putText(feature_image, 'Layer: '+str(index)+'/'+str(num_modules), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), lineType=cv2.LINE_AA)
                    cv2.putText(feature_image, 'Map: '+str(i)+'/'+str(num_maps), (0, feature_map.shape[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), lineType=cv2.LINE_AA) 
                    #cv2.imshow("feature image",feature_image)
                    group[0:274,224:,:] = feature_image

                    cv2.imshow("Detection/feature_map",group)

                    #get user input
                    key = cv2.waitKey(0)
                    if (key == 27):
                        break
                    elif (key == 83):
                        index = index
                    elif ( key == 82):
                        break
                    else:
                        skip = 1
                        break
                if (key == 27 or skip == 1):
                    break        
        else:
            break
        if (skip == 1 or key == 27):
            break

    if (key == 27):
        break

