
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
num_workers = 0
# how many samples per batch to load
batch_size = 10

# convert data to a normalized torch.FloatTensor
train_data_path= "./images/train/"
val_data_path= "./images/val/"
test_data_path = "./images/test/"
base_transform = transforms.Compose([
    transforms.Resize(333),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#use data augmentation for the training set
augmented_transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.Compose([
        transforms.RandomAffine((-15,15),translate = (0.1,0.1),scale=(0.8,1.2),shear=(-3,3)),
        transforms.RandomHorizontalFlip()
        ]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=augmented_transform)
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=base_transform)
val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=base_transform)
test_loader  = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4) 


# specify the image classes
classes = ["drums","flute","guitar","piano","trombone","violin"]

num_classes = 6

#experiment name
experiment_name = "inception_v3_conv"

#---------------define the CNN architecture------------------------

# ---   vgg16 with batch normalization
'''
model = torchvision.models.vgg16_bn(pretrained=True)
#freeze initial layers
i=0
for name, param in model.features.named_parameters():
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
'''

# ---  inception_v3

model = torchvision.models.inception_v3(pretrained=True)

print(model)

#freeze features extractors except for last ones: 270-292
i=0
for name, param in model.named_parameters():
    i+=1
    if i< 270:
        param.requires_grad = False
classifier = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(), 
                           nn.Dropout(p=0.5),
                           nn.Linear(512, num_classes))
model.fc = classifier


# --- resnet-152
'''
model = torchvision.models.resnet152(pretrained=True)

i=0
for name, param in model.named_parameters():
    i+=1
    if i< 440:
        param.requires_grad = False
classifier = nn.Sequential(
                           nn.Linear(2048, num_classes))
model.fc = classifier
'''
'''
print(model)
# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.5)

# number of epochs to train the model
n_epochs = 30

valid_loss_min = np.Inf # track change in validation loss

#number of successive epochs without validation loss as stop criteria
max_no_improvement = 10

no_improvement = 0

print("Started training...")

T_Loss = []
V_Loss = []
T_acc = []
V_acc = []
start = timer()

#for epoch in range(1, n_epochs+1):
epoch = 0
while(no_improvement<max_no_improvement):
    epoch += 1
    #update optimizer
    if (epoch == 7):
        #increase learning rate to 0.01 at epoch 7
        state = optimizer.state_dict()
        state['param_groups'][0]['lr'] = 0.01
        optimizer.load_state_dict(state)
        #halves learning rate every 10 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif (epoch > 7):
        scheduler.step()

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    train_correct = 0.0
    train_total = 0.0
    val_correct = 0.0
    val_total = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(len(correct)):
            train_correct += correct[i].item()
            train_total += 1

    T_acc.append(train_correct/train_total)

    print('\nTraining Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * train_correct / train_total,
    train_correct, train_total))

    ######################    
    # validate the model #
    ######################
    eval_start = timer()
    model.eval()
    for data, target in val_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(len(correct)):
            val_correct += correct[i].item()
            val_total += 1
    print('Eval time: ',timer()-eval_start)
    V_acc.append(val_correct/val_total)

    print('\nValidation Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * val_correct / val_total,
    val_correct, val_total))

    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(val_loader.dataset)

    T_Loss.append(train_loss)
    V_Loss.append(valid_loss)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'models/model_'+experiment_name+'.pt')
        valid_loss_min = valid_loss
        no_improvement = 0
    else:
        no_improvement += 1

    print('Total time: ',timer()-start)


T_Loss = np.array(T_Loss)
V_Loss = np.array(V_Loss)
T_acc = np.array(T_acc)
V_acc = np.array(V_acc)

# plot the training and validation loss and accuracy
N = n_epochs
N = len(V_acc)
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), T_Loss, label="train_loss")
plt.plot(np.arange(0, N), V_Loss, label="val_loss")
plt.plot(np.arange(0, N), T_acc, label="train_accuracy")
plt.plot(np.arange(0, N), V_acc, label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss and Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plots/LossAccPlot_"+experiment_name+".png")

print("Started testing...")
test_start = timer()
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))


model.load_state_dict(torch.load('models/model_'+experiment_name+'.pt'))
model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(len(correct)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(num_classes):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
print('test time: ',timer()-test_start)
'''