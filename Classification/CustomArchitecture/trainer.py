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

import blocks
import nets

from nets import *

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

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32
# percentage of training set to use as validation
valid_size = 0.1

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.Compose([
        transforms.RandomAffine((-15,15),translate = (0.1,0.1),scale=(0.8,1.2),shear=(-3,3)),
        transforms.RandomHorizontalFlip()
        ]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# convert data to a normalized torch.FloatTensor
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


# choose the training and test datasets
train_data = datasets.CIFAR10('data/CIFAR10', train=True,
                              download=True, transform=transform)
valid_data = datasets.CIFAR10('data/CIFAR10', train=True,
                              download=True, transform=transform2)
test_data = datasets.CIFAR10('data/CIFAR10', train=False,
                             download=True, transform=transform2)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# specify the image classes
classes = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,16, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(16)

        #self.FeatureExtractor = SE_BottleNeckResNet(32,32,32,2,4,4,4)
        #self.FeatureExtractor = BottleNeckResNet(32,2,4,4)
        self.FeatureExtractor = DenseNet(16,8,4,1,3) 

        
        self.classifier_0 = nn.Linear(64+32,512)
        self.classifier = nn.Linear(512,10)
        
        self.avgPool4 = nn.AvgPool2d(4,4)
        self.avgPool8 = nn.AvgPool2d(8,8)

        self.dropout = nn.Dropout(0.5)
        
        '''
        self.classifier = nn.Sequential(
            nn.AvgPool2d(8,8),
            nn.Linear(64+32,512),
            nn.Dropout(0.5),
            nn.Linear(512,10)
            )
        '''

    def forward(self, x):
        # create low level semantic representations
        x = self.bn0(F.relu(self.conv1(x)))

        x = self.FeatureExtractor(x)

        #x = self.classifier(x)
        
        x = self.avgPool8(x)
        x = x.view(-1,64+32)

        # classification
        x = self.classifier_0(x)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

# create a complete CNN
model = Net()

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum = 0.1)

# divide learning rate by gamma every scheduler step, 
# scheduler steps accordingly to the validation loss decrease
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.1)

valid_loss_min = np.Inf # track change in validation loss

print("Started training...")

button = 0

T_Loss = []
V_Loss = []
T_acc = []
V_acc = []
start = timer()

max_epochs = np.Inf  # max number of epochs before ending training 
max_no_improvement = 5  #max successive epochs without val loss reduction before ending training

epoch = 0
no_improvement = 0

# keep training until it stops improving significantly or max epochs was reached
while( (no_improvement < max_no_improvement) and (epoch < max_epochs)):
    epoch += 1
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

        #loss = new_criterion(output,train_target)
        loss = criterion(output,target)

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
    model.eval()
    for data, target in valid_loader:
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
    
    V_acc.append(val_correct/val_total)

    print('\nValidation Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * val_correct / val_total,
    val_correct, val_total))

    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

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
        torch.save(model.state_dict(), 'models/model_cifar127.pt')
        valid_loss_min = valid_loss
        no_improvement = 0
    else:
        no_improvement += 1

    scheduler.step()

    print('Total time: ',timer()-start)

T_Loss = np.array(T_Loss)
V_Loss = np.array(V_Loss)
T_acc = np.array(T_acc)
V_acc = np.array(V_acc)

# plot the training loss and accuracy
N = epoch
N = len(V_acc)

plt.style.use("ggplot")
plt.figure(0)
plt.plot(np.arange(0, N), T_acc, label="train_accuracy")
plt.plot(np.arange(0, N), V_acc, label="val_accuracy")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("plots/Plot_cifar127_Accuracy.png")

plt.style.use("ggplot")
plt.figure(1)
plt.plot(np.arange(0, N), T_Loss, label="train_loss")
plt.plot(np.arange(0, N), V_Loss, label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plots/Plot_cifar127_Loss.png")

print("Started testing...")
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))


model.load_state_dict(torch.load('models/model_cifar127.pt'))
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

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

