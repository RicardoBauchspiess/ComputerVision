import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

from timeit import default_timer as timer
import os

from SiameseNetwork import *
from utils import *


use_cuda = torch.cuda.is_available()
#use_cuda = False

if (use_cuda):
	print('cuda is available, training on GPU')
else:
	print('cuda isn\'t available, training on CPU')

batch_size = 16
num_workers = 1
num_epochs = 200

epoch = 0
experiment_number = 2

path = 'models/id'+ str(experiment_number)
model_file = path+'/model_'+str(experiment_number)+'_check.pt'
optimizer_file = path+'/optimizer_'+str(experiment_number)+'_check.pt'
epoch_file = path+'/epoch.txt'

if epoch == 0:
    path0 = os.getcwd()
    full_path = path0+'/'+path
    try:
        os.mkdir(full_path)
    except OSError:
        print ("Creation of the directory %s failed" % full_path)
    else:
        print ("Successfully created the directory %s " % full_path)



# Data loaders
train_data = NshotNwayOmniglot(root='./data', background=True, download=True, transform=transforms.ToTensor())
val_data = NshotNwayOmniglot(root='./val_data', background=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=batch_size, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_data, shuffle = False, batch_size=batch_size, num_workers=num_workers)

# Model
model = SiameseNetwork()

torch.cuda.empty_cache()
if use_cuda:
	model = model.cuda()


# criterion and optimizer
criterion = ContrastiveLoss()#nn.BCELoss() #ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0.0005)



if epoch > 0:
    file = open(epoch_file,"r")
    epoch = int(file.readline())
    file.close()
    model.load_state_dict(torch.load(model_file))
    optimizer.load_state_dict(torch.load(optimizer_file))
    
print('epoch ', epoch)

start = timer()
while(epoch < num_epochs):
	epoch += 1

	train_loss = 0.0
	val_loss = 0.0
	train_correct = 0.0
	train_total = 0.0
	val_correct = 0.0
	val_total = 0.0

	# TRAINING
	model.train()
	it = 0
	torch.cuda.empty_cache()
	#print('total iters: ', len(train_loader))
	for img1, classes in train_loader:
		
		if epoch == 1:
			it += 1
			print(it)

		optimizer.zero_grad()

		# random list whether pair of images should match or not classes
		target = torch.tensor([float(v) for v in (np.random.randint(2, size=classes.size()[0]))] )

		#print(target)
		# gets second set of images to compare
		img2 = train_data.getimages(classes,target)

		if use_cuda:
			img1, img2, target = img1.cuda(), img2.cuda(), target.cuda()

		output = model(img1, img2)
		output = output.view(-1)
		#print(output)

		# update weights
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		# update error rate
		output = torch.clip(torch.round(output), min=0, max=1)
		#print(output)

		train_total += len(target)
		train_correct += torch.sum(torch.abs(output-target)).cpu().detach().numpy()
		train_loss += loss.item()*classes.size(0)
		torch.cuda.empty_cache()
		#break

	print('epoch ',epoch)
	print('Training: loss: ',train_loss,' error rate: ',(train_correct/train_total)*100,' (', train_correct,'/',train_total,')' )

	# VALIDATION
	model.eval()
	with torch.no_grad():
		for img1, classes in val_loader:
			torch.cuda.empty_cache()

			# random list whether pair of images should match or not classes
			target = torch.tensor([float(v) for v in (np.random.randint(2, size=classes.size()[0]))] )

			# gets second set of images to compare
			img2 = val_data.getimages(classes,target)

			if use_cuda:
				img1, img2, target = img1.cuda(), img2.cuda(), target.cuda()

			output = model(img1, img2)
			output = output.view(-1)

			loss = criterion(output, target)

			# update error rate
			output = torch.clip(torch.round(output), min=0, max=1)
			#err = torch.sum(torch.abs(output-target)).cpu().detach().numpy()
			val_total += len(target)
			val_correct += torch.sum(torch.abs(output-target)).cpu().detach().numpy()
			val_loss += loss.item()*classes.size(0)
			#break

	print('Validation: loss: ',val_loss,' error rate: ',(val_correct/val_total)*100,' (', val_correct,'/',val_total,')' )

	torch.save(model.state_dict(), model_file)
	torch.save(optimizer.state_dict(), optimizer_file)
	file = open(epoch_file,"w")
	file.write(str(epoch))
	file.close()

	print('Total time: ',timer()-start)

