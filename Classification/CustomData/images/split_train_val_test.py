import os
import numpy as np
import shutil


#percentages split definition
train_percentage = 0.75
valid_percentage = 0.1
if (train_percentage+valid_percentage>1):
	print('invalid division, using default 75% train 10% val and 15% test')
	train_percentage = 0.75
	valid_percentage = 0.1
division = train_percentage+valid_percentage

#get all classes names from classes folders
classes = [dI for dI in os.listdir('classes') if os.path.isdir(os.path.join('classes',dI))]

for label in classes:
	print('spliting ',label,'images')
	#get path to images folder
	folder = 'classes/'+label+'/images'
	#get all image names
	allFileNames = os.listdir(folder)
	print('Total images: {}'.format(len(allFileNames)))
	if (len(allFileNames)>0):
		#separate images in training, validation and test sets
		np.random.shuffle(allFileNames)
		train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
        [int(len(allFileNames)*train_percentage), int(len(allFileNames)*division)])
		print('Training: {} images'.format(len(train_FileNames)), '({0:.2f}%)'.format(len(train_FileNames)/len(allFileNames))  )
		print('Validation: {} images'.format(len(val_FileNames)), '({0:.2f}%)'.format(len(val_FileNames)/len(allFileNames))  )
		print('Testing: {} images'.format(len(test_FileNames)), '({0:.2f}%)'.format(len(test_FileNames)/len(allFileNames))  )
		#create train, val and test folders for that class
		os.makedirs('train/'+label)
		os.makedirs('val/'+label)
		os.makedirs('test/'+label)
		#copy images to their respective folders
		for name in train_FileNames:
			shutil.copy(folder+'/'+name, 'train/'+label)
		for name in val_FileNames:
			shutil.copy(folder+'/'+name, 'val/'+label)
		for name in test_FileNames:
			shutil.copy(folder+'/'+name, 'test/'+label)