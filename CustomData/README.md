# Custom dataset project
This project contains a structure and scripts to generate a custom image dataset, and scripts to train CNNs on those datasets and visualize predictions and feature maps from those trained CNNs.

This project uses openCV and PyTorch, both must be installed for it to run successfully.

Project description in progress.

## Creating dataset

For each class in your custom dataset create a folder with that class name in the images/classes folders. As an example the project has the folders drums, flute, guitar, piano, trombone and violin for a custom instrument classification dataset. Within each class folder create a "images" folder and populate that folder with images of that class (images may have different sizes and ratios).

A quick way to get images of your desired class is explained by Adrian Rosebrock in https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
Follow that tutorial to get a txt with images urls, such as the ones already in the example classes. Then, from within the images folder, you may run the download_images.py script taken from Adrian's tutorial to download those images as in: 

```bash
python3 download_images.py --urls classes/<class>/<class>_urls.txt --output classes/<class>/images
```

As stated in Adrian's tutorial, you should check if the downloaded images are relevant/correct. Images that might be deleted include: images that do not contain the target class, images that contain objects of other classes besides the target class, images that contain the object name rather than an image of the object and so on.

Once you have all images separeted in your classes folders, randomly separate those images in train, validation and dataset with the following command:

```bash
python3 split_train_val_test.py
```

Currently the images will be separated as 75% train, 10% validation and 15% test. Feel free to modify the code to your desired values (command line arguments to be added in the future). After that you should have a train, a val and a test folder in your images folder, each containing a folder for each class in the dataset, filled with images of that class.

## Training CNN for custom dataset
In the CustomData folder, once you have your dataset loaded and separated, you may train a CNN with the trainCustom.py script. Modify it to adapt to your custom dataset (classes auto detection to be added in the future). Modify the CNN architecture and training parameters as you wish. Train a CNN with the command:

```bash
python3 trainCustom.py
```

The training progress will appear in the prompt. A .pt file will be created in the models folder, containing the trained wights and after training ends, an image containing the training history will appear in the training_plots folder.

## Visualizing CNNs
Visualizing the results of the trained CNN might be done with the visualizeTrained.py script. Be sure to have the same CNN architecture in this code as the one you used in trainCustom.py to train on the custom dataset and to declare the correct .pt weights file path.

To use it simply run 

```bash
python3 visualizeTrained.py
```

The code will open a detection/feature_map window. On the top left corner of the image, a loaded dataset image will appear, bellow it the groundtruth class and above it the predicted class, in green if correct and red if false. On the bottom left corner a feature map will appear and on the top right corner the image with the detected features.

Press the right arrow to move to the next feature map in the same layer, the up arrow to advance one layer, esc to quit the application and any other key to skip to the next image.
