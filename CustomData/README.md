# Custom dataset project
This project contains a structure and scripts to generate a custom image dataset

## Creating dataset

For each class in your custom dataset create a folder with that class name in the images/classes folders. As an example the project has the folders drums, flute, guitar, piano, trombone and violin for a custom instrument classification dataset. Within each class folder create a "images" folder and populate that folder with images of that class (images may have different sizes and ratios).

A quick way to get images of your desired class is explained by Adrian Rosebrock in https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
Follow that tutorial to get a txt with images urls, such as the ones already in the example classes. Then you may run the download_images.py script to download those images as in: 

```bash
python3 download_images.py --urls classes/<class>/<class>_urls.txt --output classes/<class>/images
```
