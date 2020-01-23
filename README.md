# Image Classifier
## AIM
The goal is train a classifier using 102 categories of flowers. The trained model will take an image as an input and predict the
probabilities of 10 top classes.This is also an opportunity to practise deep learning

## Method
I will be using pytorch for training. A pre-trained model 'VGG16' will be used together with the classifier I defined

## Usage
Training
* python train.py data_dir --arch "vgg13" --learning_rate 0.01 --hidden_units 512 --epochs 20 --save_dir save_directory

Prediction
* python predict.py input checkpoint --category_names cat_to_name.json
