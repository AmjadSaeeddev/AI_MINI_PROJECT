# AI_MINI_PROJECT Report

what is VGG16?

VGG16 is a convolutional neural network model proposed by K. ... Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes


what is vgg16 used for?
VGG16 convolutional neural network
VGG16 (also called OxfordNet) is a convolutional neural network architecture named after the Visual Geometry Group from Oxford, who developed it. It was used to win the ILSVRC2014 (Large Scale Visual Recognition Challenge 2014)) competition in 2014.

what is special about vgg16?

It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2.

what is the output of vgg16?
I have observed that VGG16 model predict with an output dimension of (1,512) , i understand 512 is the Features as predicted by the VGG16. ... however the inception model outputs a dimension of 1,8,8,2048.

How is VGG16 implemented?

import keras,os. from keras.models import Sequential. from keras.layers import Dense, Conv2D, MaxPool2D , Flatten. ...
trdata = ImageDataGenerator() traindata = trdata.flow_from_directory(directory="data",target_size=(224,224)) ...
model.summary()
import matplotlib.pyplot as plt. plt.plot(hist.history["acc"])

What are Vgg features?
VGG incorporates 1x1 convolutional layers to make the decision function more non-linear without changing the receptive fields. The small-size convolution filters allows VGG to have a large number of weight layers; of course, more layers leads to improved performance.

What is keras VGG16?
VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. ... It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output.

what is cifar-10 dataset?
The CIFAR-10 dataset is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.

How do we load a Cifar-10 dataset?
Utility to load cifar-10 image data into training and test data sets. Download the cifar-10 python version dataset from here, and extract the cifar-10-batches-py folder into the same directory as the load_cifar_10.py script. The code contains example usage, and runs under Python 3 only.

How is Cifar-10 data stored?
The CIFAR-10 dataset. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images.

