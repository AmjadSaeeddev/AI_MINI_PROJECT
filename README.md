# AI_MINI_PROJECT Report


What is CIFAR10?

CIFAR10 is a subset-labeled dataset produced from a collection of 80 million images. Alex Krizhevsky and Vinod Nair gathered this information. CIFAR10 contains 60,000 images of 10 labels with a resolution of 32x32 pixels in the torch package. Torchvision.datasets is the default. 

The dataset will be divided into 50,000 photos for training and 10,000 images for testing by CIFAR10.

Karen Simonyan and Andrew Zisserman created VGG16, a very deep convolutional neural network.

I utilised the VGG16 model to solve the CIFAR10 dataset in this scenario.


For this project, I used Google Collab as my primary working environment. The initial step is to choose between using a cuda or a CPU machine to train the model. The number of epochs, batch size, and learning rate for this training are then determined. The CIFAR10, as stated in the introduction, consists of ten labels that are kept in the classes variables.



what is VGG16?

VGG16 is a convolutional neural network model proposed by K. ... Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes


what is vgg16 used for?
VGG16 convolutional neural network
VGG16 (also called OxfordNet) is a convolutional neural network architecture named after the Visual Geometry Group from Oxford, who developed it. It was used to win the ILSVRC2014 (Large Scale Visual Recognition Challenge 2014)) competition in 2014.

what is special about vgg16?

It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2.

What are Vgg features?
VGG incorporates 1x1 convolutional layers to make the decision function more non-linear without changing the receptive fields. The small-size convolution filters allows VGG to have a large number of weight layers; of course, more layers leads to improved performance.
