## Traffic Sign Recognition 

The goals of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Overview

For making an SDC(Self Driving Car) make decisions on traffic signs as humans while driving we need data of the camera feed, but for making decisions we first need a way to detect and classify the traffic signs, there are many traditional ways of doing this task using image processing and computer vision techniques, using a neural network to make a classification, using deep learning techniques like CNN, etc.

Traditional ways for traffic sign recognition using image processing is computationally expensive and is not suitable for applications that need an accurate prediction stats, it is hard to make them adapt to the present day camera technology that involves generating high-resolution images, a large number of frames per second(FPS).
Normal neural networks don’t take into account the local relationships of each pixel in the image to the surrounding pixels because in normal NN’s we are just flattening the image and feeding to a NN so each pixel is connected to a neural in the next layer independently, but we need a way to maintain this relationship among pixels. And also using normal NN for tasks that include images increases the number of parameters, so it computationally expensive to train the network. Neural networks also may not adapt well to high FPS video feed.

The convolutional neural network takes into account the local relationship of pixels in the image, aims to decrease the number of parameters by sharing the parameters among the pixels by using a concept of kernels and strides. CNN is also good at the high fps video feed. 
So, by the above analysis, CNN works  better for the traffic recognition task .

## Dataset Summary and Exploration:

Dataset Exploration , I have used python’s pickle library to load the dataset from the files “train.p” , “test.p” , “valid.p” , pickle is a python library that gives us the flexibility to serialize the data in the form of python dictionaries ,while loading the data from these files as if we are accessing data from a python dictionary i.e., through keys.

Pandas and numpy are great tools for exploring the dataset, Pandas can deal with relational data , numpy can deal with numerical data like images, etc. Pandas is used for  extracting number of unique classes in the training dataset , turns out that there are 43 unique classes in the training set and also this also is the number of classes for our classification.

There are 34799 Number of training examples , 12630 number of testing examples , 43 number of classes ,  4410 number of validation data examples.

here are some classes with relatively large number of images concentrated , there are alos some classes with least number of images concentrated, so it is enough look at accuracy based model evaluation, but it would also be better to evaluate the model using precision , recall , f1-score because they provide how certain our model in classification per class so we can investigate into what examples are missclassified and why they are causing model to missclassify.

## Design and Testing Model Architecture

Different color space representation of images have different regions in image that are enhanced , this may make the  model to learn features of the image that are helpful to classify image into its labelled class very fast . 
Inorder to make our model more robust in classifying images in different conditions of image we may augment the dataset using some of the image processing techniques that doesnot remove main features from the image but just adds some small aberrations that indirectly makes network to learn even some more in depth features that are required to classify.

A “Preprocess” class has been defined with methods that can change color space of the image to a desired color space dynamically , method to add blurring to the images, method to rotate the image to certain degree, method to shift center of the image to some point in the 32X32 grid of points.

LeNet CNN architecture is being used for Traffic Sign Recognition task. LeNet  is a simple architecture with successive convolution layers with some subsampling layers in between (pooling layers). 

## Approaches Took To Train The Model:
Different color representations of images for training the model and performance of these models on validation accuracy is recorded.

Initially LeNet CNN architecture is taken as our project CNN architecture , this is with no regularization. This model is tested on different color spaced image models with batch size 128 and epochs 15 , above table shows the corresponding validation accuracies ,gray scale model is doing well on validation data but with only 93.4% . This clearly indicates that it not doing well in images that have never seen that is it not able to do generalize well on these images. So this gives the hint of employing some of the regularization techniques like l2 regularization , Dropout, EarlyStopping. 

DropOut is the standard to use in neural network because it depicts the scenario of combining different neural network models performance into one model there by increasing the accuracy of the overall model . 

After first and second fully connected layers a DropOut layer is employed that retains each node in this layer with a probability assigned .
This approach plus an increase in number of epochs significantly improved the validation accuracies of the three models. But again Gray Scaled Image out weighs other models with validation accuracy of about 96%.

With just a simple architecture of 2 convolutional layers and three fully connected layers we are able to get an accuracy of about 96%. Because of this accuracy and also the number of parameters in network, depth of the network i think this is a better network to use for this task , rather than a network with much depth this may overfit the data.

## Model Performance On Test DataSet:
Gray Image Model performed really well on the test dataset with an accuracy of about 93%.




