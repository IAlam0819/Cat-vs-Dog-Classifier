# Cat-vs-Dog-Classifier
# INTRODUCTION

Image classification is a supervised learning problem: define a set of target classes (objects to identify in images), and train a model to recognize them using labelled data. Early computer vision models relied on raw pixel data as the input to the model. We use Convolutional neural networks to build image classification models.

# BUILDING A CNN MODEL

This project includes building an image classification model, that is based on Computer Vision. Computer vision is the field of having a computer understand and label what is present in the image. One way to solve that is by using lots of pictures and tell the computer what that’s a picture of and then have the computer figure out the patterns that gives us the difference between a dog and cat. This is why the dataset is divided into two parts – one part for the training the model and other for the validation of the model.

For building the model, we will be using the concept of convolution, that is, we will be using the Conv2D layers (tf.keras.layers.Conv2D), this layer creates a convolution kernel, that is, convolued with the layer input to produce a tensor of outputs. A Conv2D layer is followed by a MaxPool2D layer (tf.keras.layers.MaxPool2D), in which we specify the pool size, which is (2 × 2) in our case. By specifying (2,2) for MaxPooling, the effect is to quarter the size of the image. The idea is to create a  (2 × 2) array of pixels and picks the biggest one, thus turning 4 pixels to 1. Effectively reducing the image by 25%.

For building a CNN model, we will be adding 3 sets of convolution pooling layer at the top. This reflects the higher complexity and size of the image, we can add more layer if we want. We will be adding a sequence of convolution layers, followed by ‘MaxPooling’ layer. Then we will flatten the image, the Flatten layer takes the image and turn it into one dimensional set. Then we will add a Dense layer of 64 neurons, with a ‘relu’ activation function. The output layer has only 1 neuron for two classes (binary classification) and it uses a different activation function sigmoid, it takes a set of values and effectively picks the biggest one. We are facing a two-class classification problem, i.e. a binary classification problem, we will end our network with a sigmoid activation, so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).

We will pre-process our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range). In keras this can be done via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you to instantiate generators of augmented image batches (and their labels) via .flow_from_directory. These generators can then be used with the keras model methods that accept data generators as inputs: fit, evaluate_generator, and predict_generator.

We used image generator and run our model for 25 epochs, we get the accuracy and loss value of training generator and validation generator as follows:

    • Training accuracy – 0.9880
    
    • Training loss – 0.0525
    
    • Validation accuracy – 0.6267
    
    • Validation loss – 1.5033
 
# TRANSFER LEARNING

Transfer learning is a machine learning technique where a model trained on one task is re-purposed on a second related task. It is an optimization that allows rapid progress or improved performance when modeling the second task.

# Building a Transfer Learning Model

There are many models for image classification with weights trained on ImageNet, such as, VGG16, VGG19, ResNet, InceptionV3, MobileNet etc. so we will be using VGG16 model, which can be easily downloaded and used by importing the library given by keras by using the following command: from keras.applications.vgg16 import VGG16. In other words, we are using a pre-trained model for our own classification and changing the output layer accordingly.

We used image generator and run our model for 5 epochs, we get the accuracy and loss value of training data and validation data as follows:

    • Training accuracy – 0.9952
    
    • Training loss – 0.0301
    
    • Validation accuracy – 0.9233
    
    • Validation loss – 0.5002
