# Colorify - Movie Colorization Application
Colorization is the process of addition of colour to a black and white video or image. A gray scale image is a scalar but A colored image is a vector valued function often represented as three separate channels. So, the colorization process requires mapping of a scalar to a vector (r,g,b) valued function which has no unique solution.
Simply put, the mission of this project is to colorize and restore old images and movies.
## Table of Contents  
1.[Prerequisites](#prerequisites)  
1.[Shot transition detection](#shot-transition-detection)  
2.[Model Architcture](#model-architcture)  
* [Generator](#generator) 
* [Discriminator](#discriminator) 


3.[Color Propagation](#color-propagation)  
4.[Results](#results)  
5.[References](#references)  
## Shot transition detection
Shot transition detection is used to split up a movie into basic temporal units called shots where the camera is fixed in each shot.
we need this aproche to make it easier to deal with a complete movie.<br/>We divide the moive into shots by using 
*Sum of absolute differences* (SAD). <br/>This is both the most obvious and most simple algorithm of all: The two consecutive grayscale frames are compared pixel by pixel, summing up the absolute values of the differences of each two corresponding pixels. 
## Model Architcture

### Generator
The generator part of a GAN learns to create fake data by incorporating feedback from the discriminator. It learns to make the discriminator classify its output as real.
it also require random input, generator network, which transforms the random input into a data instance,discriminator network, which classifies the generated data
discriminator output,generator loss, which penalizes the generator for failing to fool the discriminator
The generator can be broken down into two pieces: the encoder and decoder

### Discriminator
The discriminator is a much simpler model than its counterpart, the generator, because it’s a standard Convolutional Neural Network (CNN) that is used to predict whether the RGB channels are real or fake. It has eight 2-stride convolutional layers each which consists of dropout, leaky relu activation, and, except for the first layer, batch normalization
## Color Propagation
Color Propagation is to propagate the color of each contour in the key frame( the frame which colorized by the generator model) to other video frames of the same shot by comparing each contour in each consecutive frame. 
## Results
## References
