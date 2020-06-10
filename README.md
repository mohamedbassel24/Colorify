# Colorify - Movie Colorization Application
Colorization is the process of addition of colour to a black and white video or image. A gray scale image is a scalar but A colored image is a vector valued function often represented as three separate channels. So, the colorization process requires mapping of a scalar to a vector (r,g,b) valued function which has no unique solution.
Simply put, the mission of this project is to colorize and restore old images and movies.
## Table of Contents  
1.[Shot transition detection](#shot-transition-detection)  
2.[Model Architcture](#model-architcture)  
* [Generator](#generator) 
* [Discriminator](#discriminator) 


3.[Color Propagation](#colorpropagation)  
4.[Results](#results)  
5.[References](#references)  
## Shot transition detection
Shot transition detection is used to split up a movie into basic temporal units called shots where the camera is fixed in each shot.
we need this aproche to make it easier to deal with a complete movie.<br/>We divide the moive into shots by using 
*Sum of absolute differences* (SAD). <br/>This is both the most obvious and most simple algorithm of all: The two consecutive grayscale frames are compared pixel by pixel, summing up the absolute values of the differences of each two corresponding pixels. 
## Model Architcture

![alt text](https://ibb.co/hX6YPqV)
### Generator
### Discriminator
## ColorPropagation
## Results
## References
