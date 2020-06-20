

# Colorify - Movie Colorization Application
Colorization is the process of the addition of color to a black and white video or image. A grayscale image is a scalar but A colored image is a vector-valued function often represented as three separate channels. So, the colorization process requires mapping of a scalar to a vector (r,g,b) valued function which has no unique solution.
Simply put, the mission of this project is to **Bring new life to old photos and movies**.
## Table of Contents  
1.[How to Start ?](#how-to-run-the-project)
 - [Dependencies](#dependencies) 
 - [Run](#how-to-run-?)   

2.[Shot transition detection](#shot-transition-detection)  
3.[Model Architcture](#model-architcture)  
- [Generator](#generator) 
- [Discriminator](#discriminator) 


4.[Color Propagation](#color-propagation)  
5.[Results](#results)  
6.[License](#License)  

## How To Run The Project?
### Dependencies 
- Numpy
- Opencv
- PIL
- Keras
- moviepy

note you can view dependencies.txt for more details
### How to Run ? 
```
python argv[0] argv[1] argv[2] argv[3]  
argv[0] = main.py
argv[1] = 0 => Image Colorization , 1 : Video Colorization
argv[2] = 0 => Human model , 1 : Nature model
argv[3] = File name 
```
## Shot transition detection
<p align="center">
<img src="https://www.ibm.com/blogs/research/wp-content/uploads/2018/09/VSD2.png" width="600" height="300">
</p>
Shot transition detection is used to split up a movie into basic temporal units called shots where the camera is fixed in each shot.
we need this approach to make it easier to deal with a complete movie.<br/>We divide the movie into shots by using 
*Sum of absolute differences* (SAD).<br>This is both the most obvious and most simple algorithm of all: The two consecutive grayscale frames are compared pixel by pixel, summing up the absolute values of the differences of each two corresponding pixels. 

## Model Architcture
<p align="center">
<img src="https://bolster.ai/blog/content/images/size/w2000/2020/04/GAN-1.png" width="600" height="300">
</p>

### Generator
The generator part of a GAN learns to create fake data by incorporating feedback from the discriminator. It learns to make the discriminator classify its output as real.
it also requires random input, generator network, which transforms the random input into a data instance, discriminator network, which classifies the generated data
discriminator output, generator loss, which penalizes the generator for failing to fool the discriminator
The generator can be broken down into two pieces: the encoder and decoder
<p align="center">
<img src="https://miro.medium.com/max/3636/0*7fgHtc8fEmoC_SiZ.png" width="600" height="300">
</p>

### Discriminator
The discriminator is a much simpler model than its counterpart, the generator because it’s a standard Convolutional Neural Network (CNN) that is used to predict whether the RGB channels are real or fake. It has eight 2-stride convolutional layers each which consists of dropout, leaky relu activation, and, except for the first layer, batch normalization
## Color Propagation
Color Propagation is used to propagate the color of each contour in the keyframe( the frame which colorized by the generator model) to other video frames of the same shot by comparing each contour in each consecutive frame after converting the pre colorized frame to LAB.
### Converting from RGB to LAB

The first step was converting the images from their standard RGB color channels into CIE-LAB where the 3 new channels consist of:
- L - Represents the white to black trade off of the pixels
- A - Represents the red to green trade off of the pixels
- B - Represents the blue to yellow trade off of the pixels
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/7/7d/CIELAB_color_space_front_view.png" width="300" height="300">
</p>
The L channel is going to by the current frame and propapagte a&b channel from the previous colorized frame.

## Results

**Colorization of the Nature**
   
> [![Nature](https://img.youtube.com/vi/K83whIRUq-g/0.jpg)](https://youtu.be/K83whIRUq-g)]

**Colorization of Animals **    
  
> [![Animals](https://img.youtube.com/vi/hBtmg-5g4hM/0.jpg)](https://youtu.be/hBtmg-5g4hM)]
>
> [![Animals](https://img.youtube.com/vi/XJLXr4OUveE/0.jpg)](https://youtu.be/XJLXr4OUveE)
## License
 **[MIT license](http://opensource.org/licenses/mit-license.php)**    
