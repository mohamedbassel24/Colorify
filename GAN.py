from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.layers import ReLU, LeakyReLU, concatenate, Input, GlobalAveragePooling2D
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.activations import tanh, sigmoid
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras import Model
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout    
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
import os
import cv2

def convBlock(inp, n_filters, filter_size=4, stride=2, dropout=False, activation=True, BN=True, padding='same', alpha=.2):

    y = Conv2D(n_filters, filter_size, stride, padding, kernel_initializer=tf.keras.initializers.truncated_normal(stddev=.02))(inp)

    if BN:
        y = BatchNormalization()(y)

    if activation:
        y = LeakyReLU(alpha=alpha)(y)

    print(y.shape)
    return y

def convTransBlock(inp, n_filters, filter_size=4, stride=2, convOut=None, dropout=False, activation=True, BN=True, padding='same', alpha=.2):
   

    y = Conv2DTranspose(n_filters, filter_size, stride, padding,kernel_initializer=tf.keras.initializers.truncated_normal(stddev=.02))(concatenate([inp, convOut]) if convOut is not None else inp)

    if BN:
        y = BatchNormalization()(y)

    if dropout:
        y = Dropout(rate=dropout)(y)

    if activation:
        y = LeakyReLU(alpha=alpha)(y)

    print(y.shape)
    return y

def generator(drop_rate, alpha, inp_shape=(512, 512, 3)):

    inp = Input(inp_shape)

    n_filters = 16

    print('Encoder:')
    conv1 = convBlock(inp, n_filters, BN=False, alpha=alpha)
    conv2 = convBlock(conv1, n_filters*2, alpha=alpha)      # 128x128
    conv3 = convBlock(conv2, n_filters*4, alpha=alpha)      # 64x64
    conv4 = convBlock(conv3, n_filters*8, alpha=alpha)      # 32x32
    conv5 = convBlock(conv4, n_filters*8, alpha=alpha)      # 16x16
    conv6 = convBlock(conv5, n_filters*8, alpha=alpha)      # 8x8
    conv7 = convBlock(conv6, n_filters*8, alpha=alpha)      # 4x4
    conv8 = convBlock(conv7, n_filters*8, alpha=alpha)      # 2x2x512

    print('Decoder:')
    deconv1 = convTransBlock(conv8, n_filters*8, alpha=alpha)                                     # 4x4
    deconv2 = convTransBlock(deconv1, n_filters*8, convOut=conv7, dropout=drop_rate, alpha=alpha) # 8x8
    deconv3 = convTransBlock(deconv2, n_filters*8, convOut=conv6, dropout=drop_rate, alpha=alpha) # 16x16
    deconv4 = convTransBlock(deconv3, n_filters*8, convOut=conv5, dropout=drop_rate, alpha=alpha) # 32x32
    deconv5 = convTransBlock(deconv4, n_filters*4, convOut=conv4, alpha=alpha)                    # 64x64
    deconv6 = convTransBlock(deconv5, n_filters*2, convOut=conv3, alpha=alpha)                    # 128x128
    deconv7 = convTransBlock(deconv6, n_filters, convOut=conv2, alpha=alpha)                      # 256x256
    deconv8 = convTransBlock(deconv7, 3, convOut=conv1, activation=False, BN=False)               # 512x512

    outp = tanh(deconv8)

    model = Model(inputs=inp, outputs=outp)

    return model

def discriminator(alpha, learning_rate, inp_shape=(512, 512, 3), target_shape=(512, 512, 3)):
    
  
    n_filters = 16

    inp1 = Input(inp_shape) # sketch input
    inp2 = Input(target_shape) # colored input

    inp = concatenate([inp1, inp2]) # 512x512x6

    conv1 = convBlock(inp, n_filters, BN=False, alpha=alpha) # 256x256x64
    conv2 = convBlock(conv1, n_filters*2, alpha=alpha) # 128x128x128
    conv3 = convBlock(conv2, n_filters*4, alpha=alpha) # 64x64x256
    conv4 = convBlock(conv3, n_filters*8, alpha=alpha) # 32x32x512
    conv5 = convBlock(conv4, n_filters*8, filter_size=2, stride=1, padding='valid', alpha=alpha) # 31x31x512
    conv6 = convBlock(conv5, n_filters=1, filter_size=2, stride=1, activation=False, BN=False, padding='valid') # 30x30x1

    sigmoid_outp = sigmoid(conv6)

    outp = GlobalAveragePooling2D()(sigmoid_outp)

    model = Model(inputs=[inp1, inp2], outputs=outp)

    opt = Adam(lr=learning_rate, beta_1=.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
generator(1,2)
print("hey")
