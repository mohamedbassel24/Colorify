from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Input, BatchNormalization, GlobalAveragePooling2D, \
    concatenate
# Activations
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.activations import tanh, sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from PIL import Image
from commonfunctions import *


def ConvLayer(ConvIn, NumFilters, FilterSize=4, StrideLength=2, DropOutRate=False, Activation=True,
              BatchNormalizationON=True, Padding="same", Alpha=0.2):
    WeightsInitializer = initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
    Layer = Conv2D(NumFilters, FilterSize, StrideLength, padding=Padding, kernel_initializer=WeightsInitializer)(ConvIn)
    if BatchNormalizationON:
        Layer = BatchNormalization()(Layer)

    if Activation:
        Layer = LeakyReLU(alpha=Alpha)(Layer)

    return Layer


def ConvTransLayer(ConvTransIn, NumFilters, FilterSize=4, StrideLength=2, DropOutRate=False, convOut=None,
                   Activation=True, BatchNormalizationON=True, Padding="same", Alpha=0.2):
    WeightsInitializer = initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
    Layer = Conv2DTranspose(NumFilters, FilterSize, StrideLength, padding='same',
                            kernel_initializer=WeightsInitializer)(
        concatenate([ConvTransIn, convOut]) if convOut is not None else ConvTransIn)

    if BatchNormalizationON:
        Layer = BatchNormalization()(Layer)
    if Activation:
        Layer = LeakyReLU(alpha=Alpha)(Layer)
    if DropOutRate:
        Layer = Dropout(rate=DropOutRate)(Layer)

    return Layer


def CreateGenerator(DropOut, Alpha, InputShape=(512, 512, 3)):
    GeneratorInput = Input(InputShape)

    NumFiltersGenerator = 16

    ConvLayerOut1 = ConvLayer(GeneratorInput, NumFilters=NumFiltersGenerator, BatchNormalizationON=False, Alpha=Alpha)
    ConvLayerOut2 = ConvLayer(ConvLayerOut1, NumFilters=NumFiltersGenerator * 2, Alpha=Alpha)
    ConvLayerOut3 = ConvLayer(ConvLayerOut2, NumFilters=NumFiltersGenerator * 4, Alpha=Alpha)
    ConvLayerOut4 = ConvLayer(ConvLayerOut3, NumFilters=NumFiltersGenerator * 8, Alpha=Alpha)
    ConvLayerOut5 = ConvLayer(ConvLayerOut4, NumFilters=NumFiltersGenerator * 8, Alpha=Alpha)
    ConvLayerOut6 = ConvLayer(ConvLayerOut5, NumFilters=NumFiltersGenerator * 8, Alpha=Alpha)
    ConvLayerOut7 = ConvLayer(ConvLayerOut6, NumFilters=NumFiltersGenerator * 8, Alpha=Alpha)
    ConvLayerOut8 = ConvLayer(ConvLayerOut7, NumFilters=NumFiltersGenerator * 8, Alpha=Alpha)

    ConvTransOut1 = ConvTransLayer(ConvLayerOut8, NumFiltersGenerator * 8, Alpha=Alpha)
    ConvTransOut2 = ConvTransLayer(ConvTransOut1, NumFiltersGenerator * 8, convOut=ConvLayerOut7, DropOutRate=DropOut,
                                   Alpha=Alpha)
    ConvTransOut3 = ConvTransLayer(ConvTransOut2, NumFiltersGenerator * 8, convOut=ConvLayerOut6, DropOutRate=DropOut,
                                   Alpha=Alpha)
    ConvTransOut4 = ConvTransLayer(ConvTransOut3, NumFiltersGenerator * 8, convOut=ConvLayerOut5, DropOutRate=DropOut,
                                   Alpha=Alpha)
    ConvTransOut5 = ConvTransLayer(ConvTransOut4, NumFiltersGenerator * 4, convOut=ConvLayerOut4, Alpha=Alpha)
    ConvTransOut6 = ConvTransLayer(ConvTransOut5, NumFiltersGenerator * 2, convOut=ConvLayerOut3, Alpha=Alpha)
    ConvTransOut7 = ConvTransLayer(ConvTransOut6, NumFiltersGenerator, convOut=ConvLayerOut2, Alpha=Alpha)
    ConvTransOut8 = ConvTransLayer(ConvTransOut7, 3, convOut=ConvLayerOut1, Activation=False,
                                   BatchNormalizationON=False, Alpha=Alpha)  # bn false #activation false

    GenOut = tanh(ConvTransOut8)

    return Model(inputs=GeneratorInput, outputs=GenOut)


def Load_GAN():
    # Loading weights not the model
    drop_rate = .5
    # We need to Create the Model first
    gen_model = CreateGenerator(drop_rate, .2)
    # Load his weights
    gen_model.load_weights("Colorization/Epoch-{}".format(54))
    print("[INFO] Generator Model is Loaded Successfully..")
    custom = {'custom_loss2': 11111}
    # uncomment this if you are load_model
    # return load_model("Colorization/" + "mEpoch-{}".format(24),custom_objects={'custom_loss34': custom['custom_loss2']})
    return gen_model


def colorization(img, GeneratorModel):
    """Take input image and colorize it based on given model"""
    originalShape = img.shape
    img = imgPreProcessing(img)
    ColorArr = [img]
    ColorArr = (np.array(ColorArr, dtype='float32') - 127.5) / 127.5
    Colorized = GeneratorModel.predict(ColorArr)

    Images = (np.array(Colorized, dtype='float32') * 127.5) + 127.5
    Images = np.array(Images, dtype='int32')
    # Back to original shape
    ColorizedImage = np.copy(Images[0])
    ColorizedImage = np.array(ColorizedImage, dtype='uint8')
    ColorizedImage = cv2.resize(ColorizedImage, (originalShape[1], originalShape[0]), interpolation=cv2.INTER_AREA)
    # show_images([ColorizedImage])

    return ColorizedImage


def imgPreProcessing(img):
    dim = (512, 512)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("we.png", img)
    img = np.array(Image.open('we.png').convert('RGB'))
    return img
