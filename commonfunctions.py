#import cv2 as cv
import glob
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
import math
from scipy import stats
from skimage.filters import median
from skimage.feature import canny
import cv2
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from scipy import stats
import sys
import moviepy.editor
import time



import os
# Show the figures / plots inside the notebook
def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

def Parser(sys):
    return 1