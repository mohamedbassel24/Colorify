import cv2
import glob
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math
from scipy import stats
from skimage.filters import median
from skimage.feature import canny

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


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


def show_3d_image(img, title):
    fig = plt.figure()
    fig.set_size_inches((12, 8))
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, img.shape[0], 1)
    Y = np.arange(0, img.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = img[X, Y]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 8)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(title)
    plt.show()


def show_3d_image_filtering_in_freq(img, f):
    img_in_freq = fftpack.fft2(img)
    filter_in_freq = fftpack.fft2(f, img.shape)
    filtered_img_in_freq = np.multiply(img_in_freq, filter_in_freq)

    img_in_freq = fftpack.fftshift(np.log(np.abs(img_in_freq) + 1))
    filtered_img_in_freq = fftpack.fftshift(np.log(np.abs(filtered_img_in_freq) + 1))

    show_3d_image(img_in_freq, 'Original Image')
    show_3d_image(filtered_img_in_freq, 'Filtered Image')


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def ConvertToBinary(mPic, Threshold):
    for i in range(np.shape(mPic)[0]):
        for j in range(np.shape(mPic)[1]):
            if mPic[i][j] >= Threshold:
                mPic[i][j] = 0
            else:
                mPic[i][j] = 1
    return mPic


def ANDING_SE(Fore, Back):
    for i in range(np.shape(Fore)[0]):
        for j in range(np.shape(Fore)[1]):
            if Back[i][j] == 1:
                if Fore[i][j] != Back[i][j]:
                    return 0
    return 1


def ORI_SE(Fore, Back):
    for i in range(np.shape(Fore)[0]):
        for j in range(np.shape(Fore)[1]):
            if Back[i][j] == 1:
                if Fore[i][j] == Back[i][j]:
                    return 1
    return 0


def mErosion(mPic, SE):
    WindowWidth = np.shape(SE)[0]
    WindowHeight = np.shape(SE)[1]
    mOut = np.zeros((np.shape(mPic)[0], np.shape(mPic)[1]))
    EdgeX = (int)(WindowWidth / 2)
    EdgeY = (int)(WindowHeight / 2)
    for i in range(np.shape(mPic)[0] - EdgeX):
        for j in range(np.shape(mPic)[1] - EdgeY):
            foreground = np.zeros((WindowWidth, WindowHeight))
            for fx in range(WindowWidth):
                for fy in range(WindowHeight):
                    foreground[fx][fy] = mPic[i + fx - EdgeX][j + fy - EdgeY]
            mOut[i][j] = ANDING_SE(foreground, SE)
    return mOut


def mDilation(mPic, SE):
    WindowWidth = np.shape(SE)[0]
    WindowHeight = np.shape(SE)[1]
    mOut = np.zeros((np.shape(mPic)[0], np.shape(mPic)[1]))
    EdgeX = (int)(WindowWidth / 2)
    EdgeY = (int)(WindowHeight / 2)
    for i in range(np.shape(mPic)[0] - EdgeX):
        for j in range(np.shape(mPic)[1] - EdgeY):
            foreground = np.zeros((WindowWidth, WindowHeight))
            for fx in range(WindowWidth):
                for fy in range(WindowHeight):
                    foreground[fx][fy] = mPic[i + fx - EdgeX][j + fy - EdgeY]
            mOut[i][j] = ORI_SE(foreground, SE)
    return mOut

def Opening(mPic, SE):
    return mDilation(mErosion(mPic, SE), SE)


def Closing(mPic, SE):
    return mErosion(mDilation(mPic, SE), SE)

def PrintBinary(PIC):
    io.imshow(PIC, cmap="binary")  # 0~255 np.zeros((2, 1))
    io.show()


# TODO: HOUGHMAN  & K-Mean
def drawLine(ax, angle, dist):
    '''
    This function should draw the lines, given axis(ax), the angle and the distance parameters
    Get x1,y1,x2,y2
    '''
    x1 = dist / math.cos(angle)
    y1 = 0
    x2 = 0
    y2 = dist / math.sin(angle)

    # This line draws the line in red

    ax[1].plot((x1, y1), (x2, y2), '-r')


#im = io.imread('circuit.tif')


def Houghman(image):
    img = rgb2gray(image)
    img = canny(img, sigma=1, low_threshold=50, high_threshold=130)
    show_images([image, img])
    hspc, angles, distance = hough_line(img)
    accum, angles, dists = hough_line_peaks(hspc, angles, distance, threshold=50)

    ## This part draw the lines on the image.

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image, cmap=cm.gray)
    for angle, dist in zip(angles, dists):
        drawLine(ax, angle, dist)
    ax[1].set_xlim((0, image.shape[1]))
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.show()


def claster(k, image):
    [x, y, z] = image.shape
    img2d = image.reshape(x * y, z)
    kmeans = cluster.KMeans(k)
    kmeans.fit(img2d)
    # print(img2d[kmeans.labels_])
    for i in range(x * y):
        img2d[i] = kmeans.cluster_centers_[kmeans.labels_[i]]

    imgx = img2d.reshape(x, y, z)
    return imgx


# TODO: SEGEMENTATION

def getThreshold(rImage):
    rImage = (rImage).astype(np.uint8)
    rImage = rgb2gray(rImage)
    NumberOfPixels = []
    for Index in range(256):  # Number of GrayLevels
        count = 0
        for i in range(np.shape(rImage)[0]):
            for j in range(np.shape(rImage)[1]):
                if rImage[i][j] == Index:
                    count = count + 1
        NumberOfPixels.append(count)
    k = 255  # any k ? Numbers of GrayLevel
    rsum = 0
    for i in range(255):
        rsum += i * NumberOfPixels[i]
    T_Inital = round((rsum / (sum(NumberOfPixels) + 1)))
    print(T_Inital)
    for i in range(5):
        rsum = 0
        bsum = 0
        for X in range(T_Inital):
            rsum += X * NumberOfPixels[X]
        for Y in range(255 - T_Inital):
            bsum += (Y + T_Inital) * NumberOfPixels[Y + T_Inital]

        First_PeakMean = round((rsum) / ((sum(NumberOfPixels[:T_Inital]))))
        Sec_PeakMean = round((bsum) / (sum(NumberOfPixels[T_Inital:])))
        old = T_Inital
        T_Inital = round((First_PeakMean + Sec_PeakMean) / 2)
        print(T_Inital)
        if old == T_Inital:
            break
    return T_Inital


def LocalSegementation(r):
    M = np.shape(r)[0]
    N = np.shape(r)[1]

    LeftTop = np.copy(r[:int(M / 2), :int(N / 2)])
    LeftBot = np.copy(r[:int(M / 2), int(N / 2):])
    RightTop = np.copy(r[int(M / 2):, :int(N / 2)])
    RightBot = np.copy(r[int(M / 2):, int(N / 2):])

    io.imshow(r)  # 0~255 np.zeros((2, 1))
    io.show()

    CurrThresh = getThreshold(r)
    r = ConvertToBinary(r, CurrThresh)
    PrintBinary(r)

    CurrThresh = getThreshold(LeftTop)
    LeftTop = ConvertToBinary(LeftTop, CurrThresh)

    CurrThresh = getThreshold(LeftBot)
    LeftBot = ConvertToBinary(LeftBot, CurrThresh)

    CurrThresh = getThreshold(RightTop)
    RightTop = ConvertToBinary(RightTop, CurrThresh)

    CurrThresh = getThreshold(RightBot)
    RightBot = ConvertToBinary(RightBot, CurrThresh)

    UpperA = np.hstack((LeftTop, LeftBot))
    DownA = np.hstack((RightTop, RightBot))
    Merge = np.vstack((UpperA, DownA))
    PrintBinary(Merge)
