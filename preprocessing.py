import os
import cv2
import glob
import numpy as np


TOTAL_IMGS = 300
def Preprocessing():
    i = 1
    for image in glob.glob('train/*.jpg'):
        img_path = image
        img = cv2.imread(img_path)
        print(img_path)
        dim = (512, 512)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite("train/rgb/"+str(i)+".jpg",img)
        cv2.imwrite("train/gray/"+str(i)+".jpg",gray_img)
        i+=1

def GenerateSamples(RGBImgsPath,GrayImgsPath):
    RGBImgs = []
    GrayImgs = []

    for i in range(1,TOTAL_IMGS):
        RGBImgs.append(cv2.imread(RGBImgsPath+"/"+str(i)+".jpg"))
        GrayImgs.append(cv2.imread(GrayImgsPath+"/"+str(i)+".jpg"))


    RGBImgs = (np.array(RGBImgs, dtype='float32')-127.5)/127.5
    GrayImgs = (np.array(GrayImgs, dtype='float32')-127.5)/127.5
    return RGBImgs,GrayImgs
        
        
    

# os.mkdir('train/rgb')
# os.mkdir('train/gray')
# preprocessing()

RGBImgs,GrayImgs= GenerateSamples("train/rgb","train/gray")

# print(RGBImgs)