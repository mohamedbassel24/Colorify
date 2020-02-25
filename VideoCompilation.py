
import cv2
import re
import numpy as np
import glob

#to sort files to recompile frames in order
def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

#print(glob.glob('Test Video\\Segmented Frames GRAYSCALE\\*.jpg'))
FileArray=glob.glob('Test Video\\Segmented Frames GRAYSCALE\\*.jpg')
FileArraySorted=sorted_nicely(FileArray)

#print(FileArraySorted)

img_array = []

for filename in FileArraySorted:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('Test Video\\Video After Grayscale Conversion.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
