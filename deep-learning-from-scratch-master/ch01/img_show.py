# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2 as cv
import numpy as np
from skimage import img_as_ubyte

img = imread('../dataset/lena.png') # 画像の読み込み

cv_image = img_as_ubyte(img)
img = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
img = cv.equalizeHist(img)
plt.imshow(img)
plt.show()