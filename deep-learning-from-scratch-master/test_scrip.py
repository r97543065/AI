# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:56:28 2017

#import numpy as np
@author: me1237guy
"""
from dataset.mnist import _load_IMG, load_IMG_files
from test import *
import os.path
import os
import numpy as np
import time
from os import listdir
from os.path import isfile, isdir, join
"""
import cupy as cp

import time
"""
from matplotlib import pyplot as plt
import cv2

"""
x_cpu = np.zeros((100,100,3))
s = time.time()
x_cpu[:,:,0] = np.ones((100,100))
x_cpu[:,:,1] = np.ones((100,100))
x_cpu[:,:,2] = 200*np.ones((100,100))
x_cpu = x_cpu.astype(np.uint8)
e = time.time()
print(e-s)

s = time.time()
x_gpu = 3*cp.ones((100,100,3))
e = time.time()
print(e-s)
"""

dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir_self = dataset_dir + "\\dataset\\self"
#dataset_dir = os.path.dirname(sys.executable)

print(dataset_dir_self)

localtime = time.localtime(time.time())
filename = str(localtime[0])+str(localtime[1])+str(localtime[2])+str(localtime[3])+str(localtime[4])+str(localtime[5])
print("time :", filename)
uploadFile(filename+'.bmp',dataset_dir_self+'\\1.bmp','image/bmp')
         

#print(files)

#imageID = searchFile(100,"name contains '0000'")
#downloadFile(imageID,dataset_dir+'\\detected_image.bmp')
#uploadFile(filename+'.bmp',dataset_dir+'\\00007.bmp','image/bmp')

'''
last_imageID = ''
while (1):#input_char != "STOP"
        max_file = listFiles(10)
        imageID = searchFile(100,"name contains '"+max_file+"'")    
        
        if(last_imageID==imageID):DETECT = 0
        else:DETECT = 1 
        
        print(DETECT)
        if(DETECT):
            downloadFile(imageID,dataset_dir_self + '\\'+max_file)
            
        last_imageID = imageID     

        #imageID = searchFile(100,"name contains '0000'")        
        #downloadFile(imageID,dataset_dir_self+'\\detected_image.bmp')
        #input_char = input("")
'''
print(dataset_dir_self)
     
 

   
#

'''
input_char = ""
while (input_char != "STOP"): 
        input_char = input("")
        print(dataset_dir)

print("--------------------END PROGRAM--------------------")    
'''

'''
x_train[0,:,:,:] = 200*x_train[0,:,:,:]
X_imshow = np.zeros((90,90,3)).astype(np.uint8)
X_imshow[:,:,0] = (x_train[0,0,:,:]).astype(np.uint8)
X_imshow[:,:,1] = (x_train[0,0,:,:]).astype(np.uint8)
X_imshow[:,:,2] = (x_train[0,0,:,:]).astype(np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',X_imshow)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
