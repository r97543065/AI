# coding: utf-8
import sys, os
from os import listdir
from os.path import isfile, isdir, join
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet_self import DeepConvNet
from deep_convnet_self_single import DeepConvNet_single
from dataset.mnist import _load_IMG, load_IMG_files,load_IMG_files_single, _getpath
import cv2
from test import *
import time
from matplotlib.image import imread


def image_show(im, nx=8, margin=3, scale=10):

    N = np.shape(im);
    ny = 1

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(N[0]):
        ax = fig.add_subplot(ny, N[0], i+1, xticks=[], yticks=[])
        ax.imshow(im[i,0,:,:], cmap=plt.cm.gray_r, interpolation='nearest')
        
    plt.show()

    '''fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(N[0]):
        ax = fig.add_subplot(ny, N[0], i+1, xticks=[], yticks=[])
        ax.imshow(im[i,1,:,:], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()'''
    

last_imageID = ''
max_file = ''
mypath = "C:\\Users\\s9314\\Desktop\\git_AI\\deep-learning-from-scratch-master\\dataset\\A_self\\testing"
cnt=0    


FN1 = 7 
FN2 = 7
FN3 = 7
FN4 = 7 

F1S = 5  #3
F2S = 5  
F3S = 5
F4S = 5


PAD1 = int((F1S - 1)/2)
PAD2 = int((F2S - 1)/2)
PAD3 = int((F3S - 1)/2)
PAD4 = int((F4S - 1)/2)


stride1 = 2
stride2 = 2
stride3 = 2
stride4 = 2

pool1 = 1
pool2 = 1
pool3 = 1
pool4 = 3

network1 = DeepConvNet(input_dim=(2, 99, 99),
                 conv_param_1 = {'filter_num':FN1, 'filter_size':F1S, 'pad':PAD1, 'stride':stride1, 'pool':pool1}, #3
                 conv_param_2 = {'filter_num':FN2, 'filter_size':F2S, 'pad':PAD2, 'stride':stride2, 'pool':pool2},
                 conv_param_3 = {'filter_num':FN3, 'filter_size':F3S, 'pad':PAD3, 'stride':stride3, 'pool':pool3},
                 conv_param_4 = {'filter_num':FN4, 'filter_size':F4S, 'pad':PAD4, 'stride':stride4, 'pool':pool4},
                 hidden_size1=75, output_size=2, node_size = 63   #63          
                 )    

network2 = DeepConvNet_single(input_dim=(2, 99, 99),
                 conv_param_1 = {'filter_num':FN1, 'filter_size':F1S, 'pad':PAD1, 'stride':stride1, 'pool':pool1}, #3
                 conv_param_2 = {'filter_num':FN2, 'filter_size':F2S, 'pad':PAD2, 'stride':stride2, 'pool':pool2},
                 conv_param_3 = {'filter_num':FN3, 'filter_size':F3S, 'pad':PAD3, 'stride':stride3, 'pool':pool3},
                 conv_param_4 = {'filter_num':FN4, 'filter_size':F4S, 'pad':PAD4, 'stride':stride4, 'pool':pool4},
                 hidden_size1=45, output_size=2, node_size = 63   #63          
                 ) 

#network1.load_params("deep_convnet_params14.pkl")  
#network2.load_params("deep_convnet_params15.pkl")  
network1.load_params("deep_convnet_params_Aphtae_6_14.pkl")


img_files = listdir(mypath)
for imageID in img_files:              
    img_files[cnt] = imageID[:len(imageID)-4]            
    cnt+=1    

cnt=0            
#max_file = img_files[len(img_files)-2]
detection_result={};
P = 0;
N = 0;                           

for C in range(np.size(img_files)):
    t_test = np.zeros((2,))
    img_name, x_test = load_IMG_files(mypath + '\\' + img_files[C] + '.jpg')    
    print("calculating test accuracy ... ")
                        
    name = img_name
    
    classified_ids = []
    
    acc = 0.0
    batch_size = 1
                        
    for i in range(int(x_test.shape[0] / batch_size)):
       tx = x_test[i*batch_size:(i+1)*batch_size]
       tt = t_test[i*batch_size:(i+1)*batch_size]
       y = network1.predict(tx, train_flg=False)
       #print(y)
       y = np.argmax(y, axis=1)
       #image_show(tx)      
       classified_ids.append(y)
    
    
    classified_ids = np.array(classified_ids)
    classified_ids = classified_ids.flatten()
         
    
    mis_pairs = {}
    detected_bar = 0;
    repeat = 0;
    classified_ids[1] = 0;
    classified_ids[6] = 0;
    for i, val in enumerate(classified_ids):    
        mis_pairs[i] = classified_ids[i]     
        if i-1 >= 0:
            if mis_pairs[i-1] == 1 and mis_pairs[i] == 1 and repeat == 0:
                repeat = 1;
            elif mis_pairs[i-1] != mis_pairs[i] or repeat == 1:
                repeat = 0;
                detected_bar = detected_bar + mis_pairs[i];
            
            #if mis_pairs[i-1] != mis_pairs[i]:   
            #    detected_bar = detected_bar +  mis_pairs[i]; 
                
                
    #print(name)
    #print("======= classified result =======")
    #print("{view index: (label, inference), ...}")
    print(mis_pairs);
    #print(detected_bar);

    if detected_bar >= 2:
       # print('positive');  
       P = P+1;
       detection_result[C,0] = img_files[C] + ' is positive';
    else:
       # print('negative');
       N = N+1;
       detection_result[C,0] = img_files[C] + ' is negative';        
      
print(detection_result) 
print(P)
print(N)  
####################################################################################    

    
