# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
    
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os.path
import gzip
import pickle
import os
import sys
import numpy as np
from PIL import Image
import glob
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import random
import cv2 as cv
from skimage.util import random_noise
from skimage import img_as_ubyte

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
#dataset_dir = os.path.dirname(sys.executable)

print (dataset_dir)
save_file = dataset_dir + "/mnist.pkl"
#dataset_dir_self = dataset_dir + "/self"
dataset_dir_self = dataset_dir + "/A_self"



train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = img_dim[1]*img_dim[2]

def _getpath():
    return dataset_dir_self


def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

def download_mnist():
    for v in key_file.values():
       _download(v)

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNISTデータセットの読み込み

    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

def _load_IMG(infection=True):
    cnt = 0
    
    #IM = np.zeros((90,1,99, 99)) #120
    max_view = 461
    section_ratio = 0.8
    IM = np.zeros((max_view,2,99, 99)) #120  
    
    #im_buff = np.zeros((1,1,40, 40))
    label = np.zeros((max_view,))  #90


    
    current_view = 1 
    index = random.sample(range(0,max_view),max_view)   #90
    gain = 1
    if infection:
        image_path = dataset_dir_self
        for name in glob.glob(image_path + "/*.bmp"):
        
         if name[len(image_path)+1] == 'N' or name[len(image_path)+1] == 'n':
             label[index[cnt]] = 0 #cnt         
         elif name[len(image_path)+1] == 'P' or name[len(image_path)+1] == 'p':
             label[index[cnt]] = 1 # cnt
             
         #print(name)
         im = imread(name)
         
         siz = np.shape(im);
         if np.size(siz) == 3:
          im = np.array(cv.cvtColor(im,cv.COLOR_BGR2GRAY));
            
         cv_image = img_as_ubyte(im)                 
         im = gain*random_noise(cv.equalizeHist(cv_image), mode='gaussian', seed=None, clip=True, var=0.000005)
         
         im = np.array(im)         
         #im = im / 255
         
         ##################################
         im_buff = im
         ##################################         
         im = resize(im_buff, (99, 99),anti_aliasing=True)#im_buff                  
               
         IM_gradient_buff = np.gradient(im);                 
         IM[index[cnt],0,:,:] = IM_gradient_buff[1];
         IM[index[cnt],1,:,:] = IM_gradient_buff[0];
         #IM[index[cnt],0,:,:] = im.astype(np.float32) 
        
         
         
         current_view += 1
         if current_view > max_view:
            break
         if cnt < max_view:
          cnt=cnt+1
    
    #print(name)  
    #print(dataset_dir)            
    IM = IM.astype(np.float32)
    label = label.astype(np.int)   
    return (IM[1:round(section_ratio*max_view),:,:,:], label[1:round(section_ratio*max_view)]),(IM[round(section_ratio*max_view):max_view,:,:,:], label[round(section_ratio*max_view):max_view]),name,dataset_dir_self


def load_IMG_files(file_path):
    max_F = file_path
    im = imread(file_path)    
    siz = np.shape(im);
        
    if np.size(siz) == 3:
      im = np.array(cv.cvtColor(im,cv.COLOR_BGR2GRAY));
      
    im = random_noise(im, mode='gaussian', seed=None, clip=True, var=0.0000001)  
    #kernel = np.ones((5,5),np.float32)/25
    #im = cv.GaussianBlur(im,(3,3),0)
##############################################    
    row_sec = 3
    col_sec = 6
    row_N = 0;
    col_N = 0;
    
    row_N = round(siz[0]/row_sec);
    col_N = round(siz[1]/col_sec);

    IM = np.zeros((1*col_sec+2,2,99, 99)) #90
    cnt = 0;
    
    row = 1
    #for row in range(row_sec):
    for col in range(col_sec+2):
        
        if(col != 4 and col!=5 and col!=6 and col!=7):
            I = im[ (row)*row_N : ((row+1)*row_N) , (col)*col_N : ((col+1)*col_N) ];            
        elif(col==7):
            I = im[ (row)*row_N : ((row+1)*row_N) , (5)*col_N : ((5+1)*col_N) ];        
        elif(col==6):
            I = im[ (row)*row_N : ((row+1)*row_N) , (4)*col_N : ((4+1)*col_N) ];                    
        elif(col==4):
            I = im[ (row)*row_N : ((row+1)*row_N) , (4)*col_N-10 : ((4+1)*col_N) ];
        elif(col==5):
            I = im[ (row)*row_N : ((row+1)*row_N) , (4)*col_N-15 : ((4+1)*col_N) ];
            #I = im[ (row)*row_N : ((row+1)*row_N) , (4)*col_N-5 : ((4+1)*col_N) ];
        if(col==2 or col==4 or col ==5 or col ==7):
           I = 1.6*I; 
        
        #cv_image = img_as_ubyte(I)
        #I = random_noise(cv.equalizeHist(cv_image), mode='gaussian', seed=None, clip=True, var=0.0000001)
        I = resize(I, (99, 99),anti_aliasing=True)   #99 99
        I_gradient = np.gradient(I);  
        IM[cnt,0,:,:] = I_gradient[1];
        IM[cnt,1,:,:] = I_gradient[0];  
        cnt = cnt+1;    
##############################################		
    IM = IM.astype(np.float32)    
    return max_F, (IM[:,:,:,:])	                   

			
    
def load_IMG_files_single(file_path):
    max_F = file_path
    im = imread(file_path)
    
    siz = np.shape(im);
    if np.size(siz) == 3:
      im = np.array(cv.cvtColor(im,cv.COLOR_BGR2GRAY));
    
      
##############################################    
    IM = np.zeros((1,2,99, 99)) #90
    im = resize(im, (99, 99),anti_aliasing=True)   #99 99
    IM_gradient = np.gradient(im);                     
    IM[0,0,:,:] = IM_gradient[1];
    IM[0,1,:,:] = IM_gradient[0];
##############################################
    IM = IM.astype(np.float32)    
    return max_F, (IM[:,:,:,:]) 



if __name__ == '__main__':
    init_mnist()
