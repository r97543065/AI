# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet_self import DeepConvNet

def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    
    
    
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

network = DeepConvNet(input_dim=(2, 99, 99),
                 conv_param_1 = {'filter_num':FN1, 'filter_size':F1S, 'pad':PAD1, 'stride':stride1, 'pool':pool1}, #3
                 conv_param_2 = {'filter_num':FN2, 'filter_size':F2S, 'pad':PAD2, 'stride':stride2, 'pool':pool2},
                 conv_param_3 = {'filter_num':FN3, 'filter_size':F3S, 'pad':PAD3, 'stride':stride3, 'pool':pool3},
                 conv_param_4 = {'filter_num':FN4, 'filter_size':F4S, 'pad':PAD4, 'stride':stride4, 'pool':pool4},
                 hidden_size1=45, output_size=2, node_size = 63   #63          
                 )    
# ランダム初期化後の重み
##filter_show(network.params['W1'])

# 学習後の重み
network.load_params("deep_convnet_params14.pkl")
filter_show(network.params['W1'])
filter_show(network.params['W2'])
filter_show(network.params['W3'])
filter_show(network.params['W4'])