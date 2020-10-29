# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet_single:
    """認識率99%以上の高精度なConvNet

    ネットワーク構成は下記の通り
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """

   
   
    def __init__(self, input_dim=(2, 99, 99),   #99 99
                 conv_param_1 = {'filter_num':1, 'filter_size':1, 'pad':1, 'stride':1, 'pool':1}, #3
                 conv_param_2 = {'filter_num':1, 'filter_size':1, 'pad':1, 'stride':1, 'pool':1},
                 conv_param_3 = {'filter_num':1, 'filter_size':1, 'pad':1, 'stride':1, 'pool':1},
                 conv_param_4 = {'filter_num':1, 'filter_size':1, 'pad':1, 'stride':1, 'pool':1},
                 hidden_size1=1, output_size=1, node_size = 1):#
        # 重みの初期化===========
        # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）
        
        
        FN1 = conv_param_1['filter_num']
        FN2 = conv_param_2['filter_num']
        FN3 = conv_param_3['filter_num']
        FN4 = conv_param_4['filter_num']
        
        F1S = conv_param_1['filter_size']
        F2S = conv_param_2['filter_size']
        F3S = conv_param_3['filter_size']
        F4S = conv_param_4['filter_size']
        
        stride1 = conv_param_1['stride']
        stride2 = conv_param_2['stride']
        stride3 = conv_param_3['stride']
        stride4 = conv_param_4['stride']
               
        
        pool_h1 = conv_param_1['pool'] #
        pool_w1 = conv_param_1['pool'] #
        #################
        pool_h2 = conv_param_2['pool']
        pool_w2 = conv_param_2['pool']
        #################
        pool_h3 = conv_param_3['pool'] #
        pool_w3 = conv_param_3['pool'] #
        
        pool_h4 = conv_param_4['pool'] #
        pool_w4 = conv_param_4['pool'] #




        
        pre_node_nums = np.array([1*1*1, FN1*F1S*F1S,FN2*F2S*F2S,FN3*F3S*F3S,FN4*F4S*F4S,hidden_size1, output_size]) #FN2*F2S*F2S, FN3*F3S*F3S,  
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値
        
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4]):  #,conv_param_3  
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
            
        
        self.params['W5'] = weight_init_scales[1] * np.random.randn(node_size, hidden_size1)
        self.params['b5'] = np.zeros(hidden_size1)
        
        self.params['W6'] = weight_init_scales[2] * np.random.randn(hidden_size1, output_size)
        self.params['b6'] = np.zeros(output_size)

        # レイヤの生成===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        #self.layers.append(Pooling(pool_h1, pool_w1, stride1))
        #----------------------------------------------------------------------------------------------------------------
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        #self.layers.append(Pooling(pool_h2, pool_w2, stride2))
        #----------------------------------------------------------------------------------------------------------------        
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        #self.layers.append(Pooling(pool_h3, pool_w3, stride3))
        #----------------------------------------------------------------------------------------------------------------
        self.layers.append(Convolution(self.params['W4'], self.params['b4'], conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h4, pool_w4, stride4))
        #----------------------------------------------------------------------------------------------------------------                
        self.layers.append(Affine(self.params['W5'], self.params['b5']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.3))
    
        
        self.layers.append(Affine(self.params['W6'], self.params['b6']))
        self.layers.append(Dropout(0.3))
        #----------------------------------------------------------------------------------------------------------------
        self.last_layer = SoftmaxWithLoss()
        

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                #print(x.shape)
                x = layer.forward(x)            
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0,2,4,6,9,12)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0,2,4,6,9,12)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]
