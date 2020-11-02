# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import _load_IMG
from deep_convnet_self import DeepConvNet
from common.trainer import Trainer

#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

(x_train,t_train), (x_test, t_test), name,dataset_dir = _load_IMG(infection=True)

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
                 hidden_size1=75, output_size=2, node_size = 63   #63          
                 )  

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=500, mini_batch_size=89,#20,  100
                  optimizer='Adam', optimizer_param={'lr':0.0008},
                  evaluate_sample_num_per_epoch=10)#1000
                    ##### number of iteration = [(total_data/mini_batch_size)->(one epoch)]  *  [epochs]

trainer.train()
#Adam
# パラメータの保存
network.save_params("deep_convnet_params_Aphtae_4_14.pkl")
print("Saved Network Parameters!")
