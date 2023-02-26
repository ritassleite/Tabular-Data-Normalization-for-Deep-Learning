# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:39:58 2022

@author: ritas
"""

import numpy as np
import torch.nn as nn
import torch
import sklearn as skl
from sklearn.model_selection import ParameterSampler
import itertools
from scipy.stats import uniform
import mlp_pipeline
import joblib

def mlp_dims(dim_input,max_layers):
    
    i=1
    res=list()
    while 2**i<=dim_input:
        i=i+1
    possible_combos=list(itertools.product(range(1,i),repeat=max_layers))
    for combo in possible_combos:
        accept=True
        for j in range(0,max_layers-1):
            if combo[j+1]>combo[j]:
                accept=False
                break
        if accept==True:
            res.append(np.asarray(combo))
    return res
            
def generate_architectures(dim_input):
    res=[]
    for j in range(1,5):
        res=res+mlp_dims(dim_input,j)
    return res


    
    
    
def mlp_param_sampler(n_iters,dim_input,seed,device):
    param_grid={'hidden_layers':generate_architectures(dim_input),
                'activation_type': [nn.ReLU(),nn.LeakyReLU(),nn.Tanh()],
                'learning_rate': uniform(loc=0.00001, scale=0.01-0.00001 ),
                'dropout_rate': uniform(loc=0,scale=0.1),
                'weight_decay':uniform(loc=0.00001, scale=0.001-0.00001 ),
                'batch_size': [2**power for power in range(6,10)],
                'batch_norm': [True,False],
                
                }
    param_list = list(ParameterSampler(param_grid, n_iter=n_iters, random_state=np.random.RandomState(seed)))
    return param_list





def mlp_param_tuner(device,xtrain,xtest,ytrain,ytest,param_list,normalization,cont_cols):
    unique, counts = np.unique(ytest, return_counts=True)
    fraud_rate=counts[1]/sum(counts)
    print("Fraud Rate for test set:",fraud_rate)
    max_auc=-1000
    i=0
    for param_set in param_list:
        
                
        print('iteration: ',i)
        
        model=mlp_pipeline.pipeline(device,xtrain,xtest,ytrain,ytest,param_set,normalization,cont_cols)

        joblib.dump(model,'/content/gdrive/MyDrive/Colab Notebooks/data/mlp_2_w_target_enc/mlp{}{}.pkl'.format(normalization,i))
        

        i=i+1
    print("Done!")
    
    return i
