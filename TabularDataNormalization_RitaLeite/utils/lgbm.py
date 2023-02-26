# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:20:42 2022

@author: ritas
"""

import sklearn as skl
from sklearn.model_selection import ParameterSampler
import numpy as np
import math
from scipy.stats import uniform
import lightgbm as lgbm
import joblib
#from google.colab import files

def lgbm_param_sampler(n_iters,seed,device):
    param_grid={'num_iterations': 
                range(200, 1200),
        'learning_rate':
            uniform(loc=0.001, scale=1-0.001 ),
        'num_leaves':
            range( 5, 500 ),
        'boosting_type':
            ["goss"],
        'min_data_in_leaf':
            range( 5, 200 ),
        'max_bin':
            range( 5, 500 ),
        'device': 
            [device]}
    param_list = list(ParameterSampler(param_grid, n_iter=n_iters, random_state=np.random.RandomState(seed)))
    return param_list

def lgbm_param_tuner(xtrain,xtest,ytrain,ytest,param_list):
    unique, counts = np.unique(ytest, return_counts=True)
    fraud_rate=counts[1]/sum(counts)
    print("Fraud Rate for test set:",fraud_rate)
    max_auc=-1000
    i=0
    for param_set in param_list:
        lgb=lgbm.LGBMClassifier().set_params(**param_set)
        lgb.fit(xtrain,ytrain)
        pred=lgb.predict_proba(xtest)[:,1]
        fpr,tpr,thresholds=skl.metrics.roc_curve(ytest,pred, drop_intermediate=False)
        auc=skl.metrics.roc_auc_score(ytest, pred)
        if max_auc<auc:
            max_auc=auc
            best_params=param_set
            best_model=lgb
            roc=fpr,tpr,thresholds
        print('iteration: ',i,"------- Auc: ",auc)
        
        if auc>0.5:

            joblib.dump(lgb,'./model_bin/lgbm/model{}.pkl'.format(i))
        
        if i!=0 and i%25==0:
            files.download('./model_bin/lgbm')

        i=i+1
    print("Done!")
    print("Best param set:",best_params)
    print("AUC score: ",max_auc)
    return best_model,best_params,roc



