# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:28:18 2022

@author: ritas
"""


import numpy as np
import sklearn as skl
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import zscore
import torch
import matplotlib.pyplot as plt
import math
from early_stopping import EarlyStopping
import gc
import mlp
import custom_sampler
import time


def z_score(X,cont_cols):
    X_aux=X.copy()
    X_aux[cont_cols]=X_aux.loc[:,cont_cols].apply(lambda x: x if np.std(x) == 0 else zscore(x,nan_policy='omit'))
    return X_aux
def minmax_norm(X,cont_cols):
    cont_cols=[col for col in X.select_dtypes(float)]
    X_aux=X.copy()
    X_aux[cont_cols]=X_aux.loc[:,cont_cols].apply(lambda x: x if np.std(x) == 0 else skl.preprocessing.minmax_scale(x))
    return X_aux 
def med_norm(X,cont_cols):
    cont_cols=[col for col in X.select_dtypes(float)]
    X_aux=X.copy()
    X_aux[cont_cols]=X_aux.loc[:,cont_cols].apply(lambda x: x if np.std(x) == 0 else skl.preprocessing.robust_scale(x))
    return X_aux     

def normalization_transform(xtrain,normalization,cont_cols):    
    if normalization=='None':
        xtrain_aux=torch.tensor(xtrain.values)
    elif normalization=='Zscore':
        xtrain_aux=torch.tensor(z_score(xtrain,cont_cols).values)
    elif normalization=='MinMax':
        xtrain_aux=torch.tensor(minmax_norm(xtrain,cont_cols).values)
    elif normalization=='Median':
        xtrain_aux=torch.tensor(med_norm(xtrain,cont_cols).values)
    elif normalization=="MinMax Adj":
        xtrain_aux=torch.tensor(0.8*minmax_norm(xtrain,cont_cols).values+0.1)
    return xtrain_aux





class MultiLayerMLP(nn.Module):

    def __init__(self,input_size,dim_layers,activation_type,learning_rate,dropout_rate):
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size in dim_layers:
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(input_size, 2**size))
            input_size = 2**size  # For the next layer
            self.layers.append(activation_type)
        self.layers.append(nn.Linear(input_size,1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
    def _init_weights(self, module):
        torch.manual_seed(154783456)
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module,nn.Linear):
                module.weight.data.normal_(mean=0.0,std=1.0)
                module.bias.data.zero_()    
    def forward(self, X):
        
        for layer in self.layers:
            X=layer(X)
       
        y = torch.sigmoid(X)
        
        return y


class MultiLayerMLP_BatchNorm(nn.Module):

    def __init__(self,input_size,dim_layers,activation_type,learning_rate,dropout_rate):
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size in dim_layers:
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(input_size, 2**size))
            input_size = 2**size  # For the next layer
            self.layers.append(nn.BatchNorm1d(input_size))
            self.layers.append(activation_type)
        self.layers.append(nn.Linear(input_size,1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
    def _init_weights(self, module):
        #torch.manual_seed(154783456)
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module,nn.Linear):
                module.weight.data.normal_(mean=0.0,std=1.0)
                module.bias.data.zero_()

        
    def forward(self, X):
        
        for layer in self.layers:
            X=layer(X)
       
        y = torch.sigmoid(X)
        
        return y



def pipeline(device,xtrain,xtest,ytrain,ytest,param_grid,normalization,cont_cols):
    
    if param_grid['batch_norm']:
        model=MultiLayerMLP_BatchNorm(input_size=len(xtrain[0]),dim_layers=param_grid['hidden_layers'],activation_type=param_grid['activation_type'],
                                learning_rate=param_grid['learning_rate'],dropout_rate=param_grid['dropout_rate'])
    else:
        model=MultiLayerMLP(input_size=len(xtrain[0]),dim_layers=param_grid['hidden_layers'],activation_type=param_grid['activation_type'],
                                learning_rate=param_grid['learning_rate'],dropout_rate=param_grid['dropout_rate'])
                                
    model.apply(model._init_weights)

    print("----------Method: ",normalization,"-------------")
    trainloader,valloader=data_loader_balanced(xtrain,ytrain,xtest,ytest,param_grid['batch_size'])
    
    criterion = nn.BCELoss()
    
    model=early_stop(model,device,trainloader,valloader,model.optimizer,criterion,30,5)


    del trainloader
    del valloader
    gc.collect()
    return model





def data_loader_balanced(xtrain,ytrain,xval,yval,batch):
    unique_train, counts_train = np.unique(ytrain, return_counts=True)
    print("Train size: ",len(ytrain),"; Number of 0: ",counts_train[0],"; Number of 1:",counts_train[1])
    unique, counts = np.unique(yval, return_counts=True)
    print("Val size: ",len(yval),"; Number of 0: ",counts[0],"; Number of 1:",counts[1])
    
    trainratio = np.bincount(ytrain.values)
    classcount = trainratio.tolist()
    train_weights = 1./torch.tensor(classcount, dtype=torch.float)
    train_sampleweights = train_weights[ytrain.values]

    sampler=custom_sampler.PersonalizedWeightedRandomSampler(labels=ytrain.values,weights=train_sampleweights,num_samples=len(ytrain),batch_size=batch,replacement=True) 
    trainloader = torch.utils.data.DataLoader([[xtrain[i], ytrain.values[i]] for i in range(0,len(ytrain))],sampler=sampler,batch_size=batch, num_workers=0)
    val_loader = torch.utils.data.DataLoader([[xval[i], yval.values[i]] for i in range(0,len(yval))], batch_size=len(yval),
                                         shuffle=True, num_workers=0)
    
    return trainloader,val_loader



def early_stop(model,device,trainloader,valid_loader,optimizer,criterion,epochs,patience):
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    for epoch in range(0,epochs-1):
        avg_train_losses.append(train(model,device,trainloader,criterion))
        model.eval()
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            # calculate the loss
            loss = criterion(output.float(), target.unsqueeze(1).float())
            # record validation loss
            valid_losses.append(loss.item())

        valid_loss = np.average(valid_losses)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(epochs))
        
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    
    model.load_state_dict(torch.load('checkpoint.pt'))
    #plt.plot(avg_train_losses)
    #plt.plot(avg_valid_losses)
    #plt.show()
    avg_train_losses=[]
    avg_val_losses=[]
    return  model


def train(model, device, train_loader,criterion):
    
    losses=[]
    for i in train_loader:
        model.train()

        #LOADING THE DATA IN A BATCH
        data, target = i
        
        #MOVING THE TENSORS TO THE CONFIGURED DEVICE
        data, target = data.to(device), target.to(device)
        
        #FORWARD PASS
        output = model(data.float())
        
        #BACKWARD AND OPTIMIZE
        model.optimizer.zero_grad()
        loss = criterion(output.float(), target.unsqueeze(1).float())
        loss.backward()
        
        
        model.optimizer.step()
        losses.append(loss.cpu().detach().numpy())

    return np.average(losses)