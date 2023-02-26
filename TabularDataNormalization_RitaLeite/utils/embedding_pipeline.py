# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:07:12 2022

@author: ritas
"""


import numpy as np
#from pandas.core.dtypes.base import E
import sklearn as skl
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn import preprocessing
import category_encoders
import custom_sampler
from early_stopping import EarlyStopping
import gc



def aggregate_low_card(xtrain,xtest,xval, cat_cols):
    
    x_train=xtrain.copy(deep=True)
    x_val=xval.copy(deep=True)
    x_test=xtest.copy(deep=True)

    for col in cat_cols:
        aux=x_train[col].value_counts(normalize=False)<10
        cats=aux[aux==True].index
        if x_train[col].dtype=='int64' or x_train[col].dtype=='float64':
            x_train[col]=xtrain[col].mask(x_train[col].isin(cats),-10**10)
            x_val[col]=xval[col].mask(x_val[col].isin(cats),-10**10)
            x_test[col]=xtest[col].mask(x_test[col].isin(cats),-10**10)
        else:
            x_train[col]=x_train[col].mask(x_train[col].isin(cats),'Other')
            x_val[col]=x_val[col].mask(x_val[col].isin(cats),'Other')
            x_test[col]=x_test[col].mask(x_test[col].isin(cats),'Other')
        
    encoder=category_encoders.ordinal.OrdinalEncoder(verbose=0,cols=cat_cols,handle_unknown='value',handle_missing='value')
    x_train=encoder.fit_transform(x_train)
    x_test=encoder.transform(x_test)
    x_val=encoder.transform(x_val)
    x_train[cat_cols]=x_train[cat_cols]+2
    x_test[cat_cols]=x_test[cat_cols]+2
    x_val[cat_cols]=x_val[cat_cols]+2
    return x_train,x_test,x_val

def aggregate_low_card_BAF(x_train,x_test,x_val, cat_cols):
    xtrain=x_train.copy(deep=True)
    xval=x_val.copy(deep=True)
    xtest=x_test.copy(deep=True)    
    
    for col in cat_cols:
        aux=xtrain[col].value_counts(normalize=False)<10
        cats=aux[aux==True].index
        if xtrain[col].dtype=='int64' or xtrain[col].dtype=='float64':
            xtrain[col]=xtrain[col].mask(xtrain[col].isin(cats),-10**10)
            xval[col]=xval[col].mask(xval[col].isin(cats),-10**10)
            xtest[col]=xtest[col].mask(xtest[col].isin(cats),-10**10)
        else:
            xtrain[col]=xtrain[col].mask(xtrain[col].isin(cats),'Other')
            xval[col]=xval[col].mask(xval[col].isin(cats),'Other')
            xtest[col]=xtest[col].mask(xtest[col].isin(cats),'Other')
        
    encoder=category_encoders.ordinal.OrdinalEncoder(verbose=0,cols=cat_cols,handle_unknown='value',handle_missing='value')
    xtrain=encoder.fit_transform(xtrain)
    xtest=encoder.transform(xtest)
    xval=encoder.transform(xval)
    xtrain[cat_cols]=xtrain[cat_cols]+1
    xtest[cat_cols]=xtest[cat_cols]+1
    xval[cat_cols]=xval[cat_cols]+1
    return xtrain,xtest,xval

def get_emb_dim(xtrain,method,cat_cols):
    dims=np.zeros((len(cat_cols),2))
    i=0
    for col in cat_cols:
        print(col)
        cardinality=(max(xtrain[col]))+1
        if method=='sqrt':
            dims[i]=[cardinality,(3*round(math.sqrt(cardinality))-2)]
        elif method=='log':
            dims[i]=[cardinality,math.floor(math.log(cardinality+1)+1)]
        elif method=='4root':
            dims[i]=[cardinality,round(cardinality**0.25)]
        elif method=='ones':
            dims[i]=[cardinality,1]
        else:
            dims[i]=[cardinality,min(50,math.ceil(cardinality/2))]
        i=i+1
    return dims

def get_emb_dim_var(xtrain,cat_cols,alpha):
    dims=np.zeros((len(cat_cols),2))
    i=0
    for col in cat_cols:
        cardinality=(max(xtrain[col]))+1
        dims[i]=[cardinality,(round(alpha*math.sqrt(cardinality))-2)]
        i=i+1
    return dims

class Embedding_layers(nn.Module):

    def __init__(self,emb_dims):
      super().__init__()
      self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                     for (x, y) in emb_dims])
      
    def forward(self,x_cat):
        out= [emb_layer(x_cat[:,i])
             for i,emb_layer in enumerate(self.emb_layers)]
        aux=out[0]
        for i in range(1,len(out)):
            aux=torch.cat((aux,out[i]),dim=1)

        return aux


class LinearBlocks(nn.Module):
    def __init__(self,input_size,dim_layers,activation_type,learning_rate,dropout_rate):
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size 
        
        
        
        for size in dim_layers:
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(input_size, 2**size))
            input_size = 2**size  # For the next layer
            self.layers.append(activation_type)
        self.layers.append(nn.Linear(input_size,1))
        
       
        
    def forward(self, X):
        
        for layer in self.layers:
 
            X=layer(X)
       
        
        return X
    
class LinearBlocksBatchN(nn.Module):
    def __init__(self,input_size,dim_layers,activation_type,learning_rate,dropout_rate):
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size 
        
        
        
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
        
    def forward(self, X):
        
        for layer in self.layers:

            X=layer(X)
       
        
        return X    
    
class EmbMLP(nn.Module):
    def __init__(self,seed_it,input_size,dim_layers,activation_type,learning_rate,dropout_rate,emb_dims):
        super().__init__()
        self.embedding=Embedding_layers(emb_dims)
        self.linear_layers=LinearBlocks(input_size,dim_layers,activation_type,learning_rate,dropout_rate)
        self.seed_it=seed_it
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

    def _init_weights(self, module):
        if self.seed_it:
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
    def forward(self,x_cont,x_cat):
        embedded_cats=self.embedding(x_cat)
        
        x=torch.cat([x_cont,embedded_cats],1)
        
        y=self.linear_layers(x)
        
        output=torch.sigmoid(y)
        return output
    
class EmbMLPBatchN(nn.Module):
    def __init__(self,seed_it,input_size,dim_layers,activation_type,learning_rate,dropout_rate,emb_dims):
        super().__init__()
        self.embedding=Embedding_layers(emb_dims)
        self.linear_layers=LinearBlocksBatchN(input_size,dim_layers,activation_type,learning_rate,dropout_rate)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.seed_it=seed_it
    def _init_weights(self, module):
        if(self.seed_it):
            torch.manual_seed(1547834567)
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
                
    def forward(self,x_cont,x_cat):
        embedded_cats=self.embedding(x_cat)
        
        x=torch.cat([x_cont,embedded_cats],1)
        y=self.linear_layers(x)
        
        output=torch.sigmoid(y)
        
        return output    


def data_loader_balanced(xtrain_cat,xtrain_cont,ytrain,xval_cat,xval_cont,yval,batch):
    unique_train, counts_train = np.unique(ytrain, return_counts=True)
    print("Train size: ",len(ytrain),"; Number of 0: ",counts_train[0],"; Number of 1:",counts_train[1])
    unique, counts = np.unique(yval, return_counts=True)
    print("Val size: ",len(yval),"; Number of 0: ",counts[0],"; Number of 1:",counts[1])

    trainratio = np.bincount(ytrain.values)
    classcount = trainratio.tolist()
    train_weights = 1./torch.tensor(classcount, dtype=torch.float)
    train_sampleweights = train_weights[ytrain.values]

    sampler=custom_sampler.PersonalizedWeightedRandomSampler(labels=ytrain.values,weights=train_sampleweights,num_samples=len(ytrain),batch_size=batch,replacement=True) 
    trainloader = torch.utils.data.DataLoader(torch.cat((xtrain_cat,xtrain_cont,torch.tensor(ytrain.values).resize_(len(ytrain.values),1)),dim=1),sampler=sampler,batch_size=batch, num_workers=0)
    val_loader = torch.utils.data.DataLoader(torch.cat((xval_cat,xval_cont,torch.tensor(yval.values).resize_(len(yval.values),1)),dim=1), batch_size=len(yval),
                                         shuffle=True, num_workers=0)
    
    return trainloader,val_loader

def pipeline(device,xtrain_cat,xtrain_cont,xval_cat,xval_cont,ytrain,yval,param_grid,embedding_dim_method,emb_dims):
    seed_it=False
    if len(xtrain_cont)==0:
        l=0
    else:
        l=len(xtrain_cont[0])
    aux=sum([b for a,b in emb_dims])
    if param_grid['batch_norm']:
        model=EmbMLP(seed_it,input_size=aux+l,dim_layers=param_grid['hidden_layers'],activation_type=param_grid['activation_type'],
                                learning_rate=param_grid['learning_rate'],dropout_rate=param_grid['dropout_rate'],emb_dims=emb_dims)
    else:
        model=EmbMLPBatchN(seed_it,input_size=aux+l,dim_layers=param_grid['hidden_layers'],activation_type=param_grid['activation_type'],
                                learning_rate=param_grid['learning_rate'],dropout_rate=param_grid['dropout_rate'],emb_dims=emb_dims)
    model.apply(model._init_weights)
    print("----------Method: Embedding for Categorical",embedding_dim_method,"-------------")
    trainloader,valloader=data_loader_balanced(xtrain_cat,xtrain_cont,ytrain,xval_cat,xval_cont,yval,param_grid['batch_size'])
    
    criterion = nn.BCELoss()
    
    model=early_stop(model,device,trainloader,valloader,model.optimizer,criterion,30,5,len(xtrain_cat[0]),len(xtrain_cont[0]))
    
    

    del trainloader
    del valloader
    gc.collect()
    return model




def early_stop(model,device,trainloader,valid_loader,optimizer,criterion,epochs,patience,n_cat,n_cont):
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
        avg_train_losses.append(train(model,device,trainloader,criterion,n_cat,n_cont))
        
        for i in valid_loader:

            #LOADING THE DATA IN A BATCH
            data_cat=i[:,0:n_cat]
            data_cont=i[:,n_cat:n_cat+n_cont]
            target=i[:,n_cat+n_cont]
            data_cat,data_cont, target = data_cat.to(device),data_cont.to(device), target.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data_cont.float(),data_cat.long())
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
    
    avg_train_losses=[]
    avg_val_losses=[]
    return  model


def train(model, device, train_loader,criterion,n_cat,n_cont):
    
    model.train()
    losses=[]
    for i in train_loader:
        
        #LOADING THE DATA IN A BATCH
        data_cat=i[:,0:(n_cat)]
        data_cont=i[:,n_cat:n_cat+n_cont]
        target=i[:,n_cat+n_cont]

        
        #MOVING THE TENSORS TO THE CONFIGURED DEVICE
        data_cat,data_cont, target = data_cat.to(device),data_cont.to(device), target.to(device)
        
        #FORWARD PASS
        output = model(data_cont.float(),data_cat.long())
        
        #BACKWARD AND OPTIMIZE
        model.optimizer.zero_grad()
        loss = criterion(output.float(), target.unsqueeze(1).float())
        loss.backward()
                
        model.optimizer.step()
        losses.append(loss.cpu().detach().numpy())

    return np.average(losses)



def pipeline_no_cat(device,xtrain_cat,xtrain_cont,xval_cat,xval_cont,ytrain,yval,param_grid,embedding_dim_method,emb_dims):
    aux=sum([b for a,b in emb_dims])
    seed_it=False
    if param_grid['batch_norm']:
        model=EmbMLP(seed_it,input_size=aux,dim_layers=param_grid['hidden_layers'],activation_type=param_grid['activation_type'],
                                learning_rate=param_grid['learning_rate'],dropout_rate=param_grid['dropout_rate'],emb_dims=emb_dims)
    else:
        model=EmbMLPBatchN(seed_it,input_size=aux,dim_layers=param_grid['hidden_layers'],activation_type=param_grid['activation_type'],
                                learning_rate=param_grid['learning_rate'],dropout_rate=param_grid['dropout_rate'],emb_dims=emb_dims)

    print("----------Method: Embedding for Categorical",embedding_dim_method,"-------------")
    trainloader,valloader=data_loader_balanced(xtrain_cat,xtrain_cont,ytrain,xval_cat,xval_cont,yval,param_grid['batch_size'])
    

    criterion = nn.BCELoss()
    
    if(xtrain_cont.size()==torch.Size([0])):
      len_cont=0
    else:
      len_cont=len(xtrain_cont[0])

    model=early_stop(model,device,trainloader,valloader,model.optimizer,criterion,30,5,len(xtrain_cat[0]),len_cont)
    
    

    del trainloader
    del valloader
    gc.collect()
    return model
