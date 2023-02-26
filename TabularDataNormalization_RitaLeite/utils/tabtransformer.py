
import numpy as np
import torch.nn as nn
import torch
import sklearn as skl
from sklearn.model_selection import ParameterSampler
import itertools
from scipy.stats import uniform
import mlp_pipeline
import joblib
from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import mlp_pipeline
import time
import custom_sampler
#embedding_dims : depending on the dataset will be set according to the previously decided methods
#task='classification'
#sampler: ver melhor

from sklearn.preprocessing import LabelEncoder  # Categorical encoding for LGBM models
from sklearn import metrics                     # ROC metrics
import yaml

def tabtransformer_param_sampler(n_iters,seed,device):
    param_grid={"input_embed_dim" :[8,16,32,64],
                "num_heads":[2,4,8],
                "num_attn_blocks" :[1,2,3,4],
                "attn_dropout":uniform(loc=0,scale=0.1),
                "add_norm_dropout":uniform(loc=0,scale=0.1),
                "mlp_dropout":uniform(loc=0,scale=0.1),
                "embedding_dropout":uniform(loc=0,scale=0.1),
                "ff_dropout":uniform(loc=0,scale=0.1),
                "transformer_activation" :['ReGLU','GEGLU'],
                'activation_type': ['ReLU','LeakyReLU','Tanh'],
                'learning_rate': uniform(loc=0.00001, scale=0.01-0.00001 ),
                'weight_decay':uniform(loc=0.00001, scale=0.001-0.00001 ),
                'batch_size': [2**power for power in range(8,12)],
                'batch_norm': [True,False],
                'batch_norm_continuous_input': [True,False]
                }
    param_list = list(ParameterSampler(param_grid, n_iter=n_iters, random_state=np.random.RandomState(seed)))
    return param_list


def transformer_pipeline(param,cat_cols,cont_cols,target,xtrain,xval,ytrain,yval,device):
        
    
    l=len(cat_cols)*param['input_embed_dim']+len(cont_cols)
    
    head_config = dict(
                    layers=str(2*l)+'-'+str(4*l),
                    activation=param['activation_type'],
                    dropout=float(param['mlp_dropout']),
                    use_batch_norm=param['batch_norm'],
                    initialization="normal",
                )
    tabtransformer_config=TabTransformerConfig(input_embed_dim=param['input_embed_dim'],num_heads=param['num_heads'],
                                        num_attn_blocks=param['num_attn_blocks'],attn_dropout=float(param['attn_dropout']),
                                        add_norm_dropout=float(param['add_norm_dropout']),embedding_dropout=float(param['embedding_dropout']),
                                        ff_dropout=float(param['ff_dropout']),transformer_activation=param['transformer_activation'],
                                        head_config=head_config,task='classification',
                                        batch_norm_continuous_input=param['batch_norm_continuous_input'],
                                        learning_rate=float(param['learning_rate'])
                                        )
    
    data_config=DataConfig(target=target,continuous_cols=cont_cols,categorical_cols=cat_cols)
    
    
    optimizer_config=OptimizerConfig(optimizer='Adam', lr_scheduler=None,lr_scheduler_monitor_metric=None)
    
    trainer_config=TrainerConfig(batch_size=int(param['batch_size']),max_epochs=30,accelerator=device,early_stopping='valid_loss',early_stopping_patience=5)
        
    trainratio = np.bincount(ytrain.values)
    classcount = trainratio.tolist()
    train_weights = 1./torch.tensor(classcount, dtype=torch.float)
    train_sampleweights = train_weights[ytrain.values]

    training_sampler=custom_sampler.PersonalizedWeightedRandomSampler(labels=ytrain.values,weights=train_sampleweights,num_samples=len(ytrain),batch_size=param['batch_size'],replacement=True) 
    
    model=TabularModel(model_config=tabtransformer_config,optimizer_config=optimizer_config,data_config=data_config,trainer_config=trainer_config)
    train=xtrain.join(ytrain)
    val=xval.join(yval)
    start=time.time()
    model.fit(train=train,validation=val,train_sampler=training_sampler)
    end=time.time()
    print('5 epochs running time:',end-start)
    return model


    