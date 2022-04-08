import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import datetime as dt

def seq2graph_simple_2series(timeseries_len,num_of_examples,switch_threshold,DEVICE=None):
    
    X_linear=torch.empty((2,timeseries_len,num_of_examples,1)).normal_(mean=0,std=1)
    y_linear=torch.zeros(num_of_examples,2)
    for i in range(0,X_linear.shape[2]):
        if X_linear[0,timeseries_len-1,i,0]>switch_threshold:
            y_linear[i,0]=torch.tanh((20*X_linear[0,0,i,:] + 2*X_linear[0,1,i,:] + 3*X_linear[0,2,i,:] +4*X_linear[0,3,i,:] +5*X_linear[0,4,i,:]+ 6*X_linear[0,5,i,:])/40)
            y_linear[i,1]=torch.tanh((1*X_linear[0,0,i,:] + 1*X_linear[0,1,i,:] + 1*X_linear[0,2,i,:] +4*X_linear[0,3,i,:] +5*X_linear[0,4,i,:]+ 20*X_linear[0,5,i,:])/40)
        else:
            y_linear[i,0]=torch.tanh((20*X_linear[1,0,i,:] + 2*X_linear[1,1,i,:] + 3*X_linear[1,2,i,:] +4*X_linear[1,3,i,:] +5*X_linear[1,4,i,:]+ 6*X_linear[1,5,i,:])/40)
            y_linear[i,1]=torch.tanh((1*X_linear[1,0,i,:] + 1*X_linear[1,1,i,:] + 1*X_linear[1,2,i,:] +4*X_linear[1,3,i,:] +5*X_linear[1,4,i,:]+ 20*X_linear[1,5,i,:])/40)
            
    if DEVICE!=None:
        X=X_linear.to(DEVICE)
        y=y_linear.to(DEVICE)
    
    return X,y

def towards_data_science_time_series(len_of_timeseries=10000,add_gaussian_noise=True,mu=0, sigma=1):
    
    time_series=torch.zeros((5,len_of_timeseries))
    
    for t in range (1,len_of_timeseries):
        time_series[0,t]=0.95*np.sqrt(2)*time_series[0,t-1] - 0.9025*time_series[0,t-1] + add_gaussian_noise*np.random.normal(mu,sigma)
        time_series[1,t]=0.5*time_series[0,t-1] + add_gaussian_noise*np.random.normal(mu,sigma)
        time_series[2,t]=0.4*time_series[0,t-1] + add_gaussian_noise*np.random.normal(mu,sigma)
        time_series[3,t]=0.5*time_series[0,t-1] + 0.25*np.sqrt(2)*time_series[3,t-1] + 0.25*np.sqrt(2)*time_series[4,t-1] + add_gaussian_noise*np.random.normal(mu,sigma)
        time_series[4,t]=0.25*np.sqrt(2)*time_series[3,t-1] + 0.25*np.sqrt(2)*time_series[4,t-1] + add_gaussian_noise*np.random.normal(mu,sigma)
        
    return time_series

def model_data_from_time_series(time_series,DEVICE,window_len=6):
    
    num_of_timeseries,len_of_timeseries=time_series.shape
    num_of_examples=len_of_timeseries-window_len
    
    X=torch.empty((num_of_timeseries,window_len,num_of_examples,1)).normal_(mean=0,std=1)
    y=torch.zeros(num_of_examples,num_of_timeseries)
    
    for i in range(0,len_of_timeseries-window_len):
        X[:,:,i,0]=time_series[:,i:(i+window_len)]
        y[i,:]=time_series[:,i+window_len]
        
    
    X=X.to(DEVICE)
    y=y.to(DEVICE)
    
    return X,y