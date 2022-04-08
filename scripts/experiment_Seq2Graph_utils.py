import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

from copy import deepcopy
import datetime as dt

from model_wrapper import Seq2GraphWrapper
import data_generator as data_generator

import pickle
from tqdm import tqdm_notebook as tqdm

import pandas as pd

# test for GPU
# setting device on GPU if available, else CPU
# 0 = GeForce GTX 980
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

from TEST_config import config
from TEST_seq2graph_simple import Seq2Graph,Seq2Graph_test

def causeme_get_table_data_from_zip(zipfile_list):
    prepared_path=[]
    for zipfile_path in zipfile_list:
        archive = zipfile.ZipFile('./data/' + zipfile_path + '.zip', 'r')

        imgdata = archive.read(archive.filelist[0])

        ts=[row.split(' ') for row in str(imgdata).split('\\n')]
        master=[]
        for inner_list in ts[:-1]:
            inner=[]
            for element in inner_list:
                if 'b' in element:
                    inner.append(float(element[2:-1]))
                else:
                    inner.append(float(element))
            master.append(inner)

        master_X,master_y=data_generator.model_data_from_time_series(torch.tensor(master).T,DEVICE,6)

        for imgdata_file in archive.filelist[1:]:
            imgdata = archive.read(imgdata_file)
            ts=[row.split(' ') for row in str(imgdata).split('\\n')]
            master=[]
            for inner_list in ts[:-1]:
                inner=[]
                for element in inner_list:
                    if 'b' in element:
                        inner.append(float(element[2:-1]))
                    else:
                        inner.append(float(element))
                master.append(inner)
            X,y=data_generator.model_data_from_time_series(torch.tensor(master).T,DEVICE,6)
            master_X=torch.cat((master_X,X),2)
            master_y=torch.cat((master_y,y),0)

        path_output='./data/'+zipfile_path+'prepared.pickle'

        pickle.dump((master_X,master_y),open(path_output ,'wb'))  

        prepared_path.append(path_output)
    
    return prepared_path



def run_Seq2Graph_single_experiment(load_string,config,i,loss,folder_name,save_att_coefs=True):
    
    context_dict,timer_series=pickle.load(open(load_string , "rb" ) ) 
    config.num_of_time_series=timer_series.shape[0]
    X_train,y_train=data_generator.model_data_from_time_series(timer_series[:,1000:18000],DEVICE,config.len_of_time_series)
    X_test,y_test=data_generator.model_data_from_time_series(timer_series[:,18000:],DEVICE,config.len_of_time_series)
    NN_arh=Seq2Graph(config,DEVICE).to(DEVICE)
    model=Seq2GraphWrapper(NN_arh,config,DEVICE)
    model.train(X_train,y_train,nn.MSELoss,torch.optim.Adam,False)
    
    loss_train=loss(model.forward(X_train),y_train)
    if save_att_coefs:
        save_string="".join(load_string.split('/')[-1].split('.')[:-1]) + str(dt.datetime.today()).replace(":","_").replace(" ",'_').replace("-","_") + '_'+ str(i) +'_alpha_beta.p'
        pickle.dump((model.alpha,model.beta),open(f'./{folder_name}/train/' +save_string,'wb'))

    loss_test=loss(model.forward(X_test),y_test)
    if save_att_coefs:
        save_string="".join(load_string.split('/')[-1].split('.')[:-1]) + str(dt.datetime.today()).replace(":","_").replace(" ",'_').replace("-","_") + '_'+ str(i) +'_alpha_beta.p'
        pickle.dump((model.alpha,model.beta),open(f'./{folder_name}/test/' +save_string,'wb'))

    return loss_train.cpu().detach().numpy(),loss_test.cpu().detach().numpy()

def run_Seq2Graph_experiments(load_strings_list,folder_name,pickle_path=None,repeats=5):

    master_train_loss_mean=dict()
    master_train_loss_std=dict()
    master_test_loss_mean=dict()
    master_test_loss_std=dict()

    loss=torch.nn.MSELoss()

    for load_string in tqdm(load_strings_list):

        experiment_train_loss=[]
        experiment_test_loss=[]

        for i in range(repeats):
            test_loss=[]
            train_loss=[]

            train_loss,test_loss=run_Seq2Graph_single_experiment(load_string,config,i,loss,folder_name)

            experiment_train_loss.append(train_loss)

            experiment_test_loss.append(test_loss)

        experiment_train_loss=np.array(experiment_train_loss)
        experiment_train_loss_mean=experiment_train_loss.mean()
        experiment_train_loss_std=experiment_train_loss.std()

        experiment_test_loss=np.array(experiment_test_loss)
        experiment_test_loss_mean=experiment_test_loss.mean()
        experiment_test_loss_std=experiment_test_loss.std()

        master_train_loss_mean[load_string]=experiment_train_loss_mean
        master_train_loss_std[load_string]=experiment_train_loss_std

        master_test_loss_mean[load_string]=experiment_test_loss_mean
        master_test_loss_std[load_string]=experiment_test_loss_std

        if pickle_path is not None:
            pickle.dump( ( master_train_loss_mean,master_train_loss_std, master_test_loss_mean, master_test_loss_std), open( pickle_path, "wb" ) ) 
    
    df_master_train_loss_mean=pd.DataFrame(master_train_loss_mean.items(),columns=['time_series','train_loss_mean'])
    df_master_train_loss_std=pd.DataFrame(master_train_loss_std.items(),columns=['time_series','train_loss_std'])
    df_master_test_loss_mean=pd.DataFrame(master_test_loss_mean.items(),columns=['time_series','test_loss_mean'])
    df_master_test_loss_std=pd.DataFrame(master_test_loss_std.items(),columns=['time_series','test_loss_std'])

    results=df_master_train_loss_mean.merge(df_master_train_loss_std)
    results=results.merge(df_master_test_loss_mean)
    results=results.merge(df_master_test_loss_std)
    
    results['time_series']=results.time_series.apply(lambda x: (x.split('/')[-1]))
    
    return results


def causeme_run_Seq2Graph_single_experiment(load_string,config,loss,save_att_coefs=True):
    master_X,master_y=pickle.load(open(load_string , "rb" ) ) 
    config.num_of_time_series=master_X.shape[0]
    config.len_of_time_series=master_X.shape[1]
    X_train,y_train=master_X[:,:,:-10000,:],master_y[:-10000,:]
    X_test,y_test=master_X[:,:,-10000:,:],master_y[-10000:,:]
    NN_arh=Seq2Graph(config,DEVICE).to(DEVICE)
    model=Seq2GraphWrapper(NN_arh,config,DEVICE)
    model.train(X_train,y_train,nn.MSELoss,torch.optim.Adam,False)
    
    loss_train=loss(model.forward(X_train),y_train)
    if save_att_coefs:
        save_string="".join(load_string.split('/')[-1].split('.')[:-1]) + str(dt.datetime.today()).replace(":","_").replace(" ",'_').replace("-","_") + '_'+ str(i) +'_alpha_beta.p'
        pickle.dump((model.alpha,model.beta),open('./seq2graph_alpha_beta/train/' +save_string,'wb'))
    
    loss_test=loss(model.forward(X_test),y_test)
    if save_att_coefs:
        save_string="".join(load_string.split('/')[-1].split('.')[:-1]) + str(dt.datetime.today()).replace(":","_").replace(" ",'_').replace("-","_") + '_'+ str(i) +'_alpha_beta.p'
        pickle.dump((model.alpha,model.beta),open('./seq2graph_alpha_beta/test/' +save_string,'wb'))
    
    return loss_train.cpu().detach().numpy(),loss_test.cpu().detach().numpy()

def causeme_run_Seq2Graph_experiments(load_strings_list,pickle_path=None,repeats=5):

    master_train_loss_mean=dict()
    master_train_loss_std=dict()
    master_test_loss_mean=dict()
    master_test_loss_std=dict()
    
    loss=torch.nn.MSELoss()


    for load_string in tqdm(prepared_path):

        experiment_train_loss=[]
        experiment_test_loss=[]

        for i in range(repeats):
            test_loss=[]
            train_loss=[]

            train_loss,test_loss=causeme_run_Seq2Graph_single_experiment(load_string,config,loss)

            experiment_train_loss.append(train_loss)

            experiment_test_loss.append(test_loss)

        experiment_train_loss=np.array(experiment_train_loss)
        experiment_train_loss_mean=experiment_train_loss.mean()
        experiment_train_loss_std=experiment_train_loss.std()

        experiment_test_loss=np.array(experiment_test_loss)
        experiment_test_loss_mean=experiment_test_loss.mean()
        experiment_test_loss_std=experiment_test_loss.std()

        master_train_loss_mean[load_string]=experiment_train_loss_mean
        master_train_loss_std[load_string]=experiment_train_loss_std

        master_test_loss_mean[load_string]=experiment_test_loss_mean
        master_test_loss_std[load_string]=experiment_test_loss_std

    if pickle_path is not None:
        pickle.dump( ( master_train_loss_mean,master_train_loss_std, master_test_loss_mean, master_test_loss_std), open( pickle_path, "wb" ) ) 
    
    df_master_train_loss_mean=pd.DataFrame(master_train_loss_mean.items(),columns=['time_series','train_loss_mean'])
    df_master_train_loss_std=pd.DataFrame(master_train_loss_std.items(),columns=['time_series','train_loss_std'])
    df_master_test_loss_mean=pd.DataFrame(master_test_loss_mean.items(),columns=['time_series','test_loss_mean'])
    df_master_test_loss_std=pd.DataFrame(master_test_loss_std.items(),columns=['time_series','test_loss_std'])

    results=df_master_train_loss_mean.merge(df_master_train_loss_std)
    results=results.merge(df_master_test_loss_mean)
    results=results.merge(df_master_test_loss_std)
    
    results['time_series']=results.time_series.apply(lambda x: (x.split('/')[-1]))
    
    return results