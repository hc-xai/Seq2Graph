#Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datetime as dt

import torch
from torch import nn
import torch.nn.functional as F


def plot_grad_flow_min(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])



class Seq2GraphWrapper:
    def __init__(self, NeuralNet,config,DEVICE):
        self.NeuralNet=NeuralNet
        self.config=config
        self.DEVICE=DEVICE
        
        self.alpha=None
        self.beta=None
        
        
        
    def train(self,X,y,criterion,optimizer,plot_grads=None):
        if plot_grads==None:
            plot_grads=self.config.plot_grads
        self.criterion = criterion()
        self.optimizer = optimizer(self.NeuralNet.parameters(), lr=self.config.train_params.lr)
        
        if X.device!=self.DEVICE:
            X=X.to(self.DEVICE)
        if y.device!=self.DEVICE:
            y=y.to(self.DEVICE)
        
        for epoch in range(1, self.config.train_params.n_epochs + 1):
            if self.config.train_params.permute:
                permutation = torch.randperm(X.size()[2])
            else: 
                permutation=torch.tensor(range(0,X.size()[2]),device=self.DEVICE)

            for i in range(0,X.size()[2], self.config.train_params.batch_size):
                self.optimizer.zero_grad() # Clears existing gradients from previous epoch
                indices = permutation[i:i+self.config.train_params.batch_size]
                batch_x, batch_y = X[:,:,indices], y[indices]
                outputs = self.NeuralNet.forward(batch_x)
                self.loss = self.criterion(outputs, batch_y)
                self.loss.backward() # Does backpropagation and calculates gradients
                if plot_grads:
                    plot_grad_flow(self.NeuralNet.named_parameters())
                self.optimizer.step() # Updates the weights accordingly


            if epoch%self.config.train_params.print_every == 0 and self.config.train_params.verbose>=1:
                print('Epoch: {}/{}.............'.format(epoch, self.config.train_params.n_epochs), end=' ')
                print("Loss: {:.4f}".format(self.loss.item()))
        
        self.alpha=self.NeuralNet.alpha
        self.beta=self.NeuralNet.beta
                
    def evaluate(self,X,y,criterion):
        self.evaluate_criterion=criterion()
        output=self.NeuralNet(X)
        self.evaluate_loss=self.evaluate_criterion(output,y)
        print("Evaluate loss: %s"%str(self.evaluate_loss))
        hidden_for_plot=output[:,0].cpu().detach().numpy()
        y_for_plot=y[:,0].cpu()
        plt.plot(hidden_for_plot[0:50])
        plt.plot(y_for_plot[0:50])
        plt.savefig('./output/model_output_%s.png'%str(dt.date.today()).replace('-','_'))
        
        self.alpha=self.NeuralNet.alpha
        self.beta=self.NeuralNet.beta
        
        return self.evaluate_loss
    
    def optimize(self):
        pass
    
    def forward(self,X):
        
        output=self.NeuralNet(X)
        
        self.alpha=self.NeuralNet.alpha
        self.beta=self.NeuralNet.beta
        
        return output