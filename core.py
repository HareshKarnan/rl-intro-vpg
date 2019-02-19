import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import gym
import scipy.signal
import time
from floatingup.utils.logx import EpochLogger
import floatingup.algos.vpg.core as core
from floatingup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from floatingup.utils.mpi_torch import average_gradients,sync_all_params
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size=(64,64,64),activation=torch.tanh,output_activation=None,output_squeeze=False):
        super(NeuralNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layers = nn.ModuleList()

        layers = [obs_dim]+list(hidden_size)+[act_dim]

        for i,layer in enumerate(layers[1:]): # except the first input layer
            self.layers.append(nn.Linear(layers[i],layer))

        # self.layers.append(nn.Linear(obs_dim,hidden_size[0])) # input layer
        # for i in range(1,len(list(hidden_size))): # hidden layers
        #     self.layers.append(nn.Linear(hidden_size[i-1],hidden_size[i]))
        # self.layers.append(nn.Linear(hidden_size[-1],act_dim)) # output layer

        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze

        # if (torch.cuda.is_available):
        #     self.device = torch.device('cuda')
        # else:
        #     self.device = torch.device('cpu')


    def forward(self, inp):
        x = inp
        print(x)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            print(x)
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))

        return x.squeeze() if self.output_squeeze else x

class GaussianPolicy(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size=(64,64,64),activation=nn.Softmax,output_activation=None):
        super(GaussianPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.output_activation = output_activation
        self.mu = NeuralNetwork(obs_dim=obs_dim,
                                act_dim=act_dim,
                                hidden_size=hidden_size,
                                activation=activation,
                                output_activation=output_activation)
        self.sigma = nn.Parameter(-0.5*torch.ones(act_dim,dtype=torch.float32))

    def forward(self, x,a=None): # if a is present, then it is training, else the network is in inference mode
        mu = self.mu(x)
        policy = Normal(mu,self.sigma.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi)
        if a is not None:
            logp = policy.log_prob(a)
        else :
            logp = None

        return pi,logp,logp_pi

class ActorCritic(nn.Module):
    def __init__(self,obs_dim,act_dim,activation=None,output_activation=None):
        super(ActorCritic, self).__init__()
