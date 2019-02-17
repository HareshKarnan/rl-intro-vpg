import torch
import torch.nn as nn
import torch.distributions.normal as Normal
import torch.optim.adam as Adam
import time


class NeuralNetwork(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size=(64,64,64),activation=nn.ReLU,output_activation=None,output_squeeze=False):
        super(NeuralNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim,hidden_size[0])) # input layer
        for i in range(1,len(list(hidden_size))): # hidden layers
            self.layers.append(nn.Linear(hidden_size[i-1],hidden_size[i]))
        self.layers.append(nn.Linear(hidden_size[-1],act_dim)) # output layer

        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = activation(layer(x))

        # final output
        if self.output_activation is not None :
            x = self.output_activation(x)
        else:
            x = layers[-1](x)

        return x.squeeze() if self.output_squeeze else x

class GaussianPolicy(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size=(64,64,64),activation=nn.ReLU,output_activation=None):
        super(GaussianPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.output_activation = output_activation
        self.mu = NeuralNetwork(obs_dim,act_dim,hidden_size,activation,output_activation)
        self.sigma = nn.Parameter(-0.5*torch.ones(act_dim,dtype=torch.float64))

    def forward(self, x,a): # if a is present, then it is training, else the network is in inference mode
        mu = self.mu(x)
        sigma = self.sigma.exp()
        policy = Normal(mu,sigma)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi)
        if a is not None:
            logp = policy.log_prob(a)
        else :
            logp = None

        return pi,logp,logp_pi


def VPG(self,env,pg_lr=1e-3,vf_lr=1e-3,epochs=100,steps_per_epoch=4000):
    env = env()
    self.obs_dim = env.observation_space.shape[0]
    self.act_dim = env.action_space.shape[0]
    policy = GaussianPolicy(self.obs_dim,self.act_dim)
    valuefunction = NeuralNetwork(self.obs_dim,self.act_dim) # no output squeeze
    # define optimizer

    train_pi = Adam(policy.parameters())
    train_v = Adam(valuefunction.parameters())

    start_time = time.time()
    obs,rew,done,ep_ret,ep_len = env.reset(),0,False,0,0
    batch_obs,batch_acts,batch_rtgs,batch_rets,batch_lens = [],[],[],[],[] # buffer to store everything

    for epoch in range(epochs):
        policy.eval() # freeze policy parameters
        for t in range(steps_per_epoch):
            # perform inference from the networks for action to take and value function
            a,_,logp_t = policy(obs) # policy
            v_t = valuefunction(obs) # value function

            obs,rew,done = env.step(a.data)
            batch_acts.append(a.data)
            batch_obs.append(obs)
            batch_rets.append(rew)
















