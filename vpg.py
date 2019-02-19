import torch
import numpy as np
import torch.nn as nn
import torch.distributions.normal as Normal
import time
import scipy.signal as scipy_signal
import torch.nn.functional as F
import logging
from table_logger import TableLogger
import core
class VPG:
    def __init__(self,env,pg_lr=1e-3,vf_lr=1e-3,gamma=0.99,lam=0.95):
        super(VPG, self).__init__()
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.actor_critic = core.ActorCritic(obs_dim=self.obs_dim,act_dim=self.act_dim)

        # define optimizer
        self.train_pi = torch.optim.Adam(self.actor_critic.policy.parameters(),lr=pg_lr)
        self.train_v = torch.optim.Adam(self.actor_critic.valuefunction.parameters(),lr=vf_lr)
        self.gamma = gamma
        self.lam = lam

        self.start_time = time.time()

    def train(self,epochs=10,steps_per_epoch=40):

        self.actor_critic.eval()  # freeze policy parameters

        for epoch in range(epochs):
            batch_obs, batch_acts, batch_rew, batch_val, batch_lens, batch_adv, batch_logp = [], [], [], [], [], [], []  # buffer to store everything
            obs, rew, done, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0

            # collect experience
            for step in range(steps_per_epoch):
                obs_tensor = torch.Tensor(obs.reshape(1,-1))
                # print('observation tensor : ',obs_tensor.dtype)
                a, _, logp_t,v_t = self.actor_critic(obs_tensor)  # policy evaluation

                obs, rew, done, _ = self.env.step(a.data.numpy())  # act in the environment, then store the reward
                batch_acts.append(a.data)  # actions
                batch_obs.append(obs)  # observations
                batch_rew.append(rew)  # rewards
                batch_val.append(v_t.item()) # value function
                batch_logp.append(logp_t.item()) # logprobabilities of policy action

                ep_ret+=rew
                ep_len+=1

                if done or step==steps_per_epoch-1 : # reached terminal state
                    # implement GAE lambda advantage estimation
                    batch_val = np.asarray(batch_val) # convert to numpy to be able to multiply
                    batch_rew = np.asarray(batch_rew)
                    deltas = batch_rew + self.gamma*batch_val - batch_val
                    batch_adv = self._discount_cumsum(deltas, discount=self.gamma * self.lam)
                    # rewards to go - targets for value function
                    batch_rew = self._discount_cumsum(batch_rew,discount=self.gamma)

                    # print('Episode Return : %d Episode Length : %d'%(ep_ret,ep_len))
                    obs, rew, done = self.env.reset(), 0, False # reset everything for new episode
                    # convert numpy array to list
                    batch_rew = batch_rew.tolist()
                    batch_val = batch_val.tolist()
                    batch_adv = batch_adv.tolist()
                    break


            print('Reward in epoch %d : %d'%(epoch,ep_ret),'Lasted for : %d'%(ep_len))

            self.actor_critic.train() #unfreeze policy parameters

            # use experience to train the networks
            o = torch.Tensor(batch_obs)
            a = torch.Tensor(batch_acts)
            adv = torch.Tensor(batch_adv)
            r = torch.Tensor(batch_rew)
            logp_old = torch.Tensor(batch_logp)
            v_t = torch.Tensor(batch_val)

            # policy gradient step
            _,logp,_ = self.actor_critic.policy(o,a)
            ent = (-logp).mean() # sample estimate from entropy

            #VPG loss function
            loss_vpg_pi = -(logp*adv).mean()

            # policy gradient step
            self.train_pi.zero_grad() # set gradient buffers to zero
            loss_vpg_pi.backward()
            self.train_pi.step()

            # value function learning
            self.train_v.zero_grad() # set vf gradient buffers to zero
            v = self.actor_critic.valuefunction(o)
            loss_vpg_v = F.mse_loss(v,r)
            loss_vpg_v.backward()
            self.train_v.step()

    def _discount_cumsum(self,x,discount):
        return scipy_signal.lfilter([1],[1,float(-discount)],x[::-1],axis=0)[::-1]




















