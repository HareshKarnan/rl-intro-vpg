import torch
import torch.nn as nn
import torch.distributions.normal as Normal
import time,scipy
import torch.functional as F
import logging
import core

class VPG:
    def __init__(self,env,pg_lr=1e-3,vf_lr=1e-3,gamma=0.99,lam=0.95):
        super(VPG, self).__init__()
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.policy = core.GaussianPolicy(self.obs_dim,
                                     self.act_dim)
        self.valuefunction = core.NeuralNetwork(self.obs_dim,
                                           self.act_dim,
                                           output_squeeze=True) # no output squeeze
        # define optimizer
        self.train_pi = torch.optim.Adam(self.policy.parameters(),lr=pg_lr)
        self.train_v = torch.optim.Adam(self.valuefunction.parameters(),lr=vf_lr)
        self.gamma,self.lam = gamma,lam

        self.start_time = time.time()

    def train(self,epochs=10,steps_per_epoch=4000):
        batch_obs, batch_acts, batch_rew, batch_val, batch_lens, batch_adv,batch_logp = [],[],[],[],[],[],[] # buffer to store everything
        obs, rew, done, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
        self.policy.eval()  # freeze policy parameters
        self.valuefunction.eval()  # freeze valuefunction parameters

        for t in range(epochs):
            # collect experience
            for t in range(steps_per_epoch):
                obs_tensor = torch.Tensor(obs.reshape(1,-1))
                # print('observation tensor : ',obs_tensor.dtype)
                a, _, logp_t = self.policy(obs_tensor)  # policy evaluation
                v_t = self.valuefunction(obs_tensor)  # value function evaluation

                obs, rew, done = self.env.step(a.data.numpy())  # act in the environment, then store the reward

                batch_acts.append(a.data)  # actions
                batch_obs.append(obs)  # observations
                batch_rew.append(rew)  # rewards
                batch_val.append(v_t) # value function
                batch_logp.append(logp_t) # logprobabilities of policy action

                ep_ret+=rew
                ep_len+=1
                if done : # reached terminal state
                    # implement GAE lambda advantage estimation
                    deltas = batch_rew + self.gamma * batch_val - batch_val
                    batch_adv = self._discount_cumsum(deltas, discount=self.gamma * self.lam)

                    # rewards to go - targets for value function
                    batch_rew = self._discount_cumsum(batch_rew,discount=self.gamma)
                    print('Episode Return : %d Episode Length : %d'%(ep_ret,ep_len))
                    obs, rew, done, ep_ret,ep_len = self.env.reset(), 0, False, 0,0 # reset everything for new episode

            self.policy.train() #unfreeze policy parameters
            self.valuefunction.train() # unfreeze value function parameters

            # use experience to train the networks
            o = torch.tensor(batch_obs,dtype=torch.float64)
            a = torch.tensor(batch_acts,dtype=torch.float64)
            adv = torch.tensor(batch_adv,dtype=torch.float64)
            r = torch.tensor(batch_rew,dtype=torch.float64)
            logp_old = torch.tensor(batch_logp,dtype=torch.float64)
            v_t = torch.tensor(batch_val,dtype=torch.float64)

            # policy gradient step
            _,logp,_ = self.policy(o,a)
            ent = (-logp).mean() # sample estimate from entropy

            #VPG loss function
            loss_vpg_pi = -(logp*adv).mean()

            # policy gradient step
            self.train_pi.zero_grad() # set gradient buffers to zero
            loss_vpg_pi.backward()
            self.train_pi.step()

            # value function learning
            self.train_v.zero_grad() # set vf gradient buffers to zero
            loss_vpg_v = F.mse_loss(v_t,r)
            self.loss_vpg_v.backward()
            self.train_v.step()

    def _discount_cumsum(self,x,discount):
        return scipy.signal.lfilter([1],[1,float(-discount)],x[::-1],axis=0)[::-1]




















