
from stable_baselines3 import TD3
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

#不要改
n_states=64
n_actions=64

#设置参数
epsilon = 0.9                                   # greedy policy
gamma = 0.9                                     # reward discoun
lr=0.01


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.action_layer = nn.Sequential(
            nn.Linear(self.n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(self.n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action = F.softmax(self.action_layer(x))
        value = self.value_layer(x)
        return action, value

class AC(nn.Module):
    def __init__(self):
        super(AC, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr

        self.model = Net(self.n_states, self.n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def reset(self, env):
        if hasattr(env, 'env'):               
            self.env = env.env
        else:
            self.env = env

    def get_action(self, s):
        s=torch.FloatTensor(s)
        s = s.view(64)               # 维度变化8*8 变为 64 变为 1*64
        s = torch.unsqueeze(s, 0)
        probs, _ = self.model(s)
        print(probs)
        action = torch.multinomial(probs, 1).item()        # 根据概率采样

        if action not in self.env.possible_moves:
            action=random.choice(self.env.possible_moves)

        return action, probs.squeeze(0)

    def critic_learn(self, s, s_, reward, done):

        s=torch.FloatTensor(s)
        s=s.view(64)
        s_=torch.FloatTensor(s_)
        s_=s_.view(64)
        s = torch.unsqueeze(s, 0)
        s_ = torch.unsqueeze(s_, 0)
        reward = torch.FloatTensor([reward])

        _, value = self.model(s)                # [1, 1]
        _, value_ = self.model(s_)              # [1, 1]
        value_ = value_.detach()       #只更新Qt的（Cirtic网络)
        value, value_ = value.squeeze(0), value_.squeeze(0)

        target = 0
        target += reward
        if not done:
            target += self.gamma * value_

        loss_func = nn.MSELoss()
        loss = loss_func(value, target)
        self.critic_loss=loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        TDerror = (target - value).detach()
        return TDerror
    
    def actor_learn(self, TDerror, s, a):
        _, probs = self.get_action(s)
        log_prob = probs.log()[a]  #取ln 

        loss = -TDerror * log_prob #张量点乘
        self.actor_loss=loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()