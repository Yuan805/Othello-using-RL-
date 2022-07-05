import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np

#设置参数
n_states=64
n_actions=64
epsilon = 0.9                                   # greedy policy
gamma = 0.9                                     # reward discount
lr=0.01
memory_capacity=2000
target_replace_iter=100

class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.action_layer = nn.Sequential(
            nn.Linear(self.n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )

    def forward(self, x):
        action = F.softmax(self.action_layer(x))
        return action

class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.value_layer = nn.Sequential(
            nn.Linear(self.n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        value = self.value_layer(x)
        return value

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.target_replace_iter=target_replace_iter

        #设置网络
        self.actor = Actor()
        self.critic= Critic()
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic= torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        #设置Buffer Memory
        self.memory_capacity=memory_capacity                                            
        self.memory_counter = 0                                                 
        self.memory = np.zeros((self.memory_capacity, self.n_states * 2 + 2))

    def reset(self, env):
        self.index_all=[]     #清空index_all                  
        
        #判断PPO是否具有env的子属性
        if hasattr(env, 'env'):                
            self.env = env.env
        else:
            self.env = env
    def get_action(self, s):
        s=torch.FloatTensor(s).view(64)      # 维度变化8*8 变为 64 变为 1*64
        s = torch.unsqueeze(s, 0)

        probs = self.actor(s)

        
        action = torch.multinomial(probs, 1).item()        # 根据概率采样

        if action not in self.env.possible_moves:
            action=random.choice(self.env.possible_moves)

        return action, probs.squeeze(0)
    def store_transition(self, s, a, r, s_):                                    
        s=torch.FloatTensor(s).view(64)
        s_=torch.FloatTensor(s_).view(64)
        
        #存入memory_buffer
        transition = np.hstack((s, [a, r], s_))                               
        index = self.memory_counter % self.memory_capacity
        self.index_all.append(index)                        
        self.memory[index, :] = transition                                      
        self.memory_counter += 1    

    def update_memory(self,reward):
        #因为Othello奖励稀疏，重新设置reward
        for i in self.index_all:
            self.memory[i,self.n_states+1:self.n_states+2]=reward
    
    def learn(self):                                                           
        if self.learn_step_counter % self.target_replace_iter == 0:                  
            self.target_net.load_state_dict(self.eval_net.state_dict())         
        self.learn_step_counter += 1
        
        s=torch.FloatTensor(s).view(64)
        s_=torch.FloatTensor(s_).view(64)
        
        s = torch.unsqueeze(s, 0)
        s_ = torch.unsqueeze(s_, 0)
        reward = torch.FloatTensor([reward])

        _, value = self.model(s)                
        _, value_ = self.model(s_)              
        value_ = value_.detach()       #只更新Qt的（Cirtic网络)
        value, value_ = value.squeeze(0), value_.squeeze(0)

        target = 0
        target += reward
        target += self.gamma * value_

        loss_func = nn.MSELoss()
        loss = loss_func(value, target)
        self.critic_loss=loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        advantage = (target - value).detach()


        state_arr, action_arr, old_prob_arr, vals_arr,\
        reward_arr, dones_arr, batches = \
                self.memory.sample()
        values = vals_arr
        ### compute advantage ###
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                        (1-int(dones_arr[k])) - values[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        advantage = torch.tensor(advantage).to(self.device)
        ### SGD ###
        values = torch.tensor(values).to(self.device)
        for batch in batches:
            states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
            old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
            actions = torch.tensor(action_arr[batch]).to(self.device)
            dist = self.actor(states)
            critic_value = self.critic(states)
            critic_value = torch.squeeze(critic_value)
            new_probs = dist.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                    1+self.policy_clip)*advantage[batch]
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage[batch] + values[batch]
            critic_loss = (returns-critic_value)**2
            critic_loss = critic_loss.mean()
            total_loss = actor_loss + 0.5*critic_loss
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    self.memory.clear()