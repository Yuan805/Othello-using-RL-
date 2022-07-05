import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
import gym                                      # 导入gym
import random

BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
N_ACTIONS = 64                                  # 动作和状态的数量
N_STATES = 64      


def transfer(x):
    x=torch.FloatTensor(x)
    x=x.view(64)
    return x

class Net(nn.Module):
    def __init__(self):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()
        self.fc1 = nn.Linear(N_STATES, 640)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到640个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(640, N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 640个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value           

class DQN():
    def __init__(self):
        self.memory_capacity=2000

        self.eval_net, self.target_net = Net(), Net()                          
        self.learn_step_counter = 0                                             
        self.memory_counter = 0                                                 
        self.memory = np.zeros((self.memory_capacity, N_STATES * 2 + 2))
                    
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    
        self.loss_func = nn.MSELoss()                                          
        self.loss=0

    def reset(self, env):
        self.index_all=[]      ##储存更新的memory_batch的index                   

        if hasattr(env, 'env'):                 ##判断目标是否具有这个属性
            self.env = env.env
        else:
            self.env = env

    def get_action(self, x):                                                
        x=torch.FloatTensor(x)
        x=x.view(64)
        x = torch.unsqueeze(x, 0)     

        index=-1                                               ##加了个补丁：action从possible_moves里面选
        value=-np.inf
        if np.random.uniform() < EPSILON:                                      
            actions_value = self.eval_net.forward(x)
            for i in self.env.possible_moves:
                if actions_value[0,i]>=value:
                    value=actions_value[0,i]
                    index=i
            action=index                                              
        else:                                                                   
            action=random.choice(self.env.possible_moves)
        return action                                                         

    def store_transition(self, s, a, r, s_):                                    
        s=transfer(s)
        s_=transfer(s_)
        transition = np.hstack((s, [a, r], s_))                               
        index = self.memory_counter % self.memory_capacity
        
        self.index_all.append(index)                        
        self.memory[index, :] = transition                                      
        self.memory_counter += 1                                               

    def update_memory(self,reward):   ##奖赏 改成赢的一局动作记为1
        for i in self.index_all:
            self.memory[i,N_STATES+1:N_STATES+2]=reward

    def learn(self):                                                           
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  
            self.target_net.load_state_dict(self.eval_net.state_dict())         
        self.learn_step_counter += 1                                            

        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)    ##挑选sample        
        b_memory = self.memory[sample_index, :]                                 

        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
       

        q_eval = self.eval_net(b_s).gather(1, b_a)      ##gather函数从b_s的64个状态中取出一个来
       
        q_next = self.target_net(b_s_).detach()
       
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        
        loss = self.loss_func(q_eval, q_target)

        self.loss=loss
        
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数       
