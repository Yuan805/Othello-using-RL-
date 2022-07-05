import othello
import simple_policies
import DQN_policy
import numpy as np
from torch.utils.tensorboard import SummaryWriter   



##设置策略
def choose_policy(policy):
    if  policy=='DQN':
        dqn=DQN_policy.DQN()
        policy_chosen=dqn
        name='DQN'
    elif policy=='PPO':
        pass
        name='PPO'
    elif policy=='Greedy':
        policy_chosen=simple_policies.GreedyPolicy()
        name='Greedy'
    return policy_chosen, name

protagonist_policy=DQN_policy.DQN()
protagonist_policy_name='DQN'
opponent_policy=DQN_policy.DQN()
opponent_policy_name='DQN'

##设置先后手
protagonist=-1   
if protagonist==-1:  ##protagnoist=1,白棋采用protagnist的策略；protagnoist=-1,黑棋采用protagnist的策略；
    black_policy=protagonist_policy
    white_policy=opponent_policy
else:
    black_policy=opponent_policy
    white_policy=protagonist_policy

##设置棋盘
board_size=8
rand_seed=0
env_init_rand_steps=1
num_disk_as_reward=False
render=False
env = othello.OthelloEnv(white_policy=white_policy,
                             black_policy=black_policy,
                             protagonist=protagonist,
                             board_size=board_size,
                             seed=rand_seed,
                             initial_rand_steps=env_init_rand_steps,
                             num_disk_as_reward=num_disk_as_reward,
                             render_in_step=render)

##记录
writer=SummaryWriter('')

##训练：
train_num_rounds=1000
win_cnts = draw_cnts = lose_cnts = 0

for round in range(train_num_rounds):
    if round %100==0:
        print('Episode {}'.format(round + 1))   
 
    obs = env.reset()
    protagonist_policy.reset(env)
    opponent_policy.reset(env)
    done = False

    if protagonist_policy_name=='DQN':
        MEMORY_CAPACITY = 2000
        pro_memory_all=np.zeros((MEMORY_CAPACITY, 64 * 2 + 2))        ##最后录入样本
    if opponent_policy_name=='DQN':
        MEMORY_CAPACITY = 2000
        opp_memory_all=np.zeros((MEMORY_CAPACITY, 64 * 2 + 2)) 

    while not done:
        
        action=protagonist_policy.get_action(obs)
        obs_, reward, done, _ = env.step(action)

        if protagonist_policy_name=='DQN':
            protagonist_policy.store_transition(obs, action, reward, obs_)                 # 存储样本

        if opponent_policy_name=='DQN':
            opponent_policy.store_transition(obs, action, reward, obs_)

        obs=obs_
        if done:
                #print('reward={}'.format(reward))
                writer.add_scalar('Average_reward',reward,round)
                writer.add_scalar('Protagonist_Loss',protagonist_policy.loss,round)

                if reward == 1:
                    win_cnts += 1
                elif reward == 0:
                    draw_cnts += 1
                else:
                    lose_cnts += 1

                ##DQN 需要更新记忆
                protagonist_policy.update_memory(reward=reward)
                opponent_policy.update_memory(reward=-reward)
        if protagonist_policy.memory_counter > protagonist_policy.memory_capacity:              # 如果累计的transition数量超过了记忆库的固定容量2000 
            protagonist_policy.learn()
            opponent_policy.learn()
print('#Wins: {}, #Draws: {}, #Loses: {}'.format(win_cnts, draw_cnts, lose_cnts))
writer.close()