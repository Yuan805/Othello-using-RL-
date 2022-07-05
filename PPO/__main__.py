import othello
import simple_policies
import PPO_policy
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter   
import time
import torch

##设置策略
protagonist_policy=PPO_policy.PPO()
opponent_policy=simple_policies.GreedyPolicy()

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


#设置环境
args_exp_name='PPO'
args_seed=1
args_torch_deterministic=True  #固定卷积层，每次相同的输入得到相同的输出
args_cuda=True
args_capture_video=False
args_env_id='Othello'
args_total_time=500000
args_num_rounds=1000


#不要改动
run_name = f"{'othello'}__{args_exp_name}__{args_seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")
random.seed(args_seed)
np.random.seed(args_seed)
torch.manual_seed(args_seed)
torch.backends.cudnn.deterministic = args_torch_deterministic
device = torch.device("cuda" if torch.cuda.is_available() and args_cuda else "cpu")


train_num_rounds=100
win_cnts = draw_cnts = lose_cnts = 0

for round in range(train_num_rounds):
    if round %100==0:
        print('Episode {}'.format(round + 1))   
 
    obs = env.reset()
    protagonist_policy.reset(env)
    opponent_policy.reset(env)
    done = False

    while not done:

        action =protagonist_policy.get_action(obs)
        
        obs_, reward, done, _ = env.step(action)


        obs=obs_
        if done:
                #print('reward={}'.format(reward))
                writer.add_scalar('Average_reward',reward,round)
                if reward == 1:
                    win_cnts += 1
                elif reward == 0:
                    draw_cnts += 1
                else:
                    lose_cnts += 1


print('#Wins: {}, #Draws: {}, #Loses: {}'.format(win_cnts, draw_cnts, lose_cnts))
writer.close()