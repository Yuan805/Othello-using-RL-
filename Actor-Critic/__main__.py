import othello
import simple_policies
import Actor_Critic
import numpy as np
from torch.utils.tensorboard import SummaryWriter   

##设置策略
protagonist_policy=Actor_Critic.AC()
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

##记录
writer=SummaryWriter('')

##训练：
train_num_rounds=1000
win_cnts = draw_cnts = lose_cnts = 0

for round in range(train_num_rounds):
    print('Episode {}'.format(round + 1))  

    obs = env.reset()
    protagonist_policy.reset(env)
    opponent_policy.reset(env)
    done = False

    while not done:

        action, _ =protagonist_policy.get_action(obs)
        obs_, reward, done, _ = env.step(action)
        if reward==0:
            reward=-1e-3
        TDerror=protagonist_policy.critic_learn(obs, obs_, reward, done)
        protagonist_policy.actor_learn(TDerror=TDerror, s=obs, a=action)

        obs=obs_
        if done:
                #print('reward={}'.format(reward))
                writer.add_scalar('Average_reward',reward,round)
                writer.add_scalar('Actor_Loss',protagonist_policy.actor_loss,round)
                writer.add_scalar('Critic_Loss',protagonist_policy.critic_loss,round)

                if reward == 1:
                    win_cnts += 1
                elif reward == 0:
                    draw_cnts += 1
                else:
                    lose_cnts += 1

                ##DQN 需要更新记忆
                
print('#Wins: {}, #Draws: {}, #Loses: {}'.format(win_cnts, draw_cnts, lose_cnts))
writer.close()