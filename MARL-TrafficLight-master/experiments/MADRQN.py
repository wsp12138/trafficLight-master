import os
import sys
import timeit
from utils import save_data

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from GNG import get_gng_result
import numpy as np
from environment.env import SumoEnvironment
from agent.dqn import DQN
from config import Config
from replay_buffer import ReplayBuffer
signal = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2],[0,3],[1,3],[2,3],[3,3]]
weight=0.5

def share_info(ts_id,state,gng_result):
    for i in range(len(ts_id)):
        if i == 0:
            if gng_result[i][1] == gng_result[i + 1][1]:
                state[str(i)] = state[str(i)] + weight*state[str(i + 1)]
        elif i == (len(env.ts_ids) - 1):
            if gng_result[i][1] == gng_result[i - 1][1]:
                state[str(i)] = state[str(i)] + weight*state[str(i - 1)]
        else:
            if gng_result[i][1] == gng_result[i + 1][1]:
                state[str(i)] = state[str(i)] + weight*state[str(i + 1)]
            if gng_result[i][1] == gng_result[i - 1][1]:
                state[str(i)] = state[str(i)] + weight*state[str(i - 1)]


if __name__ == '__main__':

    cfg = Config()
    env = SumoEnvironment(net_file='../net/4x4-Lucas/4x4.net.xml',
                          route_file='../net/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                          use_gui=False,
                          num_seconds=300,
                          min_green=5,
                          delta_time=5)

    cfg.n_agents = len(env.ts_ids)
    cfg.observation_space = env.observation_space.shape[0]
    cfg.update_target = 40
    cfg.param_share=100
    replaybuffer = ReplayBuffer(cfg)
    madrqn_agents = {ts: DQN(env.observation_space.shape[0], env.action_space.n, cfg, int(ts)) for ts in env.ts_ids}
    rewards = []
    delays = []
    train_tims = []
    for i in range(100):
        print(i)
        state = env.reset()
        #gng聚类
        for ts in env.ts_ids:
            for data in state[ts]:
                signal[int(ts)].append(data)
        gng_result=get_gng_result(signal)

        #print(get_gng_result(signal))
        epsilon = 1.0 - (i / 100)
        reward_ep = 0
        delay_ep = 0
        done = {'__all__': False}
        k=0

        while not done['__all__']:
            #共享信息
            share_info(env.ts_ids,state,gng_result)

            actions = {ts: madrqn_agents[ts].choose_action(state[ts], epsilon) for ts in madrqn_agents.keys()}
            next_state, r, done, info = env.step(action=actions)
            for r_step in r.values():
                reward_ep += int(r_step)
            replaybuffer.store_episode({int(k): v for k, v in state.items()},
                                       {int(k): v for k, v in actions.items()},
                                       {int(k): v for k, v in r.items()},
                                       {int(k): v for k, v in next_state.items()})

            delay_ep += (np.sum(env.get_delay()) / 16)
            state = next_state

            #参数分享
            k+=1
            if k%cfg.param_share==0:
                best=np.argmax(list(r.values()))
                for i in range(len(r.values())):
                    if i!=best:
                        for target_param,param in zip(madrqn_agents[str(i)],madrqn_agents[str(best)]):
                            target_param.data.copy_(0.8*target_param.data+0.2*param.data)


        rewards.append(reward_ep)
        delays.append(delay_ep)
        print('reward_ep', reward_ep, 'delay_ep', delay_ep)
        start_time = timeit.default_timer()
        print("Training...")
        for k in range(400):
            transitions = replaybuffer.sample(cfg.batch_size)
            for agent in madrqn_agents.values():
                agent.update(transitions, k)

        training_time = round(timeit.default_timer() - start_time, 1)
        train_tims.append(training_time)

    save_data('./result', rewards, 'madrqn_reward')
    save_data('./result', delays, 'madrqn_reward_delay')
    save_data('./result', train_tims, 'madrqn_reward_training_time')
    env.close()






