import os
import sys
import timeit
import torch

from utils import save_data

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
from environment.env import SumoEnvironment
from agent.QMIX import QMIX
from config import Config
from replay_buffer import ReplayBuffer


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
    cfg.action_dim=env.action_space.n
    cfg.update_target = 40

    cfg.state_shape=len(env.ts_ids)*env.observation_space.shape[0]
    replaybuffer = ReplayBuffer(cfg)
    qmix_agent=QMIX(env.observation_space.shape[0],env.action_space.n,cfg,env.ts_ids)
    rewards = []
    delays = []
    train_tims = []
    for i in range(100):
        print(i)
        state = env.reset()


        epsilon = 1.0 - (i / 100)
        reward_ep = 0
        delay_ep = 0
        done = {'__all__': False}
        hidden_state = torch.zeros([cfg.n_agents, 1, cfg.hidden_dim]).cuda()
        count=0
        while not done['__all__']:
            actions = {}
            for ts in env.ts_ids:
                action,h=qmix_agent.choose_action(state[ts],hidden_state[int(ts)],epsilon)
                actions[ts] = action
                hidden_state[int(ts)] = h

            next_state, r, done, info = env.step(action=actions)
            for r_step in r.values():
                reward_ep += int(r_step)
            replaybuffer.store_episode({int(k): v for k, v in state.items()},
                                       {int(k): v for k, v in actions.items()},
                                       {int(k): v for k, v in r.items()},
                                       {int(k): v for k, v in next_state.items()})
            delay_ep += (np.sum(env.get_delay()) / 16)
            state = next_state
            count+=1
        rewards.append(reward_ep)
        delays.append(delay_ep)

        print('reward_ep', reward_ep, 'delay_ep', delay_ep)
        start_time = timeit.default_timer()
        print("Training...")


        for k in range(400):
            transitions = replaybuffer.sample(cfg.batch_size)
            qmix_agent.update(transitions,cfg.update_target)
        training_time = round(timeit.default_timer() - start_time, 1)
        train_tims.append(training_time)
    save_data('result', rewards, 'qmix_reward')
    save_data('result', delays, 'qmix_delay')
    save_data('result', train_tims, 'qmix_training_time')
    env.close()





