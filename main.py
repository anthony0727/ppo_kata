import uuid
from collections import OrderedDict as od, deque

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from gym import wrappers

from buffer import Buffer
# from pettingzoo import magent
from model import Agent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make('CartPole-v1')
num_actions = None
buffer_size = 128

def _avg(l):
    return sum(l) / len(l)

class RolloutWorker:
    def __init__(self, env, agent, max_steps=int(1e12)):
        self.max_steps = max_steps
        self.env = env
        self.agent = agent
        self.obs, self.rew, self.done = env.reset(), 0., False

    def run(self, steps):
        for step in range(steps):
            if step == self.max_steps:
                break

            ac = agent.act(self.obs)
            agent.observe(self.obs, ac, self.rew, self.done)

            if self.done:
                self.obs, self.rew, self.done = env.reset(), 0., False
            else:
                self.obs, self.rew, self.done, info = env.step(ac)

        info = {}
        try:
            num_episodes = 100
            lengths = np.array(env.get_episode_lengths()[:-num_episodes])
            returns = np.array(env.get_episode_rewards()[:-num_episodes])
            info['average_length'] = lengths.mean()
            info['average_return'] = returns.mean()
            info['max_length'] = lengths.max()
            info['max_return'] = returns.max()
        except:
            pass

        return info


if __name__ == '__main__':
    summary_writer = SummaryWriter('.')
    # wandb.init(project="ppo-v2", mode='disabled')
    wandb.init(project="ppo-v2")
    env = gym.make('CartPole-v1')
    # env = gym.make("MountainCar-v0")
    # env = gym.make("ALE/Gravitar-v5")
    # env = gym.make("Taxi-v3") failed
    # env = gym.make("FrozenLake-v1") failed
    print(env.__repr__())
    num_actions = env.action_space.n
    eg = od({
        'obs': env.observation_space.sample(),
        'action': env.action_space.sample(),
        'reward': 0.,
        'done': False,
    })
    print(f'example transition\n{eg}')
    extras = od({
        'value': np.float32,
        'log_prob': np.float32,
    })

    train_steps = int(1e8)
    train_interval = 128
    log_interval = 128
    lr = 1e-4

    buffer = Buffer(
        num_transitions=train_interval,
        example=eg,
        extras=extras
    )
    print(f'buffer spec\n{buffer.dtype}')
    in_features = 4
    agent = Agent(buffer, in_features, num_actions, device)
    print(agent)
    # summary_writer.add_graph(agent, torch.FloatTensor(eg['obs']).to('cuda'))
    env = wrappers.Monitor(
        env,
        '/tmp/' + str(uuid.uuid4()),
        video_callable=False
    )
    num_epochs = 1000
    worker = RolloutWorker(env=env, agent=agent)
    for epoch in range(num_epochs):
        info = worker.run(steps=train_interval)
        loss_dict = agent.learn()
        print(env.episode_id)
        wandb.log(loss_dict)
        wandb.log(info)
