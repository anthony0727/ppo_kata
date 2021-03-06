import random
import uuid
from collections import OrderedDict

import gym
import numpy as np
import torch
import wandb
from gym import wrappers
from torch.utils.tensorboard import SummaryWriter

from buffer import Buffer
# from pettingzoo import magent
from model import Agent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

num_actions = None


def _avg(l):
    return sum(l) / len(l)

max_length = 0.
max_return = 0.

class RolloutWorker:
    def __init__(self, env, agent, max_steps=int(1e12)):
        self.max_steps = max_steps
        self.env = env
        self.agent = agent
        self.done = True
        self.obs, self.rew, self.done = env.reset(), 0., False
        self.ac = agent.act(self.obs)
        self.global_steps = 0

    def last_transition(self):
        return (self.obs, self.ac, self.rew, self.done)

    def run(self, steps):
        agent.observe(*self.last_transition())
        ############
        for step in range(steps-1):
            if self.done:
                self.obs, self.rew, self.done = env.reset(), 0., False
            else:
                self.obs, self.rew, self.done, info = env.step(self.ac)

            self.ac = agent.act(self.obs)
            agent.observe(self.obs, self.ac, self.rew, self.done)
        #############
        info = {}
        try:
            # if self.global_step % log_interval == 0:
            num_episodes = 100
            lengths = np.array(env.get_episode_lengths()[-num_episodes:])
            returns = np.array(env.get_episode_rewards()[-num_episodes:])
            info['average_length'] = lengths.mean()
            info['average_return'] = returns.mean()
            info['max_length'] = max(max_length, lengths.max())
            info['max_return'] = max(max_return, returns.max())
        except:
            pass

        return info


DEBUG = True
WANDB = False
WANDB = True
WANDB_MODE = 'online' if WANDB else 'offline'


def _p(x):
    if DEBUG:
        print(x)


if __name__ == '__main__':
    log_dir = '/tmp/' + str(uuid.uuid4())
    summary_writer = SummaryWriter(log_dir)

    wandb.init(project="ppo-v2", mode=WANDB_MODE)
    env = gym.make('CartPole-v1')
    env.seed(seed)
    env = wrappers.Monitor(env, log_dir, video_callable=False)
    # env = gym.make('Acrobot-v1')
    _p(env.__repr__())
    _p(env.spec)
    _p(env.observation_space)
    _p(env.action_space)
    # env = gym.make("MountainCar-v0")
    # env = gym.make("ALE/Gravitar-v5")
    # env = gym.make("Taxi-v3") failed
    # env = gym.make("FrozenLake-v1") failed
    eg = OrderedDict({
        'obs': env.observation_space.sample(),
        'action': env.action_space.sample(),
        'reward': 0.,
        'done': False,
    })
    _p(f'example transition\n{eg}')
    extras = OrderedDict({
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
    _p(f'buffer spec\n{buffer.dtype}')
    in_features = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = Agent(buffer, in_features, num_actions, device)
    _p(agent)
    # summary_writer.add_graph(agent.actor, torch.FloatTensor(eg['obs']).to('cuda'))
    # summary_writer.add_graph(agent.critic, torch.FloatTensor(eg['obs']).to('cuda'))


    num_epochs = 1000
    worker = RolloutWorker(env=env, agent=agent)
    for epoch in range(num_epochs):
        info = worker.run(steps=train_interval)
        # agent.observe(*worker.last_transition())
        loss_dict = agent.learn()
        agent.buffer.reset()
        if DEBUG:
            _p(env.episode_id)
            _p(loss_dict)
            _p(info)
        wandb.log(loss_dict)
        wandb.log(info)
