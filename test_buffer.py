from collections import OrderedDict

import gym
import pytest

import numpy as np


@pytest.fixture
def env():
    return gym.make('CartPole-v1')


@pytest.fixture
def example():
    eg = OrderedDict({
        'obs': env.observation_space.sample(),
        'action': env.action_space.sample(),
        'reward': 0.,
        'done': False,
    })

    return eg


@pytest.fixture
def Buffer():
    pass


def test_axis():
    buffer = Buffer()
    num_agents = 12
    buffer = np.stack([buffer] * num_agents)
    num_envs = 32
    buffer = np.stack([buffer] * num_envs)
