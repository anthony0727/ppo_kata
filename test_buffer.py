from collections import OrderedDict

import gym
import pytest


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

def test_axis():
    pass