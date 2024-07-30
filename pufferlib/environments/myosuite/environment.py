from pdb import set_trace as T

import functools
import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.postprocess

from myosuite.utils import gym

def env_creator(name='myoElbowPose1D6MRandom-v0'):
    return functools.partial(make, name)

def make(name):
    '''Create an environment by name'''
    env = gym.make(name)

    # override observation space
    env.observation_space = gym.spaces.Box(
        low=env.observation_space.low[0],
        high=env.observation_space.high[0],
        shape=env.observation_space.shape,
        dtype=np.float64,
    )

    env = pufferlib.postprocess.ClipAction(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
