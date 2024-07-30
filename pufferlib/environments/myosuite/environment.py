import functools

import pufferlib
import pufferlib.emulation
import pufferlib.postprocess

from myosuite.utils import gym

def env_creator(name='myoElbowPose1D6MRandom-v0'):
    return functools.partial(make, name)

def make(name):
    '''Create an environment by name'''
    env = gym.make(name)

    env = pufferlib.postprocess.ClipAction(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
