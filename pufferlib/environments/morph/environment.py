import time
import argparse
import functools

from pufferlib.environments.morph.humanoid_phc import HumanoidPHC

import torch
import numpy as np

import pufferlib

def env_creator(name='morph'):
    return functools.partial(make, name)
 
def make(name, **kwargs):
    return PHCPufferEnv(**kwargs)

class PHCPufferEnv(pufferlib.PufferEnv):
    def __init__(self, motion_file, has_self_collision, num_envs=32, device_type="cuda",
            device_id=0, headless=True, log_interval=32):
        cfg = {
            'env': {
                'num_envs': num_envs,
                'motion_file': motion_file,
            },
            'robot': {
                'has_self_collision': has_self_collision,
            },
        }
        self.env = HumanoidPHC(cfg, device_type=device_type, device_id=device_id, headless=headless)
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.num_agents = self.num_envs = self.env.num_envs
        self.device = self.env.device

        # Check the buffer data types, match them to puffer
        buffers = pufferlib.namespace(
            observations=self.env.obs_buf,
            rewards=self.env.rew_buf,
            terminals=self.env.reset_buf,
            truncations=torch.zeros_like(self.env.reset_buf),
            masks=torch.ones_like(self.env.reset_buf),
            actions=torch.zeros(
                (self.num_agents, *self.single_action_space.shape), dtype=torch.float, device=self.device
            ),
        )

        super().__init__(buffers)

        self.log_interval = log_interval
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._infos = {
            "episode_return": [],
            "episode_length": [],
        }

    def reset(self, seed=None):
        self.env.reset()
        self.demo = self.env.demo
        self.state = self.env.state
        self.tick = 0
        return self.observations, []

    def step(self, actions_np):
        self.actions[:] = torch.from_numpy(actions_np)

        # obs, reward, done are put into the buffers
        self.env.step(self.actions)
        self.demo = self.env.demo
        self.state = self.env.state

        done_indices = torch.nonzero(self.terminals).squeeze(-1)
        if len(done_indices) > 0:
            self.observations[done_indices] = self.env.reset(done_indices)[done_indices]
            self._infos["episode_return"] += self.episode_returns[done_indices].tolist()
            self._infos["episode_length"] += self.episode_lengths[done_indices].tolist()
            self.episode_returns[done_indices] = 0
            self.episode_lengths[done_indices] = 0

        self.episode_returns += self.rewards
        self.episode_lengths += 1

        # TODO: self.env.extras has infos. Extract useful info?
        info = []
        self.tick += 1
        if self.tick % self.log_interval == 0:
            info = self.mean_and_log()

        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        self.env.close()

    def mean_and_log(self):
        if len(self._infos["episode_return"]) < self.log_interval:
            return []

        info = {
            "episode_return": np.mean(self._infos["episode_return"]),
            "episode_length": np.mean(self._infos["episode_length"]),
        }
        self._infos["episode_return"].clear()
        self._infos["episode_length"].clear()

        return [info]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_envs", type=int, default=32)
    parser.add_argument("-m", "--motion_file", type=str, default="sample_data/amass_train_take6_upright.pkl")
    parser.add_argument("--disable_self_collision", action="store_true")
    args = parser.parse_args()

    def test_perf(env, timeout=10):
        steps = 0
        start = time.time()
        env.reset()
        actions = env.action_space.sample()

        print("Starting perf test...")
        while time.time() - start < timeout:
            env.step(actions)
            steps += env.num_agents

        end = time.time()
        sps = int(steps / (end - start))
        print(f"Steps: {steps}, SPS: {sps}")

    cfg = {
        "env": {
            "num_envs": args.num_envs,
            "motion_file": args.motion_file,
        },
        "robot": {"has_self_collision": not args.disable_self_collision},
    }

    env = PHCPufferEnv(cfg)
    test_perf(env)
