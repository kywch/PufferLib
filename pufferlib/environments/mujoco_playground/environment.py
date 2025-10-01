import functools
import os
from typing import Callable, Optional, Tuple
import warnings

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import torch
import torch.utils.dlpack as tpack
from brax.envs.wrappers import training as brax_training
from mujoco import mjx

import pufferlib
from mujoco_playground import registry, wrapper
from mujoco_playground._src import mjx_env

# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


# TODO: Alias all mp envs to "mp_" + all lower case
# NOTE: Not supporting vision-based RL for now
ALIASES = {
    "mp_leapcubereorient": "LeapCubeReorient",
    "mp_cheetahrun": "CheetahRun",
}


def env_creator(name="LeapCubeReorient", **kwargs):
    return functools.partial(make, name)


def make(
    name, num_envs=1, device="cuda", multi_gpu=False, seed=0, action_repeat=1, **kwargs
):
    """Create an environment by name"""

    # NOTE: Not supporting vision-based RL for now
    if name in ALIASES:
        name = ALIASES[name]

    env_cfg = registry.get_default_config(name)

    # Allow few specific kwargs to go into overrides, including impl
    # NOTE: For warp, nconmax determines the vram usage. adjust according to # envs
    # env_cfg_overrides = {"impl": "warp", "nconmax": 30*2048}
    env_cfg_overrides = {}

    mp_env = registry.load(name, config=env_cfg, config_overrides=env_cfg_overrides)

    return MujocoPlaygroundPufferEnv(
        mp_env,
        num_envs,  # this is # of parallel envs
        seed,
        env_cfg.episode_length,
        action_repeat,
        device=device,
        # multi_gpu=multi_gpu,  # TODO: test multi gpu
        # randomization_fn=randomizer,
    )


def _jax_to_torch(tensor):
    return tpack.from_dlpack(tensor)


def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    full_reset: bool = False,
) -> wrapper.Wrapper:
    if vision:
        raise NotImplementedError
    elif randomization_fn is None:
        env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = wrapper.BraxAutoResetWrapper(env, full_reset=full_reset)
    return env


class MujocoPlaygroundPufferEnv(pufferlib.PufferEnv):
    """Wrapper for Brax environments that interop with torch."""

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        num_envs,
        seed,
        episode_length,
        action_repeat,
        device,
        randomization_fn=None,
        reward_scale=0.1,
        # buf=None,
    ):
        self.num_agents = num_envs  # Treat each env as an agent
        self.reward_scale = reward_scale
        self.use_privileged_obs = False

        # NOTE: Not supporting vision-based RL for now
        # NOTE: Mujoco playground envs provides asymmetric obs in dict.
        # The privileged obs is actor obs (actor obs size) + priviliged info.
        if isinstance(env.observation_size, dict):
            self.use_privileged_obs = True
            obs_shape = env.observation_size["privileged_state"]
            self._actor_obs_size = env.observation_size["state"][0]
        elif isinstance(env.observation_size, int):
            obs_shape = (env.observation_size,)
            self._actor_obs_size = env.observation_size
        else:
            raise NotImplementedError

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env.action_size,),
            dtype=np.float32,
        )

        # Handle mp/jax issues
        self.seed = seed
        self.key = jax.random.PRNGKey(self.seed)

        if "cuda" in device:
            device_rank = int(device.split(":")[-1]) if "cuda:" in device else 0
            gpu_devices = jax.devices("gpu")
            self.key = jax.device_put(self.key, gpu_devices[device_rank])
            self.device = f"cuda:{device_rank}"
            print(f"Device -- {gpu_devices[device_rank]}")
            print(f"Key device -- {self.key.devices()}")
        else:
            self.device = "cpu"
            cpu_device = jax.devices("cpu")[0]
            self.key = jax.device_put(self.key, cpu_device)
            print(f"Device -- {cpu_device}")

        # split key into two for reset and randomization
        key_reset, key_randomization = jax.random.split(self.key)

        self.key_reset = jax.random.split(key_reset, self.num_agents)

        if randomization_fn is not None:
            randomization_rng = jax.random.split(key_randomization, self.num_agents)
            v_randomization_fn = functools.partial(
                randomization_fn, rng=randomization_rng
            )
        else:
            v_randomization_fn = None

        # self.env = env
        self.env = wrap_for_brax_training(
            env,
            episode_length=episode_length,
            # NOTE: action repeat is mostly 1, except for dm suite pendulum swing up
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )

        print("JITing reset and step")
        self.reset_fn = jax.jit(self.env.reset)
        self.step_fn = jax.jit(self.env.step)
        print("Done JITing reset and step")

        # NOTE: jax jit seems to cause RuntimeError: CUDA graph capture failed. Warp error: unknown capture stream
        # .venv/lib/python3.12/site-packages/mujoco/mjx/third_party/warp/jax_experimental/ffi.py", line 619, in ffi_callback
        #  with wp.ScopedCapture() as capture:
        # self.reset_fn = self.env.reset  # jax.vmap(self.env.reset)
        # self.step_fn = self.env.step  # jax.vmap(self.env.step)

        self.env_state = None

        # Check the buffer data types, match them to puffer
        buffers = {
            "observations": torch.zeros(
                (self.num_agents, *obs_shape), dtype=torch.float32, device=self.device
            ),
            "rewards": torch.zeros(
                self.num_agents, dtype=torch.float32, device=self.device
            ),
            "terminals": torch.zeros(
                self.num_agents, dtype=torch.bool, device=self.device
            ),
            "truncations": torch.zeros(
                self.num_agents, dtype=torch.bool, device=self.device
            ),
            "masks": torch.ones(self.num_agents, dtype=torch.bool, device=self.device),
            "actions": torch.zeros(
                (self.num_agents, *self.single_action_space.shape),
                dtype=torch.float32,
                device=self.device,
            ),
        }

        super().__init__(buffers)

    def reset(self, seed=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
            self.key_reset = jax.random.split(self.key, self.num_agents)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
            self.env_state = self.reset_fn(self.key_reset)

        if self.use_privileged_obs:
            # NOTE: The policy should separate state and priviliged info
            self.observations[:] = _jax_to_torch(self.env_state.obs["privileged_state"])
        else:
            self.observations[:] = _jax_to_torch(self.env_state.obs)

        return self.observations, []

    def step(self, action: npt.NDArray[np.float32]):
        action = jnp.array(action)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
            self.env_state = self.step_fn(self.env_state, action)

        if self.use_privileged_obs:
            # NOTE: The policy should separate state and priviliged info
            self.observations[:] = _jax_to_torch(self.env_state.obs["privileged_state"])
        else:
            self.observations[:] = _jax_to_torch(self.env_state.obs)

        self.terminals[:] = _jax_to_torch(self.env_state.done)
        self.truncations[:] = _jax_to_torch(self.env_state.info["truncation"])

        self.rewards[:] = _jax_to_torch(self.env_state.reward) # * self.reward_scale

        # NOTE: exclude truncation steps from training
        self.masks[:] = ~self.truncations

        info_ret = {}
        # These metrics come out each step
        for k, v in self.env_state.metrics.items():
            if k not in info_ret:
                info_ret[k] = _jax_to_torch(v).float().mean().item()

        done_envs = self.env_state.info["episode_done"].astype(bool)
        if jnp.any(done_envs):
            info_ret["episode_length"] = self.env_state.info["episode_metrics"]["length"][done_envs].mean().item()
            info_ret["episode_return"] = self.env_state.info["episode_metrics"]["sum_reward"][done_envs].mean().item()

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            [info_ret],
        )

    def render(self):
        # Probably pass an rgb array?
        pass

    def close(self):
        pass


if __name__ == "__main__":
    import cProfile
    import pstats

    import pufferlib.vector
    from pufferlib import pufferl
    from pufferlib.environments.mujoco_playground.policy import Policy

    # env_name = "mp_leapcubereorient"
    env_name = "mp_cheetahrun"

    vecenv = pufferlib.vector.make(env_creator(env_name), env_kwargs={"num_envs": 4096})
    policy = Policy(vecenv.driver_env).cuda()
    args = pufferl.load_config("default")
    args["train"]["env"] = env_name
    args["train"]["total_timesteps"] = 20_000_000
    # args["train"]["learning_rate"] = 0.0003
    # args["train"]["update_epochs"] = 3
    # args["train"]["gamma"] = 0.98
    # args["train"]["gae_lambda"] = 0.95
    # args["train"]["ent_coef"] = 0.001

    # args["train"]["compile"] = True

    # logger = pufferl.WandbLogger(args)
    logger = None

    trainer = pufferl.PuffeRL(args["train"], vecenv, policy, logger)

    while trainer.global_step < args["train"]["total_timesteps"]:
        trainer.evaluate()
        logs = trainer.train()

    # cProfile.run('trainer.evaluate()', 'stats.prof')
    # p = pstats.Stats('stats.prof')
    # p.sort_stats('cumtime')
    # p.print_stats(30)

    trainer.print_dashboard()
    trainer.close()
