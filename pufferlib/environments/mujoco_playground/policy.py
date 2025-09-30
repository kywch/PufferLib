import torch
from torch import nn

from pufferlib.pytorch import layer_init
from pufferlib.models import LSTMWrapper as Recurrent


class Policy(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size

        self.actor_obs_size = env._actor_obs_size
        self.critic_obs_size = env.single_observation_space.shape[0]
        self.action_size = env.single_action_space.shape[0]

        # Match the network to the brax defaults: [512, 256, 128] with SiLU and layernorm
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.actor_obs_size, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
        )
        self.actor_mu = nn.Sequential(
            nn.SiLU(),
            layer_init(nn.Linear(self.hidden_size, self.action_size), std=0.01),
        )
        self.actor_sigma = nn.Parameter(torch.zeros(1, self.action_size))

        # Separate asymmetric critic that uses the priviliged info
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.critic_obs_size, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128),
            nn.SiLU(),
            layer_init(nn.Linear(128, 1), std=0.01),
        )

        self.privileged_obs = None

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        self.privileged_obs = observations
        # NOTE: actor only uses the non-privileged
        return self.actor_mlp(observations[:, : self.actor_obs_size])

    def decode_actions(self, hidden):
        """Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers)."""
        mean = self.actor_mu(hidden)
        logstd = self.actor_sigma.expand_as(mean)
        std = torch.exp(logstd)
        logits = torch.distributions.Normal(mean, std)
        values = self.critic_mlp(self.privileged_obs)
        return logits, values
