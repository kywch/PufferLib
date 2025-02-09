import torch
from torch import nn
from pufferlib.pytorch import layer_init

import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class Policy(nn.Module):
    def __init__(self, env, input_dim, action_dim, demo_dim, hidden):
        super().__init__()
        self.is_continuous = True

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(input_dim, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden)),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.sigma = nn.Parameter(
            torch.zeros(action_dim, requires_grad=False, dtype=torch.float32),
            requires_grad=False,
        )
        nn.init.constant_(self.sigma, -2.9)
        #self.mu = pufferlib.pytorch.layer_init(
        #    nn.Linear(hidden, action_dim), std=0.01)
        #self.sigma = nn.Parameter(torch.zeros(1, action_dim))

        ### Separate Critic
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(input_dim, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, action_dim)),
            nn.SiLU(),
        )
        self.value = nn.Linear(hidden, 1)

        ### Discriminator
        self._disc_mlp = nn.Sequential(
            layer_init(nn.Linear(demo_dim, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, hidden)),
            nn.ReLU(),
        )
        self._disc_logits = layer_init(torch.nn.Linear(hidden, 1))

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, obs):
        return self.actor_mlp(obs), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)
        probs = torch.distributions.Normal(mu, std)
        value = self.value(hidden)
        return probs, value

    def discriminate(self, amp_obs):
        disc_mlp_out = self._disc_mlp(amp_obs)
        disc_logits = self._disc_logits(disc_mlp_out)
        return disc_logits

    def disc_logit_weights(self):
        return torch.flatten(self._disc_logits.weight)

    def disc_weights(self):
        weights = []
        for m in self._disc_mlp.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self._disc_logits.weight))
        return weights


