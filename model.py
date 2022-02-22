import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from utils import standardize


def _t(x, device='cuda', cls=torch.FloatTensor):
    return cls(x).to(device)


def _dynamic_zeros_like(x):
    if isinstance(x, np.ndarray):
        return np.zeros_like(x)
    elif isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    else:
        raise RuntimeError('Unknown type')


def gae(rews, vals, dones, gam=0.99, lamb=0.95):
    advs = _dynamic_zeros_like(vals)
    masks = 1 - dones  # n + 1
    for i in reversed(range(len(vals) - 1)):
        delta = -vals[i] + (rews[i] + masks[i] * (gam * vals[i + 1]))
        advs[i] = delta + masks[i] * (gam * lamb * advs[i + 1])

    return advs


def surrogate_loss(ratios, advs, clip_param=0.1):
    clipped_ratios = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
    policy_gradients = torch.min(clipped_ratios * advs, ratios * advs)

    return policy_gradients.mean()


class Agent(nn.Module):
    def __init__(
            self,
            buffer,
            in_features,
            num_actions,
            device,
            lr=1e-4,
            local_epochs=4,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_actions = num_actions
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs

        self.actor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=-1)
        ).to(device)
        self.critic = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        ).to(device)
        self.optim = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.log_prob = 0.
        self.value = 0.

    def forward(self, x):
        ac_dist = self._policy(x)
        val = self._value(x)

        return ac_dist, val

    def act(self, obs):
        with torch.no_grad():
            ac_dist = self._policy(obs)
            self.value = self._value(obs).cpu().numpy()

        ac = ac_dist.sample()
        self.log_prob = ac_dist.log_prob(ac).cpu().numpy()

        return ac.item()

    def _value(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = _t(obs)
        logits = self.critic(obs).squeeze()

        return logits

    def _policy(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = _t(obs)
        probs = self.actor(obs)

        return Categorical(probs.cpu())

    def observe(self, obs, action, reward, done):
        if done:
            self.log_prob = 0.
            self.value = 0.

        self.buffer.append(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            log_prob=self.log_prob,
            value=self.value,
        )

    def learn(self):
        advs = gae(self.buffer['reward'], self.buffer['value'], self.buffer['done'])
        returns = _t(self.buffer['value'] + advs)
        advs = standardize(_t(advs))
        num_batches = 4
        idxes = np.arange(0, len(self.buffer))
        idxes = np.split(idxes, num_batches)

        for local_epoch in range(self.local_epochs):
            for batch_idx in idxes:
                batch = self.buffer[batch_idx].as_tensor()
                # inference
                ac_dist, values = self(batch['obs'])

                acs = ac_dist.sample()
                log_probs = ac_dist.log_prob(acs).to(self.device)
                ratios = log_probs - batch['log_prob']
                ratios = torch.exp(ratios)
                batch_advs = advs[batch_idx]
                # loss
                vf_coef = 0.5
                ent_coef = 0.01
                entropy = ac_dist.entropy().mean()
                actor_loss = surrogate_loss(ratios, batch_advs) + ent_coef * entropy
                critic_loss = vf_coef * F.smooth_l1_loss(values, returns[batch_idx])

                utility = actor_loss - critic_loss
                loss = -utility
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                # self.actor_optim.zero_grad()
                # actor_loss.backward()
                # self.actor_optim.step()
                # self.critic_optim.zero_grad()
                # critic_loss.backward()
                # self.critic_optim.step()
        self.buffer.reset()

        return {
            # 'ppo loss': loss.item(),
            'pg loss': actor_loss.item(),
            'value loss': critic_loss.item(),
            'entropy': entropy.item()
        }
