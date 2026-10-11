import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def mlp(inp, out, hidden):
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out),
    )


class SharedPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, init_log_std=-0.5):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs):
        return torch.distributions.Normal(self.net(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs):
        """Deterministic actions for a batch of buildings, shape (n, act_dim), in [-1, 1]."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        return np.tanh(self.net(x).numpy())


@dataclass
class PPOConfig:
    total_steps: int = 300_000
    rollout_steps: int = 2048
    hidden: int = 128
    lr: float = 5e-4
    gamma: float = 0.95
    gae_lambda: float = 0.9
    clip_eps: float = 0.1
    update_epochs: int = 10
    minibatch_size: int = 512
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    init_log_std: float = -0.5


def train(env, cfg: PPOConfig, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n, obs_dim, act_dim = env.num_agents, env.obs_dim, env.act_dim

    policy = SharedPolicy(obs_dim, act_dim, cfg.hidden, cfg.init_log_std)
    value_net = mlp(obs_dim, 1, cfg.hidden)
    params = list(policy.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    r_count, r_sum, r_sumsq = 0, 0.0, 0.0
    history = []          
    obs = env.reset(seed=seed)
    steps, start = 0, time.time()

    while steps < cfg.total_steps:
        obs_b, u_b, logp_b, val_b, rew_b, done_b = [], [], [], [], [], []
        for _ in range(cfg.rollout_steps):
            o = torch.as_tensor(obs)
            with torch.no_grad():
                d = policy.dist(o)
                u = d.sample()
                logp = d.log_prob(u).sum(-1)
                v = value_net(o).squeeze(-1)
            obs, reward, done, _ = env.step(np.tanh(u.numpy()))
            obs_b.append(o); u_b.append(u); logp_b.append(logp); val_b.append(v)
            rew_b.append(reward); done_b.append(done)
            steps += 1
            if done:
                obs = env.reset()
            if steps >= cfg.total_steps:
                break

        with torch.no_grad():
            last_v = value_net(torch.as_tensor(obs)).squeeze(-1).numpy()

        rews = np.stack(rew_b)                             
        vals = torch.stack(val_b).numpy()               
        dones = np.array(done_b)
        r_count += rews.size
        r_sum += float(rews.sum())
        r_sumsq += float((rews ** 2).sum())
        r_std = np.sqrt(max(r_sumsq / r_count - (r_sum / r_count) ** 2, 0.0)) + 1e-8
        rews_s = rews / r_std

        T = len(rews)
        adv = np.zeros_like(rews_s)
        last = np.zeros(n, dtype=np.float32)
        for t in reversed(range(T)):
            nonterminal = 0.0 if dones[t] else 1.0
            next_v = last_v if t == T - 1 else vals[t + 1]
            delta = rews_s[t] + cfg.gamma * next_v * nonterminal - vals[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
            adv[t] = last
        ret = adv + vals

        b_obs = torch.stack(obs_b).reshape(-1, obs_dim)
        b_u = torch.stack(u_b).reshape(-1, act_dim)
        b_logp = torch.stack(logp_b).reshape(-1)
        b_adv = torch.as_tensor(adv.reshape(-1))
        b_ret = torch.as_tensor(ret.reshape(-1))
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        N = len(b_obs)
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(N)
            for i in range(0, N, cfg.minibatch_size):
                idx = perm[i : i + cfg.minibatch_size]
                d = policy.dist(b_obs[idx])
                ratio = torch.exp(d.log_prob(b_u[idx]).sum(-1) - b_logp[idx])
                s1 = ratio * b_adv[idx]
                s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_adv[idx]
                pl = -torch.min(s1, s2).mean()
                vl = ((value_net(b_obs[idx]).squeeze(-1) - b_ret[idx]) ** 2).mean()
                optimizer.zero_grad()
                (pl + cfg.value_coef * vl).backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()

        mean_r = float(rews.sum(axis=1).mean())
        history.append((steps, mean_r))
        print(f"steps {steps:7d} | {time.time() - start:6.1f}s | mean reward/step (all buildings) {mean_r:8.3f}")

    return policy, history