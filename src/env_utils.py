import numpy as np
from citylearn.citylearn import CityLearnEnv


class SharedCityLearn:
    def __init__(self, schema, reward_mode="cost", **kwargs):
        self.env = CityLearnEnv(schema, central_agent=False, **kwargs)
        self.num_agents = len(self.env.buildings)
        self.reward_mode = reward_mode

        self.obs_lows = [np.asarray(s.low, dtype=np.float32) for s in self.env.observation_space]
        self.obs_highs = [np.asarray(s.high, dtype=np.float32) for s in self.env.observation_space]
        self.act_lows = [np.asarray(s.low, dtype=np.float32) for s in self.env.action_space]
        self.act_highs = [np.asarray(s.high, dtype=np.float32) for s in self.env.action_space]

        self.obs_dims = [len(lo) for lo in self.obs_lows]
        self.act_dims = [len(lo) for lo in self.act_lows]
        self.obs_dim = max(self.obs_dims) + 1 # to tell the buildings apart
        self.act_dim = max(self.act_dims)

        names = self.env.observation_names
        self.idx_net = [n.index("net_electricity_consumption") for n in names]
        self.idx_price = [n.index("electricity_pricing") for n in names]

    def zero_action(self):
        """Normalized actions that map to a raw action of 0 (do nothing) for every building."""
        out = np.zeros((self.num_agents, self.act_dim), dtype=np.float32)
        for i in range(self.num_agents):
            lo, hi = self.act_lows[i], self.act_highs[i]
            out[i, : self.act_dims[i]] = 2.0 * (0.0 - lo) / (hi - lo + 1e-8) - 1.0
        return out

    def _process(self, obs):
        out = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        for i in range(self.num_agents):
            x = np.asarray(obs[i], dtype=np.float32)
            lo, hi = self.obs_lows[i], self.obs_highs[i]
            x = 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0
            out[i, : len(x)] = np.clip(x, -5.0, 5.0)
            out[i, -1] = self.obs_dims[i] / (self.obs_dim - 1)
        return out

    def _cost_reward(self, obs):
        r = np.zeros(self.num_agents, dtype=np.float32)
        for i in range(self.num_agents):
            net = float(obs[i][self.idx_net[i]])
            price = float(obs[i][self.idx_price[i]])
            r[i] = -max(net, 0.0) * price
        return r

    def reset(self, seed=None):
        try:
            result = self.env.reset(seed=seed)
        except TypeError:
            result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return self._process(obs)

    def step(self, actions):
        """actions: array (num_agents, act_dim) in [-1, 1]."""
        env_actions = []
        for i in range(self.num_agents):
            a = np.clip(actions[i, : self.act_dims[i]], -1.0, 1.0)
            lo, hi = self.act_lows[i], self.act_highs[i]
            env_actions.append((lo + (a + 1.0) / 2.0 * (hi - lo)).tolist())
        result = self.env.step(env_actions)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated) or bool(truncated)
        else:
            obs, reward, done, info = result
            done = bool(done)
        if self.reward_mode == "cost":
            reward = self._cost_reward(obs)
        else:
            reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        return self._process(obs), reward, done, info