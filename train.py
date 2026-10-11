import json
from pathlib import Path

import torch

from src.env_utils import SharedCityLearn
from src.ppo import PPOConfig, train

SCHEMA = "citylearn_challenge_2022_phase_1"
SEED = 0

Path("results").mkdir(exist_ok=True)
env = SharedCityLearn(SCHEMA)
cfg = PPOConfig()
policy, history = train(env, cfg, seed=SEED)

torch.save(policy.state_dict(), f"results/policy_seed{SEED}.pt")
with open(f"results/history_seed{SEED}.json", "w") as f:
    json.dump({"config": cfg.__dict__, "history": history,
               "obs_dim": env.obs_dim, "act_dim": env.act_dim}, f, indent=2)
print("Saved policy and training history to results/")