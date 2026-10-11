import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.env_utils import SharedCityLearn
from src.ppo import SharedPolicy

SCHEMA = "citylearn_challenge_2022_phase_1"
SEED = 0


def run(env, controller):
    obs = env.reset()
    total, curve, done = 0.0, [], False
    while not done:
        obs, reward, done, _ = env.step(controller(obs))
        total += float(reward.sum())
        curve.append(total)
    kpis = None
    try:
        kpis = env.env.evaluate()
    except Exception as e:
        print("evaluate() not available:", e)
    return total, curve, kpis


with open(f"results/history_seed{SEED}.json") as f:
    meta = json.load(f)
env = SharedCityLearn(SCHEMA)
policy = SharedPolicy(meta["obs_dim"], meta["act_dim"], meta["config"]["hidden"])
policy.load_state_dict(torch.load(f"results/policy_seed{SEED}.pt"))
policy.eval()

controllers = {
    "No control": lambda o: env.zero_action(),
    "PPO": policy.act,
}

plt.figure(figsize=(8, 4))
for name, ctrl in controllers.items():
    total, curve, kpis = run(env, ctrl)
    print(f"\n{name}: total reward {total:.1f}")
    if kpis is not None:
        print(kpis.head(20).to_string())
    plt.plot(curve, label=f"{name} ({total:.0f})")

plt.xlabel("Time step (hour)")
plt.ylabel("Cumulative reward, all buildings")
plt.legend()
plt.tight_layout()
plt.savefig("results/cumulative_reward.png", dpi=150)
print("\nSaved results/cumulative_reward.png")