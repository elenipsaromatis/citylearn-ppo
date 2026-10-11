from src.env_utils import SharedCityLearn

SCHEMA = "citylearn_challenge_2022_phase_1"

try:
    from citylearn.data import DataSet
    ds = DataSet()
    names = ds.get_dataset_names() if hasattr(ds, "get_dataset_names") else ds.get_names()
    print("Available datasets:")
    for name in names:
        print("  ", name)
except Exception as e:
    print("Could not list datasets:", e)

env = SharedCityLearn(SCHEMA)
print("\nBuildings:", env.num_agents)
print("Observation dims per building:", env.obs_dims)
print("Action dims per building:", env.act_dims)
print("Observation names (building 0):", env.env.observation_names[0])
print("Action names (building 0):", env.env.action_names[0])
print("Episode length (time steps):", env.env.time_steps)