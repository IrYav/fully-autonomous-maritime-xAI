import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from Agent.Wrappers.ship_wrapper import ShipEnv
from Agent.callback import StatsCallback
from Simulation.Auxilaries.sim_config import sim_config
#from network import MaritimeCNN, NetworkConfig

#Gym Environment
env = ShipEnv(sim_config)

instance_name = "less_entrophy_bigger_batches2"
#Add custom logging
callback = StatsCallback()

net_config = env.net_config

# Optional but VERY useful early on
check_env(env)

model = PPO(
  policy="MultiInputPolicy",
  env=env,

  # ----------------------------------------------  
  # Learning
  # ----------------------------------------------

  learning_rate=1e-4,
  n_steps=4096,
  batch_size=128,
  gamma=0.99,
  gae_lambda=0.95,

  # ----------------------------------------------
  # PPO specifics
  # ----------------------------------------------

  clip_range=0.2,
  ent_coef=0.001,

  # ----------------------------------------------
  # Logging
  # ----------------------------------------------

  verbose=1,
  tensorboard_log=f"./models/{instance_name}/",
)
# model = PPO.load(
#     "ppo_ship.zip",
#     env=env
# )

# --------------------------------------------------
# Train
# --------------------------------------------------
try:
  for i in range(15):
    model.learn(
      total_timesteps=100_000,
      reset_num_timesteps=False,
      callback=callback,
    )

    model.save(f"./models/{instance_name}/ppo_ship_{model.num_timesteps}")
except KeyboardInterrupt:
  pass
finally:
  model.save(f"./models/{instance_name}/ppo_ship")