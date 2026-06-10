import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from Agent.Wrappers.ship_wrapper import ShipEnv
from Agent.callback import StatsCallback
from Simulation.Auxilaries.sim_config import sim_config
#from network import MaritimeCNN, NetworkConfig

#Gym Environment
env = ShipEnv(sim_config)

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

  learning_rate=3e-4,
  n_steps=2048,
  batch_size=64,
  gamma=0.99,
  gae_lambda=0.95,

  # ----------------------------------------------
  # PPO specifics
  # ----------------------------------------------

  clip_range=0.2,
  ent_coef=0.01,

  # ----------------------------------------------
  # Logging
  # ----------------------------------------------

  verbose=1,
  tensorboard_log="./tensorboard/",
)

# --------------------------------------------------
# Train
# --------------------------------------------------
model.learn(
  total_timesteps=500_000,
  #reset_num_timesteps=False,
  callback=callback,
)

# --------------------------------------------------
# Save
# --------------------------------------------------

model.save("ppo_ship")