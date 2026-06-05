import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from Agent.Wrappers.ship_wrapper import ShipEnv
from Controller.controller import SimulationConfig
from Simulation.Auxilaries.ship_instances import ferry
#from network import MaritimeCNN, NetworkConfig

#Learning environment specifics
sim_config = SimulationConfig(
  boundary = np.array([[-2500, -2500], [2500, 2500]], dtype=float),
  ship_params = ferry(),
  lview_size = np.array([300,300]), 
  lview_res = np.array([64,64])
  )

#Gym Environment
env = ShipEnv(sim_config)

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
  total_timesteps=1_000_000,
)

# --------------------------------------------------
# Save
# --------------------------------------------------

model.save("ppo_ship")