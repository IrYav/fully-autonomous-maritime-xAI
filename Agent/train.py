#One thread per env.
import math
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from Agent.Wrappers.ship_wrapper import ShipEnv
from Agent.callback import StatsCallback
from Simulation.Auxilaries.sim_config import sim_config

ROLLOUT_SIZE = 4096
BATCH_SIZE = 128

n_steps = math.ceil(ROLLOUT_SIZE / sim_config.n_envs / 32) * 32



# Checks for some errors
check_env(ShipEnv(sim_config))
if __name__ == "__main__":

  #Gym Environments
  def make_env():
    def _init():
      return Monitor(ShipEnv(sim_config))
    return _init

  env = SubprocVecEnv([make_env() for _ in range(sim_config.n_envs)])


  instance_name = "11halved_n_epochs/run0"
  #Add custom logging
  callback = StatsCallback()

  model = PPO(
    policy="MultiInputPolicy",
    env=env,
    device="cuda",
    # ----------------------------------------------  
    # Learning
    # ----------------------------------------------

    learning_rate=1e-4,
    n_epochs=5,
    n_steps=n_steps,
    batch_size=BATCH_SIZE,
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
    tensorboard_log=f"./experiments/{instance_name}/",
  )
  # model = PPO.load(
  #     "ppo_ship.zip",
  #     env=env
  # )

  # --------------------------------------------------
  # Train
  # --------------------------------------------------
  try:
    for i in range(20):
      model.learn(
        total_timesteps=200_000,
        reset_num_timesteps=False,
        callback=callback,
      )

      model.save(f"./experiments/{instance_name}/checkpoints/ppo_ship_{model.num_timesteps}")
  except KeyboardInterrupt:
    pass
  finally:
    model.save(f"./experiments/{instance_name}/ppo_ship")