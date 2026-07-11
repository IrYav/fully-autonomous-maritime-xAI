#One thread per env.
import math
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from Agent.Wrappers.ship_wrapper import ShipEnv
from Agent.callback import StatsCallback
from Simulation.Auxilaries.sim_config import sim_config

ROLLOUT_SIZE = 4096
BATCH_SIZE = 128

n_steps = math.ceil(ROLLOUT_SIZE / sim_config.n_envs / 32) * 32



if __name__ == "__main__":
  # Checks for some errors
  check_env(ShipEnv(sim_config))
  
  #Gym Environments
  def make_env():
    def _init():
      return Monitor(ShipEnv(sim_config))
    return _init

  env = SubprocVecEnv([make_env() for _ in range(sim_config.n_envs)])


  instance_name = "SAC/05lowerGamma/run0"
  #Add custom logging
  callback = StatsCallback()

  model = SAC(
    policy="MultiInputPolicy",
    env=env,
    device="cuda",

    # ------------------------------------------------
    # Core SAC learning
    # ------------------------------------------------
    learning_rate=5e-5,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.98,

    # ------------------------------------------------
    # SAC specifics
    # ------------------------------------------------
    train_freq=4,
    gradient_steps=1,
    ent_coef="auto",
    # target_entropy="auto",

    # ------------------------------------------------
    # Logging
    # ------------------------------------------------
    verbose=1,
    tensorboard_log=f"./experiments/{instance_name}/",
  )

  # --------------------------------------------------
  # Train
  # --------------------------------------------------
  try:
    for i in range(10):
      model.learn(
        total_timesteps=200_000,
        reset_num_timesteps=False,
        callback=callback,
      )

      model.save(f"./experiments/{instance_name}/checkpoints/ship_{model.num_timesteps}")
  except KeyboardInterrupt:
    pass
  finally:
    model.save(f"./experiments/{instance_name}/ship")