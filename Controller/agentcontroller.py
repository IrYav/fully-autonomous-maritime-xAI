
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO, SAC


from Agent.Observation.observation import build_observation, dict_to_obs
from Agent.Wrappers.env_wrapper import get_reward
from Controller.controller import BaseController
from Simulation.Auxilaries.io import log_result, log_steps, save_env_ship
from Simulation.Auxilaries.sim_config import SimulationConfig
from Simulation.Environment import agent


class AgentController(BaseController):
  '''
    Controller for running a trained agent in the environment.  
  '''
  def __init__(self, config : SimulationConfig,
               model: PPO | SAC ):
    super().__init__(config)

    self.model = model

  def finish_episode(self) -> float:
    '''
      Finish the current episode and return the total reward.
    '''
    total_reward, cols = 0.0, []
    while not cols: #Any collision finishes (reaching the end is also a collision)
      res = self.model_step()
      total_reward += res["reward"]
      cols = res["cols"]
      if self.sim_state.steps >= self.config.max_steps:
        break
      
    return total_reward, self.sim_state.steps, cols

  def model_step(self):
    '''
      Does a model step on the current environment.
      Requires a valid episode.
    '''
    obs = build_observation(self)
    action, _ = self.model.predict(obs, deterministic=True)
    action = agent.Action(*action)

    pos0 = self.sim_state.ship.state.position
    ship_state0 = self.sim_state.ship.state
    _, cols = (
        self.simulate_step(action, self.config.dt)
      )
    pos1 = self.sim_state.ship.state.position

    reward = get_reward(cols, pos0, pos1, self.config, self.sim_state)["reward"]
    return {
      "cols": cols,
      "action": action,
      "reward": reward,
      "ship_state0": ship_state0,
    }
  def get_step_log_row(dict) -> list[float]:
    return
  
  def generate_samples(self, path, num_step_samples: int) -> None:
    '''
      Generates and runs environments.

      It will run until at least `num_step_samples` steps have been collected.
      The last episode will always be finished, even if it exceeds the number of steps.

      Each run contains the following:
      - Stores `env` and `ship` as json.
      - All steps, consisting of pos, action and step_reward
      - Result: Reward, Stepcount, Cols
    '''
    steps, run = 0, 0
    while steps < num_step_samples:
      step_log = []
      cur_path = Path(path) / f"run{run:04d}"
      self.new_training_instance()

      #Log start
      save_env_ship(cur_path / "scenario.json", self.sim_state.env, self.sim_state.ship)

      total_reward = 0.0
      cols = []
      while not cols: #Any collision finishes (reaching the end is also a collision)
        res = self.model_step()
        total_reward += res["reward"]
        cols = res["cols"]
        #Collect all steps
        step_log.append([res["ship_state0"], res["action"], res["reward"]])
        steps += 1
        if self.sim_state.steps >= self.config.max_steps:
          break

      #log all steps and result
      log_steps(cur_path / "steps.npz", step_log)
      log_result(cur_path / "result.npz", total_reward, self.sim_state.steps, cols)
      run += 1

  def _get_state_dict(self):
    obs = build_observation(self, True)
    return obs["state"]

  