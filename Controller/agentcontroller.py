
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO, SAC


from Agent.Observation.observation import build_observation, dict_to_obs
from Agent.Observation.rasterize import update_local_view
from Agent.Wrappers.env_wrapper import get_reward
from Controller.controller import BaseController
from Simulation.Auxilaries.io import log_result, log_steps, save_env_ship
from Simulation.Auxilaries.sim_config import SimulationConfig
from Simulation.Environment import agent


class AgentController(BaseController):
  '''
    Controller for running a trained agent in the environment.

    Has a few comfort functions for repeatable perturbations or large scale sampling.
  '''
  def __init__(self, config : SimulationConfig,
               model: PPO | SAC ):
    super().__init__(config)

    self.model = model

  def set_ship(self, ship):
    '''
      Setting the ship requires a local view update.
      Direct accesses should be thus avoided.
    '''
    self.sim_state.ship = ship
    update_local_view(self.lview, self.sim_state.env, ship, 
                            self.config.lview_size, self.config.lview_res)

  def finish_episode(self, ghost_finish = False) -> tuple[float,int,list]:
    '''
      Finish the current episode and return the total reward.
    '''
    if ghost_finish:
      #Aside of the ship, nothing changes in a run.
      ship_backup = self.sim_state.ship.copy()

    total_reward, cols = 0.0, []
    while not cols: #Any collision finishes (reaching the end is also a collision)
      res = self.model_step()
      total_reward += res["reward"]
      cols = res["cols"]
      if self.sim_state.steps >= self.config.max_steps:
        break
      
    if ghost_finish:
      #Reset simulation to base state
      self.set_ship(ship_backup)    
    
    return total_reward, self.sim_state.steps, cols

  def model_step(self, ghost_step = False, state = None):
    '''
      Does a model step on the current environment.
      Requires a valid episode.

      Iff `ghost_step`, the action will be executed, 
      otherwise it will only 
    '''
    obs = build_observation(self) # Normal obs

    #Overwrite states
    if state is not None:
      obs["state"] = state

    action, _ = self.model.predict(obs, deterministic=True)

    pos0 = self.sim_state.ship.state.position
    aaction = agent.Action(*action)

    if ghost_step:
      #Ship is the only thing that changes.
      ship_backup = self.sim_state.ship.copy()

    #Run step
    _, cols = (
        self.simulate_step(aaction, self.config.dt)
      )
    pos1 = self.sim_state.ship.state.position

    if ghost_step:
      #Undo the ship movement.
      self.set_ship(ship_backup)
    
    

    reward = get_reward(cols, pos0, pos1, self.config, self.sim_state)["reward"]
    return {
      "cols": cols,
      "action": action,
      "reward": reward,
    }

  def states_predict(self, states):
    '''
      Returns a model prediction for each of the states, 
      given the currently stored environment data. 
    '''
    obs = build_observation(self) # Normal obs
    n = states.shape[0]

    #Overwrite states
    obs["state"] = states.astype(np.float32)

    #Performance friendly way to pretend to have n fake-duplicates
    #of the image/gimage. (Probably)
    obs["image"] = np.broadcast_to(obs["image"], (n, *obs["image"].shape))
    obs["gimage"] = np.broadcast_to(obs["gimage"], (n, *obs["gimage"].shape))

    return self.model.predict(obs, deterministic=True)[0]

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
      while not cols: #Any collision finishes (reaching the end is also a "collision")
        ship_state = build_observation(self)["state"]
        res = self.model_step()
        total_reward += res["reward"]
        cols = res["cols"]
        #Collect all steps
        step_log.append([ship_state, res["action"], res["reward"]])
        steps += 1

        #Timeout exit, so we don't sample all 
        #num_step_samples from a single ep
        if self.sim_state.steps >= self.config.max_steps:
          break

      #log all steps and result
      log_steps(cur_path / "steps.npz", step_log)
      log_result(cur_path / "result.npz", total_reward, self.sim_state.steps, cols)
      run += 1

  def _get_state_dict(self):
    obs = build_observation(self, True)
    return obs["state"]
