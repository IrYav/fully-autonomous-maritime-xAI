import math

import gymnasium as gym
import numpy as np

from gymnasium import spaces

from Agent.Curriculum.curriculum_instance import start_diff
from Agent.Observation.observation import build_observation, get_vectorgrad_alignment
from Agent.Observation.rasterize import get_local_view
from Agent.network import NetworkConfig
from Controller.MultiThread.buffered_controller import BufferedMTController
from Simulation.Auxilaries.auxiliaries import SimulationState
from Simulation.Auxilaries.sim_config import SimulationConfig
from Simulation.Environment.agent import Action

from collections import deque


#How many episodes are relevant for the curriculums difficulty changes.
#May be slightly bigger.
CURRICULUM_WINDOW = 100

class ShipEnv(gym.Env):

  metadata = {
    "render_modes": [],
  }

  def __init__(self, config: SimulationConfig):
    super().__init__()


    self.controller = BufferedMTController(config, start_diff(),4,2)

    self.sim_config = config
    
    self.difficulty = start_diff()

    self.image_size = self.sim_config.lview_res[0]
    self.gimage_size = self.sim_config.gview_res[0]

    self.net_config = get_net_conf(self.sim_config)

    self.grad_align = []
    self.results = deque(maxlen=(CURRICULUM_WINDOW + config.n_envs - 1) // config.n_envs)

    self.observation_space = spaces.Dict({

      "image": spaces.Box(
        low=0,
        high=255,
        shape=(3, self.image_size, self.image_size),
        dtype=np.uint8,
      ),

      "gimage": spaces.Box(
        low=0,
        high=255,
        shape=(2, self.gimage_size, self.gimage_size),
        dtype=np.uint8,
      ),

      "state": spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(self.net_config.vector_size,),
        dtype=np.float32,
      ),
    })

    self.action_space = spaces.Box(
      low=-1.0,
      high=1.0,
      shape=(self.net_config.num_actions,),
      dtype=np.float32,
    )

  # =================================================
  # Reset Episode
  # =================================================

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.controller.new_training_instance(self.difficulty)
    self.grad_align = []
    obs = build_observation(self.controller)

    info = {}

    return obs, info

  # =================================================
  # One RL Step
  # =================================================

  def step(self, action):
    steps = self.controller.sim_state.steps
    #Translate to Simulation action
    action = Action(*action)

    pos0 = self.controller.sim_state.ship.state.position
    #collisions include benign cases of reaching the end.
    _, collisions = self.controller.simulate_step(action, self.sim_config.dt)
    pos1 = self.controller.sim_state.ship.state.position

    # ---------------------------------------------
    # Build observation
    # ---------------------------------------------

    obs = build_observation(self.controller)

    truncated = steps >= self.sim_config.max_steps
    terminated = len(collisions) > 0
    
    if terminated or truncated:
      self._update_counter(collisions)

    reward = get_reward(collisions,pos0,pos1,
                        self.sim_config,self.controller.sim_state)["reward"]

    info = {}

    return (
      obs,
      reward,
      terminated,
      truncated,
      info,
    )

  # =================================================
  # Optional Rendering
  # =================================================

  def render(self):

    pass

  # =================================================
  # Cleanup
  # =================================================

  def close(self):
    self.controller.close()


  def _print_ep_info(self, terminated, truncated, collisions):
    steps = self.controller.sim_state.steps
    if(terminated or truncated):
      grads = np.array(self.grad_align)
      print("-----Episode Over-----")
      if steps < 2:
        print("Stopped run instantly.")
      else:
        print(f"Step count: {steps} of {self.sim_config.max_steps}")
        print(f"Alignment mean: {grads.mean()}")
      print(f"Collisions: {collisions}")

  def _update_counter(self, collisions):
    if collisions == []:
      self.results.append("Truncated")
    elif collisions[0] == "end":
      self.results.append("Success")
    elif collisions[0][0:6] == "island":
      self.results.append("Crashed into island")
    else:
      self.results.append("Left map")

  def clear_results(self):
    self.results.clear()

  def set_difficulty(self, difficulty):
    self.difficulty = difficulty

  def get_difficulty(self):
    return self.difficulty

@staticmethod
def get_net_conf(sim_config: SimulationConfig) -> NetworkConfig:
  return NetworkConfig(sim_config.lview_res[0],12,2)

@staticmethod
def get_reward(
  collisions: list[str], 
  pos0, 
  pos1, 
  sim_config: SimulationConfig, 
  sim_state: SimulationState) -> dict[str, float]:
  '''
    How much reward do we get for the current step?  
  '''
  res = {
    "reward" : 0,
    "time_penalty" : 0,
    "progress_reward" : 0,
    "finished" : 0,
    "crashed" : 0,
  }

  #Linear time penalty per step.
  reward = -0.15 * sim_state.steps/sim_config.max_steps
  res["time_penalty"] = reward

  if collisions == []:
    alignment, dist = get_vectorgrad_alignment(sim_state, pos0, pos1)
    
    # Alleviate reward loss for minor deviations from a good trajectory
    # Increase reward loss for minor deviations from a bad trajectory
    #alignment = math.copysign(abs(alignment) ** 0.75, alignment)

    # How far do we travel, and how good is the alignment?
    progress_reward = alignment * dist
    # Scale progress bonus down with map size.
    progress_reward /= sim_config.map_diagonal
    # Weighting factor for following the harmonic field, for complex maps the 
    # harmonic field "flows" past islands, and hence progress rewards may 
    # sum up to multiples of the following factor, 
    # as a good path may be longer than the diagonal.
    progress_reward *= 400
    res["progress_reward"] = progress_reward
    reward += progress_reward

  #End is the first entry, iff it's the only entry.
  elif collisions[0] == "end":
    reward = 100 #Reached end
    res["finished"] = reward
  else:
    reward = -150 #Hard crash
    res["crashed"] = reward

  res["reward"] = reward

  return res