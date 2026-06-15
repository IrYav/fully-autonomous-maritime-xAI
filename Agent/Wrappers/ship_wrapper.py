import math

import gymnasium as gym
import numpy as np

from gymnasium import spaces

from Agent.Observation.observation import build_observation, get_posgrad_alignment, get_vectorgrad_alignment
from Agent.Observation.rasterize import get_local_view
from Agent.network import NetworkConfig
from Controller.controller import BaseController
from Simulation.Auxilaries.sim_config import SimulationConfig
from Simulation.Environment.agent import Action

from collections import deque




class ShipEnv(gym.Env):

  metadata = {
    "render_modes": [],
  }

  def __init__(self, config: SimulationConfig):
    super().__init__()


    self.controller = BaseController(config)

    self.sim_config = config
    
    self.difficulty = (0,)

    self.image_size = self.sim_config.lview_res[0]
    self.net_config = get_net_conf(self.sim_config)

    self.grad_align = []
    self.results = deque(maxlen=100)

    self.observation_space = spaces.Dict({

      "image": spaces.Box(
        low=0,
        high=255,
        shape=(3, self.image_size, self.image_size),
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
    self.controller.new_training_instance(*self.difficulty)
    self.steps = 0
    self.grad_align = []
    obs = build_observation(self.controller)

    info = {}

    return obs, info

  # =================================================
  # One RL Step
  # =================================================

  def step(self, action):
    self.steps += 1
    #Translate to Simulation action
    action = Action(*action)

    pos0 = self.controller.sim_state.ship.state.position
    #collisions include benign cases of reaching the end.
    path, collisions = self.controller.simulate_step(action, self.sim_config.dt)
    pos1 = self.controller.sim_state.ship.state.position

    # ---------------------------------------------
    # Build observation
    # ---------------------------------------------

    obs = build_observation(self.controller)

    truncated = self.steps >= self.sim_config.max_steps
    terminated = len(collisions) > 0
    
    if terminated or truncated:
      self._update_counter(collisions)

    #Flat time penalty
    reward = -0.05

    if collisions == []:
      alignment, dist = get_vectorgrad_alignment(self.controller.sim_state, pos0, pos1)
      # How far do we travel, and how good is the alignment?
      progress_reward = alignment * dist
      # Scale progress bonus down with map size.
      progress_reward /= self.sim_config.map_diagonal()
      # Weighting factor for following the harmonic field, for complex maps the 
      # harmonic field "flows" past islands, and hence progress rewards may 
      # sum up to multiples of the following factor, 
      # as a good path may be longer than the diagonal.
      progress_reward *= 50

      # Store gradient quality (for logging)
      self.grad_align.append(alignment)

      reward += progress_reward

    #End is the first entry, iff it's the only entry.
    elif collisions[0] == "end":
      reward = 100 #Reached end
    else:
      reward = -100 #Hard crash

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

    pass


  def _print_ep_info(self, terminated, truncated, collisions):
    if(terminated or truncated):
      grads = np.array(self.grad_align)
      print("-----Episode Over-----")
      if self.steps < 2:
        print("Stopped run instantly.")
      else:
        print(f"Step count: {self.steps} of {self.sim_config.max_steps}")
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

@staticmethod
def get_net_conf(sim_config: SimulationConfig) -> NetworkConfig:
  return NetworkConfig(sim_config.lview_res[0],11,2)