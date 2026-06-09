import math

import gymnasium as gym
import numpy as np

from gymnasium import spaces

from Agent.Observation.observation import build_observation
from Agent.Observation.rasterize import get_local_view
from Agent.network import NetworkConfig
from Controller.controller import BaseController, SimulationConfig
from Simulation.Auxilaries.auxiliaries import SimulationState
from Simulation.Environment.agent import Action


class ShipEnv(gym.Env):

  metadata = {
    "render_modes": [],
  }

  def __init__(self, config: SimulationConfig):
    super().__init__()

    self.controller = BaseController(config)

    self.sim_config = config
    
    self.image_size = self.sim_config.lview_res[0]
    self.net_config = get_net_conf(self.sim_config)

    self.grad_align = []
    
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
    self.controller.new_training_instance()
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

    #collisions include benign cases of reaching the end.
    path, collisions = self.controller.simulate_step(action, self.sim_config.dt)

    # ---------------------------------------------
    # Build observation
    # ---------------------------------------------

    obs = build_observation(self.controller)

    truncated = self.steps >= self.sim_config.max_steps
    terminated = len(collisions) > 0

    reward = -0.05

    if collisions == []:
      if truncated:
        reward = 0
      else:
        # Store gradient quality
        alignment = self._get_grad_alignment(self.controller.sim_state)
        self.grad_align.append(alignment)
        #Time penalty gets alleviated, iff we go towards the goal. 
        reward *= 1 - 1.5 * max(alignment, 0)
    #End is the first entry, iff it's the only entry.
    elif collisions[0] == "end":
      reward = 100
    else:
      reward = -100



    if(terminated or truncated):
      grads = np.array(self.grad_align)
      print("-----Episode Over-----")
      if self.steps < 2:
        print("Stopped run instantly.")
      else:
        print(f"Step count: {self.steps} of {self.sim_config.max_steps}")
        print(f"Alignment mean: {grads.mean()}")
      print(f"Collisions: {collisions}")

    info = {}

    return (
      obs,
      reward,
      terminated,
      truncated,
      info,
    )

  def _get_grad_alignment(self, sim_state: SimulationState) -> float:
    heading = sim_state.ship.state.heading
    pos = sim_state.ship.state.position

    ship_dir = np.array([math.cos(heading), math.sin(heading)])
    local_grad = sim_state.field.gradient_at_(pos)

    reward_alignment = np.dot(
        ship_dir,
        local_grad,
    )
    return reward_alignment

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

@staticmethod
def get_net_conf(sim_config: SimulationConfig) -> NetworkConfig:
  return NetworkConfig(sim_config.lview_res[0],11,2)