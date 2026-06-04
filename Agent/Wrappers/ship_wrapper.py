import math

import gymnasium as gym
import numpy as np

from gymnasium import spaces

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
    
    image_size = self.sim_config.lview_res[0]
    self.net_config = NetworkConfig(image_size,11,2)

    self.grad_align = []
    
    self.observation_space = spaces.Dict({

      "image": spaces.Box(
        low=0,
        high=1,
        shape=(3, image_size, image_size),
        dtype=np.float32,
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
    obs = self._get_obs()

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

    obs = self._get_obs()

    reward = -0.05
    if collisions == []:
      # Store gradient quality
      alignment = self._get_grad_alignment(self.controller.sim_state)
      self.grad_align.append(alignment)
      #Time penalty gets alleviated, iff we go into 
      reward *= 1 - 0.5 * max(alignment, 0)
    #End is the first entry, iff it's the only entry.
    elif collisions[0] == "end":
      reward = 100
    else:
      reward = -100

    truncated = self.steps >= self.sim_config.max_steps
    terminated = len(collisions) > 0

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

  # =================================================
  # Observation Construction
  # =================================================

  def _get_obs(self):
    #Global View:
    #TODO

    #Local View:
    env = self.controller.sim_state.env
    ship = self.controller.sim_state.ship
    size, res = self.sim_config.lview_size, self.sim_config.lview_res
    local_view = get_local_view(env, ship, size, res)

    #State Vector:
    # TODO:
    end_pos = np.asarray(
        env.end_poly.centroid.coords[0]
    )
    ship_pos = ship.state.position

    heading = ship.state.heading

    rel_pos = end_pos - ship_pos
    rel_angle = (
        math.atan2(rel_pos[1], rel_pos[0])
        - heading
    )

    state_obs = np.array([

        *ship_pos,

        math.cos(heading),
        math.sin(heading),

        ship.state.velocity,

        *rel_pos,

        math.cos(rel_angle),
        math.sin(rel_angle),

        *end_pos,
    ], dtype=np.float32)

    return {
      "image": local_view,
      "state": state_obs,
    }

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