from __future__ import annotations

import math

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from Controller.controller import BaseController

from Agent.Observation.rasterize import get_global_view, get_local_view
from Simulation.Auxilaries.auxiliaries import SimulationState
from Simulation.Auxilaries.sim_config import SimulationConfig


def build_observation(controller : BaseController, debugging=False):
  steps = controller.sim_state.steps
  env = controller.sim_state.env
  ship = controller.sim_state.ship 
  sim_config = controller.config

  local_view = controller.lview
  global_view = controller.gview

  map_min = sim_config.boundary[0]
  map_size = sim_config.boundary[1] - sim_config.boundary[0]

  #--Normalizing the State Vector--
  # Normalized to [-1, 1]
  end_pos_norm = np.asarray(
    (env.end_poly.centroid.coords[0] - map_min) / map_size
  ) * 2 - 1

  # Normalized to [-1, 1]
  ship_pos_norm = (
    (ship.state.position - map_min) / map_size
  ) * 2 - 1

  heading = ship.state.heading

  rel_pos_norm = (end_pos_norm - ship_pos_norm) / 2

  rel_angle = (
    math.atan2(rel_pos_norm[1], rel_pos_norm[0])
    - heading
  )
  
  #--Constructing the observation--
  state_dict = {
    "ship_x_norm": ship_pos_norm[0],
    "ship_y_norm": ship_pos_norm[1],

    "heading_cos": math.cos(heading),
    "heading_sin": math.sin(heading),
    # Normalized to [-1 * k, 1] with k = max_rev_vel/max_vel. 
    # 0 meaning no movement is more important.
    "velocity_norm": ship.state.velocity / ship.params.max_velocity,

    "time_elapsed_norm": steps/sim_config.max_steps,

    "rel_x_norm": rel_pos_norm[0],
    "rel_y_norm": rel_pos_norm[1],

    "rel_angle_cos": math.cos(rel_angle),
    "rel_angle_sin": math.sin(rel_angle),

    "end_x_norm": end_pos_norm[0],
    "end_y_norm": end_pos_norm[1],
  }
  if not debugging:
    state_obs = np.array(list(state_dict.values()), dtype=np.float32)
  else:
    state_obs = state_dict
  return {
    "image": local_view,
    "gimage" : global_view,
    "state": state_obs,
  }

def dict_to_obs(d: dict[str, float]):
  '''
    Turning a dict of observations into an array, of correct order.
  '''
  return np.array([
    d["ship_x_norm"],
    d["ship_y_norm"],
    d["heading_cos"],
    d["heading_sin"],
    d["velocity_norm"],
    d["time_elapsed_norm"],
    d["rel_x_norm"],
    d["rel_y_norm"],
    d["rel_angle_cos"],
    d["rel_angle_sin"],
    d["end_x_norm"],
    d["end_y_norm"],
  ])

def get_posgrad_alignment(sim_state: SimulationState) -> float:
  '''
  Alignment between the ship's heading and the local potential field gradient.
  Ignores the actual movement of the ship.
  '''
  heading = sim_state.ship.state.heading
  pos = sim_state.ship.state.position

  ship_dir = np.array([math.cos(heading), math.sin(heading)])
  local_grad = sim_state.field.gradient_at_(pos)

  reward_alignment = np.dot(
    ship_dir,
    local_grad,
  )
  return reward_alignment

def get_vectorgrad_alignment(sim_state: SimulationState, pos0, pos1) -> tuple[float,float]:
  '''
    How much does a movement from `pos0` to `pos1` align with the 
    direction of the gradient at `pos0`?
    
    And how much distance did it cover?
  '''
  delta_pos = pos1 - pos0

  dist = np.linalg.norm(delta_pos)
  if dist == 0:
    return 0.0, 0.0

  move_dir = delta_pos / dist

  grad_dir = sim_state.field.gradient_at_(pos0)

  return np.dot(move_dir, grad_dir), dist