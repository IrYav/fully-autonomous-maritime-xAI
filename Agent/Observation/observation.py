import math

import numpy as np

from Agent.Observation.rasterize import get_local_view
from Simulation.Auxilaries.auxiliaries import SimulationState


def build_observation(controller):
  #Global View:
  #TODO

  #Local View:
  env = controller.sim_state.env
  ship = controller.sim_state.ship
  sim_config = controller.config
  size, res = sim_config.lview_size, sim_config.lview_res
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

def get_posgrad_alignment(sim_state: SimulationState) -> float:
    heading = sim_state.ship.state.heading
    pos = sim_state.ship.state.position

    ship_dir = np.array([math.cos(heading), math.sin(heading)])
    local_grad = sim_state.field.gradient_at_(pos)

    reward_alignment = np.dot(
        ship_dir,
        local_grad,
    )
    return reward_alignment

def get_vectorgrad_alignment(sim_state: SimulationState, pos0, pos1) -> float:
    delta_pos = pos1 - pos0

    dist = np.linalg.norm(delta_pos)
    if dist == 0:
        return 0.0

    move_dir = delta_pos / dist

    grad_dir = sim_state.field.gradient_at_(pos0)

    return np.dot(move_dir, grad_dir), dist