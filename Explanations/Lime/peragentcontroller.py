from Controller.agentcontroller import AgentController

from stable_baselines3 import PPO, SAC
from Simulation.Auxilaries.sim_config import SimulationConfig

from dataclasses import dataclass

import numpy as np
import math

@dataclass
class Perturbation:
  '''
    All values are normalized as outlined in, observation.py `build_observation`

    Mini-container for perturbations, 
    heading (radian) gets clipped to the nearest point on the unit circle.

    Heading of (0,0) is undefined but accepted. In that case, the use-case should
    just not be perturbed.
  '''
  ship_x: float
  ship_y: float

  heading_cos: float # Radian
  heading_sin: float # Radian 

  velocity: float
  time_elapsed: float

  end_x: float
  end_y: float

  def __post_init__(self):
    # Auf gültigen Heading projizieren
    theta = math.atan2(self.heading_sin, self.heading_cos)
    self.heading_cos = math.cos(theta)
    self.heading_sin = math.sin(theta)


class PerAgentController(AgentController):
  '''
    Controller for running a trained agent in an environment, as well as
    applying and evaluating `Perturbation`s.
  '''
  def __init__(self, config : SimulationConfig,
               model: PPO | SAC ):
    super().__init__(config, model)

  def apply_perturbation(self, p: Perturbation):

    new_ship = self.sim_state.ship.copy()
    new_ship.state.position = [p.ship_x, p.ship_y]
    new_ship.state.heading = math.atan2(p.heading_sin, p.heading_cos)
    new_ship.state.velocity = p.velocity
    #sim_state stores non-normalized ones.
    self.sim_state.steps = round(p.time_elapsed * self.config.max_steps)
    

  def eval_step_perturbations(self, pers: Perturbation) -> np.ndarray:
    #TODO
    self.basePert = 2
    for per in pers:
      return