from Controller.agentcontroller import AgentController

from stable_baselines3 import PPO, SAC
from Simulation.Auxilaries.sim_config import SimulationConfig

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import math
from shapely.affinity import translate

from Simulation.Environment.environment import Environment

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
    self.base_perturbation : Perturbation = None

  def store_base_state(self) -> Perturbation:
    '''Store and return the current simulation state in 8D form.'''
    map_min = np.asarray(self.config.boundary[0], dtype=float)
    map_size = np.asarray(self.config.boundary[1], dtype=float) - map_min

    def normalize_position(position):
      return ((np.asarray(position, dtype=float) - map_min) / map_size) * 2 - 1

    ship_position = normalize_position(self.sim_state.ship.state.position)
    end_position = normalize_position(self.sim_state.env.end_poly.centroid.coords[0])
    heading = self.sim_state.ship.state.heading

    self.base_perturbation = Perturbation(
      ship_x=float(ship_position[0]),
      ship_y=float(ship_position[1]),
      heading_cos=math.cos(heading),
      heading_sin=math.sin(heading),
      velocity=self.sim_state.ship.state.velocity / self.sim_state.ship.params.max_velocity,
      time_elapsed=self.sim_state.steps / self.config.max_steps,
      end_x=float(end_position[0]),
      end_y=float(end_position[1]),
    )
    return self.base_perturbation

  def apply_perturbation(self, p: Perturbation):
    '''Apply an 8D normalized perturbation as a fresh training instance.'''
    map_min = np.asarray(self.config.boundary[0], dtype=float)
    map_size = np.asarray(self.config.boundary[1], dtype=float) - map_min

    def denormalize_position(x, y):
      return (np.array([x, y], dtype=float) + 1) * map_size / 2 + map_min

    ship_position = denormalize_position(p.ship_x, p.ship_y)
    end_position = denormalize_position(p.end_x, p.end_y)
    new_ship = self.sim_state.ship.copy()
    new_ship.state.position = ship_position
    new_ship.state.heading = math.atan2(p.heading_sin, p.heading_cos)
    new_ship.state.velocity = p.velocity * new_ship.params.max_velocity

    old_end_position = np.asarray(
      self.sim_state.env.end_poly.centroid.coords[0], dtype=float
    )
    new_end_poly = translate(
      self.sim_state.env.end_poly,
      xoff=float(end_position[0] - old_end_position[0]),
      yoff=float(end_position[1] - old_end_position[1]),
    )
    new_env = Environment(
      self.sim_state.env.islands,
      self.sim_state.env.corridor,
      self.sim_state.env.start,
      new_end_poly,
      self.sim_state.env.boundary,
    )

    # This rebuilds the mesh, harmonic field, and observation views. It also
    # resets the step counter, so restore the perturbed elapsed time after it.
    self.new_training_instance(
      pre_made_env=new_env,
      pre_made_ship=new_ship,
    )
    self.sim_state.steps = round(p.time_elapsed * self.config.max_steps)
    

  def eval_step_perturbations(
      self, pers: Iterable[Perturbation]
  ) -> np.ndarray:
    '''Evaluate one model step for each perturbation from the same base state.

    Returns an array with columns ``[rudder, acceleration, reward]``.
    '''
    base_state = self.store_base_state()
    
    pers = list(pers)
    result = np.empty((len(pers), 3), dtype=float)

    try:
      for i, per in enumerate(pers):
        # ``apply_perturbation`` rebuilds the training instance, including
        # its environment-dependent mesh and field.
        self.apply_perturbation(per)
        step = self.model_step()
        action = np.asarray(step["action"], dtype=float).reshape(-1)
        result[i] = [action[0], action[1], step["reward"]]

        # Reset before the next perturbation so no ship movement or elapsed
        # time from this evaluation leaks into the following one.
        self.apply_perturbation(base_state)
    finally:
      # Also restore the base state if evaluating a perturbation raises.
      self.apply_perturbation(base_state)

    return result

  @staticmethod
  def _collision_category(cols: list[str]) -> int:
    '''Map the first collision to an int representing
       end, island, boundary, or time-out.'''
    if not cols:
      return 3  # time_out
    if cols[0] == "end":
      return 0
    if cols[0].startswith("island_"):
      return 1
    if cols[0] == "boundary":
      return 2
    raise ValueError(f"Unknown collision type: {cols[0]}")

  def _evaluate_run_perturbations(
      self, pers: Iterable[Perturbation]
  ) -> np.ndarray:
    '''Return run metrics and a numeric collision category for each input.'''
    pers = list(pers)
    base_state = self.store_base_state()
    result = np.empty((len(pers), 3), dtype=float)

    try:
      self.apply_perturbation(base_state)
      base_reward, _, _ = self.finish_episode()
      self.apply_perturbation(base_state)

      for i, per in enumerate(pers):
        self.apply_perturbation(per)
        start_steps = self.sim_state.steps
        reward, stepcount, cols = self.finish_episode()
        result[i] = [
          reward - base_reward,
          stepcount - start_steps,
          self._collision_category(cols),
        ]
        self.apply_perturbation(base_state)
    finally:
      self.apply_perturbation(base_state)

    return result

  def eval_run_perturbations(
      self, pers: Iterable[Perturbation]
  ) -> np.ndarray:
    '''
      Evaluate complete runs from the same base state.

      Returns columns ``[reward_delta, stepcount, collision_category]``.
      Collision categories are ``end=0``, ``island=1``, ``boundary=2``,
      and ``time_out=3``.
    '''
    return self._evaluate_run_perturbations(pers)

  def eval_run_collision_probabilities(
      self, pers: Iterable[Perturbation]
  ) -> np.ndarray:
    '''
      Evaluate runs and return one-hot collision probabilities.

      Columns are ordered ``[end, island, boundary, time_out]``.
    '''
    categories = self._evaluate_run_perturbations(pers)[:, 2].astype(int)
    probabilities = np.zeros((len(categories), 4), dtype=float)
    probabilities[np.arange(len(categories)), categories] = 1.0
    return probabilities

  def eval_run_perturbation_array(
      self, states: np.ndarray
  ) -> np.ndarray:
    '''Array adapter for ``eval_run_perturbations`` with shape ``(n, 8)``.'''
    states = np.asarray(states)

    if states.ndim != 2 or states.shape[1] != 8:
      raise ValueError(
        f"Expected states with shape (n, 8), got {states.shape}."
      )

    return self.eval_run_perturbations(
      Perturbation(*state) for state in states
    )

  def eval_run_collision_probability_array(
      self, states: np.ndarray
  ) -> np.ndarray:
    '''Array adapter for run collision probabilities with shape ``(n, 8)``.'''
    states = np.asarray(states)

    if states.ndim != 2 or states.shape[1] != 8:
      raise ValueError(
        f"Expected states with shape (n, 8), got {states.shape}."
      )

    return self.eval_run_collision_probabilities(
      Perturbation(*state) for state in states
    )

  def eval_step_perturbation_array(
      self, states: np.ndarray
  ) -> np.ndarray:
    '''
      Does conceptually the same as `eval_step_perturbations`, but
      allows the use of an states array of shape (n,8),
      rather than an iterable for perturbations.
    '''
    states = np.asarray(states)

    if states.ndim != 2 or states.shape[1] != 8:
      raise ValueError(
        f"Expected states with shape (n, 8), got {states.shape}."
      )

    return self.eval_step_perturbations(
      Perturbation(*state) for state in states
    )
