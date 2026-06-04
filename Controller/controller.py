
from dataclasses import dataclass
from math import ceil
import sys
from PySide6.QtWidgets import QApplication
import numpy as np

from Agent.Observation.rasterize import get_local_view
from Simulation.Environment.harmonic import HarmonicField
from Simulation.Visualization.visualization import EnvironmentViewer
from Simulation.Auxilaries.auxiliaries import SimulationState
import Simulation.Environment.environment as envi
import Simulation.Environment.mesh_prep as mesher_prep
import Simulation.Environment.mesh as mesher
import Simulation.Environment.agent as agent

@dataclass
class SimulationConfig:
  '''
    Args:
      boundary: Numpy (2,) array delimiting the area the ship may travel, islands may protrude.
      ship_params: Variables defining the ship instance.
      lview_size: Geometrical size of the local view rasterized to...
      lview_res: Pixel size of the local view. 
      h: optional, mesh resolution. Use `None` to use a boundary-size derived value.
      dt: optional, time that passes per simulation step.
      max_eps: Maximum episodes allowed per training run, 
        `None` generates a plausible value using world size, ships max speed and dt, iff `None`
  '''
  boundary: np.ndarray
  ship_params: agent.ShipParams
  lview_size: np.ndarray
  lview_res: np.ndarray
  h: float | None = None
  dt: float = 5.0
  max_steps: int | None = None
  def __post_init__(self):
    if self.h is None:
      self.h = np.max(self.boundary[1] - self.boundary[0]) / 40
    if self.max_steps == None:
      #Ship gets enough steps to traverse the world multiple times.
      dist = np.max(self.boundary)
      stepLen = self.ship_params.max_velocity * self.dt
      self.max_steps = ceil(20 * dist / stepLen)

class BaseController:
  def __init__(self, config : SimulationConfig):
    self.config = config
    self.sim_state = None
    self.lview = None

  def new_training_instance(self):
    ship_width = self.config.ship_params.width
    ship_len = self.config.ship_params.length
    ship_kappa = self.config.ship_params.max_curvature
    turning_radius = 1 / self.config.ship_params.max_curvature

    #-----------------
    # Simulation Data
    #-----------------
    env = envi.new_environment(
      self.config.boundary, 
      num_islands = 15, 
      end_radius = max(5 * self.config.h, 1.5 * ship_len),
      corridor_width = 2 * ship_width,
      min_border_dist = 1.5 * turning_radius,
      max_curvature = ship_kappa
    )
    mesh_raw = mesher_prep.prepare_domain(env.islands, env.end_poly, self.config.boundary, h=self.config.h)
    mesh = mesher.build_mesh(mesh_raw)
    field = HarmonicField(mesh)
    ship_state = agent.get_start_state(env)
    ship = agent.Ship(ship_state, self.config.ship_params)
    self.lview = get_local_view(env, ship, self.config.lview_size, self.config.lview_res)
    self.sim_state = SimulationState(env, mesh, field, ship)


    
  def simulate_step(self, action: agent.Action, dt):
    ship, env = self.sim_state.ship, self.sim_state.env
    state, params = ship.state, ship.params

    traj = agent.compute_trajectory(state, action, params, dt)
    ship_path = agent.get_path(ship.state, traj, ship.params)
    cols = agent.check_for_collision(ship_path, env)
    newstate = agent.apply_trajectory(state, traj)

    ship.state = newstate
    self.lview = get_local_view(env, ship, self.config.lview_size, self.config.lview_res)
    return ship_path, cols


class InteractiveController(BaseController):
  def __init__(self, config):
    super().__init__(config)
    self.app = QApplication(sys.argv)
    self.viewer = EnvironmentViewer()
    self._setup_buttons()
    self.new_training_instance()

  def new_training_instance(self):
    super().new_training_instance()
    self.viewer.set_sim_state(self.sim_state)
    self.viewer.set_lviews(self.lview)

  def simulate_step(self):
    action, dt = self.viewer.get_sim_input()
    ship_path, cols = super().simulate_step(action, dt)

    if cols:
      if cols[0] == "end":
        print("Reached the goal")
      else:
        print(f"Collided with: {cols}")

    self.viewer.update_step(ship_path)
    self.viewer.set_lviews(self.lview)

  def run(self):
    sys.exit(self.app.exec())

  def _setup_buttons(self):
    self.viewer.add_main_button("New Instance", self.new_training_instance)
    self.viewer.add_sidebar_button("Simulate Step", self.simulate_step)
    self.viewer.show()


#TODO: Might require changes so it can be used for training.
#      Or maybe plain useless, as BaseController might provide everything.
class TrainingController(BaseController):
  pass