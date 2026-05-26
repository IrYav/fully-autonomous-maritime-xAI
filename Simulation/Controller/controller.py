
from dataclasses import dataclass
import sys
from PySide6.QtWidgets import QApplication
import numpy as np

from Simulation.Environment.harmonic import HarmonicField
from Simulation.Visualization.vis_environment import EnvironmentViewer
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
      h: optional, mesh resolution. Leave `None` to use a boundary-derived dynamic default value. 
  '''
  boundary: np.ndarray
  ship_params: agent.ShipParams
  h: float | None = None

  def __post_init__(self):
    if self.h is None:
      self.h = np.max(self.boundary[1] - self.boundary[0]) / 40

class BaseController:
  def __init__(self, config):
    self.config = config
    self.app = QApplication(sys.argv)
    self.sim_state = None

  def new_training_instance(self):
    env = envi.new_environment(self.config.boundary, num_islands=15, end_radius= 5 * self.config.h)
    mesh_raw = mesher_prep.prepare_domain(env.islands, env.end_poly, self.config.boundary, h=self.config.h)
    mesh = mesher.build_mesh(mesh_raw)
    bc = HarmonicField.get_boundary_conditions(mesh)
    field = HarmonicField(mesh, bc)
    ship_state = agent.get_start_state(env)
    ship = agent.Ship(ship_state, self.config.ship_params)

    self.sim_state = SimulationState(env, mesh, field, ship)
    
  def simulate_step(self, action, dt):
    ship, env = self.sim_state.ship, self.sim_state.env
    state, params = ship.state, ship.params

    traj = agent.compute_trajectory(state, action, params, dt)
    ship_path = agent.get_path(ship.state, traj, ship.params)
    cols = agent.check_for_collision(ship_path, env)
    newstate = agent.apply_trajectory(state, traj)

    ship.state = newstate
    return ship_path, cols


class InteractiveController(BaseController):
  def __init__(self, config):
    super().__init__(config)
    self.viewer = EnvironmentViewer()
    self._setup_buttons()
    self.new_training_instance()

  def new_training_instance(self):
    super().new_training_instance()
    self.viewer.set_sim_state(self.sim_state)

  def simulate_step(self):
    action, dt = self.viewer.get_sim_input()
    ship_path, cols = super().simulate_step(action, dt)

    if cols:
      if cols[0] == "end":
        print("Reached the goal")
      else:
        print(f"Collided with: {cols}")

    self.viewer.update_step(ship_path)

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