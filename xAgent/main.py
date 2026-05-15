#External
import numpy as np
import sys
from PySide6.QtWidgets import QApplication

#Internal
from xAgent.Auxilaries.ship_instances import ferry
from xAgent.Environment.harmonic import HarmonicField
from xAgent.Visualization.vis_environment import EnvironmentViewer
from xAgent.Auxilaries.auxiliaries import SimulationState
import xAgent.Environment.environment as envi
import xAgent.Environment.mesh_prep as mesher_prep
import xAgent.Environment.mesh as mesher
import xAgent.Environment.agent as agent

#Rectangle for agent.
boundary = np.array([[-2500, -2500], [2500, 2500]], dtype=float)
h = np.max(boundary[1] - boundary[0]) / 40 # Mesh Resolution
eps = 0.5 # Affects island simplification before mesher input. Higher means simpler.
ship_class = ferry()

def main():
  # Initialize
  app = SimulationApp()
  app.run()
  return 

class SimulationApp:
  def __init__(self):
    self.app = QApplication(sys.argv)
    self.viewer = EnvironmentViewer()
    self.sim_state = None
    self._setup_buttons()
    self.new_training_instance()

  def new_training_instance(self):
    env = envi.new_environment(boundary, num_islands=15, end_radius= 5 * h)
    mesh_raw = mesher_prep.prepare_domain(env.islands, env.end_poly, boundary, h=h, eps = eps)
    mesh = mesher.build_mesh(mesh_raw)
    bc = HarmonicField.get_boundary_conditions(mesh)
    field = HarmonicField(mesh, bc)
    ship_state = agent.get_start_state(env)
    ship = agent.Ship(ship_state, ship_class)

    self.sim_state = SimulationState(env, mesh, field, ship)
    self.viewer.set_sim_state(self.sim_state)
  
  def run(self):
    sys.exit(self.app.exec())

  def _setup_buttons(self):
    self.viewer.add_main_button("New Instance", self.new_training_instance)
    self.viewer.add_sidebar_button("Simulate Step", self.simulate_step)
    self.viewer.show()

  
  def simulate_step(self):
    ship, env = self.sim_state.ship, self.sim_state.env
    action, dt = self.viewer.get_sim_input()
    state, params = ship.state, ship.params

    traj = agent.compute_trajectory(state, action, params, dt)
    ship_path = agent.get_path(ship.state, traj, ship.params)
    cols = agent.check_for_collision(ship_path, env)
    newstate = agent.apply_trajectory(state, traj)

    if cols:
      if cols[0] == "end":
        print("Reached the goal")
      else:
        print(f"Collided with: {cols}")

    ship.state = newstate

    self.viewer.update_step(ship_path)
main()