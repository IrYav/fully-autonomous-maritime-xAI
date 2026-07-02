
from dataclasses import dataclass
from math import ceil
import sys
from PySide6.QtWidgets import QApplication
import numpy as np
from stable_baselines3 import PPO
from PySide6.QtCore import QTimer

from Agent.Observation.observation import build_observation, get_posgrad_alignment
from Agent.Observation.rasterize import get_local_view, get_global_view, update_global_view, update_local_view
from Simulation.Environment.harmonic import HarmonicField
from Simulation.Visualization.visualization import EnvironmentViewer
from Simulation.Auxilaries.auxiliaries import SimulationState
from Simulation.Auxilaries.sim_config import SimulationConfig
import Simulation.Environment.environment as envi
import Simulation.Environment.mesh_prep as mesher_prep
import Simulation.Environment.mesh as mesher
import Simulation.Environment.agent as agent
from enum import Enum, auto

#Used for passively looping simulation. 
class RunMode(Enum):
  MANUAL = auto()
  POLICY = auto()

class BaseController:
  def __init__(self, config : SimulationConfig):
    self.config = config
    self.sim_state = None
    self.lview = None
    self.gview = None
  def new_training_instance(self, num_islands = 15, scenario = None):
    ship_width = self.config.ship_params.width
    ship_len = self.config.ship_params.length
    ship_kappa = self.config.ship_params.max_curvature
    turning_radius = 1 / self.config.ship_params.max_curvature
    #-----------------
    # Simulation Data
    #-----------------
    env = envi.new_environment(
      self.config.boundary, 
      num_islands = num_islands, 
      end_radius = max(5 * self.config.h, 1.5 * ship_len),
      corridor_width = 2 * ship_width,
      min_border_dist = 1.5 * turning_radius,
      max_curvature = ship_kappa,
      scenario = scenario,
    )
    mesh_raw = mesher_prep.prepare_domain(env.islands, env.end_poly, self.config.boundary, h=self.config.h)
    mesh = mesher.build_mesh(mesh_raw)
    field = HarmonicField(mesh)
    ship_state = agent.get_start_state(env)
    ship = agent.Ship(ship_state, self.config.ship_params)
    self.lview = get_local_view( env, ship, 
                                 self.config.lview_size, self.config.lview_res)
    self.gview = get_global_view(env, ship, 
                                 self.config.boundary, self.config.gview_res, self.config.lview_size)
    self.sim_state = SimulationState(env, mesh, field, ship)


  from line_profiler import profile
  @profile  
  def simulate_step(self, action: agent.Action, dt):
    ship, env = self.sim_state.ship, self.sim_state.env
    state, params = ship.state, ship.params

    traj = agent.compute_trajectory(state, action, params, dt)
    ship_path = agent.get_path(ship.state, traj, ship.params)
    cols = agent.check_for_collision(ship_path, env)
    newstate = agent.apply_trajectory(state, traj)

    ship.state = newstate
    update_local_view(self.lview, env, ship, 
                                self.config.lview_size, self.config.lview_res)
    # update_global_view(self.gview, env, ship, 
    #                             self.config.boundary, self.config.gview_res, self.config.lview_size)
    return ship_path, cols

  
class InteractiveController(BaseController):
  def __init__(self, config: SimulationConfig, model: PPO | None):
    super().__init__(config)
    self.app = QApplication(sys.argv)
    self.viewer = EnvironmentViewer()
    self.model = model
    self.run_timer = QTimer()
    self.run_timer.timeout.connect(self._timer_step)
    self.run_mode = RunMode.MANUAL
    self._setup_buttons()
    self.new_training_instance()

  def new_training_instance(self):
    super().new_training_instance()
    self.viewer.set_sim_state(self.sim_state)
    self.viewer.set_lviews(self.lview)
    self.viewer.set_gviews(self.gview)
    self.viewer.set_state_vector(self._get_state_dict())
  
  def simulate_step(self, action, dt):
    ship_path, cols = super().simulate_step(action, dt)

    if cols:
      if cols[0] == "end":
        print("Reached the goal")
      else:
        print(f"Collided with: {cols}")

    self.viewer.update_step(ship_path)
    self.viewer.set_lviews(self.lview)
    self.viewer.set_gviews(self.gview)
    self.viewer.set_state_vector(self._get_state_dict())
    return ship_path, cols

  def simulate_manual_step(self):
    action, dt = self.viewer.get_sim_input()
    self.simulate_step(action, dt)

  def simulate_policy_step(self):
    if self.model is None:
      raise RuntimeError("Trying to use a non-existing model to generate a step")
      
    obs = build_observation(self, 0)

    action, _ = self.model.predict(obs, deterministic=True)
    action = agent.Action(*action)
    self.simulate_step(action, self.config.dt) # Important, use fixed dt.

  def run(self):
    sys.exit(self.app.exec())
  
  def _toggle_manual_run(self, enabled: bool):
    if enabled:
      self.run_timer.start(100)
    else:
      self.run_timer.stop()

  def _setup_buttons(self):
    run_modes = {"Manual": self._set_manual_mode}
    self.viewer.add_main_button("New Instance", self.new_training_instance)
    self.viewer.add_sidebar_button("Manual Step", self.simulate_manual_step)
    if self.model is not None:
      self.viewer.add_sidebar_button("Policy Step", self.simulate_policy_step)
      run_modes["Policy"] = self._set_policy_mode
    self.viewer.add_sidebar_button("Auto Run",self._toggle_manual_run,True)
    self.viewer.add_radio_group("Run Mode",run_modes)
    self.viewer.show()

  def _timer_step(self):
    if self.run_mode == RunMode.MANUAL:
      self.simulate_manual_step()

    elif self.run_mode == RunMode.POLICY:
      self.simulate_policy_step()

  def _set_manual_mode(self):
    self.run_mode = RunMode.MANUAL

  def _set_policy_mode(self):
    self.run_mode = RunMode.POLICY

  def _get_state_dict(self):
    obs = build_observation(self, 0, True)
    return obs["state"]