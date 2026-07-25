
#Used for passively looping simulation. 
from enum import Enum, auto
import sys

from stable_baselines3 import PPO, SAC

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from Agent.Observation.observation import build_observation
from Controller.controller import BaseController
from Simulation.Auxilaries.sim_config import SimulationConfig
from Simulation.Environment import agent
from Simulation.Environment.environment import Environment
from Simulation.Visualization.RightHalf.AdminTab.editor_controller import EditorController
from Simulation.Visualization.visualization import EnvironmentViewer


class RunMode(Enum):
  MANUAL = auto()
  POLICY = auto()

class InteractiveController(BaseController):
  def __init__(self, config: SimulationConfig, model: PPO | SAC | None):

    super().__init__(config)
    self.app = QApplication(sys.argv)
    self.ec = EditorController(self.new_training_instance)
    self.viewer = EnvironmentViewer(self.ec)
    self.model = model
    self.run_timer = QTimer()
    self.run_timer.timeout.connect(self._timer_step)
    self.run_mode = RunMode.MANUAL
    self._setup_buttons()
    self.new_training_instance()

  def new_training_instance(self, env: Environment | None = None):
    super().new_training_instance(pre_made_env=env)
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
      
    obs = build_observation(self)

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
    obs = build_observation(self, True)
    return obs["state"]