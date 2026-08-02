
#Used for passively looping simulation. 
from enum import Enum, auto
import sys

from stable_baselines3 import PPO, SAC

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from Agent.Observation.observation import build_observation, dict_to_obs
from Agent.Wrappers.env_wrapper import get_reward
from Controller.controller import BaseController
from Simulation.Auxilaries.sim_config import SimulationConfig
from Simulation.Environment import agent
from Simulation.Environment.environment import Environment
from Simulation.Visualization.RightHalf.AdminTab.editor_controller import EditorController
from Simulation.Visualization.visualization import EnvironmentViewer


class RunMode(Enum):
  MANUAL = auto()
  POLICY = auto()
  POLICY_STOP = auto()

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

  def new_training_instance(self, env: Environment | None = None, ship=None, new_episode=True):
    pre_made_ship = ship.as_ship() if ship is not None else None
    super().new_training_instance(pre_made_env=env, pre_made_ship=pre_made_ship)
    self.viewer.set_sim_state(self.sim_state, new_episode=new_episode)
    self.viewer.set_lviews(self.lview)
    self.viewer.set_gviews(self.gview)
    self.viewer.set_state_vector(self._get_state_dict())
    self.viewer.set_reward_dict(self._get_reward_dict(True))

  def simulate_step(self, action, dt):
    self.ship_path, self.cols = super().simulate_step(action, dt)

    if self.cols:
      if self.cols[0] == "end":
        print("Reached the goal")
      else:
        print(f"Collided with: {self.cols}")

    self.ec.sync_ship(self.sim_state.ship)
    self.viewer.update_step(self.ship_path)
    self.viewer.set_lviews(self.lview)
    self.viewer.set_gviews(self.gview)
    self.viewer.set_state_vector(self._get_state_dict())
    self.viewer.set_reward_dict(self._get_reward_dict())
    return self.ship_path, self.cols

  def simulate_manual_step(self):
    action, dt = self.viewer.get_sim_input()
    self.simulate_step(action, dt)

  def simulate_policy_step(self):
    if self.model is None:
      raise RuntimeError("Trying to use a non-existing model to generate a step")
      
    obs = build_observation(self) 
    obs_state_dict = self.viewer.get_agent_observation()
    obs["state"] = dict_to_obs(obs_state_dict)
    action, _ = self.model.predict(obs, deterministic=True)
    action = agent.Action(*action)
    _, colls = self.simulate_step(action, self.config.dt) # Important, use fixed dt.
    if not colls == [] and self.run_mode is RunMode.POLICY_STOP:
      self.auto_run.toggle()

  def run(self):
    sys.exit(self.app.exec())
  
  def set_auto_run(self, enabled: bool):
    if enabled:
      self.run_timer.start(50)
    else:
      self.run_timer.stop()

  def _setup_buttons(self):
    run_modes = {"Manual": self._set_manual_mode}
    self.viewer.add_main_button("New Instance", self.new_training_instance)
    self.viewer.add_sidebar_button("Manual Step", self.simulate_manual_step)
    if self.model is not None:
      self.viewer.add_sidebar_button("Policy Step", self.simulate_policy_step)
      run_modes["Policy"] = self._set_policy_mode
      run_modes["Policy Stop"] = self._set_policy_stop_mode
    self.auto_run = self.viewer.add_sidebar_button("Auto Run",self.set_auto_run,True)
    self.viewer.add_radio_group("Run Mode",run_modes)
    self.viewer.show()

  def _timer_step(self):
    if self.run_mode is RunMode.MANUAL:
      self.simulate_manual_step()

    elif self.run_mode in [RunMode.POLICY, RunMode.POLICY_STOP]:
      self.simulate_policy_step()

  def _set_manual_mode(self):
    self.run_mode = RunMode.MANUAL

  def _set_policy_mode(self):
    self.run_mode = RunMode.POLICY

  def _set_policy_stop_mode(self):
    self.run_mode = RunMode.POLICY_STOP

  def _get_state_dict(self):
    obs = build_observation(self, True)
    return obs["state"]

  def _get_reward_dict(self, first_call = False):
    if first_call:
      self.res = {
        "reward" : "Undefined",
        "time_penalty" : "Undefined",
        "progress_reward" : "Undefined",
        "finished" : "Undefined",
        "crashed" : "Undefined",
        "total_reward" : 0,
      }
      
    else:
      curr_pos = self.sim_state.ship.state.position
      self.res.update(get_reward(
          self.cols,
          self.old_pos, 
          curr_pos,
          self.config,
          self.sim_state,
        )
      )
      self.res["total_reward"] += self.res["reward"]

    #For next call.
    self.old_pos = self.sim_state.ship.state.position
    return self.res
