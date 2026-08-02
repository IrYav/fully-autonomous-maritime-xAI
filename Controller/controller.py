from Agent.Curriculum.curriculum import Difficulty
from Agent.Curriculum.curriculum_instance import final_diff
from Agent.Observation.rasterize import get_local_view, get_global_view, update_global_view, update_local_view
from Controller.scenario_build import build_scenario, build_scenario_from_env
from Simulation.Auxilaries.auxiliaries import SimulationState
from Simulation.Auxilaries.sim_config import SimulationConfig
import Simulation.Environment.agent as agent
import gmsh

from Simulation.Environment.environment import Environment

class BaseController:
  def __init__(self, config : SimulationConfig):
    self.config = config
    self.sim_state = None
    self.lview = None
    self.gview = None
    _init_gmsh()
    
  def new_training_instance(self, 
                            difficulty: Difficulty = None, 
                            pre_made_env: Environment | None = None,
                            pre_made_ship: agent.Ship | None = None):
    
    if difficulty is None:
      difficulty = final_diff()

    #-----------------
    # Simulation Data
    #-----------------
    env, mesh, field = self._get_scenario(difficulty, pre_made_env)
    if pre_made_env is None:
      ship_state = agent.get_start_state(env)
      ship = agent.Ship(ship_state, self.config.ship_params)
    elif pre_made_ship is not None:
      state = pre_made_ship.state
      ship_state = agent.ShipState(
        state.position.copy(), state.heading, 0.0)
      ship = agent.Ship(ship_state, pre_made_ship.params)
    else:
      ship = self.sim_state.ship

    self.lview = get_local_view( env, ship, 
                                 self.config.lview_size, self.config.lview_res)
    self.gview = get_global_view(env, ship, 
                                 self.config.boundary, self.config.gview_res, self.config.lview_size)
    self.sim_state = SimulationState(env, mesh, field, ship, 0)

  def _get_scenario(self, difficulty: Difficulty, pre_made_env: Environment | None = None):
    if pre_made_env is None:
      return build_scenario(self.config, difficulty)
    else:
      return build_scenario_from_env(self.config, pre_made_env)
    
  def simulate_step(self, action: agent.Action, dt):
    ship, env = self.sim_state.ship, self.sim_state.env
    state, params = ship.state, ship.params

    self.sim_state.steps += 1
    traj = agent.compute_trajectory(state, action, params, dt)
    ship_path = agent.get_path(ship.state, traj, ship.params)
    cols = agent.check_for_collision(ship_path, env)
    ship.state = agent.apply_trajectory(state, traj)

    update_local_view(self.lview, env, ship, 
                                self.config.lview_size, self.config.lview_res)
    return ship_path, cols
  
  def close(self):
    '''
      Closes gmsh.
    '''
    gmsh.finalize()

@staticmethod
def _init_gmsh():
  gmsh.initialize()
  gmsh.option.setNumber("General.NumThreads", 2)
  gmsh.option.setNumber("General.Verbosity", 1)
