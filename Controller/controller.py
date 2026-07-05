from Agent.Curriculum.curriculum import Difficulty
from Agent.Curriculum.curriculum_instance import final_diff
from Agent.Observation.rasterize import get_local_view, get_global_view, update_global_view, update_local_view
from Controller.scenario_build import build_scenario
from Simulation.Auxilaries.auxiliaries import SimulationState
from Simulation.Auxilaries.sim_config import SimulationConfig
import Simulation.Environment.agent as agent
import gmsh

class BaseController:
  def __init__(self, config : SimulationConfig):
    self.config = config
    self.sim_state = None
    self.lview = None
    self.gview = None
    _init_gmsh()
    
  def new_training_instance(self, difficulty: Difficulty = None):
    if difficulty is None:
      difficulty = final_diff()
    #-----------------
    # Simulation Data
    #-----------------
    env, mesh, field = self._get_scenario(difficulty)
    ship_state = agent.get_start_state(env)
    ship = agent.Ship(ship_state, self.config.ship_params)
    self.lview = get_local_view( env, ship, 
                                 self.config.lview_size, self.config.lview_res)
    self.gview = get_global_view(env, ship, 
                                 self.config.boundary, self.config.gview_res, self.config.lview_size)
    self.sim_state = SimulationState(env, mesh, field, ship)

  def _get_scenario(self, difficulty: Difficulty):
    return build_scenario(self.config, difficulty)
  
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