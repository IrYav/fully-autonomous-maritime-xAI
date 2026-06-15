from dataclasses import dataclass
from math import ceil

import numpy as np


from Simulation.Auxilaries.ship_instances import ferry
from Simulation.Environment.agent import ShipParams


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
  ship_params: ShipParams
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
    self.verifylview()

    
  def verifylview(self):
    '''
      Verifies is the ship has reasonable amounts of foresight 
      to see a crash, before he loses the ability to adress it.
      
      Or, differently phrased, is it possibly too late to do anything, 
      once we see an island ahead?
    '''
    turn_rad_dist = 1 / self.ship_params.max_curvature # We need enough dist to turn,
    velo_dist = self.ship_params.max_velocity * self.dt # After a time step towards an island
    ship_nose_offset =  self.ship_params.length / 2 #Accounting for ship length
    required_vr = turn_rad_dist + velo_dist + ship_nose_offset 
    view_range = self.lview_size[0] / 2
    if required_vr > view_range:
      print("Ship may not have sufficient local view range allocated:")
      print(f"{turn_rad_dist} for turning.")
      print(f"{velo_dist} for max velocity.")
      print(f"{ship_nose_offset} for max velocity.")
      print("Yielding a required local view range of "
        "and compared to the actual local viewrange:")
      print(f"{required_vr} > {view_range}")
    return

  def map_diagonal(self):
    return np.linalg.norm(
      self.boundary[1] - 
      self.boundary[0]
      )

sim_config = SimulationConfig(
  boundary = np.array([[-2500, -2500], [2500, 2500]], dtype=float),
  ship_params = ferry(),
  lview_size = np.array([600,600]), 
  lview_res = np.array([64,64])
  )
