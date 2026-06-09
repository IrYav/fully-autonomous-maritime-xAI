import numpy as np

from Controller.controller import SimulationConfig
from Simulation.Auxilaries.ship_instances import ferry


sim_config = SimulationConfig(
  boundary = np.array([[-2500, -2500], [2500, 2500]], dtype=float),
  ship_params = ferry(),
  lview_size = np.array([300,300]), 
  lview_res = np.array([64,64])
  )
