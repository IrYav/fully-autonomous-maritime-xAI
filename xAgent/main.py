#External
import numpy as np

#Internal
from xAgent.Auxilaries.ship_instances import ferry
from xAgent.Controller.controller import InteractiveController, SimulationConfig

config = SimulationConfig(
  boundary = np.array([[-2500, -2500], [2500, 2500]], dtype=float),
  ship_params = ferry()
  )

def main():
  # Initialize
  app = InteractiveController(config)
  app.run()
  return 

main()