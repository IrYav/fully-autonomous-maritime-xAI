#External
import numpy as np

#Internal
from Simulation.Auxilaries.ship_instances import ferry
from Simulation.Controller.controller import InteractiveController, SimulationConfig

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