import numpy as np

from Simulation.Auxilaries.ship_instances import ferry
from Controller.controller import InteractiveController, SimulationConfig

config = SimulationConfig(
  boundary = np.array([[-2500, -2500], [2500, 2500]], dtype=float),
  ship_params = ferry(),
  lview_size = np.array([250,250]),
  lview_res = np.array([32,32])
  )

def main():
  # Initialize
  app = InteractiveController(config)
  app.run()
  return

if __name__ == "__main__":
  main()