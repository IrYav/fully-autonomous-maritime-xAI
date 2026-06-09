from stable_baselines3 import PPO

from Simulation.Auxilaries.sim_config import sim_config
from Controller.controller import InteractiveController

def main():
  try:
    model = PPO.load("ppo_ship")
  except FileNotFoundError:
    model = None
  # Initialize
  app = InteractiveController(sim_config, model)
  app.run()
  return

if __name__ == "__main__":
  main()