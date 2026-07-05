from stable_baselines3 import PPO

from Simulation.Auxilaries.sim_config import sim_config
from Controller.interactive_controller import InteractiveController
from pathlib import Path

def main():
  try:
    model_path = next(Path(".").glob("ppo_ship*.zip"))
    model = PPO.load(model_path)
  except FileNotFoundError:
    model = None
  # Initialize
  app = InteractiveController(sim_config, model)
  app.run()
  app.close()
  return

if __name__ == "__main__":
  main()