from stable_baselines3 import PPO, SAC

from Simulation.Auxilaries.sim_config import sim_config
from Controller.interactive_controller import InteractiveController
from pathlib import Path

ship = "PPO"
ship = "SAC"


def main():
  try:
    if ship == "PPO":
      model_path = next(Path(".").glob("ppo_ship*.zip"))
      model = PPO.load(model_path)
    elif ship == "SAC":
      model_path = next(Path(".").glob("sac_ship*.zip"))
      model = SAC.load(model_path)
    else:
      print("Need to select a ship to load.")
      exit()
  except FileNotFoundError:
    model = None
  # Initialize
  app = InteractiveController(sim_config, model)
  app.run()
  app.close()
  return

if __name__ == "__main__":
  main()