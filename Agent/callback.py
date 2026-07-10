from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter
from Agent.Curriculum.curriculum import Difficulty
from Agent.Curriculum.curriculum_instance import build_curriculum

class StatsCallback(BaseCallback):
  def __init__(self, verbose: int = 0):
    super().__init__(verbose)
    self.curriculum = build_curriculum()
    self.difficulty = self.curriculum.get_start_diff()

  def _on_step(self):
    return True
  
  def _on_rollout_end(self):
    #Attributes collected over all environments.
    results = self.training_env.get_attr("results")

    #Logging
    outcomes = ["Truncated", 
                "Success", 
                "Crashed into island", 
                "Left map"]
    
    #Frequencies of certain events (e.g, Success) collected.
    all_results = []

    for env_results in results:
        all_results.extend(env_results)

    freq = Counter(all_results)
    n = len(all_results)

    for outcome in outcomes:
      self.logger.record(
        f"custom/{outcome}",
        freq[outcome] / n if n > 0 else 0
      )

    #If we have enough samples and a large friction of them is successful, we graduate.
    if n >= 100 and self.curriculum.passed(self.difficulty, freq["Success"] / n):
      self.difficulty = self.curriculum.next_difficulty(self.difficulty)
      #Set difficulty for all envs
      self.training_env.env_method("clear_results")
      self.training_env.env_method("set_difficulty", self.difficulty)
      #SAC receives a distribution shock upon curriculum change.
      #By emptying the buffer, we ensure that only up-to-date information
      #gets processed into updates.
      #self.model.replay_buffer.reset()
    #Record numerical difficulty
    self.logger.record("custom/difficulty",self.difficulty.index)
    self.logger.record("custom/num_results", n)