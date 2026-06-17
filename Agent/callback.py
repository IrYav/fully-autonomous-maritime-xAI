from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter
class StatsCallback(BaseCallback):
  def _on_step(self):
    return True
  
  def _on_rollout_end(self):
    env = self.training_env.envs[0].unwrapped

    #Logging
    outcomes = ["Truncated", 
                "Success", 
                "Crashed into island", 
                "Left map"]
    
    freq = Counter(env.results)
    n = len(env.results)
    for outcome in outcomes:
      self.logger.record(
        f"custom/{outcome}",
        freq[outcome] / n if n > 0 else 0
      )

    #Curriculum Learning
    difficulty, new_diff = _get_difficulty(self.num_timesteps, freq["Success"], env.difficulty[0])
    if new_diff: 
      freq.clear()
      env.results.clear()

    self.logger.record("custom/difficulty",difficulty[0])

    for env in self.training_env.envs:
      env.unwrapped.difficulty = difficulty

  # def _on_training_start(self):
  #   difficulty = _get_difficulty(self.num_timesteps)

  #   for env in self.training_env.envs:
  #     env.unwrapped.difficulty = difficulty

#TODO Refactor, this is annoying to code and change
#Probably iterating over two lists or something.
#Probably can work over an auxiliary iterator function, that works via next or successor,
#here it should ideally only do "Level up or not" logic. 
def _get_difficulty(num_timesteps: int, success_count: int, curr_diff : int) -> tuple[tuple[int, str], bool]:
  if curr_diff == 0: 
    if success_count < 95:
      diff = 0, None
    else:
      diff = 1, "blocking"
  elif curr_diff == 1:
    if success_count < 90:
      diff = 1,"blocking"
    else:
      diff = 2, None
  else:
    if success_count < 80 or curr_diff == 15: #Too bad or max diff
      diff = curr_diff, None
    else:
      diff = curr_diff + 1, None

  return diff, diff[0] != curr_diff