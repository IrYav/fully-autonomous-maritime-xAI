from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter
class StatsCallback(BaseCallback):
  def _on_step(self):
    return True
  
  def _on_rollout_end(self):
    #Attributes collected over all environments.
    results = self.training_env.get_attr("results")
    difficulty = self.training_env.env_method("get_difficulty")
    difficulty_num = difficulty[0][0]
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

    #Curriculum Learning
    new_diff, diff_changed = _pick_difficulty(self.num_timesteps, freq["Success"], difficulty_num)
    if diff_changed:    
      #Set difficulty for all envs
      self.training_env.env_method("clear_results")
      self.training_env.env_method("set_difficulty", new_diff)

    #Record numerical difficulty
    self.logger.record("custom/difficulty",new_diff[0])

#TODO Refactor, this is annoying to code and change
#Probably iterating over two lists or something.
#Probably can work over an auxiliary iterator function, that works via next or successor,
#here it should ideally only do "Level up or not" logic. 
def _pick_difficulty(num_timesteps: int, success_count: int, curr_diff : int) -> tuple[tuple[int, str], bool]:
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