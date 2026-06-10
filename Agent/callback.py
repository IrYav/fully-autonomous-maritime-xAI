from stable_baselines3.common.callbacks import BaseCallback

class StatsCallback(BaseCallback):
  def _on_rollout_end(self):
    env = self.training_env.envs[0]

    for key, value in env.resultCounter.items():
      self.logger.record(f"custom/{key}", value)