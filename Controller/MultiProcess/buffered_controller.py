import multiprocessing as mp
import queue

from Agent.Curriculum.curriculum import Difficulty
from Controller.controller import BaseController
from Controller.scenario_build import build_scenario
from Controller.MultiProcess.scenario_worker import scenario_worker
from Simulation.Auxilaries.sim_config import SimulationConfig


class BufferedMPController(BaseController):
  """
  Process-based buffered variant of BaseController.

  Has been abandoned as `SubprocVecEnv`s are daemon processes
  and thus can not create workers.
  (daemonic processes cant create their own processes).
  Pretty fast in scenario throughput in benchmarks however.

  Semantics:
  - A set of worker processes continuously prepares complete scenarios
    for the currently active difficulty.
  - _get_scenario(difficulty) prefers buffered results, discards outdated
    results, and falls back to a local synchronous build if nothing suitable
    is currently available.
  """

  def __init__(
    self,
    config: SimulationConfig,
    difficulty: Difficulty,
    buffer_size: int = 16,
    num_workers: int = 2,
  ):
    super().__init__(config)

    self.buffer = mp.Queue(maxsize=buffer_size)
    self._free_slots = mp.BoundedSemaphore(buffer_size)

    self._manager = mp.Manager()
    self._shared_state = self._manager.Namespace()
    self._shared_state.current_difficulty = difficulty

    self._stop_event = mp.Event()
    self._workers: list[mp.Process] = []
    self.buffer_hits = 0
    self.buffer_misses = 0
    self.buffer_discards = 0
    for i in range(num_workers):
      p = mp.Process(
        target=scenario_worker,
        args=(
          config,
          self._shared_state,
          self.buffer,
          self._free_slots,
          self._stop_event,
        ),
        daemon=True,
        name=f"scenario-worker-{i}",
      )
      p.start()
      self._workers.append(p)

  def close(self):
    """
    Best-effort shutdown.
    Running builds may still finish in the worker before the process exits.
    """
    self._stop_event.set()

    for p in self._workers:
      p.join(timeout=1.0)
    print("Hits, misses, discards:",
          self.buffer_hits,
          self.buffer_misses,
          self.buffer_discards)
    # Optional cleanup. Safe to keep even if not strictly required.
    self.buffer.close()
    self.buffer.join_thread()
    self._manager.shutdown()
    super().close()
    
  def set_difficulty(self, difficulty: Difficulty):
    self._shared_state.current_difficulty = difficulty

  def _build_scenario(self, difficulty: Difficulty):
    """
    Local synchronous fallback build in the main process.
    """
    return build_scenario(self.config, difficulty)

  def _get_scenario(self, difficulty: Difficulty):
    """
    Prefer buffered scenarios for the requested difficulty.
    Outdated scenarios are discarded.
    If the buffer is empty, build synchronously in the main process.
    """
    self.set_difficulty(difficulty)

    while True:
      try:
        item_diff, env, mesh, field = self.buffer.get_nowait()
      except queue.Empty:
        self.buffer_misses += 1
        return self._build_scenario(difficulty)

      # We removed one item from the queue, so one buffer slot became free.
      self._free_slots.release()

      if item_diff == difficulty:
        self.buffer_hits += 1
        return env, mesh, field
      self.buffer_discards += 1