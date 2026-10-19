import queue
import threading

from Agent.Curriculum.curriculum import Difficulty
from Controller.controller import BaseController
from Controller.scenario_build import build_scenario
from Simulation.Auxilaries.sim_config import SimulationConfig


class BufferedMTController(BaseController):
  """
  Thread-based buffered variant of BaseController.

  Intended for use inside SubprocVecEnv workers, where spawning child
  processes from the environment process is not allowed.

  Semantics:
  - A set of background threads continuously prepares complete scenarios
    for the currently active difficulty.
  - _get_scenario(difficulty) prefers buffered results, discards outdated
    results, and falls back to a local synchronous build if nothing suitable
    is currently available.
  - Multiple producer threads are allowed, but gmsh access must be serialized
    elsewhere (see scenario_build / mesh wrapper).
  """

  def __init__(
    self,
    config: SimulationConfig,
    difficulty: Difficulty,
    buffer_size: int = 16,
    num_threads: int = 2,
  ):
    super().__init__(config)

    self.buffer = queue.Queue(maxsize=buffer_size)
    self._free_slots = threading.BoundedSemaphore(buffer_size)

    # Single writer, multiple readers; stale reads are acceptable because
    # outdated scenarios are discarded on consumption.
    self.current_difficulty = difficulty

    self._stop_event = threading.Event()
    self._threads: list[threading.Thread] = []

    self.buffer_hits = 0
    self.buffer_misses = 0
    self.buffer_discards = 0

    for i in range(num_threads):
      t = threading.Thread(
        target=self._worker_loop,
        name=f"scenario-thread-{i}",
        daemon=True,
      )
      t.start()
      self._threads.append(t)

  def close(self):
    """
    Best-effort shutdown.
    Running builds may still finish in a worker thread before it exits.
    """
    self._stop_event.set()

    for t in self._threads:
      t.join()

    #All threads done, close gmsh
    super().close()
    
    print(
      "ThreadBufferedController hits, misses, discards:",
      self.buffer_hits,
      self.buffer_misses,
      self.buffer_discards
    )

  def set_difficulty(self, difficulty: Difficulty):
    self.current_difficulty = difficulty

  def _build_scenario(self, difficulty: Difficulty):
    """
    Local synchronous fallback build in the current thread.
    """
    return build_scenario(self.config, difficulty)

  def _get_scenario(self, difficulty: Difficulty, pre_made_env = None):
    """
    `pre_made_env` is ignored!

    Prefer buffered scenarios for the requested difficulty.
    Outdated scenarios are discarded.
    If the buffer is empty, build synchronously in the current thread.
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

  def _worker_loop(self):
    """
    Background producer thread.

    Semantics:
    - Wait until a free buffer slot is available.
    - Read current difficulty.
    - Build one complete scenario.
    - Push (difficulty, env, mesh, field) into the queue.
    """
    while not self._stop_event.is_set():
      # Do not block forever, so shutdown can be noticed without fake releases.
      acquired = self._free_slots.acquire(timeout=0.1)
      if not acquired:
        continue

      # We may have been woken shortly before shutdown.
      if self._stop_event.is_set():
        self._free_slots.release()
        break

      difficulty = self.current_difficulty
      scenario = build_scenario(self.config, difficulty)
      item = (difficulty, *scenario)

      try:
        self.buffer.put_nowait(item)
      except queue.Full:
        # Should not happen if free_slots and queue stay consistent.
        self._free_slots.release()