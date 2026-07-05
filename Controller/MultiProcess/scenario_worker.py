import queue

from Controller.scenario_build import build_scenario


def scenario_worker(
  config,
  shared_state,
  scenario_queue,
  free_slots,
  stop_event,
):
  """
  Background producer process.

  Semantics:
  - Wait until a free buffer slot is available.
  - Read current difficulty.
  - Build one complete scenario.
  - Push (difficulty, env, mesh, field) into the queue.
  """

  while not stop_event.is_set():
    # Do not block forever, so shutdown can be noticed without fake releases.
    acquired = free_slots.acquire(timeout=0.1)
    if not acquired:
      continue

    # We may have been woken shortly before shutdown.
    if stop_event.is_set():
      free_slots.release()
      break

    difficulty = shared_state.current_difficulty
    scenario = build_scenario(config, difficulty)
    item = (difficulty, *scenario)

    try:
      scenario_queue.put_nowait(item)
    except queue.Full:
      # Should not happen if free_slots and queue stay consistent.
      free_slots.release()