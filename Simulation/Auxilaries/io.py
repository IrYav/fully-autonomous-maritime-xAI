"""Human-readable persistence for editor scenarios."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from os import PathLike
from pathlib import Path
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from Simulation.Auxilaries.sim_config import SimulationConfig
from Simulation.Environment.agent import Ship, ShipParams, ShipState
from Simulation.Environment.environment import Environment
from Simulation.Visualization.RightHalf.AdminTab.draft_model import (
  AreaObject, DraftShip, EnvironmentDraft, Material,
)


SCHEMA_VERSION = 1


@dataclass
class Run:
  """Decoded contents of one generated sample run."""

  scenario: tuple[Environment, Ship]
  steps: list[tuple[np.ndarray, np.ndarray, float]]
  result: tuple[float, int, list[str]]


def _polygon_to_dict(polygon: Polygon) -> dict[str, list[list[float]]]:
  return {
    "exterior": [[float(x), float(y)] for x, y in polygon.exterior.coords],
    "interiors": [
      [[float(x), float(y)] for x, y in ring.coords]
      for ring in polygon.interiors
    ],
  }


def _polygon_from_dict(data: Any) -> Polygon:
  if not isinstance(data, dict) or "exterior" not in data:
    raise ValueError("A polygon must contain an exterior ring.")
  polygon = Polygon(data["exterior"], data.get("interiors", []))
  if polygon.is_empty or not polygon.is_valid:
    raise ValueError("Scenario contains an invalid polygon.")
  return polygon


def draft_to_dict(draft: EnvironmentDraft) -> dict[str, Any]:
  result: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "objects": [
      {
        "id": obj.id,
        "material": obj.material.name,
        "polygon": _polygon_to_dict(obj.polygon),
      }
      for obj in draft
    ],
    "order": [int(object_id) for object_id in draft.order],
    "ship": None,
  }
  if draft.ship is not None:
    result["ship"] = {
      "state": {
        "position": [float(x) for x in draft.ship.state.position],
        "heading": float(draft.ship.state.heading),
        "velocity": float(draft.ship.state.velocity),
      },
      "params": {
        name: float(getattr(draft.ship.params, name))
        for name in ShipParams.__dataclass_fields__
      },
    }
  return result


def draft_from_dict(data: Any) -> EnvironmentDraft:
  if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
    raise ValueError(f"Unsupported scenario format (expected version {SCHEMA_VERSION}).")

  draft = EnvironmentDraft()
  objects = data.get("objects")
  if not isinstance(objects, list):
    raise ValueError("Scenario objects must be a list.")
  for item in objects:
    try:
      object_id = int(item["id"])
      material = Material[item["material"]]
      polygon = _polygon_from_dict(item["polygon"])
    except (KeyError, TypeError, ValueError) as exc:
      raise ValueError("Scenario contains an invalid object.") from exc
    draft.add(AreaObject(object_id, polygon, material))

  order = data.get("order")
  try:
    order = [int(object_id) for object_id in order]
  except (TypeError, ValueError) as exc:
    raise ValueError("Scenario object order must be a list of IDs.") from exc
  if set(order) != set(draft.objects) or len(order) != len(draft.objects):
    raise ValueError("Scenario object order does not match its objects.")
  draft.order = order

  ship = data.get("ship")
  if ship is not None:
    try:
      state_data = ship["state"]
      params_data = ship["params"]
      state = ShipState(
        np.asarray(state_data["position"], dtype=float),
        float(state_data["heading"]),
        float(state_data["velocity"]),
      )
      params = ShipParams(**{
        field.name: float(params_data[field.name])
        for field in fields(ShipParams)
      })
    except (KeyError, TypeError, ValueError) as exc:
      raise ValueError("Scenario contains an invalid ship.") from exc
    if state.position.shape != (2,):
      raise ValueError("Ship position must contain two coordinates.")
    draft.ship = DraftShip(state, params)
  return draft


def save_env_ship(
    path: str | PathLike[str],
    env: Environment,
    ship: Ship,
) -> None:
  """Save an environment and runtime ship in the editor scenario format."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  draft = EnvironmentDraft.draft_from_env(env, ship)
  with open(path, "w", encoding="utf-8") as scenario_file:
    json.dump(draft_to_dict(draft), scenario_file, indent=2)
    scenario_file.write("\n")

def load_env_ship(
    path: str | PathLike[str],
    config: SimulationConfig,
) -> tuple[Environment, Ship]:
  """Load a scenario file as an environment and runtime ship.

  Scenario files contain the editable geometry and ship, while the boundary
  is supplied by the simulation configuration. The corridor is intentionally
  empty because it is not part of the editor scenario format.
  """
  with open(path, "r", encoding="utf-8") as scenario_file:
    draft = draft_from_dict(json.load(scenario_file))

  if draft.ship is None:
    raise ValueError("Scenario must contain a ship to load an environment and ship.")

  base_env = Environment(
    Polygon(),
    Polygon(),
    draft.ship.state.position.copy(),
    Polygon(),
    config.get_boundary(),
  )
  env = draft.draft_to_env(base_env, update_draft=False)
  return env, draft.ship.as_ship()


def _required_npz_value(
  data: np.lib.npyio.NpzFile, name: str, path: Path,
) -> Any:
  if name not in data:
    raise ValueError(f"Run log {path} is missing '{name}'.")
  return data[name]


def _load_run(run_path: Path, config: SimulationConfig) -> Run:
  scenario_path = run_path / "scenario.json"
  steps_path = run_path / "steps.npz"
  result_path = run_path / "result.npz"
  for artifact in (scenario_path, steps_path, result_path):
    if not artifact.is_file():
      raise FileNotFoundError(f"Run {run_path} is missing '{artifact.name}'.")

  scenario = load_env_ship(scenario_path, config)

  try:
    with np.load(steps_path, allow_pickle=False) as data:
      states = _required_npz_value(data, "states", steps_path)
      actions = _required_npz_value(data, "actions", steps_path)
      rewards = _required_npz_value(data, "rewards", steps_path)
  except (OSError, ValueError) as exc:
    if isinstance(exc, ValueError) and str(exc).startswith("Run log"):
      raise
    raise ValueError(f"Could not read steps log {steps_path}.") from exc

  if states.size == 0:
    states = states.reshape(0, 12)
  if actions.size == 0:
    actions = actions.reshape(0, 2)
  if states.ndim != 2 or states.shape[1] != 12:
    raise ValueError(f"Steps log {steps_path} has invalid observations shape {states.shape}; expected (n, 12).")
  if actions.ndim != 2 or actions.shape[1] != 2:
    raise ValueError(f"Steps log {steps_path} has invalid actions shape {actions.shape}; expected (n, 2).")
  if len(states) != len(actions) or len(states) != len(rewards):
    raise ValueError(f"Steps log {steps_path} has mismatched states, actions, and rewards lengths.")

  steps = [
    (
      state.copy(),
      action.astype(float).copy(),
      float(reward),
    )
    for state, action, reward in zip(states, actions, rewards)
  ]

  try:
    with np.load(result_path, allow_pickle=False) as data:
      reward = _required_npz_value(data, "reward", result_path)
      step_count = _required_npz_value(data, "steps", result_path)
      collisions = _required_npz_value(data, "cols", result_path)
  except (OSError, ValueError) as exc:
    if isinstance(exc, ValueError) and str(exc).startswith("Run log"):
      raise
    raise ValueError(f"Could not read result log {result_path}.") from exc

  try:
    reward = float(np.asarray(reward).item())
    step_count = int(np.asarray(step_count).item())
    collisions = np.asarray(collisions).tolist()
  except (TypeError, ValueError) as exc:
    raise ValueError(f"Result log {result_path} contains invalid values.") from exc
  if not isinstance(collisions, list):
    collisions = [collisions]

  return Run(scenario, steps, (reward, step_count, collisions))


def load_runs(path: str | PathLike[str], config: SimulationConfig) -> Iterator[Run]:
  """Lazily yield decoded runs from a directory created by ``generate_samples``."""
  root = Path(path)
  run_paths = sorted(
    (
      child for child in root.iterdir()
      if child.is_dir() and re.fullmatch(r"run\d+", child.name)
    ),
    key=lambda child: int(child.name[3:]),
  )
  for run_path in run_paths:
    yield _load_run(run_path, config)

def log_steps(path, 
              steps: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
  """Save observation-array, action, and reward step logs to a npz file."""
  states = np.asarray([
    step[0]
    for step in steps
  ], dtype=np.float32)

  actions = np.asarray([
    step[1]
    for step in steps
  ], dtype=np.float32)

  rewards = np.asarray([
    step[2]
    for step in steps
  ], dtype=np.float32)

  np.savez(
    path,
    states=states,
    actions=actions,
    rewards=rewards,
  )

def log_result(path, reward: float, steps: int, cols: list) -> None:
  """Save the result of a simulation run."""
  np.savez(
    path,
    reward=reward,
    steps=steps,
    cols=cols
  )
