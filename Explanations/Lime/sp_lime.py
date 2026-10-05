"""Candidate-aware SP-LIME for run-level explanations.

The bundled :class:`SubmodularPick` implementation assumes that every
candidate is evaluated by the same ``predict_fn``.  That is not suitable for
the run samples used here: every candidate owns its own environment and ship.
This module therefore keeps the candidate explanation loop separate from the
SP-LIME selection step, while retaining the same coverage criterion.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, TYPE_CHECKING

import numpy as np

from Explanations.Lime.lime_model import RunValueClass
from Simulation.Auxilaries.io import load_env_ship
from Simulation.Auxilaries.sim_config import SimulationConfig
from Simulation.Environment.agent import Ship
from Simulation.Environment.environment import Environment

if TYPE_CHECKING:
  from Explanations.Lime.lime import LimeExplainer


@dataclass
class RunCandidate:
  """One run start state that can be given to SP-LIME."""

  id: str
  state: np.ndarray
  environment: Environment
  ship: Ship
  source_path: Path


@dataclass
class RunExplanation:
  """A selected explanation together with the run it explains."""

  id: str
  explanation: object
  state: np.ndarray
  environment: Environment
  ship: Ship


def _normalized_position(position, config: SimulationConfig) -> np.ndarray:
  map_min = np.asarray(config.boundary[0], dtype=float)
  map_size = np.asarray(config.boundary[1], dtype=float) - map_min
  return ((np.asarray(position, dtype=float) - map_min) / map_size) * 2 - 1


def build_run_start_state(
    environment: Environment,
    ship: Ship,
    config: SimulationConfig,
) -> np.ndarray:
  """Reconstruct the reduced 8-D observation at the start of a run.

  ``AgentController.generate_samples`` logs the observation immediately after
  creating a new training instance and before taking the first action.  At
  that point the elapsed step count is zero, so the state can be reconstructed
  from the saved environment and ship without requiring ``steps.npz``.
  """
  ship_position = _normalized_position(ship.state.position, config)
  end_position = _normalized_position(
    environment.end_poly.centroid.coords[0], config,
  )
  heading = float(ship.state.heading)
  return np.asarray([
    ship_position[0],
    ship_position[1],
    np.cos(heading),
    np.sin(heading),
    ship.state.velocity / ship.params.max_velocity,
    0.0,
    end_position[0],
    end_position[1],
  ], dtype=float)


def load_run_candidates(
    samples_path: str | Path,
    config: SimulationConfig,
) -> list[RunCandidate]:
  """Load one candidate per sample-run directory.

  Only ``scenario.json`` is loaded.  The recorded trajectory is not needed
  because candidates represent run starts rather than individual steps.
  """
  root = Path(samples_path)
  run_paths = sorted(
    (
      child for child in root.iterdir()
      if child.is_dir() and re.fullmatch(r"run\d+", child.name)
    ),
    key=lambda child: int(child.name[3:]),
  )

  candidates = []
  for run_path in run_paths:
    scenario_path = run_path / "scenario.json"
    if not scenario_path.is_file():
      raise FileNotFoundError(
        f"Run {run_path} is missing 'scenario.json'."
      )
    environment, ship = load_env_ship(scenario_path, config)
    candidates.append(RunCandidate(
      id=run_path.name,
      state=build_run_start_state(environment, ship, config),
      environment=environment,
      ship=ship,
      source_path=run_path,
    ))
  return candidates


def _predict_fn_for_target(
    pac,
    target: RunValueClass,
):
  if target is RunValueClass.REWARD_DELTA:
    return lambda states: pac.eval_run_perturbation_array(states)[:, 0]
  if target is RunValueClass.STEPCOUNT:
    return lambda states: pac.eval_run_perturbation_array(states)[:, 1]
  if target is RunValueClass.COLLISIONS:
    return pac.eval_run_collision_probability_array
  raise TypeError(f"Unsupported run target: {target!r}")


def _configure_pac(pac, environment: Environment, ship: Ship) -> None:
  pac.new_training_instance(
    pre_made_env=environment,
    pre_made_ship=ship,
  )


def _restore_pac(pac, environment: Environment, ship: Ship, steps: int) -> None:
  pac.new_training_instance(
    pre_made_env=environment,
    pre_made_ship=ship,
  )
  pac.sim_state.steps = steps


def _explain_candidate(
    lime: LimeExplainer,
    candidate: RunCandidate,
    target: RunValueClass,
    num_samples: int,
    discretize_continuous: bool,
):
  pac = lime.pac
  _configure_pac(pac, candidate.environment, candidate.ship)

  if target is RunValueClass.COLLISIONS:
    explainer = lime.get_explainer(
      mode="classification",
      class_names=["end", "island", "boundary", "time_out"],
      discretize_continuous=discretize_continuous,
    )
    kwargs = {"top_labels": 4}
  else:
    explainer = lime.get_explainer(
      mode="regression",
      discretize_continuous=discretize_continuous,
    )
    kwargs = {}

  explanation = explainer.explain_instance(
    candidate.state,
    predict_fn=_predict_fn_for_target(pac, target),
    num_features=8,
    num_samples=num_samples,
    sampling_method="lhs",
    **kwargs,
  )
  explanation._discretize_continuous = discretize_continuous
  explanation._island_mode = False
  return explanation


def _explanation_matrix(explanations: list, classification: bool):
  """Convert LIME feature weights into the SP-LIME matrix ``W``."""
  features: dict[str, int] = {}
  rows = []

  for explanation in explanations:
    labels = explanation.available_labels() if classification else [1]
    row: list[tuple[str, float]] = []
    for label in labels:
      row.extend(explanation.as_list(label=label))
    rows.append(row)
    for feature, _ in row:
      if feature not in features:
        features[feature] = len(features)

  matrix = np.zeros((len(rows), len(features)), dtype=float)
  for row_index, row in enumerate(rows):
    for feature, weight in row:
      matrix[row_index, features[feature]] += float(weight)
  return matrix


def _pick_indices(weights: np.ndarray, k: int) -> list[int]:
  """Apply the greedy coverage criterion used by bundled SP-LIME."""
  if k <= 0 or len(weights) == 0:
    return []
  k = min(k, len(weights))
  importance = np.sum(np.abs(weights), axis=0) ** 0.5
  remaining = set(range(len(weights)))
  selected: list[int] = []

  for _ in range(k):
    best_score = -np.inf
    best_index = None
    for index in remaining:
      covered = np.sum(np.abs(weights[selected + [index]]), axis=0) > 0
      score = float(np.dot(covered, importance))
      if score >= best_score:
        best_score = score
        best_index = index
    if best_index is None:
      break
    selected.append(best_index)
    remaining.remove(best_index)
  return selected


def select_run_explanations(
    lime: LimeExplainer,
    candidates: Iterable[RunCandidate],
    target: RunValueClass,
    k: int,
    num_samples: int = 500,
    discretize_continuous: bool = False,
) -> list[RunExplanation]:
  """Select up to ``k`` representative run-start explanations.

  Every candidate is explained in its own environment and ship.  The
  controller is restored to its original context after the complete operation
  and after any candidate-level failure.
  """
  if not isinstance(target, RunValueClass):
    raise TypeError(f"target must be a RunValueClass, got {type(target).__name__}")
  if k < 0:
    raise ValueError("k must be non-negative")
  if num_samples < 1:
    raise ValueError("num_samples must be positive")

  candidates = list(candidates)
  if not candidates or k == 0:
    return []

  pac = lime.pac
  if pac.sim_state is None:
    raise RuntimeError("The PerAgentController must have a training instance.")
  original_env = pac.sim_state.env
  original_ship = pac.sim_state.ship.copy()
  original_steps = pac.sim_state.steps
  explanations = []

  try:
    for candidate in candidates:
      explanations.append(_explain_candidate(
        lime, candidate, target, num_samples, discretize_continuous,
      ))
  finally:
    _restore_pac(pac, original_env, original_ship, original_steps)

  weights = _explanation_matrix(
    explanations,
    classification=target is RunValueClass.COLLISIONS,
  )
  selected_indices = _pick_indices(weights, k)

  return [
    RunExplanation(
      id=candidates[index].id,
      explanation=explanations[index],
      state=candidates[index].state.copy(),
      environment=deepcopy(candidates[index].environment),
      ship=candidates[index].ship.copy(),
    )
    for index in selected_indices
  ]


__all__ = [
  "RunCandidate",
  "RunExplanation",
  "build_run_start_state",
  "load_run_candidates",
  "select_run_explanations",
]
