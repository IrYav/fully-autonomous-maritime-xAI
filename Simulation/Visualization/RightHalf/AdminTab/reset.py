"""Reset-point support for the environment editor.

The reset point is kept by :class:`EditorController`; this module contains
the small actions used by the editor buttons.  Keeping the actions here keeps
the UI free of knowledge about how a reset is committed to the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

from Simulation.Environment.agent import ShipParams, ShipState
from Simulation.Environment.environment import Environment


@dataclass
class ResetState:
  """A copy of the state to which an episode can be reset."""

  env: Environment | None = None
  ship_state: ShipState | None = None
  ship_params: ShipParams | None = None

  def copy(self) -> ResetState:
    return deepcopy(self)


def reset_state(controller) -> None:
  """Commit the saved environment while retaining the current ship pose."""
  controller.commit_reset_state()


def reset_ship(controller) -> None:
  """Commit the saved ship state while retaining the current environment."""
  controller.commit_reset_ship()
