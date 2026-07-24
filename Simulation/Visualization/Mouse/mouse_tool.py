from __future__ import annotations

import numpy as np

from Simulation.Visualization.Mouse.mouse_mode import MouseMode


class MouseTool:

  def __init__(self):
    self.mode: MouseMode | None = None
    self._dragging = False
    self._last_drag_pos: np.ndarray | None = None

  def set_mode(self, mode: MouseMode | None) -> None:
    """
    Change the active mouse mode.
    """

    if self.mode is mode:
      return

    if self.mode is not None:
      self.mode.on_deactivate()

    self.mode = mode

    if self.mode is not None:
      self._dragging = False
      self.mode.on_activate()

  def on_press(self, pos: np.ndarray) -> None:
    if self.mode is not None:
      self.mode.on_press(pos)

  def on_move(self, pos: np.ndarray, pressed: bool) -> None:
    if self.mode is None:
      return

    if pressed:
      if not self._dragging:
        self._dragging = True
        self._last_drag_pos = pos
        self.mode.on_drag_begin(pos)
        return

      delta = pos - self._last_drag_pos
      self._last_drag_pos = pos
      self.mode.on_drag(delta)

    else:
      self._dragging = False
      self._last_drag_pos = None
      self.mode.on_move(pos)

  def on_release(self, pos: np.ndarray) -> None:
    if self.mode is not None:
      if self._dragging:
        self.mode.on_drag_end(pos)
        self._dragging = False
        self._last_drag_pos = None
        
      self.mode.on_release(pos)

  @property
  def can_finish(self) -> bool:
    if self.mode is not None:
      return self.mode.can_finish
    else:
      return False
      
  def on_finish(self):
    if self.mode is not None:
      self.mode.on_finish()

