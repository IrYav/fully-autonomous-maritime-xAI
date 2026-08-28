from enum import Enum, auto

from dataclasses import dataclass
import numpy as np

from PySide6.QtCore import QObject, Signal

class Source(Enum):
  LIME = auto()
  GROUND_TRUTH = auto()
  ERROR = auto()
  
class StepValueClass(Enum):
  HEADING = auto()
  THROTTLE = auto()
  LREWARD = auto()

class RunValueClass(Enum):
  REWARD_DELTA = auto()
  STEPCOUNT = auto()
  COLLISIONS = auto() #TODO Check prob dist, might need more than 1 col

@dataclass
class ValueField:
  positions: np.ndarray      # Shape: (N, 2)
  theta_values: np.ndarray   # Shape: (T,), maps index to angle, c.f. `selected_theta`
  values: np.ndarray         # Shape: (N, T, S, C)


class HeatmapVM(QObject):
  sourceChanged = Signal()
  valueClassChanged = Signal()
  thetaChanged = Signal()

  def __init__(
    self,
    field: ValueField,
    value_enum: type[Enum]
  ):
    super().__init__()

    self._field = field
    self._value_enum = value_enum

    self._selected_source = Source.LIME
    self._selected_class = list(value_enum)[0]
    self._selected_theta = 0

    self._source_idx = {
      src: i for i, src in enumerate(Source)
    }

    self._class_idx = {
      cls: i for i, cls in enumerate(value_enum)
    }
  
  def set_selected_source(self, source: Source):
    if source == self._selected_source:
      return

    self._selected_source = source
    self.sourceChanged.emit()

  def set_selected_class(self, cls: Enum):
    if cls == self._selected_class:
      return

    self._selected_class = cls
    self.valueClassChanged.emit()

  def set_selected_theta(self, theta_index: int):
    if theta_index == self._selected_theta:
      return

    self._selected_theta = theta_index
    self.thetaChanged.emit()

  @property
  def positions(self):
    return self._field.positions
  
  @property
  def theta_values(self):
    return self._field.theta_values

  @property
  def selected_theta(self):
    '''
      `theta_values[self.selected_theta]` has the actual angle.
    '''
    return self._selected_theta

  @property
  def visible_values(self):
    t = self._selected_theta
    s = self._source_idx[self._selected_source]
    c = self._class_idx[self._selected_class]

    return self._field.values[:, t, s, c]