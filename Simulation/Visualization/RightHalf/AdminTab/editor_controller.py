from collections.abc import Callable

import numpy as np
from shapely import Point, Polygon, LineString

from PySide6.QtCore import QObject, Signal

from Simulation.Environment.environment import Environment, island
from Simulation.Visualization.RightHalf.AdminTab.draft_model import AreaObject, Material, EnvironmentDraft


class EditorController(QObject):
  '''
    Centralizes all data change methods for the editor to access and connect.
  '''
  env: Environment | None
  draft : EnvironmentDraft
  material : Material
  selected : AreaObject | None
  changed = Signal()
  preview_changed = Signal(object) #Contains list[PathPatch]
  env_changed = Signal()
  def __init__(
      self,
      on_commit):
    super().__init__()
    self._on_commit = on_commit
    self.selected = None
    self.material = Material.LAND

  #--- Viewer API ---
  def on_auto(self):
    '''
      Create a randomly generated polygon, similar to the distribution used for `Environment`
    '''
    ao = self._create_rand_area()

    self.draft.add(ao)
    self.selected = ao
    self.changed.emit()

  def on_set_material(self, new_mat : Material):
    '''
      Change material for the controller and for the currently selected island.
    '''
    self.material = new_mat
    if self.selected is not None:
      self.selected.material = new_mat
    self.changed.emit()

  def on_delete(self):
    self.draft.remove(self.selected.id)
    self.selected = None
    self.changed.emit()
    self.preview_changed.emit([]) # Remove the selected marker

  def on_commit(self):
    new_env = self.draft.draft_to_env(self.env)
    self._on_commit(new_env)

  #--- Controller functions ---
  def add_man_area(self, points: list[Point]):
    polygon = Polygon(points)
    self.draft.add(AreaObject(self.draft.next_id, polygon, self.material))
    self.changed.emit()

  def set_env(self, env: Environment | None):
    if env is not None:
      self.draft = EnvironmentDraft.draft_from_env(env)
      self.env = env
    else:
      self.draft = EnvironmentDraft() # Empty container.
      
    self.changed.emit()

  def to_front(self, ao: AreaObject):
    self.draft.move_to_front(ao.id)

  #--- Local Methods ---
  def _create_rand_area(self):
    def isleGen():
      minx, miny, maxx, maxy = self.env.boundary.bounds
      boundary = np.array([(minx, miny),(maxx, maxy)])
      
      scale = np.abs(boundary[1] - boundary[0])
      islePoints = np.random.random((2,2)) * scale + boundary[0]
      radius = np.clip(
        np.random.uniform(0.2, 0.35) * np.linalg.norm(islePoints[1] - islePoints[0]),
        0.05 * np.mean(scale),
        0.25 * np.mean(scale)
      )
      
      poly, _ = island(islePoints,radius)
      return poly
    
    poly = isleGen()
    while not poly.is_valid:
      poly = isleGen()

    return AreaObject(self.draft.next_id, poly, self.material)