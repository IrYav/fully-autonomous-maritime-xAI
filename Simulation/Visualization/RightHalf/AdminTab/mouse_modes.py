
from enum import Enum, auto

import numpy as np
from matplotlib.patches import PathPatch
from shapely import LineString, Point, Polygon
from shapely.affinity import rotate, scale, translate

from Simulation.Environment.environment import Points
from Simulation.Visualization.Mouse.mouse_mode import MouseMode
from Simulation.Visualization.RightHalf.AdminTab.draft_model import AreaObject, DraftShip
from Simulation.Visualization.RightHalf.AdminTab.editor_controller import EditorController
from Simulation.Visualization.VisHelpers.helpers import line_patch


class TransformMode(Enum):
  MOVE = auto()
  ROTATE = auto()
  SCALE = auto()


class CreateMode(MouseMode):
  '''
    CreateMode is used to create a polygon by sequentially
    creating the exterior points of it.
  '''
  curr_points: list[Points]
  curr_pos : Point | None
  ec: EditorController

  def __init__(self, ec: EditorController):
    super().__init__()
    self.ec = ec

  def on_activate(self):
    self._clear()
  
  def on_deactivate(self):
    self._clear()
    self.ec.clear_pv()

  def on_press(self, pos):
    self.curr_pos = Point(pos)

    if self._can_add:
      self.curr_points.append(self.curr_pos)
      self.ec.preview_changed.emit(self._get_mode_artist())

  def on_move(self, pos):
    self.curr_pos = Point(pos)
    artists = self._get_mode_artist()
    self.ec.preview_changed.emit(artists)

  def on_finish(self):
    if self.can_finish:
      self.ec.add_man_area(self.curr_points)
      self._clear()
      self.ec.clear_pv()

  @property
  def can_finish(self):
    return (
      len(self.curr_points) >= 3 and 
      Polygon(self.curr_points).is_valid
    )
  
  def _clear(self):
    self.curr_points = []
    self.curr_pos = None

  def _get_mode_artist(self) -> list[PathPatch]:

    if len(self.curr_points) == 0:
      #No draft at all
      return []
    
    if self._can_add:
      #Valid point
      last_part = line_patch(
        LineString([self.curr_points[-1], self.curr_pos]),
        color="#0072B2",
        alpha=0.5,
        linewidth=2,
      )
    else:
      #Invalid point
      last_part = line_patch(
        LineString([self.curr_points[-1], self.curr_pos]),
        color="#D55E00",
        alpha=0.5,
        linewidth=2,
      )
    if self.can_finish:
      #Valid draft
      parts = line_patch(
        LineString(self.curr_points),
        color="#0072B2",
        alpha=0.8,
        linewidth=3,
      )
    
    elif len(self.curr_points) > 1 :
      #Finish currently not possible, but additional points may correct it.
      parts = line_patch(
        LineString(self.curr_points),
        color="#E69F00",
        alpha=0.8,
        linewidth=3,
      )
    else: #No parts available
      return [last_part]
    
    return [parts, last_part]
    
  @property
  def _can_add(self) -> bool:

    #Basically no self-cuts
    return (
      len(self.curr_points) == 0 or
      LineString([*self.curr_points, self.curr_pos]).is_simple
    )
  
class SelectMode(MouseMode):
  ec: EditorController

  def __init__(self, ec: EditorController):
    super().__init__()
    self.ec = ec
    self.transform_mode = TransformMode.MOVE
    self._drag_start_pos: np.ndarray | None = None
    self._drag_last_pos: np.ndarray | None = None
    self._start_polygon: Polygon | None = None
    self._start_centroid: Point | None = None
    self._start_distance: float = 0.0

  def set_transform_mode(self, mode: TransformMode):
    self.transform_mode = mode

  def on_deactivate(self):
    self._clear_drag_state()
    self.ec.selected = None
    self.ec.clear_pv()

  def on_move(self, pos):
    '''
      Highlight the currently hovered polygon, iff None is selected.
    '''
    if self.ec.selected is not None:
      return #Only highlight selected one
    elif self._top_ao(pos) is not None:
      self.ec.preview_changed.emit(self._get_mode_artist())
    else:
      self.ec.clear_pv()

  def on_press(self, pos):
    top_ao = self._top_ao(pos)
    if self.ec.selected is not top_ao:
      self.ec.selected = top_ao #Can assign None again.
      if self.ec.selected is not None:
        self.ec.to_front(self.ec.selected)
      self.ec.preview_changed.emit(self._get_mode_artist())

  def on_drag_begin(self, pos):
    if self.ec.selected is None:
      self.ec.selected = self._top_ao(pos)
    if self.ec.selected is None:
      return

    self._drag_start_pos = np.asarray(pos, dtype=float)
    self._drag_last_pos = self._drag_start_pos.copy()
    self._start_polygon = self.ec.selected.polygon
    self._start_centroid = self._start_polygon.centroid
    self._start_distance = self._distance_to_centroid(self._drag_start_pos)

  def on_drag(self, delta):
    if self.ec.selected is None:
      return

    delta_arr = np.asarray(delta, dtype=float)
    self._drag_last_pos = self._drag_last_pos + delta_arr if self._drag_last_pos is not None else delta_arr

    if self.transform_mode is TransformMode.MOVE:
      if isinstance(self.ec.selected, DraftShip):
        self.ec.selected.state.position = self.ec.selected.state.position + delta_arr
      else:
        self.ec.selected.polygon = translate(self.ec.selected.polygon, *delta_arr)
    elif self.transform_mode is TransformMode.ROTATE:
      if self._start_centroid is None or self._drag_start_pos is None:
        return
      angle_start = self._angle_to_centroid(self._drag_start_pos)
      angle_current = self._angle_to_centroid(self._drag_last_pos)
      delta_angle = angle_current - angle_start
      if isinstance(self.ec.selected, DraftShip):
        #Ship rotation is treated differently, we set the angle rather than rotating the polygon.
        self.ec.selected.state.heading = angle_current
      else:
        self.ec.selected.polygon = rotate(
          self._start_polygon,
          delta_angle,
          origin=self._start_centroid,
          use_radians=True,
        )
    elif self.transform_mode is TransformMode.SCALE:
      if isinstance(self.ec.selected, DraftShip):
        return
      if self._start_centroid is None or self._drag_start_pos is None:
        return
      start_vector = self._drag_start_pos - np.asarray((self._start_centroid.x, self._start_centroid.y), dtype=float)
      current_vector = self._drag_last_pos - np.asarray((self._start_centroid.x, self._start_centroid.y), dtype=float)
      start_length = float(np.linalg.norm(start_vector))
      if start_length == 0:
        return
      factor = float(np.linalg.norm(current_vector)) / start_length
      self.ec.selected.polygon = scale(
        self._start_polygon,
        xfact=factor,
        yfact=factor,
        origin=self._start_centroid,
      )

    self.ec.draft_changed.emit()
    self.ec.preview_changed.emit(self._get_mode_artist())

  def on_drag_end(self, pos):
    self._clear_drag_state()

  def on_release(self, pos):
    self._clear_drag_state()

  def _clear_drag_state(self):
    self._drag_start_pos = None
    self._drag_last_pos = None
    self._start_polygon = None
    self._start_centroid = None
    self._start_distance = 0.0

  def _distance_to_centroid(self, pos: np.ndarray) -> float:
    if self._start_centroid is None:
      return 0.0
    centroid = np.asarray((self._start_centroid.x, self._start_centroid.y), dtype=float)
    return float(np.linalg.norm(pos - centroid))

  def _angle_to_centroid(self, pos: np.ndarray) -> float:
    if self._start_centroid is None:
      return 0.0
    centroid = np.asarray((self._start_centroid.x, self._start_centroid.y), dtype=float)
    v = pos - centroid
    return float(np.arctan2(v[1], v[0]))

  #def _set_no_selection

  def _top_ao(self, pos) -> AreaObject | DraftShip | None:
    '''
      Returns first `AreaObject` from the top (or `None`)
    '''
    pos = Point(pos)

    if self.ec.draft.ship is not None and pos.within(self.ec.draft.ship.polygon):
      return self.ec.draft.ship

    for ao in reversed(self.ec.draft):
      if pos.within(ao.polygon):
        return ao
      
    return None
  
  def _get_mode_artist(self) -> list[PathPatch]:
      if self.ec.selected is None:
        return []
      return [line_patch(
        self.ec.selected.polygon.exterior,
        color="#0072B2",
      )]
